import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.encoders.egnn_attn import GAEncoder
from src.modules.encoders.pair import ResiduePairEncoder
from src.modules.encoders.single import PerResidueEncoder
from src.utils.protein.constants import BBHeavyAtom, HeavyAtom2int
from .rde import CircularSplineRotamerDensityEstimator
from src.models.rde_mlm import MaskedLanguageModelingDensityEstimator
from src.models.pdc import ProbabilityDensityCloud
from src.modules.encoders.egnn import EGNN_Network


class DDG_PDC_Network(nn.Module):

    def __init__(self, cfg, max_aa_types=22):
        super().__init__()
        self.use_plm = cfg.use_plm
        self.resolution = cfg.resolution
        dim = 1280 if self.use_plm else cfg.encoder.node_feat_dim
        self.learnable_var = cfg.encoder.learnable_var
        if self.learnable_var:
            self.var_embed = nn.Embedding(max_aa_types, 3)    # each residue has a different positional variance

        # Pretrain
        self.rde = None
        if cfg.checkpoint.path:
            self.ckpt_type = cfg.checkpoint.type
            print(f'Loading {cfg.checkpoint.type} from {cfg.checkpoint.path}')
            ckpt = torch.load(cfg.checkpoint.path, map_location='cpu')
            if self.ckpt_type == 'CircularSplineRotamerDensityEstimator':
                self.rde = CircularSplineRotamerDensityEstimator(ckpt['config'].model)
            elif self.ckpt_type == 'MaskedLanguageModelingDensityEstimator':
                self.rde = MaskedLanguageModelingDensityEstimator(ckpt['config'].model)
            elif self.ckpt_type == 'ProbabilityDensityCloud':
                self.rde = ProbabilityDensityCloud(ckpt['config'].model)
            self.rde.load_state_dict(ckpt['model'])
            for p in self.rde.parameters():
                p.requires_grad_(False)
            self.single_fusion = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))

        # Encoding
        self.single_encoder = PerResidueEncoder(feat_dim=dim, max_num_atoms=5)  # N, CA, C, O, CB,
        self.mut_bias = nn.Embedding(num_embeddings=2, embedding_dim=dim, padding_idx=0, )
        self.pair_encoder = ResiduePairEncoder(feat_dim=cfg.encoder.pair_feat_dim, max_num_atoms=5,)  # N, CA, C, O, CB,
        if self.resolution != 'CA':
            self.token_emb = nn.Embedding(len(HeavyAtom2int) + 1, dim)  # 6 atom types
        self.attn_encoder = GAEncoder(node_feat_dim=dim, pair_feat_dim=cfg.encoder.pair_feat_dim, num_layers=cfg.encoder.num_layers,)

        # Refinement module
        if cfg.pos.mask_length > 0:
            self.recycle = cfg.pos.recycle
            self.mask_wt = cfg.pos.mask_wt
            self.spatial_project = EGNN_Network(dim=dim, depth=cfg.encoder.refine_num_layers, num_nearest_neighbors=cfg.encoder.num_nearest_neighbors,
                                                norm_coors=cfg.encoder.norm_coors, update_coors_mean=cfg.encoder.update_coors_mean, update_coors_var=cfg.encoder.update_coors_var, )
        else:
            self.recycle = 0
            self.mask_wt = False

        # Pred
        self.ddg_readout = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim // 4), nn.ReLU(), nn.Linear(dim // 4, 1))

    def _encode_rde(self, batch, mask_extra=None):
        batch = {k: v for k, v in batch.items()}
        batch['chi_corrupt'] = batch['chi']
        batch['chi_masked_flag'] = batch['mut_flag']
        if mask_extra is not None:
            batch['mask_atoms'] = batch['mask_atoms'] * mask_extra[:, :, None]
        with torch.no_grad():
            return self.rde.encode(batch)

    def encode(self, batch, mode):
        mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
        chi = batch['chi'] * (1 - batch['mut_flag'].float())[:, :, None]

        res_feat = self.single_encoder(aa=batch['aa'], phi=batch['phi'], phi_mask=batch['phi_mask'], psi=batch['psi'], psi_mask=batch['psi_mask'], chi=chi,
                                       chi_mask=batch['chi_mask'], mask_residue=mask_residue, )  # (N, L, F)
        if self.use_plm:
            res_feat += batch['plm_wt'] if mode == 'wt' else batch['plm_mut']

        if self.rde is not None:
            x_pret = self._encode_rde(batch)
            if self.ckpt_type == 'ProbabilityDensityCloud':
                x_pret = x_pret[0]
            res_feat = self.single_fusion(torch.cat([res_feat, x_pret], dim=-1))

        res_feat = res_feat + self.mut_bias(batch['mut_flag'].long())
        z = self.pair_encoder(aa=batch['aa'], res_nb=batch['res_nb'], chain_nb=batch['chain_nb'], pos_atoms=batch['pos_atoms'], mask_atoms=batch['mask_atoms'], )
        res_feat = self.attn_encoder(pos_atoms=batch['pos_atoms'], res_feat=res_feat, pair_feat=z, mask=mask_residue)
        return res_feat

    def refine(self, res_feat, pos_change_flag, batch):
        if self.resolution != 'CA':
            res_feat = self.token_emb(batch['type_atoms']) + res_feat.unsqueeze(-2)  # (N, L, A, F)
            pos_atom = torch.flatten(batch['pos_atoms'], start_dim=1, end_dim=2)  # (N, L, A, 3) -> (N, L * A, 3)
            if not self.learnable_var:
                pos_atom_var = torch.flatten(batch['pos_atom_var'], start_dim=1, end_dim=2)  # (N, L, A, 3) -> (N, L * A, 3)
            else:
                pos_atom_var = self.var_embed(batch['aa']).repeat(1, batch['pos_atoms'].shape[-2], 1)  # (B, L * A, 3)

            res_feat = torch.flatten(res_feat, start_dim=1, end_dim=2)
            mask = torch.flatten(batch['mask_atoms'], start_dim=1, end_dim=2)
        else:
            pos_atom = batch['pos_atoms'][:, :, BBHeavyAtom.CA]
            mask = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
            if not self.learnable_var:
                pos_atom_var = batch['pos_atom_var'][:, :, BBHeavyAtom.CA]
            else:
                pos_atom_var = self.var_embed(batch['aa'])  # (B, L, 3)

        coors_out = self.spatial_project(coors_mean=pos_atom, coors_var=pos_atom_var, feats=res_feat, mask=mask, pos_change_flag=pos_change_flag)[1]
        return coors_out

    def forward(self, batch, return_pos=False):
        batch_wt = {k: v for k, v in batch.items()}
        batch_mt = {k: v for k, v in batch.items()}

        ###############################################
        ## wide type
        ###############################################
        loss_coors = []
        if self.recycle > 0:
            if self.resolution != 'CA':     # (B, 3) or (B, 5, 3) for different resolution
                pos_change_flag = batch['pos_change_flag'].repeat(1, batch['pos_atoms'].shape[-2])
                pos_gt = torch.flatten(batch['pos_gt'], start_dim=1, end_dim=2)[pos_change_flag]
            else:
                pos_change_flag = batch['pos_change_flag']
                pos_gt = batch['pos_gt'][:, :, BBHeavyAtom.CA][pos_change_flag]

            for _ in range(self.recycle):
                h_wt_0 = self.encode(batch_wt, 'wt')
                c_wt = self.refine(h_wt_0, pos_change_flag, batch_wt)
                loss_coors.append(torch.sqrt(((pos_gt - c_wt[pos_change_flag]) ** 2).sum(dim=-1) + 1e-10).mean())   # loss for each iteration
                batch_wt['pos_atoms'] = c_wt.detach().clone().reshape(batch_wt['pos_atoms'].shape)

            batch_wt['pos_atoms'] = batch['pos_gt'].clone()                            # use crystal structures (given)

        loss_dict = {'pos_refine': torch.stack(loss_coors).sum() if len(loss_coors) > 0 else torch.tensor(0.0)}
        h_wt = self.encode(batch_wt, 'wt')

        ###############################################
        ## mutation type
        ###############################################
        batch_mt['aa'] = batch_mt['aa_mut']
        if self.mask_wt:
            for _ in range(self.recycle):
                with torch.no_grad():   # no gradient
                    h_mt_0 = self.encode(batch_mt, 'mut')
                    c_mt = self.refine(h_mt_0, pos_change_flag, batch_mt)
                    batch_mt['pos_atoms'] = c_mt.detach().clone().reshape(batch_mt['pos_atoms'].shape)
        h_mt = self.encode(batch_mt, 'mut')

        ###############################################
        ## ddG
        ###############################################
        H_mt, H_wt = h_mt.max(dim=1)[0], h_wt.max(dim=1)[0]
        ddg_pred = self.ddg_readout(H_mt - H_wt).squeeze(-1)
        ddg_pred_inv = self.ddg_readout(H_wt - H_mt).squeeze(-1)
        loss = (F.mse_loss(ddg_pred, batch['ddG']) + F.mse_loss(ddg_pred_inv, -batch['ddG'])) / 2

        loss_dict['regression'] = loss
        out_dict = {'ddG_pred': ddg_pred.detach().clone(), 'ddG_true': batch['ddG'], }

        if return_pos:  # no pos_mt_true
            out_dict.update({'pos': batch['pos_atoms'], 'pos_wt_pred': batch_wt['pos_atoms'], 'pos_mt_pred': batch_mt['pos_atoms'], 'aa': batch_wt['aa'],
                             'aa_mut': batch_mt['aa'], 'resseq': batch['resseq'], 'chain_nb': batch['chain_nb']})

        return loss_dict, out_dict
