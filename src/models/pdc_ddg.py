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
        try:   # code version compatibility
            self.learnable_var = cfg.encoder.learnable_var
        except:
            self.learnable_var = cfg.learnable_var

        if self.learnable_var:
            self.var_embed = nn.Embedding(max_aa_types, 3)    # each residue has a different positional variance

        # Pretrain
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
        else:
            self.rde = None

        # Encoding
        self.single_encoder = PerResidueEncoder(feat_dim=dim, max_num_atoms=5)  # N, CA, C, O, CB,
        self.mut_bias = nn.Embedding(num_embeddings=2, embedding_dim=dim, padding_idx=0, )
        self.pair_encoder = ResiduePairEncoder(feat_dim=cfg.encoder.pair_feat_dim, max_num_atoms=5,)  # N, CA, C, O, CB,

        if self.resolution != 'CA':
            self.token_emb = nn.Embedding(len(HeavyAtom2int) + 1, dim)  # 6 atom types
        self.attn_encoder = GAEncoder(node_feat_dim=dim, pair_feat_dim=cfg.encoder.pair_feat_dim, num_layers=cfg.encoder.num_layers,)

        self.spatial_project = EGNN_Network(dim=dim, depth=cfg.encoder.num_layers, num_nearest_neighbors=cfg.encoder.num_nearest_neighbors, norm_coors=cfg.encoder.norm_coors,
                                            update_coors_mean=cfg.encoder.update_coors_mean, update_coors_var=cfg.encoder.update_coors_var)

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

    def encode(self, batch, mode='wt'):
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
        pos_atom = batch['pos_atoms']

        res_feat = self.attn_encoder(pos_atoms=pos_atom, res_feat=res_feat, pair_feat=z, mask=mask_residue)

        # Update coordinates
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

        res_feat, coors_out, coors_var = self.spatial_project(coors_mean=pos_atom, coors_var=pos_atom_var, feats=res_feat, mask=mask)

        return res_feat, coors_var

    def forward(self, batch):
        loss_dict = {'pos_refine': torch.tensor(0.0)}
        batch_wt = {k: v for k, v in batch.items()}
        batch_mt = {k: v for k, v in batch.items()}
        batch_mt['aa'] = batch_mt['aa_mut']

        h_wt = self.encode(batch_wt, 'wt')[0]
        h_mt = self.encode(batch_mt, 'mut')[0]

        H_mt, H_wt = h_mt.max(dim=1)[0], h_wt.max(dim=1)[0]
        ddg_pred = self.ddg_readout(H_mt - H_wt).squeeze(-1)
        ddg_pred_inv = self.ddg_readout(H_wt - H_mt).squeeze(-1)
        loss = (F.mse_loss(ddg_pred, batch['ddG']) + F.mse_loss(ddg_pred_inv, -batch['ddG'])) / 2

        loss_dict['regression'] = loss
        out_dict = {'ddG_pred': ddg_pred.detach().clone(), 'ddG_true': batch['ddG'], }
        return loss_dict, out_dict
