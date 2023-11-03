import torch
import torch.nn as nn
import torch.nn.functional as F

from rde.modules.encoders.egnn_attn import GAEncoder
from rde.modules.encoders.pair import ResiduePairEncoder
from rde.modules.encoders.single import PerResidueEncoderV1
from rde.utils.protein.constants import BBHeavyAtom, HeavyAtom2int
from rde.modules.encoders.egnn import EGNN_Network


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
        self.feat_extractor = None
        if cfg.checkpoint.path:
            self.ckpt_type = cfg.checkpoint.type
            print(f'Loading {cfg.checkpoint.type} from {cfg.checkpoint.path}')
            ckpt = torch.load(cfg.checkpoint.path, map_location='cpu')
            if self.ckpt_type == 'RDE':
                from rde.models import CircularSplineRotamerDensityEstimator
                self.feat_extractor = CircularSplineRotamerDensityEstimator(ckpt['config'].model)
            elif self.ckpt_type == 'BackboneNetwork':
                from rde.models.backbone import BackboneNetwork
                self.feat_extractor = BackboneNetwork(ckpt['config'].model)
            elif self.ckpt_type == 'Equiformer':
                from rde.models.equiformer import EquiformerNet
                self.feat_extractor = EquiformerNet(ckpt['config'].model)
            elif self.ckpt_type == 'PDCNet':
                from rde.models.pdc import PDC_Network
                self.feat_extractor = PDC_Network(ckpt['config'].model)

            # https://stackoverflow.com/questions/63057468/how-to-ignore-and-initialize-missing-keys-in-state-dict
            self.feat_extractor.load_state_dict(ckpt['model'], strict=False)   # ignore unmatched keys
            for p in self.feat_extractor.parameters():
                p.requires_grad_(False)
            if 'use_plm' in ckpt['config'].model and ckpt['config'].model.use_plm and self.ckpt_type != 'Equiformer':
                self.single_fusion = nn.Sequential(nn.Linear(dim + 1280, dim), nn.ReLU(), nn.Linear(dim, dim))
            else:
                self.single_fusion = nn.Sequential(nn.Linear(dim + ckpt['config'].model.encoder.node_feat_dim, dim), nn.ReLU(), nn.Linear(dim, dim))

        # Encoding
        self.single_encoder = PerResidueEncoderV1(feat_dim=dim, max_num_atoms=5)  # N, CA, C, O, CB,
        self.mut_bias = nn.Embedding(num_embeddings=2, embedding_dim=dim, padding_idx=0, )
        self.pair_encoder = ResiduePairEncoder(feat_dim=cfg.encoder.pair_feat_dim, max_num_atoms=5,)  # N, CA, C, O, CB,

        if self.resolution != 'CA':
            self.token_emb = nn.Embedding(len(HeavyAtom2int) + 1, dim)  # 6 atom types
        self.attn_encoder = GAEncoder(node_feat_dim=dim, pair_feat_dim=cfg.encoder.pair_feat_dim, num_layers=cfg.encoder.num_layers,)

        # self.spatial_project = EGNN_Network(dim=dim, depth=cfg.encoder.num_layers, num_nearest_neighbors=cfg.encoder.num_nearest_neighbors, norm_coors=cfg.encoder.norm_coors,
        #                                     update_coors_mean=cfg.encoder.update_coors_mean, update_coors_var=cfg.encoder.update_coors_var)

        # Pred
        self.ddg_readout = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim // 4), nn.ReLU(), nn.Linear(dim // 4, 1))

    def _encode_pretrain(self, batch, mask_extra=None):
        batch = {k: v for k, v in batch.items()}
        # for RDE
        if self.ckpt_type in ['RDE', 'PDCNet']:
            batch['chi_corrupt'] = batch['chi']                    # use real chi with no corruption
            batch['chi_masked_flag'] = batch['mut_flag']

        if mask_extra is not None:
            batch['mask_atoms'] = batch['mask_atoms'] * mask_extra[:, :, None]

        with torch.no_grad():
            if self.ckpt_type == 'PDCNet':
                return self.feat_extractor.encode(batch, task='chi_angle')
            return self.feat_extractor.encode(batch)

    def encode(self, batch, mode='wt'):
        mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
        chi = batch['chi'] * (1 - batch['mut_flag'].float())[:, :, None]
        res_feat = self.single_encoder(aa=batch['aa'], phi=batch['phi'], phi_mask=batch['phi_mask'], psi=batch['psi'], psi_mask=batch['psi_mask'], chi=chi,
                                       chi_mask=batch['chi_mask'], mask_residue=mask_residue, )  # (N, L, F)
        if self.use_plm:
            res_feat += batch['plm_wt'] if mode == 'wt' else batch['plm_mut']

        if self.feat_extractor is not None:
            x_pret = self._encode_pretrain(batch)
            res_feat = self.single_fusion(torch.cat([res_feat, x_pret], dim=-1))

        res_feat = res_feat + self.mut_bias(batch['mut_flag'].long())
        z = self.pair_encoder(aa=batch['aa'], res_nb=batch['res_nb'], chain_nb=batch['chain_nb'], pos_atoms=batch['pos_atoms'], mask_atoms=batch['mask_atoms'], )
        res_feat = self.attn_encoder(pos_atoms=batch['pos_atoms'], res_feat=res_feat, pair_feat=z, mask=mask_residue)

        # # Update coordinates
        # if self.resolution != 'CA':
        #     res_feat = self.token_emb(batch['type_atoms']) + res_feat.unsqueeze(-2)  # (N, L, A, F)
        #     pos_atom = torch.flatten(batch['pos_atoms'], start_dim=1, end_dim=2)  # (N, L, A, 3) -> (N, L * A, 3)
        #     if not self.learnable_var:
        #         pos_atom_var = torch.flatten(batch['pos_atom_var'], start_dim=1, end_dim=2)  # (N, L, A, 3) -> (N, L * A, 3)
        #     else:
        #         pos_atom_var = self.var_embed(batch['aa']).repeat(1, batch['pos_atoms'].shape[-2], 1)  # (B, L * A, 3)
        #
        #     res_feat = torch.flatten(res_feat, start_dim=1, end_dim=2)
        #     mask = torch.flatten(batch['mask_atoms'], start_dim=1, end_dim=2)
        # else:
        #     pos_atom = batch['pos_atoms'][:, :, BBHeavyAtom.CA]
        #     mask = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
        #     if not self.learnable_var:
        #         pos_atom_var = batch['pos_atom_var'][:, :, BBHeavyAtom.CA]
        #     else:
        #         pos_atom_var = self.var_embed(batch['aa'])  # (B, L, 3)
        #
        # res_feat, coors_out, coors_var = self.spatial_project(coors_mean=pos_atom, coors_var=pos_atom_var, feats=res_feat, mask=mask)
        coors_var = 0
        return res_feat, coors_var

    def forward(self, batch):
        batch_wt = {k: v for k, v in batch.items()}
        batch_mt = {k: v for k, v in batch.items()}
        batch_mt['aa'] = batch_mt['aa_mut']

        h_wt = self.encode(batch_wt, 'wt')[0]
        h_mt = self.encode(batch_mt, 'mut')[0]

        H_mt, H_wt = h_mt.max(dim=1)[0], h_wt.max(dim=1)[0]
        ddg_pred = self.ddg_readout(H_mt - H_wt).squeeze(-1)
        ddg_pred_inv = self.ddg_readout(H_wt - H_mt).squeeze(-1)
        loss = (F.mse_loss(ddg_pred, batch['ddG']) + F.mse_loss(ddg_pred_inv, -batch['ddG'])) / 2

        loss_dict = {'regression': loss}
        out_dict = {'ddG_pred': ddg_pred.detach().clone(), 'ddG_true': batch['ddG'], }
        return loss_dict, out_dict
