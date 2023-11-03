import torch
import torch.nn as nn
import torch.nn.functional as F

from rde.modules.common.geometry import construct_3d_basis, angstrom_to_nm
from rde.modules.encoders.attn import GAEncoder
from rde.modules.encoders.pair import ResiduePairEncoder
from rde.modules.encoders.single import PerResidueEncoderV2
from rde.utils.protein.constants import BBHeavyAtom, HeavyAtom2int
from .rde import CircularSplineRotamerDensityEstimator
from rde.models.backbone import BackboneNetwork
from rde.modules.encoders.egnn import EGNN_Network


class DG_PDC_Network(nn.Module):

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
        self.rde = None
        if cfg.checkpoint.path:
            self.ckpt_type = cfg.checkpoint.type
            print(f'Loading {cfg.checkpoint.type} from {cfg.checkpoint.path}')
            ckpt = torch.load(cfg.checkpoint.path, map_location='cpu')
            if self.ckpt_type == 'CircularSplineRotamerDensityEstimator':
                self.rde = CircularSplineRotamerDensityEstimator(ckpt['config'].model)
            elif self.ckpt_type == 'BackboneNetwork':
                self.rde = BackboneNetwork(ckpt['config'].model)
            elif self.ckpt_type == 'Equiformer':
                from rde.models.equiformer import EquiformerNet
                self.rde = EquiformerNet(ckpt['config'].model)

            self.rde.load_state_dict(ckpt['model'], strict=False)
            for p in self.rde.parameters():
                p.requires_grad_(False)
            if 'use_plm' in ckpt['config'].model and ckpt['config'].model.use_plm and self.ckpt_type != 'Equiformer':
                self.single_fusion = nn.Sequential(nn.Linear(dim + 1280, dim), nn.ReLU(), nn.Linear(dim, dim))
            else:
                self.single_fusion = nn.Sequential(nn.Linear(dim + ckpt['config'].model.encoder.node_feat_dim, dim), nn.ReLU(), nn.Linear(dim, dim))

        # Encoding
        self.single_encoder = PerResidueEncoderV2(feat_dim=dim, max_num_atoms=5)  # N, CA, C, O, CB,
        self.pair_encoder = ResiduePairEncoder(feat_dim=cfg.encoder.pair_feat_dim, max_num_atoms=5,)  # N, CA, C, O, CB,

        if self.resolution != 'CA':
            self.token_emb = nn.Embedding(len(HeavyAtom2int) + 1, dim)  # 6 atom types
        self.attn_encoder = GAEncoder(node_feat_dim=dim, pair_feat_dim=cfg.encoder.pair_feat_dim, num_layers=cfg.encoder.num_layers,)

        self.spatial_project = EGNN_Network(dim=dim, depth=cfg.encoder.num_layers, num_nearest_neighbors=cfg.encoder.num_nearest_neighbors, norm_coors=cfg.encoder.norm_coors,
                                            update_coors_mean=cfg.encoder.update_coors_mean, update_coors_var=cfg.encoder.update_coors_var, dropout=cfg.encoder.dropout)

        # Pred
        self.dg_readout = nn.Sequential(nn.Dropout(cfg.encoder.dropout), nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim // 4), nn.ReLU(), nn.Linear(dim // 4, 1))

    def _encode_rde(self, batch, mask_extra=None):
        batch = {k: v for k, v in batch.items()}
        if self.ckpt_type == 'CircularSplineRotamerDensityEstimator':
            batch['chi_corrupt'] = batch['chi']
            batch['chi_masked_flag'] = torch.zeros_like(batch['aa'])

        if mask_extra is not None:
            batch['mask_atoms'] = batch['mask_atoms'] * mask_extra[:, :, None]
        with torch.no_grad():
            return self.rde.encode(batch)

    def encode(self, batch):
        R = construct_3d_basis(batch['pos_heavyatom'][:, :, BBHeavyAtom.CA], batch['pos_heavyatom'][:, :, BBHeavyAtom.C], batch['pos_heavyatom'][:, :, BBHeavyAtom.N], )
        t = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        t = angstrom_to_nm(t)

        x = self.single_encoder(aa=batch['aa'], res_nb=batch['res_nb'], chain_nb=batch['chain_nb'], pos_atoms=batch['pos_atoms'], mask_atoms=batch['mask_atoms'], R=R, t=t,
                                fragment_type=batch['fragment_type'])
        if self.use_plm:             # comply with downstream ddG task
            x += batch['plm_wt']

        if self.rde is not None:
            x_pret = self._encode_rde(batch)
            x = self.single_fusion(torch.cat([x, x_pret], dim=-1))

        z = self.pair_encoder(aa=batch['aa'], res_nb=batch['res_nb'], chain_nb=batch['chain_nb'], pos_atoms=batch['pos_atoms'], mask_atoms=batch['mask_atoms'], )
        mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
        x = self.attn_encoder(res_feat=x, pair_feat=z, mask=mask_residue, R=R, t=t)

        # Update coordinates
        if self.resolution != 'CA':
            x = self.token_emb(batch['type_atoms']) + x.unsqueeze(-2)  # (N, L, A, F)
            pos_atom = torch.flatten(batch['pos_atoms'], start_dim=1, end_dim=2)  # (N, L, A, 3) -> (N, L * A, 3)
            if not self.learnable_var:
                pos_atom_var = torch.flatten(batch['pos_atom_var'], start_dim=1, end_dim=2)  # (N, L, A, 3) -> (N, L * A, 3)
            else:
                pos_atom_var = self.var_embed(batch['aa']).repeat(1, batch['pos_atoms'].shape[-2], 1)  # (B, L * A, 3)

            x = torch.flatten(x, start_dim=1, end_dim=2)
            mask = torch.flatten(batch['mask_atoms'], start_dim=1, end_dim=2)
        else:
            pos_atom = batch['pos_atoms'][:, :, BBHeavyAtom.CA]
            mask = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
            if not self.learnable_var:
                pos_atom_var = batch['pos_atom_var'][:, :, BBHeavyAtom.CA]
            else:
                pos_atom_var = self.var_embed(batch['aa'])  # (B, L, 3)

        x, coors_out, coors_var = self.spatial_project(coors_mean=pos_atom, coors_var=pos_atom_var, feats=x, mask=mask)
        return x

    def forward(self, batch):
        h_wt = self.encode(batch)
        H_wt = h_wt.max(dim=1)[0]
        dg_pred = self.dg_readout(H_wt).squeeze(-1)
        loss = F.mse_loss(dg_pred, batch['dG'])

        loss_dict = {'regression': loss}
        out_dict = {'dG_pred': dg_pred.detach().clone(), 'dG_true': batch['dG'], }
        return loss_dict, out_dict
