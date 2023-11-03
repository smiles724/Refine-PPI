import torch
import torch.nn as nn
import torch.nn.functional as F

from rde.models.backbone import BackboneNetwork
from rde.modules.common.geometry import construct_3d_basis, angstrom_to_nm
from rde.modules.encoders.single import PerResidueEncoderV2
from rde.utils.protein.constants import BBHeavyAtom
from .rde import CircularSplineRotamerDensityEstimator


class DG_Equiformer(nn.Module):

    def __init__(self, cfg, ):
        super().__init__()
        self.use_plm = cfg.use_plm
        self.cfg.type = cfg.type
        dim = cfg.encoder.node_feat_dim
        input_dim = 1280 if self.use_plm else cfg.encoder.input_feat_dim
        dropout_ = cfg.encoder.dropout

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
                self.single_fusion = nn.Sequential(nn.Linear(input_dim + 1280, dim), nn.ReLU(), nn.Linear(dim, dim))
            else:
                self.single_fusion = nn.Sequential(nn.Linear(input_dim + ckpt['config'].model.encoder.node_feat_dim, dim), nn.ReLU(), nn.Linear(dim, dim))

        # Encoding
        self.single_encoder = PerResidueEncoderV2(feat_dim=input_dim, max_num_atoms=5)  # N, CA, C, O, CB
        if cfg.type == 'Equiformerv2':
            from rde.modules.encoders.equiformer import EquiformerV2
            self.transformer = EquiformerV2(dim=dim, num_layers=cfg.encoder.num_layers)
        else:
            from rde.modules.encoders.equiformer import Equiformer  # pip install beartype   / pip install opt_einsum
            self.transformer = Equiformer(dim=dim, dim_head=dim // 2, heads=2, num_degrees=1, depth=cfg.encoder.num_layers, num_neighbors=cfg.encoder.num_nearest_neighbors)

        # Pred
        self.dg_readout = nn.Sequential(nn.Dropout(dropout_), nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim // 4), nn.ReLU(), nn.Linear(dim // 4, 1))

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
        if self.use_plm:
            x += batch['plm_wt']
        if self.rde is not None:
            x_pret = self._encode_rde(batch)
            x = self.single_fusion(torch.cat([x, x_pret], dim=-1))

        # PPIformer
        pos_atom = batch['pos_atoms'][:, :, BBHeavyAtom.CA]
        mask = batch['mask_atoms'][:, :, BBHeavyAtom.CA]

        if self.cfg.type == 'Equiformerv2':
            x = self.transformer(x, pos_atom, mask)
        else:
            x = self.transformer(x, pos_atom, mask)[0]
        return x

    def forward(self, batch):

        h_wt = self.encode(batch)
        H_wt = h_wt.max(dim=1)[0]
        dg_pred = self.dg_readout(H_wt).squeeze(-1)
        loss = F.mse_loss(dg_pred, batch['dG'])

        loss_dict = {'regression': loss}
        out_dict = {'dG_pred': dg_pred.detach().clone(), 'dG_true': batch['dG'], }
        return loss_dict, out_dict
