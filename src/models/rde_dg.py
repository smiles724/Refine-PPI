import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.encoders.single import PerResidueEncoder
from src.modules.encoders.pair import ResiduePairEncoder
from src.modules.encoders.attn import GAEncoder
from src.utils.protein.constants import BBHeavyAtom
from .rde import CircularSplineRotamerDensityEstimator
from src.models.rde_mlm import MaskedLanguageModelingDensityEstimator
from src.models.pdc import ProbabilityDensityCloud


class DG_RDE_Network(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        dim = cfg.encoder.node_feat_dim

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
        self.pair_encoder = ResiduePairEncoder(feat_dim=cfg.encoder.pair_feat_dim, max_num_atoms=5)  # N, CA, C, O, CB,
        self.attn_encoder = GAEncoder(**cfg.encoder)

        # Pred
        self.dg_readout = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def _encode_rde(self, batch, mask_extra=None):
        batch = {k: v for k, v in batch.items()}
        batch['chi_corrupt'] = batch['chi']
        batch['chi_masked_flag'] = torch.zeros(size=batch['aa'].shape, dtype=torch.bool).cuda()   # no mutation and mask
        if mask_extra is not None:
            batch['mask_atoms'] = batch['mask_atoms'] * mask_extra[:, :, None]
        with torch.no_grad():
            return self.rde.encode(batch)

    def encode(self, batch):
        mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA]

        x = self.single_encoder(aa=batch['aa'], phi=batch['phi'], phi_mask=batch['phi_mask'], psi=batch['psi'], psi_mask=batch['psi_mask'], chi=batch['chi'],
                                chi_mask=batch['chi_mask'], mask_residue=mask_residue, )
        if self.rde is not None:
            x_pret = self._encode_rde(batch)
            if self.ckpt_type == 'ProbabilityDensityCloud':
                x_pret = x_pret[0]
            x = self.single_fusion(torch.cat([x, x_pret], dim=-1))

        z = self.pair_encoder(aa=batch['aa'], res_nb=batch['res_nb'], chain_nb=batch['chain_nb'], pos_atoms=batch['pos_atoms'], mask_atoms=batch['mask_atoms'], )
        x = self.attn_encoder(pos_atoms=batch['pos_atoms'], res_feat=x, pair_feat=z, mask=mask_residue)

        return x

    def forward(self, batch):
        h = self.encode(batch)
        H = h.max(dim=1)[0]

        dg_pred = self.dg_readout(H).squeeze(-1)
        loss = F.mse_loss(dg_pred, batch['dG'])

        loss_dict = {'regression': loss, }
        out_dict = {'dG_pred': dg_pred, 'dG_true': batch['dG'], }
        return loss_dict, out_dict
