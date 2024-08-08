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


class DDG_RDE_Network(nn.Module):

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
        self.mut_bias = nn.Embedding(num_embeddings=2, embedding_dim=dim, padding_idx=0, )
        self.pair_encoder = ResiduePairEncoder(feat_dim=cfg.encoder.pair_feat_dim, max_num_atoms=5)  # N, CA, C, O, CB,
        self.attn_encoder = GAEncoder(node_feat_dim=dim, pair_feat_dim=cfg.encoder.pair_feat_dim, num_layers=cfg.encoder.num_layers,)

        # Pred
        self.ddg_readout = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def _encode_rde(self, batch, mask_extra=None):
        batch = {k: v for k, v in batch.items()}
        batch['chi_corrupt'] = batch['chi']
        batch['chi_masked_flag'] = batch['mut_flag']
        if mask_extra is not None:
            batch['mask_atoms'] = batch['mask_atoms'] * mask_extra[:, :, None]
        with torch.no_grad():
            return self.rde.encode(batch)

    def encode(self, batch):
        mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
        chi = batch['chi'] * (1 - batch['mut_flag'].float())[:, :, None]

        x_single = self.single_encoder(aa=batch['aa'], phi=batch['phi'], phi_mask=batch['phi_mask'], psi=batch['psi'], psi_mask=batch['psi_mask'], chi=chi,
                                       chi_mask=batch['chi_mask'], mask_residue=mask_residue, )
        if self.rde is not None:
            x_pret = self._encode_rde(batch)
            if self.ckpt_type == 'ProbabilityDensityCloud':
                x_pret = x_pret[0]
            x_single = self.single_fusion(torch.cat([x_single, x_pret], dim=-1))

        b = self.mut_bias(batch['mut_flag'].long())
        x = x_single + b

        z = self.pair_encoder(aa=batch['aa'], res_nb=batch['res_nb'], chain_nb=batch['chain_nb'], pos_atoms=batch['pos_atoms'], mask_atoms=batch['mask_atoms'], )
        x = self.attn_encoder(pos_atoms=batch['pos_atoms'], res_feat=x, pair_feat=z, mask=mask_residue)

        return x

    def forward(self, batch):
        batch_wt = {k: v for k, v in batch.items()}
        batch_mt = {k: v for k, v in batch.items()}
        batch_mt['aa'] = batch_mt['aa_mut']

        h_wt = self.encode(batch_wt)
        h_mt = self.encode(batch_mt)

        H_mt, H_wt = h_mt.max(dim=1)[0], h_wt.max(dim=1)[0]
        ddg_pred = self.ddg_readout(H_mt - H_wt).squeeze(-1)
        ddg_pred_inv = self.ddg_readout(H_wt - H_mt).squeeze(-1)
        loss = (F.mse_loss(ddg_pred, batch['ddG']) + F.mse_loss(ddg_pred_inv, -batch['ddG'])) / 2

        loss_dict = {'regression': loss, }
        out_dict = {'ddG_pred': ddg_pred, 'ddG_true': batch['ddG'], }
        return loss_dict, out_dict
