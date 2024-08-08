import numpy as np
import torch
import torch.nn as nn

from src.modules.encoders.attn import GAEncoder
from src.modules.encoders.pair import ResiduePairEncoder
from src.modules.encoders.single import PerResidueEncoder
from src.utils.protein.constants import BBHeavyAtom, num_aa_types, chi_angles_atoms


def _latent_log_prob(z, num_chis):
    assert z.size(-1) == num_chis
    volume = (2 * np.pi) ** num_chis
    logp = np.log(1 / volume)
    shape = list(z.size())[:-1]
    return torch.full(shape, logp, device=z.device, dtype=torch.float)


class MaskedLanguageModelingDensityEstimator(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dim = cfg.encoder.node_feat_dim

        # Encoding
        self.single_encoder = PerResidueEncoder(feat_dim=dim, max_num_atoms=5,  # N, CA, C, O, CB,
                                                )
        self.masked_bias = nn.Embedding(num_embeddings=2, embedding_dim=dim, padding_idx=0, )
        self.pair_encoder = ResiduePairEncoder(feat_dim=cfg.encoder.pair_feat_dim, max_num_atoms=5,  # N, CA, C, O, CB,
                                               )
        self.attn_encoder = GAEncoder(**cfg.encoder)

        self.angle_predictor = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 4), nn.Sigmoid())
        self.loss_fn = nn.MSELoss()
        self.register_buffer('num_chis_of_aa',
                             torch.tensor(data=[len(chi_angles_atoms[i]) if i < len(chi_angles_atoms) else 0 for i in range(num_aa_types + 1)], dtype=torch.long, ))

    def encode(self, batch):
        mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
        chi = batch['chi_corrupt']

        x = self.single_encoder(aa=batch['aa'], phi=batch['phi'], phi_mask=batch['phi_mask'], psi=batch['psi'], psi_mask=batch['psi_mask'], chi=chi, chi_mask=batch['chi_mask'],
                                mask_residue=mask_residue, )
        b = self.masked_bias(batch['chi_masked_flag'].long())
        x = x + b
        z = self.pair_encoder(aa=batch['aa'], res_nb=batch['res_nb'], chain_nb=batch['chain_nb'], pos_atoms=batch['pos_atoms'], mask_atoms=batch['mask_atoms'], )
        x = self.attn_encoder(pos_atoms=batch['pos_atoms'], res_feat=x, pair_feat=z, mask=mask_residue)
        return x

    def sample(self, batch):
        c = self.encode(batch)
        angle_hat = self.angle_predictor(c) * 2 * np.pi - np.pi  # range between [-pi, pi]
        return angle_hat

    def forward(self, batch):
        c = self.encode(batch)
        angle_hat = self.angle_predictor(c) * 2 * np.pi - np.pi  # range between [-pi, pi]
        n_chis_data = batch['chi_mask'].sum(-1)  # (N, L, 4) -> (N, L)
        chi_complete = batch['chi_complete']  # (N, L), only consider complete chi-angles
        loss_dict = {}
        for n_chis in range(1, 4 + 1):  # iterate through residues with 1 - 4 chi angles
            supervise_mask = torch.logical_and(torch.logical_and(chi_complete, n_chis_data >= n_chis), batch['chi_corrupt_flag'])  # (N, L)
            loss = self.loss_fn(angle_hat[:, :, n_chis - 1] * supervise_mask, batch['chi'][:, :, n_chis - 1])
            loss_dict['mse_%dchis' % n_chis] = loss

        return loss_dict
