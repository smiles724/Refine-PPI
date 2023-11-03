import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rde.modules.common.geometry import construct_3d_basis, angstrom_to_nm
from rde.modules.common.layers import AccumulatedNormalization
from rde.modules.encoders.equiformer import Equiformer  # pip install beartype   / pip install opt_einsum
from rde.modules.encoders.single import PerResidueEncoder
from rde.utils.protein.constants import BBHeavyAtom, num_aa_types, chi_angles_atoms, num_chi_angles


class EquiformerNet(nn.Module):

    def __init__(self, cfg, mask_token=20):
        super().__init__()
        self.use_plm = cfg.use_plm
        self.target = cfg.target
        dim = cfg.encoder.node_feat_dim
        input_dim = 1280 if self.use_plm else cfg.encoder.input_feat_dim
        dropout_ = cfg.encoder.dropout

        # Encoding, https://github.com/lucidrains/equiformer-pytorch
        self.single_encoder = PerResidueEncoder(feat_dim=input_dim, max_num_atoms=5)  # N, CA, C, O, CB
        self.input_mlp = nn.Sequential(nn.Dropout(dropout_), nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, dim))
        num_degrees = cfg.encoder.num_degrees
        self.transformer = Equiformer(dim=tuple([dim] * num_degrees), dim_head=tuple([dim] * num_degrees), heads=tuple([cfg.encoder.heads] * num_degrees), num_degrees=num_degrees,
                                      depth=cfg.encoder.num_layers, num_neighbors=cfg.encoder.num_nearest_neighbors)
        # Side-chain recovery
        self.angle_mode = cfg.angle_mode
        if self.angle_mode == 'cos_sin':
            self.angle_predictor = nn.Sequential(nn.Dropout(dropout_), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 8))
        else:
            self.angle_predictor = nn.Sequential(nn.Dropout(dropout_), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 4), nn.Sigmoid())

        # Denoising
        self.noise_scale = cfg.noise_scale
        self.pre_reduce = nn.Sequential(nn.Dropout(dropout_), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 3))
        self.pos_normalizer = AccumulatedNormalization(accumulator_shape=(128, 3))  # (L, 3)

        # MLM
        self.mask_token = mask_token
        self.mlm_predictor = nn.Sequential(nn.Dropout(dropout_), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 20))

        # Contrastive
        # self.temperature = cfg.temperature if 'temperate' in cfg else 0.07
        # assert self.temperature > 0.0, 'The temperature must be a positive float!'
        self.loss_fn = nn.MSELoss()
        self.register_buffer('num_chis_of_aa',
                             torch.tensor(data=[len(chi_angles_atoms[i]) if i < len(chi_angles_atoms) else 0 for i in range(num_aa_types + 1)], dtype=torch.long, ))

        # Init weights
        for name, param in self.transformer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def encode(self, batch, mode='wt'):
        R = construct_3d_basis(batch['pos_heavyatom'][:, :, BBHeavyAtom.CA], batch['pos_heavyatom'][:, :, BBHeavyAtom.C], batch['pos_heavyatom'][:, :, BBHeavyAtom.N], )
        t = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        t = angstrom_to_nm(t)
        x = self.single_encoder(aa=batch['aa'], res_nb=batch['res_nb'], chain_nb=batch['chain_nb'], pos_atoms=batch['pos_atoms'], mask_atoms=batch['mask_atoms'], R=R, t=t)
        if self.use_plm:  # comply with downstream ddG task
            x += batch['plm_wt'] if mode == 'wt' else batch['plm_mut']

        # PPIformer
        pos_atom = batch['pos_atoms'][:, :, BBHeavyAtom.CA]
        mask = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
        x = self.input_mlp(x)
        x = self.transformer(x, pos_atom, mask)[0]
        return x

    def forward(self, batch, mode='train'):
        loss_dict = {}

        if 'chi_angle' in self.target:
            res_feat = self.encode(batch)
            n_chis_data = batch['chi_mask'].sum(-1)  # (N, L, 4) -> (N, L)
            chi_complete = batch['chi_complete']  # (N, L), only consider complete chi-angles

            if self.angle_mode == 'cos_sin':
                angle_hat = self.angle_predictor(res_feat)  # (N, L, 4, 2)
                angle_hat = angle_hat.unflatten(-1, (4, 2))
                angle_hat = angle_hat / torch.norm(angle_hat, dim=-1, keepdim=True)  # /= is a replaced operation
            else:
                angle_hat = self.angle_predictor(res_feat) * 2 * np.pi - np.pi  # joint prediction, range between [-pi, pi]

            if mode in ['train', 'val']:
                if self.angle_mode == 'cos_sin':
                    label = torch.stack([torch.cos(batch['chi_native']), torch.sin(batch['chi_native'])], dim=-1)
                for n_chis in range(4):  # iterate through residues with 1 - 4 chi angles
                    supervise_mask = torch.logical_and(chi_complete, n_chis_data >= n_chis + 1)  # (N, L)
                    if self.angle_mode == 'cos_sin':
                        loss = self.loss_fn(angle_hat[:, :, n_chis] * supervise_mask.unsqueeze(-1), label[:, :, n_chis])
                    else:
                        loss = self.loss_fn(angle_hat[..., n_chis] * supervise_mask, batch['chi_native'][..., n_chis])
                    loss_dict['mse_%dchis' % (n_chis + 1)] = loss

            elif mode == 'test':
                if self.angle_mode == 'cos_sin':
                    angle_hat = torch.atan(angle_hat[..., 1] / angle_hat[..., 0]) + (angle_hat[..., 0] < 0) * (angle_hat[..., 1] > 0) * np.pi - (angle_hat[..., 0] < 0) * (
                            angle_hat[..., 1] < 0) * np.pi  # tangent
                for res_name, num_chis in num_chi_angles.items():
                    if num_chis < 1:
                        continue
                    loss_mask = torch.logical_and(chi_complete, batch['aa'] == res_name._value_)
                    for n_chis in range(num_chis):
                        loss = nn.functional.l1_loss(angle_hat.squeeze(-1)[..., n_chis][loss_mask], batch['chi_native'][..., n_chis][loss_mask], reduction='none')
                        loss_dict[f'mae_{res_name._name_}_{n_chis + 1}'] = loss

        if 'mlm' in self.target:
            aa_ = batch['aa'].clone()
            mask_ = (batch['aa_masked'] == self.mask_token)

            batch['aa'] = batch['aa_masked'].clone()
            res_feat = self.encode(batch)
            aa_pred = self.mlm_predictor(res_feat)

            loss_dict['mlm'] = nn.functional.cross_entropy(aa_pred[mask_], aa_[mask_])
            batch['aa'] = aa_

        if 'denoise' in self.target:
            noise = torch.randn_like(batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]) * self.noise_scale  # (N, L, 3)
            batch['pos_heavyatom'] += noise.unsqueeze(-2)

            res_feat = self.encode(batch)
            noise_hat = self.pre_reduce(res_feat)
            normalized_noise = self.pos_normalizer(noise)
            loss_dict['denoise'] = self.loss_fn(normalized_noise, noise_hat)

        if 'contrastive' in self.target:
            # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
            prot_feat_0 = res_feat.mean(1)
            prot_feat_1 = self.encode(batch['patch_1']).mean(1)
            feats = torch.cat([prot_feat_0, prot_feat_1], dim=0)  # (2B, F)

            # Calculate cosine similarity
            cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)  # (2B, 2B)

            # Mask out cosine similarity to itself
            self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
            cos_sim.masked_fill_(self_mask, -1e15)

            # Find positive example -> batch_size away from the original example
            positive_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

            # InfoNCE loss
            cos_sim = cos_sim / self.temperature
            nll = -cos_sim[positive_mask] + torch.logsumexp(cos_sim, dim=-1)
            loss_dict['contrastive'] = nll.mean()

        if len(loss_dict) == 0:
            raise ValueError('Illegal self-supervised target.')

        return loss_dict
