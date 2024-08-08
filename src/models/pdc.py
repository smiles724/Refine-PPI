import numpy as np
import torch
import torch.nn as nn

from src.modules.encoders.attn import GAEncoder
from src.modules.encoders.pair import ResiduePairEncoder
from src.modules.encoders.single import PerResidueEncoder
from src.utils.protein.constants import BBHeavyAtom, num_aa_types, chi_angles_atoms, HeavyAtom2int, num_chi_angles
from src.modules.encoders.egnn import EGNN_Network


class ProbabilityDensityCloud(nn.Module):

    def __init__(self, cfg, max_aa_types=22):
        super().__init__()
        self.cfg = cfg
        self.use_plm = cfg.use_plm
        self.target = cfg.target
        self.resolution = cfg.resolution
        if self.target == 'refine':
            self.recycle = cfg.pos.recycle
        dim = 1280 if self.use_plm else cfg.encoder.node_feat_dim
        # dim = cfg.encoder.node_feat_dim
        # if self.use_plm:
        #     self.plm_linear = nn.Linear(1280, dim)

        try:  # code version alignment
            self.learnable_var = cfg.encoder.learnable_var
            dropout_ = cfg.encoder.dropout
        except:
            self.learnable_var = cfg.learnable_var
            dropout_ = 0.2

        # Encoding
        self.single_encoder = PerResidueEncoder(feat_dim=dim, max_num_atoms=5)  # N, CA, C, O, CB,   # TODO: only use backbone geometries rather than side-chain
        self.pair_encoder = ResiduePairEncoder(feat_dim=cfg.encoder.pair_feat_dim, max_num_atoms=5)  # N, CA, C, O, CB
        self.attn_encoder = GAEncoder(node_feat_dim=dim, pair_feat_dim=cfg.encoder.pair_feat_dim, num_layers=cfg.encoder.num_layers, )
        self.spatial_project = EGNN_Network(dim=dim, depth=cfg.encoder.num_layers, num_nearest_neighbors=cfg.encoder.num_nearest_neighbors, norm_coors=cfg.encoder.norm_coors,
                                            update_coors_mean=cfg.encoder.update_coors_mean, update_coors_var=cfg.encoder.update_coors_var, dropout=dropout_)
        self.masked_bias = nn.Embedding(num_embeddings=2, embedding_dim=dim, padding_idx=0, )
        # self.angle_predictor = nn.Sequential(nn.Dropout(dropout_), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 4), nn.Sigmoid())
        self.angle_predictor = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 4), nn.Sigmoid())

        if self.learnable_var: self.var_embed = nn.Embedding(max_aa_types, 3)  # each residue has a different positional variance
        if self.resolution != 'CA':
            self.token_emb = nn.Embedding(len(HeavyAtom2int) + 1, dim)  # 6 atom types

        self.loss_fn = nn.MSELoss()
        self.register_buffer('num_chis_of_aa',
                             torch.tensor(data=[len(chi_angles_atoms[i]) if i < len(chi_angles_atoms) else 0 for i in range(num_aa_types + 1)], dtype=torch.long, ))

    def encode(self, batch, mode='wt'):
        mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
        if self.target == 'chi_angle':
            chi = batch['chi_corrupt']
        else:
            chi = batch['chi']

        x = self.single_encoder(aa=batch['aa'], phi=batch['phi'], phi_mask=batch['phi_mask'], psi=batch['psi'], psi_mask=batch['psi_mask'], chi=chi, chi_mask=batch['chi_mask'],
                                mask_residue=mask_residue, )
        if self.use_plm:  # comply with downstream ddG task
            # x += self.plm_linear(batch['plm_wt']) if mode == 'wt' else self.plm_linear(batch['plm_mut'])
            x += batch['plm_wt'] if mode == 'wt' else batch['plm_mut']

        if self.target == 'chi_angle':
            b = self.masked_bias(batch['chi_masked_flag'].long())
            x = x + b
        z = self.pair_encoder(aa=batch['aa'], res_nb=batch['res_nb'], chain_nb=batch['chain_nb'], pos_atoms=batch['pos_atoms'], mask_atoms=batch['mask_atoms'], )
        x = self.attn_encoder(pos_atoms=batch['pos_atoms'], res_feat=x, pair_feat=z, mask=mask_residue)

        if self.target == 'refine':
            return x
        else:
            pos_atom = batch['pos_atoms'][:, :, BBHeavyAtom.CA]
            if self.learnable_var:
                pos_atom_var = self.var_embed(batch['aa'])  # (B, N, 3)
            else:
                pos_atom_var = batch['pos_atom_var'][:, :, BBHeavyAtom.CA]
            res_feat, coors_out, coors_var = self.spatial_project(coors_mean=pos_atom, coors_var=pos_atom_var, feats=x, mask=mask_residue)
            return res_feat, coors_out, coors_var

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

    def forward(self, batch, mode='train'):
        loss_dict = {}

        if 'rmsf' in self.target:  # recover RMSF
            res_feat, coors_out, coors_var = self.encode(batch)
            loss_dict['rmsf'] = self.loss_fn(coors_var, batch['rmsf'])

        if 'chi_angle' in self.target:
            res_feat, coors_out, coors_var = self.encode(batch)
            n_chis_data = batch['chi_mask'].sum(-1)  # (N, L, 4) -> (N, L)
            chi_complete = batch['chi_complete']  # (N, L), only consider complete chi-angles

            # angle_hat = torch.remainder(self.angle_predictor(res_feat), 2 * torch.pi) - torch.pi      # joint prediction, range between [-pi, pi]

            angle_hat = self.angle_predictor(res_feat) * 2 * np.pi - np.pi  # joint prediction, range between [-pi, pi]
            if mode != 'test':   # TODO: periodic of some special chi-angles
                for n_chis in range(1, 4 + 1):  # iterate through residues with 1 - 4 chi angles
                    supervise_mask = torch.logical_and(torch.logical_and(chi_complete, n_chis_data >= n_chis), batch['chi_corrupt_flag'])  # (N, L)
                    loss = self.loss_fn(angle_hat.squeeze(-1)[..., n_chis - 1] * supervise_mask, batch['chi'][..., n_chis - 1])
                    loss_dict['mse_%dchis' % n_chis] = loss
            else:
                for res_name, num_chis in num_chi_angles.items():
                    if num_chis < 1:
                        continue
                    loss_mask = torch.logical_and(chi_complete, batch['aa'] == res_name._value_)
                    for n_chis in range(1, num_chis + 1):
                        loss = nn.functional.l1_loss(angle_hat.squeeze(-1)[..., n_chis - 1][loss_mask], batch['chi'][..., n_chis - 1][loss_mask], reduction='none')
                        loss_dict[f'mae_{res_name._name_}_{n_chis}'] = loss

        if 'refine' in self.target:
            if self.resolution != 'CA':
                pos_change_flag = batch['pos_change_flag'].repeat(1, batch['pos_atoms'].shape[-2])
            else:
                pos_change_flag = batch['pos_change_flag']

            if self.resolution == 'CA':  # (B, 3) or (B, 5, 3) for different resolution
                pos_gt = batch['pos_gt'][:, :, BBHeavyAtom.CA][pos_change_flag]
            else:
                pos_gt = torch.flatten(batch['pos_gt'], start_dim=1, end_dim=2)[pos_change_flag]

            loss_coors = []
            for _ in range(self.recycle):
                res_feat = self.encode(batch)
                coors = self.refine(res_feat, pos_change_flag, batch)
                # distance loss (instead of RMSE) for each iteration, https://discuss.pytorch.org/t/function-mselossbackward-returned-nan-values-in-its-0th-output/94875
                loss_coors.append(torch.sqrt(((pos_gt - coors[pos_change_flag]) ** 2).sum(dim=-1) + 1e-10).mean())  # sum() leads to explosion
                batch['pos_atoms'] = coors.detach().clone().reshape(batch['pos_atoms'].shape)

            if mode == 'train':
                loss_dict['pos_refine'] = torch.stack(loss_coors).sum()
            else:
                loss_dict['pos_refine'] = loss_coors[-1]

        if len(loss_dict) == 0:
            raise ValueError('Illegal self-supervised target.')

        return loss_dict
