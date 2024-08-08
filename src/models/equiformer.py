import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.encoders.equiformer import Equiformer  # pip install beartype   / pip install opt_einsum
from src.modules.encoders.attn import GAEncoder
from src.modules.encoders.pair import ResiduePairEncoder
from src.modules.encoders.single import PerResidueEncoder
from src.utils.protein.constants import BBHeavyAtom, num_aa_types, chi_angles_atoms, HeavyAtom2int, num_chi_angles
from src.modules.encoders.egnn import EGNN_Network


class EquiformerNet(nn.Module):

    def __init__(self, cfg, max_aa_types=22):
        super().__init__()
        self.cfg = cfg
        self.use_plm = cfg.use_plm
        self.target = cfg.target
        self.resolution = cfg.resolution
        self.temperature = cfg.temperature if 'temperate' in cfg else 0.07
        assert self.temperature > 0.0, 'The temperature must be a positive float!'
        dim = cfg.encoder.node_feat_dim
        if self.use_plm:
            self.plm_linear = nn.Linear(1280, dim)

        # Encoding
        self.single_encoder = PerResidueEncoder(feat_dim=dim, max_num_atoms=5)  # N, CA, C, O, CB
        # self.pair_encoder = ResiduePairEncoder(feat_dim=cfg.encoder.pair_feat_dim, max_num_atoms=5)  # N, CA, C, O, CB
        # self.attn_encoder = GAEncoder(node_feat_dim=dim, pair_feat_dim=cfg.encoder.pair_feat_dim, num_layers=cfg.encoder.num_layers, )

        self.transformer = Equiformer(dim=(dim, dim // 2, dim // 4, dim // 4), dim_head=(dim, dim // 2, dim // 2, dim // 2), heads=(4, 2, 2, 2), num_degrees=4, linear_out=True,
                                      depth=cfg.encoder.num_layers, reversible=False, attend_self=True, num_neighbors=cfg.encoder.num_nearest_neighbors)
        self.masked_bias = nn.Embedding(num_embeddings=2, embedding_dim=dim, padding_idx=0, )  # https://github.com/lucidrains/equiformer-pytorch
        self.angle_predictor = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 4), nn.Sigmoid())

        if self.resolution != 'CA':
            self.token_emb = nn.Embedding(len(HeavyAtom2int) + 1, dim)  # 6 atom types
        # if self.target.pos_refine:
        #     self.var_embed = nn.Embedding(max_aa_types, 3)
        #     self.spatial_project = EGNN_Network(dim=dim, depth=1, num_nearest_neighbors=cfg.encoder.num_nearest_neighbors, norm_coors=True,
        #                                         update_coors_mean=True, update_coors_var=True, )

        self.loss_fn = nn.MSELoss()
        self.register_buffer('num_chis_of_aa',
                             torch.tensor(data=[len(chi_angles_atoms[i]) if i < len(chi_angles_atoms) else 0 for i in range(num_aa_types + 1)], dtype=torch.long, ))

    def encode(self, batch, mode='wt'):
        mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
        if self.target.chi_angle:
            chi = batch['chi_corrupt']
        else:
            chi = batch['chi']

        res_feat = self.single_encoder(aa=batch['aa'], phi=batch['phi'], phi_mask=batch['phi_mask'], psi=batch['psi'], psi_mask=batch['psi_mask'], chi=chi,
                                       chi_mask=batch['chi_mask'], mask_residue=mask_residue, )
        if self.use_plm:  # comply with downstream ddG task
            res_feat += self.plm_linear(batch['plm_wt']) if mode == 'wt' else self.plm_linear(batch['plm_mut'])

        if self.target.chi_angle:
            b = self.masked_bias(batch['chi_masked_flag'].long())
            res_feat = res_feat + b

        # z = self.pair_encoder(aa=batch['aa'], res_nb=batch['res_nb'], chain_nb=batch['chain_nb'], pos_atoms=batch['pos_atoms'], mask_atoms=batch['mask_atoms'], )
        # res_feat = self.attn_encoder(pos_atoms=batch['pos_atoms'], res_feat=res_feat, pair_feat=z, mask=mask_residue)

        return res_feat

    def forward(self, batch, mode='train'):
        loss_dict = {}
        res_feat = self.encode(batch)
        if self.resolution != 'CA':
            res_feat = self.token_emb(batch['type_atoms']) + res_feat.unsqueeze(-2)  # (N, L, A, F)
            res_feat = torch.flatten(res_feat, start_dim=1, end_dim=2)  # (N, L, A, 3) -> (N, L * A, 3)
            pos_atom = torch.flatten(batch['pos_atoms'], start_dim=1, end_dim=2)
            mask = torch.flatten(batch['mask_atoms'], start_dim=1, end_dim=2)
            # pos_atom_var = self.var_embed(batch['aa']).repeat(1, batch['pos_atoms'].shape[-2], 1)  # (B, L * A, 3)

        else:
            pos_atom = batch['pos_atoms'][:, :, BBHeavyAtom.CA]
            mask = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
            # pos_atom_var = self.var_embed(batch['aa'])  # (B, L, 3)

        res_feat = self.transformer(res_feat, pos_atom, mask)[0]

        if self.target.chi_angle:
            n_chis_data = batch['chi_mask'].sum(-1)  # (N, L, 4) -> (N, L)
            chi_complete = batch['chi_complete']  # (N, L), only consider complete chi-angles

            if self.resolution != 'CA':      # joint prediction, range between [-pi, pi]
                angle_hat = self.angle_predictor(res_feat.unflatten(1, (chi_complete.shape[-1], -1))[:, :, BBHeavyAtom.CA]) * 2 * np.pi - np.pi
            else:
                angle_hat = self.angle_predictor(res_feat) * 2 * np.pi - np.pi

            if mode != 'test':
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

        if self.target.pos_refine:
            if self.resolution != 'CA':
                pos_change_flag = batch['pos_change_flag'].repeat(1, batch['pos_atoms'].shape[-2])
                pos_gt = torch.flatten(batch['pos_gt'], start_dim=1, end_dim=2)[pos_change_flag]
            else:
                pos_change_flag = batch['pos_change_flag']
                pos_gt = batch['pos_gt'][:, :, BBHeavyAtom.CA][pos_change_flag]

            # if self.resolution != 'CA':
            #     res_feat = self.token_emb(batch['type_atoms']) + res_feat.unsqueeze(-2)  # (N, L, A, F)
            #
            #     res_feat = torch.flatten(res_feat, start_dim=1, end_dim=2)  # (N, L, A, 3) -> (N, L * A, 3)
            #     pos_atom = torch.flatten(batch['pos_atoms'], start_dim=1, end_dim=2)
            #     mask = torch.flatten(batch['mask_atoms'], start_dim=1, end_dim=2)
            #     pos_atom_var = self.var_embed(batch['aa']).repeat(1, batch['pos_atoms'].shape[-2], 1)  # (B, L * A, 3)
            #
            # else:
            #     pos_atom = batch['pos_atoms'][:, :, BBHeavyAtom.CA]
            #     mask = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
            #     pos_atom_var = self.var_embed(batch['aa'])  # (B, L, 3)

            # coors_out = self.spatial_project(coors_mean=pos_atom, coors_var=pos_atom_var, feats=res_feat, mask=mask, pos_change_flag=pos_change_flag)[1]
            loss_dict['pos_refine'] = torch.sqrt(((pos_gt - coors_out[pos_change_flag]) ** 2).sum(dim=-1) + 1e-10).mean()

        if self.target.contrastive:
            # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
            prot_feat_0 = res_feat.mean(1)
            prot_feat_1 = self.encode(batch['patch_1']).mean(1)
            feats = torch.cat([prot_feat_0, prot_feat_1], dim=0)   # (2B, F)

            # Calculate cosine similarity
            cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)   # (2B, 2B)

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
