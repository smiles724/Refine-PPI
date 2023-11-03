import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rde.modules.encoders.egnn_attn import GAEncoder
from rde.modules.encoders.pair import ResiduePairEncoder
from rde.modules.encoders.single import PerResidueEncoderV1
from rde.utils.protein.constants import BBHeavyAtom, HeavyAtom2int, num_chi_angles
from rde.modules.encoders.egnn import EGNN_Network


class PDC_Network(nn.Module):

    def __init__(self, cfg, fine_tune=False, mask_token=20, max_aa_types=22):
        super().__init__()
        # Configurations
        self.use_sasa = cfg.get('sasa', None)
        self.use_plm = cfg.use_plm
        self.resolution = cfg.resolution
        self.target = 'ddG' if fine_tune else cfg.target
        dim = 1280 if self.use_plm else cfg.encoder.node_feat_dim
        try:
            dropout_ = cfg.encoder.dropout
        except:
            dropout_ = 0.2

        # Encoding
        if self.resolution != 'CA':
            self.token_emb = nn.Embedding(len(HeavyAtom2int) + 1, dim)  # 6 atom types
        self.var_embed = nn.Embedding(max_aa_types, 3)                  # each residue has a different positional variance
        self.single_encoder = PerResidueEncoderV1(feat_dim=dim, max_num_atoms=5, use_sasa=self.use_sasa)                              # N, CA, C, O, CB,
        self.pair_encoder = ResiduePairEncoder(feat_dim=cfg.encoder.pair_feat_dim, max_num_atoms=5,)                                  # N, CA, C, O, CB,
        self.attn_encoder = GAEncoder(node_feat_dim=dim, pair_feat_dim=cfg.encoder.pair_feat_dim, num_layers=cfg.encoder.num_layers,)
        self.spatial_project = EGNN_Network(dim=dim, depth=cfg.encoder.num_layers, num_nearest_neighbors=cfg.encoder.num_nearest_neighbors, norm_coors=cfg.encoder.norm_coors,
                                            update_coors_mean=cfg.encoder.update_coors_mean, update_coors_var=cfg.encoder.update_coors_var)

        # Side-chain recovery
        if 'chi_angle' in self.target:
            self.masked_bias = nn.Embedding(num_embeddings=2, embedding_dim=dim, padding_idx=0, )
            self.loss_fn = nn.MSELoss()
            self.angle_predictor = nn.Sequential(nn.Dropout(dropout_), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 4), nn.Sigmoid())

        # MLM
        if 'mlm' in self.target:
            self.mlm_masked_bias = nn.Embedding(num_embeddings=2, embedding_dim=dim, padding_idx=0, )
            self.mask_token = mask_token
            self.mlm_predictor = nn.Sequential(nn.Dropout(dropout_), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 20))

    def encode(self, batch, task=None, mode='wt'):
        mask = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
        chi = batch['chi_corrupt'] if task == 'chi_angle' else batch['chi']
        aa = batch['aa_masked'] if task == 'mlm' else batch['aa']
        res_feat = self.single_encoder(aa=aa, phi=batch['phi'], phi_mask=batch['phi_mask'], psi=batch['psi'], psi_mask=batch['psi_mask'], chi=chi,
                                       chi_mask=batch['chi_mask'], mask_residue=mask, sasa=batch['sasa'] if self.use_sasa else None,)            # (N, L, F)
        if self.use_plm:
            res_feat += batch['plm_wt'] if mode == 'wt' else batch['plm_mut']
        pair_feat = self.pair_encoder(aa=batch['aa'], res_nb=batch['res_nb'], chain_nb=batch['chain_nb'], pos_atoms=batch['pos_atoms'], mask_atoms=batch['mask_atoms'], )
        res_feat = self.attn_encoder(pos_atoms=batch['pos_atoms'], res_feat=res_feat, pair_feat=pair_feat, mask=mask)

        if task == 'chi_angle':
            b = self.masked_bias(batch['chi_masked_flag'].long())
            res_feat = res_feat + b

        if task == 'mlm':
            mask_flag = (batch['aa_masked'] == self.mask_token)
            b = self.mlm_masked_bias(mask_flag.long())
            res_feat = res_feat + b

        if self.resolution != 'CA':
            res_feat = self.token_emb(batch['type_atoms']) + res_feat.unsqueeze(-2)                # (N, L, A, F)
            pos_atom = torch.flatten(batch['pos_atoms'], start_dim=1, end_dim=2)                   # (N, L, A, 3) -> (N, L * A, 3)
            pos_atom_var = self.var_embed(batch['aa']).repeat(1, batch['pos_atoms'].shape[-2], 1)  # (B, L * A, 3)
            res_feat = torch.flatten(res_feat, start_dim=1, end_dim=2)
            mask = torch.flatten(batch['mask_atoms'], start_dim=1, end_dim=2)
        else:
            pos_atom = batch['pos_atoms'][:, :, BBHeavyAtom.CA]
            pos_atom_var = self.var_embed(batch['aa'])  # (B, L, 3)

        res_feat, coors_out, coors_var = self.spatial_project(coors_mean=pos_atom, coors_var=pos_atom_var, feats=res_feat, mask=mask)
        return res_feat

    def forward(self, batch, mode='train'):   # do not remove mode
        loss_dict = {}

        if 'chi_angle' in self.target:
            res_feat = self.encode(batch, task='chi_angle')
            n_chis_data = batch['chi_mask'].sum(-1)  # (N, L, 4) -> (N, L)
            chi_complete = batch['chi_complete']     # (N, L), only consider complete chi-angles

            angle_hat = self.angle_predictor(res_feat) * 2 * np.pi - np.pi  # joint prediction, range between [-pi, pi]
            if mode != 'test':
                for n_chis in range(4):  # iterate through residues with 1 - 4 chi angles
                    supervise_mask = torch.logical_and(chi_complete, n_chis_data >= n_chis + 1)  # (N, L)
                    loss = self.loss_fn(angle_hat[..., n_chis] * supervise_mask, batch['chi_native'][..., n_chis])
                    loss_dict['mse_%dchis' % (n_chis + 1)] = loss
            else:
                for res_name, num_chis in num_chi_angles.items():
                    if num_chis < 1:
                        continue
                    loss_mask = torch.logical_and(chi_complete, batch['aa'] == res_name._value_)
                    for n_chis in range(num_chis):
                        loss = nn.functional.l1_loss(angle_hat.squeeze(-1)[..., n_chis][loss_mask], batch['chi_native'][..., n_chis][loss_mask], reduction='none')
                        loss_dict[f'mae_{res_name._name_}_{n_chis + 1}'] = loss

        if 'mlm' in self.target:
            res_feat = self.encode(batch, task='mlm')
            aa_pred = self.mlm_predictor(res_feat)
            mask_flag = (batch['aa_masked'] == self.mask_token)
            loss_dict['mlm'] = nn.functional.cross_entropy(aa_pred[mask_flag], batch['aa'][mask_flag])

        if 'ddG' in self.target:   # TODO: rewrite this part with mut_flag feature
            batch_wt = {k: v for k, v in batch.items()}
            batch_mt = {k: v for k, v in batch.items()}
            batch_mt['aa'] = batch_mt['aa_mut']

            h_wt = self.encode(batch_wt, task='ddG', mode='wt')
            h_mt = self.encode(batch_mt, task='ddG', mode='mut')
            aa_pred_wt = self.mlm_predictor(h_wt)
            aa_pred_mt = self.mlm_predictor(h_mt)

            ddg_pred = (torch.gather(aa_pred_wt, -1, (batch_wt['aa'] * batch['mut_flag']).unsqueeze(-1)).squeeze(-1) * batch['mut_flag']).sum(-1) - (
                        torch.gather(aa_pred_mt, -1, (batch_mt['aa'] * batch['mut_flag']).unsqueeze(-1)).squeeze(-1) * batch['mut_flag']).sum(-1)

            loss = F.mse_loss(ddg_pred, batch['ddG'])
            loss_dict = {'regression': loss}
            out_dict = {'ddG_pred': ddg_pred.detach().clone(), 'ddG_true': batch['ddG'], }
            return loss_dict, out_dict

        if len(loss_dict) == 0:
            raise ValueError('Illegal self-supervised target.')
        return loss_dict
