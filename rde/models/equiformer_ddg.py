import torch
import torch.nn as nn
import torch.nn.functional as F

from rde.modules.encoders.equiformer import Equiformer  # pip install beartype   / pip install opt_einsum
from rde.modules.encoders.single import PerResidueEncoderV1
from rde.utils.protein.constants import BBHeavyAtom


class DDG_Equiformer(nn.Module):

    def __init__(self, cfg, ):
        super().__init__()
        self.use_plm = cfg.use_plm
        dim = cfg.encoder.node_feat_dim
        input_dim = 1280 if self.use_plm else cfg.encoder.input_feat_dim

        # Pretrain
        self.rde = None
        if cfg.checkpoint.path:
            self.ckpt_type = cfg.checkpoint.type
            print(f'Loading {cfg.checkpoint.type} from {cfg.checkpoint.path}')
            ckpt = torch.load(cfg.checkpoint.path, map_location='cpu')
            if self.ckpt_type == 'CircularSplineRotamerDensityEstimator':
                from rde.models.rde import CircularSplineRotamerDensityEstimator
                self.rde = CircularSplineRotamerDensityEstimator(ckpt['config'].model)
            elif self.ckpt_type == 'BackboneNetwork':
                from rde.models.backbone import BackboneNetwork
                self.rde = BackboneNetwork(ckpt['config'].model)
            elif self.ckpt_type == 'Equiformer':
                from rde.models.equiformer import EquiformerNet
                self.rde = EquiformerNet(ckpt['config'].model)

            # https://stackoverflow.com/questions/63057468/how-to-ignore-and-initialize-missing-keys-in-state-dict
            self.rde.load_state_dict(ckpt['model'], strict=False)  # ignore unmatched keys
            for p in self.rde.parameters():
                p.requires_grad_(False)
            if 'use_plm' in ckpt['config'].model and ckpt['config'].model.use_plm and self.ckpt_type != 'Equiformer':
                self.single_fusion = nn.Sequential(nn.Linear(input_dim + 1280, dim), nn.ReLU(), nn.Linear(dim, dim))
            else:
                self.single_fusion = nn.Sequential(nn.Linear(input_dim + ckpt['config'].model.encoder.node_feat_dim, dim), nn.ReLU(), nn.Linear(dim, dim))

        # Encoding
        self.single_encoder = PerResidueEncoderV1(feat_dim=input_dim, max_num_atoms=5)  # N, CA, C, O, CB,
        num_degrees = cfg.encoder.num_degrees
        self.transformer = Equiformer(dim=tuple([dim] * num_degrees), dim_head=tuple([dim] * num_degrees), heads=tuple([cfg.encoder.heads] * num_degrees), num_degrees=num_degrees,
                                      depth=cfg.encoder.num_layers, num_neighbors=cfg.encoder.num_nearest_neighbors)
        self.mut_bias = nn.Embedding(num_embeddings=2, embedding_dim=dim, padding_idx=0, )

        # Init weights
        for name, param in self.transformer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

        # Pred
        self.ddg_readout = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim // 4), nn.ReLU(), nn.Linear(dim // 4, 1))

    def _encode_rde(self, batch, mask_extra=None):
        batch = {k: v for k, v in batch.items()}
        # for RDE
        if self.ckpt_type == 'CircularSplineRotamerDensityEstimator':
            batch['chi_corrupt'] = batch['chi']  # use real chi with no corruption
            batch['chi_masked_flag'] = batch['mut_flag']

        if mask_extra is not None:
            batch['mask_atoms'] = batch['mask_atoms'] * mask_extra[:, :, None]
        with torch.no_grad():
            return self.rde.encode(batch)

    def encode(self, batch, mode='wt'):
        chi = batch['chi'] * (1 - batch['mut_flag'].float())[:, :, None]
        x = self.single_encoder(aa=batch['aa'], phi=batch['phi'], phi_mask=batch['phi_mask'], psi=batch['psi'], psi_mask=batch['psi_mask'], chi=chi, chi_mask=batch['chi_mask'],
                                mask_residue=batch['mask_atoms'][:, :, BBHeavyAtom.CA], )  # (N, L, F)
        if self.use_plm:
            x += batch['plm_wt'] if mode == 'wt' else batch['plm_mut']
        if self.rde is not None:
            x_pret = self._encode_rde(batch)
            x = self.single_fusion(torch.cat([x, x_pret], dim=-1))

        # PPIformer
        pos_atom = batch['pos_atoms'][:, :, BBHeavyAtom.CA]
        mask = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
        x = self.transformer(x, pos_atom, mask)[0]
        return x

    def forward(self, batch):
        batch_wt = {k: v for k, v in batch.items()}
        batch_mt = {k: v for k, v in batch.items()}
        batch_mt['aa'] = batch_mt['aa_mut']

        h_wt = self.encode(batch_wt, 'wt')
        h_mt = self.encode(batch_mt, 'mut')

        H_mt, H_wt = h_mt.max(dim=1)[0], h_wt.max(dim=1)[0]
        ddg_pred = self.ddg_readout(H_mt - H_wt).squeeze(-1)
        ddg_pred_inv = self.ddg_readout(H_wt - H_mt).squeeze(-1)
        loss = (F.mse_loss(ddg_pred, batch['ddG']) + F.mse_loss(ddg_pred_inv, -batch['ddG'])) / 2

        loss_dict = {'regression': loss}
        out_dict = {'ddG_pred': ddg_pred.detach().clone(), 'ddG_true': batch['ddG'], }
        return loss_dict, out_dict
