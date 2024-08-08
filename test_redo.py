import argparse

import pandas as pd
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from src.utils.misc import get_logger, load_config
from src.utils.train import *
from src.utils.data import PaddingCollate
from src.utils.protein.constants import num_chi_angles
from src.datasets.pdbredo_chain import get_pdbredo_chain_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='ga', choices=['egnn', 'ga'])
    parser.add_argument('--ckpt', type=str, default='trained_models/RDE.pt')
    parser.add_argument('-o', '--output', type=str, default='redo_results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    logger = get_logger('test', None)

    ckpt = torch.load(args.ckpt)
    if args.backbone == 'ga':
        config, config_name = load_config('configs/pdc_redo.yml')
    else:
        config = ckpt['config']
    print(config)

    logger.info('Loading datasets...')
    use_plm = config.model.use_plm
    val_dataset = get_pdbredo_chain_dataset('val', config.data, use_plm=use_plm)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size * 4, shuffle=False, collate_fn=PaddingCollate(), num_workers=args.num_workers)

    logger.info('Building model...')
    if 'type' in config.model and config.model.type == 'equiformer':
        from src.models.equiformer import EquiformerNet
        model = EquiformerNet(config.model).to(args.device)
    elif 'pos' in config.model:
        from src.models.pdc_ddg_refine import DDG_PDC_Network
        model = DDG_PDC_Network(config.model).to(args.device)  # use the same architecture as the downstream task
    elif args.backbone == 'egnn':
        from src.models.pdc import ProbabilityDensityCloud
        model = ProbabilityDensityCloud(config.model).to(args.device)
    else:
        from src.models.rde import CircularSplineRotamerDensityEstimator
        model = CircularSplineRotamerDensityEstimator(ckpt['config'].model).to(args.device)
        loss_dict = {}

    logger.info('Loading state dict...')
    model.load_state_dict(ckpt['model'])

    chis_logger = {'1': [], '2': [], '3': [], '4': []}
    with torch.no_grad():
        model.eval()

        for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
            batch = recursive_to(batch, args.device)
            if args.backbone == 'ga':   # RDE
                xs = model(batch)    # (N, L, 4)
                xs = torch.cat(xs, dim=-1)
                print(xs.shape)
                xs -= np.pi
                n_chis_data = batch['chi_mask'].sum(-1)  # (N, L, 4) -> (N, L)
                chi_complete = batch['chi_complete']  # (N, L), only consider complete chi-angles

                print(xs.max(), xs.min(), batch['chi'].max(), batch['chi'].min())
                for res_name, num_chis in num_chi_angles.items():
                    if num_chis < 1:
                        continue
                    loss_mask = torch.logical_and(chi_complete, batch['aa'] == res_name._value_)
                    for n_chis in range(1, num_chis + 1):
                        loss = torch.nn.functional.l1_loss(xs.squeeze(-1)[..., n_chis - 1][loss_mask], batch['chi'][..., n_chis - 1][loss_mask], reduction='none')
                        # raise ValueError
                        loss_dict[f'mae_{res_name._name_}_{n_chis}'] = loss

            else:
                loss_dict = model(batch, mode='test')
            for key in loss_dict.keys():
                if len(chis_logger[key[-1]]) == 0:
                    chis_logger[key[-1]] = loss_dict[key]
                else:
                    chis_logger[key[-1]] = torch.cat([chis_logger[key[-1]], loss_dict[key]])

                if key not in chis_logger:
                    chis_logger[key] = loss_dict[key]
                else:
                    chis_logger[key] = torch.cat([chis_logger[key], loss_dict[key]])

    chis_logger = {k: v.mean().item() / (2 * np.pi) * 360 for k, v in chis_logger.items()}

    results = pd.DataFrame.from_dict(chis_logger, orient='index')
    results.to_csv(args.ckpt.split('.')[0] + args.output + '.csv', index=True)







































