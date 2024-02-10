import os
import argparse
import pandas as pd
import torch.utils.tensorboard
from tqdm.auto import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from rde.models.rde_ddg import DDG_RDE_Network
from rde.utils.misc import get_logger
from rde.utils.train import *
from rde.utils.skempi import SkempiDatasetManager, eval_skempi_three_modes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--feature', action='store_true', help='save output feature')
    parser.add_argument('-o', '--output', type=str, default='skempi_results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    if args.ckpt.endswith('.pt'):
        logger = get_logger('test', None)
        ckpt = torch.load(args.ckpt)
        config = ckpt['config']
        num_cvfolds = len(ckpt['model']['models'])
        print(config)

        dataset_mgr = SkempiDatasetManager(config, num_cvfolds=num_cvfolds, num_workers=args.num_workers, logger=logger, )
        cv_mgr = CrossValidation(model_factory=DDG_RDE_Network, config=config, num_cvfolds=num_cvfolds).to(args.device)
        print('Loading state dict...')
        # cv_mgr.load_state_dict(ckpt['model'])

        results, feats = [], {}
        with torch.no_grad():
            for fold in range(num_cvfolds):
                model, _, _ = cv_mgr.get(fold)
                model.eval()
                for i, batch in enumerate(tqdm(dataset_mgr.get_val_loader(fold), desc=f'Fold {fold + 1}/{num_cvfolds}', dynamic_ncols=True)):
                    batch = recursive_to(batch, args.device)
                    if args.feature:   # to obtain feature
                        feat_dict = model(batch, return_feat=True)
                        for complex, mutstr, feat_wt, feat_mt in zip(batch['complex'], batch['mutstr'], feat_dict['feat_wt'], feat_dict['feat_mt']):
                            feats[complex + mutstr] = (feat_wt.cpu(), feat_mt.cpu())
                    else:
                        loss_dict, output_dict = model(batch)
                        for complex, mutstr, ddg_true, ddg_pred in zip(batch['complex'], batch['mutstr'], output_dict['ddG_true'], output_dict['ddG_pred']):
                            results.append({'complex': complex, 'mutstr': mutstr, 'num_muts': len(mutstr.split(',')), 'ddG': ddg_true.item(), 'ddG_pred': ddg_pred.item(), })
        if results:
            results = pd.DataFrame(results)
            results.to_csv(args.ckpt.split('.')[0] + args.output + '.csv', index=False)
    else:
        results = pd.read_csv(args.ckpt)

    if args.feature:
        torch.save(feats, os.path.join(config.data.cache_dir, f'skempi_{config.model.type}_feat.pt'))
    else:
        df_metrics = eval_skempi_three_modes(results)
        print(df_metrics)
        df_metrics.to_csv(args.ckpt.split('.')[0] + args.output + '_metrics.csv', index=False)
