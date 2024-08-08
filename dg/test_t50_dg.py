import argparse
import pandas as pd
import torch
import torch.utils.tensorboard
from tqdm.auto import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from src.utils.misc import get_logger
from src.utils.train import *
from src.utils.t50 import T50DatasetManager, per_complex_corr_dg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--backbone', type=str, default='egnn', choices=['egnn', 'ga'])
    parser.add_argument('--ckpt', type=str, default='trained_models/DDG_RDE_Network_30k.pt')
    parser.add_argument('-o', '--output', type=str, default='skempi_results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    logger = get_logger('test', None)

    ckpt = torch.load(args.ckpt)
    config = ckpt['config']
    num_cvfolds = len(ckpt['model']['models'])

    logger.info('Loading datasets...')   # TODO: test using held-out set
    dataset_mgr = T50DatasetManager(config, num_workers=args.num_workers, logger=logger, )
    logger.info('Building model...')
    from src.models.rde_dg import DG_RDE_Network
    cv_mgr = CrossValidation(model_factory=DG_RDE_Network, config=config, num_cvfolds=num_cvfolds).to(args.device)
    logger.info('Loading state dict...')
    cv_mgr.load_state_dict(ckpt['model'])

    scalar_accum = ScalarMetricAccumulator()
    results = []
    with torch.no_grad():
        for fold in range(num_cvfolds):
            model, _, _ = cv_mgr.get(fold)
            for i, batch in enumerate(tqdm(dataset_mgr.get_val_loader(fold), desc=f'Fold {fold + 1}/{num_cvfolds}', dynamic_ncols=True)):
                batch = recursive_to(batch, args.device)

                loss_dict, output_dict = model(batch)
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                scalar_accum.add(name='loss', value=loss, batchsize=batch['size'], mode='mean')

                for ddg_true, ddg_pred in zip(output_dict['dG_true'], output_dict['dG_pred']):
                    results.append({'dG': ddg_true.item(), 'dG_pred': ddg_pred.item()})

    results = pd.DataFrame(results)
    results.to_csv(args.ckpt.split('.')[0] + args.output + '.csv', index=False)
    pearson_all = results[['dG', 'dG_pred']].corr('pearson').iloc[0, 1]
    spearman_all = results[['dG', 'dG_pred']].corr('spearman').iloc[0, 1]
    print(f'test/all_pearson: {pearson_all} test/all_spearman: {spearman_all}')
