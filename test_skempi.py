import argparse
import pandas as pd
import torch.utils.tensorboard
from tqdm.auto import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from src.utils.misc import get_logger
from src.utils.train import *
from src.utils.skempi import SkempiDatasetManager, eval_skempi_three_modes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='egnn', choices=['egnn', 'ga'])
    parser.add_argument('--ckpt', type=str, default='trained_models/DDG_RDE_Network_30k.pt')
    parser.add_argument('-o', '--output', type=str, default='skempi_results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    logger = get_logger('test', None)

    ckpt = torch.load(args.ckpt)
    config = ckpt['config']
    num_cvfolds = len(ckpt['model']['models'])
    print(config)

    logger.info('Loading datasets...')
    dataset_mgr = SkempiDatasetManager(config, num_cvfolds=num_cvfolds, num_workers=args.num_workers, logger=logger, )
    logger.info('Building model...')
    if args.backbone == 'ga':
        from src.models.rde_ddg import DDG_RDE_Network
        cv_mgr = CrossValidation(model_factory=DDG_RDE_Network, config=config, num_cvfolds=num_cvfolds).to(args.device)
    else:
        if 'pos' in config.model:
            from src.models.pdc_ddg_refine import DDG_PDC_Network
        else:
            from src.models.pdc_ddg import DDG_PDC_Network
        cv_mgr = CrossValidation(model_factory=DDG_PDC_Network, config=config, num_cvfolds=num_cvfolds).to(args.device)
    logger.info('Loading state dict...')
    cv_mgr.load_state_dict(ckpt['model'])

    scalar_accum = ScalarMetricAccumulator()
    results = []
    with torch.no_grad():
        for fold in range(num_cvfolds):
            model, _, _ = cv_mgr.get(fold)
            model.eval()
            for i, batch in enumerate(tqdm(dataset_mgr.get_val_loader(fold), desc=f'Fold {fold + 1}/{num_cvfolds}', dynamic_ncols=True)):
                batch = recursive_to(batch, args.device)

                loss_dict, output_dict = model(batch, return_pos=True)
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                scalar_accum.add(name='loss', value=loss, batchsize=batch['size'], mode='mean')

                pos_mse = (((output_dict['pos'] - output_dict['pos_wt_pred']) ** 2).sum(-1) ** 0.5)[:, :, 1].sum(-1)

                for complex, mutstr, ddg_true, ddg_pred, mse in zip(batch['complex'], batch['mutstr'], output_dict['ddG_true'], output_dict['ddG_pred'], pos_mse):
                    results.append({'complex': complex, 'mutstr': mutstr, 'num_muts': len(mutstr.split(',')), 'ddG': ddg_true.item(), 'ddG_pred': ddg_pred.item(), 'pos_mse': mse.item()})

    results = pd.DataFrame(results)
    results['method'] = 'PDC-Net'
    results.to_csv(args.ckpt.split('.')[0] + args.output + '.csv', index=False)
    df_metrics = eval_skempi_three_modes(results)
    print(df_metrics)
    df_metrics.to_csv(args.ckpt.split('.')[0] + args.output + '_metrics.csv', index=False)
