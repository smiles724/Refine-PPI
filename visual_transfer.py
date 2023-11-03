import argparse
import pandas as pd
import torch.utils.tensorboard
from tqdm.auto import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from rde.utils.misc import get_logger
from rde.utils.train import *
from rde.utils.skempi import SkempiDatasetManager, eval_skempi_three_modes

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
        from rde.models.rde_ddg import DDG_RDE_Network

        cv_mgr = CrossValidation(model_factory=DDG_RDE_Network, config=config, num_cvfolds=num_cvfolds).to(args.device)
    else:
        from rde.models.pdc_ddg import DDG_PDC_Network

        cv_mgr = CrossValidation(model_factory=DDG_PDC_Network, config=config, num_cvfolds=num_cvfolds).to(args.device)
    logger.info('Loading state dict...')
    cv_mgr.load_state_dict(ckpt['model'])

    results = []
    with torch.no_grad():
        for fold in range(num_cvfolds):
            model, _, _ = cv_mgr.get(fold)
            model.eval()
            for i, batch in enumerate(tqdm(dataset_mgr.get_val_loader(fold), desc=f'Fold {fold + 1}/{num_cvfolds}', dynamic_ncols=True)):
                batch = recursive_to(batch, args.device)

                batch_wt = {k: v for k, v in batch.items()}
                batch_mt = {k: v for k, v in batch.items()}
                batch_mt['aa'] = batch_mt['aa_mut']

                h_wt = model._encode_pretrain(batch_wt)[0]
                h_mt = model._encode_pretrain(batch_mt)[0]

                # H_mt, H_wt = h_mt.max(dim=1)[0], h_wt.max(dim=1)[0]
                # ddg_pred = model.ddg_readout(H_mt - H_wt).squeeze(-1)

                for complex, feat_wt, feat_mt, mutstr, dg_wt, dg_mt, ddg in zip(batch['complex'], h_wt, h_mt, batch['mutstr'], batch['dG_wt'], batch['dG_mt'], batch['ddG']):
                    results.append({'complex': complex, 'feat_wt': feat_wt.cpu().numpy(), 'feat_mt': feat_mt.cpu().numpy(), 'num_muts': len(mutstr.split(',')),
                                    'dG_wt': dg_wt.item(), 'dG_mt': dg_mt.item(), 'ddG': ddg.item()})

    torch.save(results, f'transfer_{config.data.get("predicted_structure", True)}.pt')

    results = pd.DataFrame(results)
    df_metrics = eval_skempi_three_modes(results)
    print(df_metrics)