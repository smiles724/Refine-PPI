import argparse

import torch.utils.tensorboard
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from src.utils.train import *
from src.utils.skempi import SkempiDatasetManager
from src.datasets.pdbredo_chain import get_pdbredo_chain_dataset
from src.utils.misc import load_config, seed_all, get_logger
from src.utils.data import PaddingCollate

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, default='')
parser.add_argument('--data', type=str, default='skempi', choices=['skempi', 'redo'])
parser.add_argument('--backbone', type=str, default='egnn', choices=['egnn', 'ga'])
parser.add_argument('--ckpt', type=str, default='trained_models/DDG_RDE_Network_30k.pt')
parser.add_argument('-o', '--output', type=str, default='skempi_results')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()
logger = get_logger('test', None)

if args.data == 'redo':
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    train_dataset = get_pdbredo_chain_dataset('train', config.data)
    val_dataset = get_pdbredo_chain_dataset('val', config.data)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=PaddingCollate(), num_workers=args.num_workers)

    pos, pos_gt, aa, resseq, res_nb, chain_nb = [], [], [], [], [], []
    loss = []
    for i, batch in enumerate(tqdm(train_loader, desc='Train', dynamic_ncols=True)):
        batch = recursive_to(batch, args.device)

        # raise ValueError
        for j in range(len(batch['pos_change_flag'])):
            pos_change_flag = batch['pos_change_flag'][j]
            l = (((batch['pos_gt'][j][pos_change_flag] - batch['pos_atoms'][j][pos_change_flag]) ** 2).sum(dim=-1) ** 0.5)

            if torch.max(l) > 1000:
                loss.append(l.sum().item())
                pos.append(batch['pos_atoms'][j])
                pos_gt.append(batch['pos_gt'][j])
                aa.append(batch['aa'][j])
                resseq.append(batch['resseq'][j])
                res_nb.append(batch['res_nb'][j])
                chain_nb.append(batch['chain_nb'][j])

                print(l.sum(), torch.max(l), torch.min(l))

        if len(loss) > 100:
            torch.save(loss, 'loss_redo.pt')
            break
    torch.save(loss, 'loss_redo.pt')

    pos = torch.stack(pos).cpu()
    pos_gt = torch.stack(pos_gt).cpu()
    aa = torch.stack(aa).cpu()
    resseq = torch.stack(resseq).cpu()
    res_nb = torch.stack(res_nb).cpu()
    chain_nb = torch.stack(chain_nb).cpu()

    torch.save([pos, pos_gt, aa, resseq, res_nb, chain_nb], 'pos_redo.pt')

else:
    ckpt = torch.load(args.ckpt)
    config = ckpt['config']
    num_cvfolds = len(ckpt['model']['models'])
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

    pos_wt_true, pos_wt_pred, pos_mt_pred, wt_aa, mt_aa, resseq, res_nb, chain_nb = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for fold in range(num_cvfolds):
            model, _, _ = cv_mgr.get(fold)
            for i, batch in enumerate(tqdm(dataset_mgr.get_val_loader(fold), desc=f'Fold {fold + 1}/{num_cvfolds}', dynamic_ncols=True)):
                batch = recursive_to(batch, args.device)

                _, output_dict = model(batch, return_pos=True)
                pos_wt_true.append(output_dict['pos'])   # TODO: from batch
                pos_wt_pred.append(output_dict['pos_wt_pred'])
                pos_mt_pred.append(output_dict['pos_mt_pred'])
                wt_aa.append(output_dict['aa'])
                mt_aa.append(output_dict['aa_mut'])
                resseq.append(output_dict['resseq'])
                chain_nb.append(output_dict['chain_nb'])
                break
            break

        pos_wt_true = torch.cat(pos_wt_true).cpu()
        pos_wt_pred = torch.cat(pos_wt_pred).cpu()
        pos_mt_pred = torch.cat(pos_mt_pred).cpu()
        wt_aa = torch.cat(wt_aa).cpu()
        mt_aa = torch.cat(mt_aa).cpu()
        resseq = torch.cat(resseq).cpu()
        chain_nb = torch.cat(chain_nb).cpu()

    torch.save([pos_wt_true, pos_wt_pred, pos_mt_pred, wt_aa, mt_aa, resseq, chain_nb], 'pos_visual.pt')

