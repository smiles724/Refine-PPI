import argparse
import os
import shutil

import pandas as pd
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm as tq
from tqdm.auto import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from rde.utils.misc import load_config, seed_all, get_logger, get_new_log_dir
from rde.utils.train import *
from rde.utils.t50 import T50DatasetManager, per_complex_corr_dg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs_t50')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
        ckpt_dir = None
    else:
        if args.resume:
            log_dir = get_new_log_dir(args.logdir, prefix='%s-resume' % config_name)
        else:
            log_dir = get_new_log_dir(args.logdir, prefix='%s' % config_name)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)

    logger.info('Loading datasets...')
    train_iterator, val_loader = T50DatasetManager(config, num_workers=args.num_workers, logger=logger, )
    logger.info('Building model...')
    if config.model.type == 'ga':
        from rde.models.rde_dg import DG_RDE_Network
        cv_mgr = CrossValidation(model_factory=DG_RDE_Network, config=config, num_cvfolds=1).to(args.device)
    elif config.model.type.lower() == 'pdc':
        from rde.models.pdc_dg import DG_PDC_Network
        cv_mgr = CrossValidation(model_factory=DG_PDC_Network, config=config, num_cvfolds=1).to(args.device)
    elif config.model.type.lower() == 'equiformer':
        from rde.models.equiformer_dg import DG_Equiformer
        cv_mgr = CrossValidation(model_factory=DG_Equiformer, config=config, num_cvfolds=1).to(args.device)
    model, optimizer, scheduler = cv_mgr.get(0)
    print(f'Number of parameters: {count_parameters(model) / 1e6:.2f}M')
    it_first = 1

    if args.resume is not None:
        logger.info('Resuming from checkpoint: %s' % args.resume)
        ckpt = torch.load(args.resume, map_location=args.device)
        it_first = ckpt['iteration']  # + 1
        cv_mgr.load_state_dict(ckpt['model'], )


    def train(it):
        model.train()
        batch = recursive_to(next(train_iterator), args.device)
        loss_dict, _ = model(batch)
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        scalar_dict = {'grad': orig_grad_norm, 'lr(1e5)': optimizer.param_groups[0]['lr'] * 1e5, }
        logstr = '[%s] Iter %05d | loss %.4f ' % ('train', it, loss.item())
        for k, v in scalar_dict.items():
            logstr += ' | %s %.3f' % (k, v.item() if isinstance(v, torch.Tensor) else v)
        return logstr


    def validate(it, best_it, best_metric):
        scalar_accum = ScalarMetricAccumulator()
        results = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validate', dynamic_ncols=True):
                batch = recursive_to(batch, args.device)
                loss_dict, output_dict = model(batch)
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                scalar_accum.add(name='loss', value=loss, batchsize=batch['size'], mode='mean')

                for complex, dg_true, dg_pred in zip(batch['complex'], output_dict['dG_true'], output_dict['dG_pred']):
                    results.append({'complex': complex, 'dG': dg_true.item(), 'dG_pred': dg_pred.item()})

        results = pd.DataFrame(results)
        pearson_pc, spearman_pc, corr_table = per_complex_corr_dg(results)
        logger.info(f'[PC]  Pearson {pearson_pc:.6f} Spearman {spearman_pc:.6f}')
        writer.add_scalar('val/pc_pearson', pearson_pc, it)
        writer.add_scalar('val/pc_spearman', spearman_pc, it)

        avg_loss = scalar_accum.get_average('loss')
        if spearman_pc > best_metric:
            best_metric = spearman_pc
            best_it = i
            if not args.debug:
                corr_table.to_csv(os.path.join(ckpt_dir, f'pc_{it}.csv'))
        scalar_accum.log(it, 'val', best_it=best_it, best_metric=best_metric, logger=logger, writer=writer)
        if it != it_first:
            if config.train.scheduler.type == 'plateau':
                scheduler.step(avg_loss)
            else:
                scheduler.step()
        return avg_loss, best_metric, best_it

    try:
        best_spearman_val, best_i = -1.0, 0
        it_tqdm = tq(range(it_first, config.train.max_iters + 1))
        for i in it_tqdm:
            message = train(i)
            it_tqdm.set_description(message)

            if i % config.train.val_freq == 0:
                avg_val_loss, best_spearman_val, best_i = validate(i, best_i, best_spearman_val)
                if not args.debug and best_i == i:   # only save the best iteration
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % i)
                    torch.save({'config': config, 'model': cv_mgr.state_dict(), 'iteration': i, 'avg_val_loss': avg_val_loss, }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')
