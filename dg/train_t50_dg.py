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

from src.utils.misc import load_config, seed_all, get_logger, get_new_log_dir, current_milli_time
from src.utils.train import *
from src.utils.t50 import T50DatasetManager

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs_t50')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pc', action='store_true', default=False)
    args = parser.parse_args()

    # Load configs
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
    dataset_mgr = T50DatasetManager(config, num_workers=args.num_workers, logger=logger, )
    num_cvfolds = len(dataset_mgr.train_iterators)
    logger.info('Building model...')
    if config.model.type == 'ga':
        from src.models.rde_dg import DG_RDE_Network
        cv_mgr = CrossValidation(model_factory=DG_RDE_Network, config=config, num_cvfolds=num_cvfolds).to(args.device)
    else:  # TODO
        from src.models.pdc_ddg import DG_RDE_Network
        cv_mgr = CrossValidation(model_factory=DG_RDE_Network, config=config, num_cvfolds=num_cvfolds).to(args.device)
    print(f'Number of parameters: {count_parameters(cv_mgr.get(0)[0]) / 1e6:.2f}M')
    it_first = 1

    # Resume
    if args.resume is not None:
        logger.info('Resuming from checkpoint: %s' % args.resume)
        ckpt = torch.load(args.resume, map_location=args.device)
        it_first = ckpt['iteration']  # + 1
        cv_mgr.load_state_dict(ckpt['model'], )


    def train(it):
        fold = it % num_cvfolds
        model, optimizer, scheduler = cv_mgr.get(fold)
        time_start = current_milli_time()
        model.train()

        batch = recursive_to(next(dataset_mgr.get_train_iterator(fold)), args.device)
        loss_dict, _ = model(batch)
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        time_forward_end = current_milli_time()
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        time_backward_end = current_milli_time()

        # Logging
        scalar_dict = {'grad': orig_grad_norm, 'lr(1e5)': optimizer.param_groups[0]['lr'] * 1e5, 'time_forward': (time_forward_end - time_start) / 1000,
                       'time_backward': (time_backward_end - time_forward_end) / 1000, }
        logstr = '[%s] Iter %05d | loss %.4f | fold %.0f' % ('train', it, loss.item(), fold)
        for k, v in scalar_dict.items():
            logstr += ' | %s %.3f' % (k, v.item() if isinstance(v, torch.Tensor) else v)
        return logstr


    def validate(it, best_it, best_metric):
        scalar_accum = ScalarMetricAccumulator()
        results = []
        with torch.no_grad():
            for fold in range(num_cvfolds):
                model, optimizer, scheduler = cv_mgr.get(fold)
                for batch in tqdm(dataset_mgr.get_val_loader(fold), desc='Validate', dynamic_ncols=True):
                    batch = recursive_to(batch, args.device)

                    loss_dict, output_dict = model(batch)
                    loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                    scalar_accum.add(name='loss', value=loss, batchsize=batch['size'], mode='mean')

                    for dg_true, dg_pred in zip(output_dict['dG_true'], output_dict['dG_pred']):
                        results.append({'dG': dg_true.item(), 'dG_pred': dg_pred.item()})

        results = pd.DataFrame(results)   # dG has no per-structure metrics
        pearson_all = results[['dG', 'dG_pred']].corr('pearson').iloc[0, 1]
        spearman_all = results[['dG', 'dG_pred']].corr('spearman').iloc[0, 1]

        logger.info(f'[All] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f}')
        writer.add_scalar('val/all_pearson', pearson_all, it)
        writer.add_scalar('val/all_spearman', spearman_all, it)

        # if args.pc:
        #     pearson_pc, spearman_pc = per_complex_corr_dg(results)
        #     logger.info(f'[PC]  Pearson {pearson_pc:.6f} Spearman {spearman_pc:.6f}')
        #     writer.add_scalar('val/pc_pearson', pearson_pc, it)
        #     writer.add_scalar('val/pc_spearman', spearman_pc, it)

        avg_loss = scalar_accum.get_average('loss')
        scalar_accum.log(it, 'val', best_it=best_it, best_metric=best_metric, logger=logger, writer=writer)
        # Trigger scheduler
        for fold in range(num_cvfolds):
            _, _, scheduler = cv_mgr.get(fold)
            if it != it_first:  # Don't step optimizers after resuming from checkpoint
                if config.train.scheduler.type == 'plateau':
                    scheduler.step(avg_loss)
                else:
                    scheduler.step()

        # if args.pc:
        #     return avg_loss, pearson_pc, spearman_pc
        return avg_loss, pearson_all, spearman_all

    try:
        best_spearman_val, best_i = 0.0, 0
        it_tqdm = tq(range(it_first, config.train.max_iters + 1))
        for i in it_tqdm:
            message = train(i)
            it_tqdm.set_description(message)

            if i % config.train.val_freq == 0:
                avg_val_loss, pearson_val, spearman_val = validate(i, best_i, best_spearman_val)
                if spearman_val > best_spearman_val:
                    best_spearman_val = spearman_val
                    best_i = i
                if not args.debug:
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % i)
                    torch.save({'config': config, 'model': cv_mgr.state_dict(), 'iteration': i, 'avg_val_loss': avg_val_loss, }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')
