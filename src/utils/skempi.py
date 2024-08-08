import functools

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets import SkempiABbindDataset
from src.utils.data import PaddingCollate
from src.utils.misc import inf_iterator, BlackHole
from src.utils.transforms import get_transform


def per_complex_corr(df, pred_attr='ddG_pred', limit=10):
    corr_table = []
    for cplx in df['complex'].unique():
        df_cplx = df.query(f'complex == "{cplx}"')    # use 'complex' to separate per-structure
        if len(df_cplx) < limit:
            continue
        corr_table.append(
            {'complex': cplx, 'pearson': df_cplx[['ddG', pred_attr]].corr('pearson').iloc[0, 1], 'spearman': df_cplx[['ddG', pred_attr]].corr('spearman').iloc[0, 1], })
    corr_table = pd.DataFrame(corr_table)
    avg = corr_table[['pearson', 'spearman']].mean()
    return avg['pearson'], avg['spearman']


class SkempiDatasetManager(object):

    def __init__(self, cfg, num_cvfolds, num_workers=4, logger=BlackHole()):
        super().__init__()
        self.cfg = cfg
        self.num_cvfolds = num_cvfolds
        self.train_iterators = []
        self.val_loaders = []
        self.logger = logger
        self.num_workers = num_workers
        for fold in range(num_cvfolds):
            train_iterator, val_loader = self.init_loaders(fold)
            self.train_iterators.append(train_iterator)
            self.val_loaders.append(val_loader)

    def init_loaders(self, fold):
        cfg = self.cfg
        dataset_ = functools.partial(SkempiABbindDataset, skempi_csv_path=cfg.data.skempi_csv_path, skempi_pdb_dir=cfg.data.skempi_pdb_dir, cache_dir=cfg.data.cache_dir,
                                     abbind_csv_path=cfg.data.get('abbind_csv_path', None), abbind_pdb_dir=cfg.data.get('abbind_pdb_dir', None), num_cvfolds=self.num_cvfolds,
                                     cvfold_index=fold, transform=get_transform(cfg.data.transform), use_plm=cfg.model.use_plm, reset=cfg.data.reset,
                                     mask_length=cfg.model.pos.mask_length if 'pos' in cfg.model else 0,
                                     mask_noise_scale=cfg.model.pos.mask_noise_scale if 'pos' in cfg.model else 1.0)
        train_dataset = dataset_(split='train')
        val_dataset = dataset_(split='val')

        train_cplx = set([e['complex'] for e in train_dataset.entries])
        val_cplx = set([e['complex'] for e in val_dataset.entries])
        leakage = train_cplx.intersection(val_cplx)
        assert len(leakage) == 0, f'data leakage {leakage}'

        train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=PaddingCollate(), num_workers=self.num_workers)
        train_iterator = inf_iterator(train_loader)
        val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=PaddingCollate(), num_workers=self.num_workers)
        self.logger.info('Fold %d: Train %d, Val %d' % (fold, len(train_dataset), len(val_dataset)))
        return train_iterator, val_loader

    def get_train_iterator(self, fold):
        return self.train_iterators[fold]

    def get_val_loader(self, fold):
        return self.val_loaders[fold]


def overall_correlations(df):
    pearson = df[['ddG', 'ddG_pred']].corr('pearson').iloc[0, 1]
    spearman = df[['ddG', 'ddG_pred']].corr('spearman').iloc[0, 1]
    return {'overall_pearson': pearson, 'overall_spearman': spearman, }


def percomplex_correlations(df, return_details=False):
    corr_table = []
    for cplx in df['complex'].unique():
        df_cplx = df.query(f'complex == "{cplx}"')
        if len(df_cplx) < 10:
            continue
        corr_table.append(
            {'complex': cplx, 'pearson': df_cplx[['ddG', 'ddG_pred']].corr('pearson').iloc[0, 1], 'spearman': df_cplx[['ddG', 'ddG_pred']].corr('spearman').iloc[0, 1], })
    corr_table = pd.DataFrame(corr_table)
    average = corr_table[['pearson', 'spearman']].mean()
    out = {'percomplex_pearson': average['pearson'], 'percomplex_spearman': average['spearman'], }
    if return_details:
        return out, corr_table
    return out


def overall_auroc(df):
    score = roc_auc_score((df['ddG'] > 0).to_numpy(), df['ddG_pred'].to_numpy())
    return {'auroc': score, }


def overall_rmse_mae(df):
    true = df['ddG'].to_numpy()
    pred = df['ddG_pred'].to_numpy()[:, None]
    reg = LinearRegression().fit(pred, true)
    pred_corrected = reg.predict(pred)
    rmse = np.sqrt(((true - pred_corrected) ** 2).mean())
    mae = np.abs(true - pred_corrected).mean()
    return {'rmse': rmse, 'mae': mae, }


def analyze_all_results(df):
    methods = df['method'].unique()
    funcs = [overall_correlations, overall_rmse_mae, overall_auroc, percomplex_correlations, ]
    analysis = []
    for method in tqdm(methods):
        df_this = df[df['method'] == method]
        result = {'method': method, }
        for f in funcs:
            result.update(f(df_this))
        analysis.append(result)
    analysis = pd.DataFrame(analysis)
    return analysis


def analyze_all_percomplex_correlations(df):
    methods = df['method'].unique()
    df_corr = []
    for method in tqdm(methods):
        df_this = df[df['method'] == method]
        _, df_corr_this = percomplex_correlations(df_this, return_details=True)
        df_corr_this['method'] = method
        df_corr.append(df_corr_this)
    df_corr = pd.concat(df_corr).reset_index()
    return df_corr


def eval_skempi(df_items, mode, ddg_cutoff=None):
    assert mode in ('all', 'single', 'multiple')
    if mode == 'single':
        df_items = df_items.query('num_muts == 1')
    elif mode == 'multiple':
        df_items = df_items.query('num_muts > 1')

    if ddg_cutoff is not None:
        df_items = df_items.query(f"ddG >= {-ddg_cutoff} and ddG <= {ddg_cutoff}")

    df_metrics = analyze_all_results(df_items)
    df_metrics['mode'] = mode
    return df_metrics


def eval_skempi_three_modes(results, ddg_cutoff=None):
    df_all = eval_skempi(results, mode='all', ddg_cutoff=ddg_cutoff)
    df_single = eval_skempi(results, mode='single', ddg_cutoff=ddg_cutoff)
    df_multiple = eval_skempi(results, mode='multiple', ddg_cutoff=ddg_cutoff)
    df_metrics = pd.concat([df_all, df_single, df_multiple], axis=0)
    df_metrics.reset_index(inplace=True, drop=True)
    return df_metrics
