import functools

import pandas as pd
from torch.utils.data import DataLoader

from rde.datasets import T50DGDataset
from rde.utils.data import PaddingCollate
from rde.utils.misc import inf_iterator, BlackHole
from rde.utils.transforms import get_transform


def per_complex_corr_dg(df, pred_attr='dG_pred', limit=10):
    corr_table = []
    for cplx in df['complex'].unique():
        df_cplx = df.query(f'complex == "{cplx}"')
        assert len(df_cplx) > limit, f'{cplx} has {len(df_cplx)} < {limit} items...'
        corr_table.append({'complex': cplx, 'pearson': df_cplx[['dG', pred_attr]].corr('pearson').iloc[0, 1], 'spearman': df_cplx[['dG', pred_attr]].corr('spearman').iloc[0, 1]})
    corr_table = pd.DataFrame(corr_table)
    avg = corr_table[['pearson', 'spearman']].mean()
    return avg['pearson'], avg['spearman'], corr_table


def T50DatasetManager(config, num_workers=4, logger=BlackHole()):
    dataset_ = functools.partial(T50DGDataset, csv_path=config.data.train_csv_path, pdb_root=config.data.train_pdb_root, test_root=config.data.test_root,
                                 cache_dir=config.data.cache_dir, transform=get_transform(config.data.transform), reset=config.data.reset, use_plm=config.data.use_plm)
    train_dataset, val_dataset = dataset_(split='train'), dataset_(split='val')

    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=PaddingCollate(), num_workers=num_workers)
    train_iterator = inf_iterator(train_loader)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size * 2, shuffle=False, collate_fn=PaddingCollate(), num_workers=num_workers)
    logger.info('Train %d, Val %d' % (len(train_dataset), len(val_dataset)))
    return train_iterator, val_loader



