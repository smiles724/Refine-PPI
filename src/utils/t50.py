import os
import functools
import pandas as pd
from torch.utils.data import DataLoader

from src.datasets import T50DDGDataset, T50DGDataset
from src.utils.data import PaddingCollate
from src.utils.misc import inf_iterator, BlackHole
from src.utils.transforms import get_transform


def per_complex_corr_dg(df, pred_attr='dG_pred', limit=10):
    corr_table = []
    for cplx in df['complex'].unique():
        df_cplx = df.query(f'complex == "{cplx}"')
        if len(df_cplx) < limit:
            continue
        corr_table.append({'complex': cplx, 'pearson': df_cplx[['dG', pred_attr]].corr('pearson').iloc[0, 1], 'spearman': df_cplx[['dG', pred_attr]].corr('spearman').iloc[0, 1]})
    corr_table = pd.DataFrame(corr_table)
    avg = corr_table[['pearson', 'spearman']].mean()
    return avg['pearson'], avg['spearman']


class T50DatasetManager(object):

    def __init__(self, config, num_workers=4, logger=BlackHole()):
        super().__init__()
        self.target = config.data.target
        self.config = config
        self.loaders = []
        self.logger = logger
        self.num_workers = num_workers
        self.train_iterators = []
        self.val_loaders = []

        all_folds = os.listdir(config.data.root_path)
        for fold in all_folds:
            fold_path = os.path.join(config.data.root_path, fold)
            if os.path.isdir(fold_path):
                train_iterator, val_loader = self.init_loaders(fold_path)
                self.train_iterators.append(train_iterator)
                self.val_loaders.append(val_loader)

    def init_loaders(self, fold_path):
        config = self.config
        if self.target.lower() == 'ddg':  # TODO
            dataset_ = functools.partial(T50DDGDataset, csv_path=config.data.csv_path, cache_dir=config.data.cache_dir, transform=get_transform(config.data.transform),
                                         reset=config.data.reset)
        elif self.target.lower() == 'dg':
            dataset_ = functools.partial(T50DGDataset, fold_path=fold_path, cache_dir=config.data.cache_dir, transform=get_transform(config.data.transform),
                                         reset=config.data.reset)
        else:
            raise ValueError(f'Target can be only dg or ddg.')
        train_dataset = dataset_(split='train')
        val_dataset = dataset_(split='valid')
        # test_dataset = dataset_(split='test')

        train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=PaddingCollate(), num_workers=self.num_workers)
        train_iterator = inf_iterator(train_loader)
        val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=PaddingCollate(), num_workers=self.num_workers)
        # test_loader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=PaddingCollate(), num_workers=self.num_workers)
        self.logger.info('Train %d, Val %d' % (len(train_dataset), len(val_dataset)))
        return train_iterator, val_loader

    def get_train_iterator(self, fold):
        return self.train_iterators[fold]

    def get_val_loader(self, fold):
        return self.val_loaders[fold]
