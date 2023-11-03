import copy
import os
import pickle

import lmdb
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

_TRANSFORM_DICT = {}


def get_transform(cfg):
    if cfg is None or len(cfg) == 0:
        return None
    tfms = []
    for t_dict in cfg:
        t_dict = copy.deepcopy(t_dict)
        cls = _TRANSFORM_DICT[t_dict.pop('type')]
        tfms.append(cls(**t_dict))
    return Compose(tfms)


class SAbDabDataset(Dataset):
    MAP_SIZE = 32 * (1024 * 1024 * 1024)  # 32GB

    def __init__(self, processed_dir='./data/processed', split='train', transform=None, reset=False, use_plm=False):
        super().__init__()
        self.use_plm = use_plm
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)

        # entries
        self.sabdab_entries = None
        self._load_sabdab_entries(reset)

        # structures
        self.db_conn, self.db_ids = None, None
        self._load_structures()

        # splits
        self.ids_in_split = None
        self._load_split(split)

        # plm features
        if self.use_plm:
            self._load_plm_feature()

        self.transform = transform

    def _load_sabdab_entries(self, reset):
        if not os.path.exists(self._entry_cache_path) or reset:
            raise ValueError('Please run preprocess.py first to generate entries.')
        with open(self._entry_cache_path, 'rb') as f:
            self.sabdab_entries = pickle.load(f)

    def _load_structures(self):
        path = self._structure_cache_path
        with open(path + '-ids', 'rb') as f:
            self.db_ids = pickle.load(f)
        self.sabdab_entries = list(filter(lambda e: e['id'] in self.db_ids, self.sabdab_entries))

    @property
    def _entry_cache_path(self):
        return os.path.join(self.processed_dir, 'entry')

    @property
    def _structure_cache_path(self):
        return os.path.join(self.processed_dir, 'structures.lmdb')

    @property
    def _structure_esmfold_cache_path(self):
        return os.path.join(self.processed_dir, 'structures_esmfold.lmdb')

    @property
    def _split_path(self):
        return os.path.join(self.processed_dir, 'sequence_only_split')

    def _load_split(self, split):
        assert split in ('train', 'val', 'test')
        with open(self._split_path, 'rb') as f:
            val_test_split = pickle.load(f)

        val_test_split['val'] = [entry['id'] for entry in self.sabdab_entries if entry['id'] in val_test_split['val']]
        val_test_split['test'] = [entry['id'] for entry in self.sabdab_entries if entry['id'] in val_test_split['test']]
        val_test_split['train'] = [entry['id'] for entry in self.sabdab_entries if entry['id'] not in val_test_split['val'] + val_test_split['test']]
        self.ids_in_split = val_test_split[split]

    def _load_plm_feature(self):
        plm_feature_path = os.path.join(self.processed_dir, 'esm2_embeddings.pt')
        self.plm_feature = torch.load(plm_feature_path)
        self.ids_in_split = [i for i in self.ids_in_split if i in self.plm_feature.keys()]

    def _connect_db(self):
        if self.db_conn is not None:
            return
        lmdb_path = self._structure_cache_path
        self.db_conn = lmdb.open(lmdb_path, map_size=self.MAP_SIZE, create=False, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, )

    def get_structure(self, id, ):
        self._connect_db()
        with self.db_conn.begin() as txn:
            return pickle.loads(txn.get(id.encode()))

    def __len__(self):
        return len(self.ids_in_split)

    def __getitem__(self, index):
        id = self.ids_in_split[index]
        data = self.get_structure(id)

        if self.use_plm:
            data['antigen']['plm_feature'] = self.plm_feature[id]

        if self.transform is not None:
            data = self.transform(data)
        return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--processed_dir', type=str, default='./data/processed')
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args()
    if args.reset:
        sure = input('Sure to reset? (y/n): ')
        if sure != 'y':
            exit()
    dataset = SAbDabDataset(processed_dir=args.processed_dir, split=args.split, reset=args.reset, use_plm=True)
    print(dataset[0])
    print(len(dataset), len(dataset.clusters))
