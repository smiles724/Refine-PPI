import copy
import os
import pickle

import torch
import numpy as np
import pandas as pd
from Bio.PDB.PDBParser import PDBParser
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from rde.utils.protein.constants import Fragment
from rde.utils.protein.parsers import parse_biopython_structure


def load_t50_train_entries(train_pdb_root, csv_path):
    df = pd.read_csv(csv_path)
    df['dG'] = df['label']

    entries = []
    for i, row in df.iterrows():
        pdb_path = os.path.join(train_pdb_root, row['source_hash'].strip() + '.pdb')
        assert os.path.exists(pdb_path) and np.isfinite(row['dG'])
        entry = {'complex': row['source_hash'], 'dG': np.float32(row['dG']), 'pdb_path': pdb_path, }    # B: vh, C: vl, D: ag
        entries.append(entry)
    return entries


def load_t50_test_entries(test_root):
    entries = []
    for file in os.listdir(test_root):
        if file.endswith('.csv'):
            seed = file.split('_')[0]
            df = pd.read_csv(os.path.join(test_root, file))
            for i, row in df.iterrows():
                pdb_path = os.path.join(test_root, seed, row[' hash_filepre'].strip() + '.pdb')
                assert os.path.exists(pdb_path) and np.isfinite(row[' dG (Kcal/mol)'])
                entry = {'complex': seed, 'dG': np.float32(row[' dG (Kcal/mol)']), 'pdb_path': pdb_path, }   # A：vh，B：vl，C：ag
                entries.append(entry)
    return entries


class T50DGDataset(Dataset):

    def __init__(self, csv_path, pdb_root, test_root, cache_dir, split='train', use_plm=False, transform=None, blocklist=frozenset({'1KBH'}), reset=False):
        super().__init__()
        self.csv_path = csv_path
        self.pdb_root = pdb_root
        self.cache_dir = cache_dir
        self.test_root = test_root
        os.makedirs(cache_dir, exist_ok=True)
        self.blocklist = blocklist
        self.transform = transform
        self.use_plm = use_plm
        self.split = split
        assert split in ('train', 'val', 'test')   # no val

        self.entries_cache = os.path.join(cache_dir, f'entries_{split}.pkl')
        self.entries = None
        self._load_entries(reset)

        self.structures_cache = os.path.join(cache_dir, f'structures_{split}.pkl')
        self.structures = None
        self._load_structures(reset)
        if use_plm:
            self.plm_feature = torch.load(os.path.join(self.cache_dir, f'esm2_embeddings_{split}.pt'))

    def _load_entries(self, reset):
        if not os.path.exists(self.entries_cache) or reset:
            self.entries = self._preprocess_entries()
        else:
            with open(self.entries_cache, 'rb') as f:
                self.entries = pickle.load(f)

    def _preprocess_entries(self):
        if self.split == 'train':
            entries = load_t50_train_entries(self.pdb_root, self.csv_path)
        else:
            entries = load_t50_test_entries(self.test_root)

        with open(self.entries_cache, 'wb') as f:
            pickle.dump(entries, f)
        return entries

    def _load_structures(self, reset):
        if not os.path.exists(self.structures_cache) or reset:
            self.structures = self._preprocess_structures()
        else:
            with open(self.structures_cache, 'rb') as f:
                self.structures = pickle.load(f)

    def _preprocess_structures(self):
        path_list = [e['pdb_path'] for e in self.entries]

        structures = {}
        for path in tqdm(path_list, desc='Structures'):
            parser = PDBParser(QUIET=True)
            if path not in structures.keys():
                model = parser.get_structure(None, path)[0]
                data, _ = parse_biopython_structure(model)
                structures[path] = data

        with open(self.structures_cache, 'wb') as f:
            pickle.dump(structures, f)
        return structures

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        dG, pdb_path = entry['dG'], entry['pdb_path']
        data = copy.deepcopy(self.structures[pdb_path])
        data['dG'], data['complex'] = dG, entry['complex']

        group_id = []
        if 'A' not in data['chain_id']:
            domain = {'group_ag': ['D'], 'group_vh': ['B'], 'group_vl': ['C']}
        else:
            domain = {'group_ag': ['C'], 'group_vh': ['A'], 'group_vl': ['B']}
        for ch in data['chain_id']:
            if ch in domain['group_vh']:
                group_id.append(Fragment.Heavy)
            elif ch in domain['group_vl']:
                group_id.append(Fragment.Light)
            elif ch in domain['group_ag']:
                group_id.append(Fragment.Antigen)
            else:
                raise ValueError(f"group ID {ch} not in {domain['group_vh'] + domain['group_vl'] + domain['group_ag']}. Path: {entry['pdb_path']}")
        data['fragment_type'] = torch.LongTensor(group_id)
        if self.use_plm:
            data['plm_wt'] = copy.deepcopy(self.plm_feature[entry['pdb_path'].split('/')[-1][:-4]])

        if self.transform is not None:
            data = self.transform(data)
        return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='/nfs_baoding_os/haokai/data/affinity/dG/0608_bidock/bidock/splited/train/scenario-4_case-2/fold-0/')
    parser.add_argument('--cache_dir', type=str, default='./data/T50_cache')
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args()

    # dataset = T50DGDataset(csv_path=args.csv_path, cache_dir=args.cache_dir, split='valid', reset=args.reset, )
    # print(dataset[0])
    # print(len(dataset))
