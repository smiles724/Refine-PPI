import copy
import os
import pickle

import numpy as np
import pandas as pd
from Bio.PDB.PDBParser import PDBParser
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.utils.protein.parsers import parse_biopython_structure


def load_t50_entries(csv_path):
    df = pd.read_csv(csv_path)
    df['dG'] = df['label']

    entries = []
    for i, row in df.iterrows():
        pdb_path = row['pdb_path'].split('#')[0]
        if not os.path.exists(pdb_path) or not np.isfinite(row['dG']):
            continue

        entry = {'id': i, 'dG': np.float32(row['dG']), 'pdb_path': pdb_path, 'group_receptor': list(row['protein_b_chain']),
                 'group_ligand': list(row['protein_a_chain'])}
        entries.append(entry)
    return entries


class T50DGDataset(Dataset):

    def __init__(self, fold_path, cache_dir, split='train', transform=None, blocklist=frozenset({'1KBH'}), reset=False):
        super().__init__()
        self.csv_path = os.path.join(fold_path, f'{split}.csv')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.blocklist = blocklist
        self.transform = transform
        assert split in ('train', 'valid', 'test')

        self.entries_cache = os.path.join(cache_dir, f'entries_{split}.pkl')
        self.entries = None
        self._load_entries(reset)

        self.structures_cache = os.path.join(cache_dir, f'structures_{split}.pkl')
        self.structures = None
        self._load_structures(reset)

    def _load_entries(self, reset):
        if not os.path.exists(self.entries_cache) or reset:
            self.entries = self._preprocess_entries()
        else:
            with open(self.entries_cache, 'rb') as f:
                self.entries = pickle.load(f)

    def _preprocess_entries(self):
        entries = load_t50_entries(self.csv_path)
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
                data, seq_map = parse_biopython_structure(model)
                structures[path] = data

        with open(self.structures_cache, 'wb') as f:
            pickle.dump(structures, f)
        return structures

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        dG, pdb_path, ag_chain = entry['dG'], entry['pdb_path'], entry['group_receptor']
        data = copy.deepcopy(self.structures[pdb_path])
        data['dG'], data['ag_chain'] = dG, ag_chain
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
