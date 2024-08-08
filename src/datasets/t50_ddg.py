import os
import copy
import random
import pickle
import math
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import one_to_index

from src.utils.protein.parsers import parse_biopython_structure


def load_t50_entries(csv_path):
    df = pd.read_csv(csv_path)
    df['ddG'] = df['diff']
    # df['dG_wt'] = (8.314 / 4184) * (273.15 + 25.0) * np.log(df['Affinity_wt_parsed'])
    # df['dG_mut'] = (8.314 / 4184) * (273.15 + 25.0) * np.log(df['Affinity_mut_parsed'])
    # df['ddG'] = df['dG_mut'] - df['dG_wt']

    # def _parse_mut(mut_name):
    #     wt_type, mutchain, mt_type = mut_name[0], mut_name[1], mut_name[-1]
    #     mutseq = int(mut_name[2:-1])
    #     return {'wt': wt_type, 'mt': mt_type, 'chain': mutchain, 'resseq': mutseq, 'icode': ' ', 'name': mut_name}

    entries = []
    for i, row in df.iterrows():
        # pdbcode, group1, group2 = row['#Pdb'].split('_')
        # if pdbcode in block_list:
        #     continue
        # mut_str = row['Mutation(s)_cleaned']
        # muts = list(map(_parse_mut, row['Mutation(s)_cleaned'].split(',')))
        # if muts[0]['chain'] in group1:
        #     group_ligand, group_receptor = group1, group2
        # else:
        #     group_ligand, group_receptor = group2, group1
        wt_path, mut_path = row['pdb_path'].split('#')
        # pdb_path = os.path.join(pdb_dir, '{}.pdb'.format(pdbcode.upper()))
        if not os.path.exists(wt_path) or not os.path.exists(mut_path):
            continue

        if not np.isfinite(row['ddG']):
            continue

        entry = {'id': i, 'ddG': np.float32(row['ddG']), 'wt_path': wt_path, 'mut_path': mut_path}
        entries.append(entry)

    return entries


class T50DDGDataset(Dataset):

    def __init__(self, csv_path, cache_dir, split='train', transform=None, blocklist=frozenset({'1KBH'}), reset=False):
        super().__init__()
        self.csv_path = os.path.join(csv_path, f'{split}.csv')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.blocklist = blocklist
        self.transform = transform
        assert split in ('train', 'valid', 'test')

        self.entries_cache = os.path.join(cache_dir, f'entries_{split}.pkl')
        self.entries = None
        self._load_entries(reset)

        self.structures_cache = os.path.join(cache_dir, 'structures.pkl')
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
        # pdbcodes = list(set([e['pdbcode'] for e in self.entries_full]))
        path_list = [[e['wt_path'], e['mut_path']] for e in self.entries]

        structures = {}
        for path_pair in tqdm(path_list, desc='Structures'):
            parser = PDBParser(QUIET=True)
            # pdb_path = os.path.join(self.pdb_dir, '{}.pdb'.format(pdbcode.upper()))

            for path in path_pair:
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
        ddG, wt_path, mut_path = entry['ddG'], entry['wt_path'], entry['mut_path']
        wt_data, mut_data = copy.deepcopy(self.structures[wt_path]), copy.deepcopy(self.structures[mut_path])

        # aa_mut = data['aa'].clone()
        # for mut in entry['mutations']:
        #     ch_rs_ic = (mut['chain'], mut['resseq'], mut['icode'])
        #     if ch_rs_ic not in seq_map: continue
        #     aa_mut[seq_map[ch_rs_ic]] = one_to_index(mut['mt'])
        # data['aa_mut'] = aa_mut
        # data['mut_flag'] = (data['aa'] != data['aa_mut'])

        if self.transform is not None:
            wt_data = self.transform(wt_data)
            mut_data = self.transform(mut_data)

        return wt_data, mut_data, ddG


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='/nfs_baoding_os/haokai/data/affinity/ddG/0609/ntimes_improve/hard/3/')
    parser.add_argument('--cache_dir', type=str, default='./data/T50_cache')
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args()

    dataset = T50Dataset(csv_path=args.csv_path, cache_dir=args.cache_dir, split='valid', reset=args.reset, )
    print(dataset[0])
    print(len(dataset))
