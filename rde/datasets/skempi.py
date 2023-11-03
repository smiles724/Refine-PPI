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

from rde.utils.protein.parsers import parse_biopython_structure


def load_skempi_entries(csv_path, pdb_dir, block_list, predicted_structure=False):
    df = pd.read_csv(csv_path, sep=';')
    df['dG_wt'] = (8.314 / 4184) * (273.15 + 25.0) * np.log(df['Affinity_wt_parsed'])
    df['dG_mut'] = (8.314 / 4184) * (273.15 + 25.0) * np.log(df['Affinity_mut_parsed'])
    df['ddG'] = df['dG_mut'] - df['dG_wt']   # kcal/mol

    def _parse_mut(mut_name):
        wt_type, mutchain, mt_type = mut_name[0], mut_name[1], mut_name[-1]
        mutseq = int(mut_name[2:-1])
        return {'wt': wt_type, 'mt': mt_type, 'chain': mutchain, 'resseq': mutseq, 'icode': ' ', 'name': mut_name}  # no icode in skempi

    entries = []
    for i, row in df.iterrows():
        pdbcode, group1, group2 = row['#Pdb'].split('_')
        if pdbcode in block_list:
            continue
        mut_str = row['Mutation(s)_cleaned']
        muts = list(map(_parse_mut, row['Mutation(s)_cleaned'].split(',')))
        if muts[0]['chain'] in group1:    # identify ligand and receptor
            group_ligand, group_receptor = group1, group2
        else:
            group_ligand, group_receptor = group2, group1
        if predicted_structure:
            pdb_path = os.path.join(pdb_dir, '{}_wt_esmfold.pdb'.format(pdbcode.upper()))
        else:
            pdb_path = os.path.join(pdb_dir, '{}.pdb'.format(pdbcode.upper()))

        if not os.path.exists(pdb_path) or not np.isfinite(row['ddG']):
            continue

        # do not use 'pdbcode' for 'complex'!
        # some entries have the same mutation but different ddG, e.g., 3N0P_A_B 3NCB_A_B 3N06_A_B
        entry = {'id': i, 'complex': row['#Pdb'], 'mutstr': mut_str, 'num_muts': len(muts), 'pdbcode': pdbcode + '+skempi', 'group_ligand': list(group_ligand),
                 'group_receptor': list(group_receptor), 'mutations': muts, 'ddG': np.float32(row['ddG']), 'dG_wt': np.float32(row['dG_wt']), 'dG_mt': np.float32(row['dG_mut']), 'pdb_path': pdb_path, }
        entries.append(entry)
    return entries


def load_abbind_entries(csv_path, pdb_dir, block_list=('3NPS', )):
    df = pd.read_csv(csv_path, encoding="ISO-8859-1")

    def _parse_mut(mut_name):
        if not mut_name.isupper():
            wt_type, mutchain, mt_type, icode = mut_name[2], mut_name[0], mut_name[-1], mut_name[-2].upper()    # uppercase icode
            mutseq = int(mut_name[3:-2])
        else:
            wt_type, mutchain, mt_type, icode = mut_name[2], mut_name[0], mut_name[-1], ' '
            mutseq = int(mut_name[3:-1])
        return {'wt': wt_type, 'mt': mt_type, 'chain': mutchain, 'resseq': mutseq, 'icode': icode, 'name': mut_name}  # there is icode in abbind

    entries = []
    for i, row in df.iterrows():
        pdbcode = row['#PDB']
        group1, group2 = row['Partners(A_B)'].split('_')

        if pdbcode in block_list:
            continue
        mut_str = row['Mutation']
        if 'delta' in mut_str:   # skip
            continue
        muts = list(map(_parse_mut, row['Mutation'].split(',')))
        if muts[0]['chain'] in group1:  # identify ligand and receptor
            group_ligand, group_receptor = group1, group2
        else:
            group_ligand, group_receptor = group2, group1

        pdb_path = os.path.join(pdb_dir, '{}.pdb'.format(pdbcode.upper()))
        if not os.path.exists(pdb_path) or not np.isfinite(row['ddG(kcal/mol)']):
            continue

        entry = {'id': i, 'complex': pdbcode, 'mutstr': mut_str, 'num_muts': len(muts), 'pdbcode': pdbcode + '+abbind', 'group_ligand': list(group_ligand),
                 'group_receptor': list(group_receptor), 'mutations': muts, 'ddG': np.float32(row['ddG(kcal/mol)']), 'pdb_path': pdb_path, }
        entries.append(entry)
    return entries


class SkempiABbindDataset(Dataset):

    # '3N06', '3N0P', '3NCB' have same ddG with different mutations, but deleting them does not lead to improvement...      # '3NPS', '1DVF', '2JEL'
    def __init__(self, skempi_csv_path, skempi_pdb_dir, cache_dir, abbind_csv_path=None, abbind_pdb_dir=None, use_sasa=False, use_plm=False, cvfold_index=0, num_cvfolds=3,
                 split='train', split_seed=2023, transform=None, blocklist=frozenset({'1KBH', }), reset=False, predicted_structure=False):
        super().__init__()
        self.skempi_csv_path = skempi_csv_path
        self.skempi_pdb_dir = skempi_pdb_dir
        self.abbind_csv_path = abbind_csv_path
        self.abbind_pdb_dir = abbind_pdb_dir
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        prefix = 'skempi_' if abbind_csv_path is None else 'skempi_abbind_'
        if predicted_structure:
            with open(os.path.join(cache_dir, prefix + 'structures.pkl'), 'rb') as f:   # PDBs are needed to obtain seq_map
                self.real_structures = pickle.load(f)
            prefix += 'esmfold_'
            self.skempi_pdb_dir = os.path.join(cache_dir, 'esmfold')

        self.entries_cache = os.path.join(cache_dir, prefix + 'entries.pkl')
        self.structures_cache = os.path.join(cache_dir, prefix + 'structures.pkl')
        self.use_sasa = use_sasa
        self.use_plm = use_plm
        self.predicted_structure = predicted_structure

        self.blocklist = blocklist
        self.transform = transform
        self.cvfold_index = cvfold_index
        self.num_cvfolds = num_cvfolds
        assert split in ('train', 'val', 'all')
        self.split = split
        self.split_seed = split_seed

        self.entries = None
        self.entries_full = None
        self._load_entries(reset)

        self.structures = None
        self._load_structures(reset)
        if use_plm:
            self.plm_feature = torch.load(os.path.join(self.cache_dir, 'esm2_embeddings.pt'))

    def _load_entries(self, reset):
        if not os.path.exists(self.entries_cache) or reset:
            self.entries_full = self._preprocess_entries()
        else:
            with open(self.entries_cache, 'rb') as f:
                self.entries_full = pickle.load(f)

        complex_to_entries = {}
        for e in self.entries_full:
            if e['complex'] not in complex_to_entries:
                complex_to_entries[e['complex']] = []
            complex_to_entries[e['complex']].append(e)

        complex_list = sorted(complex_to_entries.keys())
        random.Random(self.split_seed).shuffle(complex_list)

        if self.split == 'all':
            complexes_this = complex_list
        else:
            split_size = math.ceil(len(complex_list) / self.num_cvfolds)
            complex_splits = [complex_list[i * split_size: (i + 1) * split_size] for i in range(self.num_cvfolds)]

            val_split = complex_splits.pop(self.cvfold_index)
            train_split = sum(complex_splits, start=[])
            if self.split == 'val':
                complexes_this = val_split
            else:
                complexes_this = train_split

        entries = []
        for cplx in complexes_this:
            if cplx in complex_to_entries.keys():
                entries += complex_to_entries[cplx]
        self.entries = entries

    def _preprocess_entries(self):
        skempi_entries = load_skempi_entries(self.skempi_csv_path, self.skempi_pdb_dir, self.blocklist, self.predicted_structure)
        if self.abbind_pdb_dir is not None:
            abbind_entries = load_abbind_entries(self.abbind_csv_path, self.abbind_pdb_dir, self.blocklist)
        else:
            abbind_entries = []
        entries = skempi_entries + abbind_entries

        with open(self.entries_cache, 'wb') as f:
            pickle.dump(entries, f)
        return entries

    def _load_structures(self, reset):
        if not os.path.exists(self.structures_cache) or reset:
            self.structures = self._preprocess_structures(self.use_sasa)
        else:
            with open(self.structures_cache, 'rb') as f:
                self.structures = pickle.load(f)

    @property
    def _sasa_path(self):
        return os.path.join(self.cache_dir, 'SASA')

    def _preprocess_structures(self, sasa):
        pdbcodes = list(set([e['pdbcode'] for e in self.entries_full]))

        structures = {}
        for pdbcode_source in tqdm(pdbcodes, desc='Structures'):
            parser = PDBParser(QUIET=True)
            pdbcode, source = pdbcode_source.split('+')  # HM_xxxx
            if source == 'skempi':
                if self.predicted_structure:
                    pdb_path = os.path.join(self.skempi_pdb_dir, '{}_wt_esmfold.pdb'.format(pdbcode.upper()))
                else:
                    pdb_path = os.path.join(self.skempi_pdb_dir, '{}.pdb'.format(pdbcode.upper()))
            else:
                pdb_path = os.path.join(self.abbind_pdb_dir, '{}.pdb'.format(pdbcode.upper()))

            sasa_dict = None
            if sasa:
                with open(os.path.join(self._sasa_path, pdbcode), 'rb') as f:
                    sasa_dict = pickle.load(f)

            model = parser.get_structure(None, pdb_path)[0]
            data, seq_map = parse_biopython_structure(model, name=pdbcode, sasa=sasa_dict)
            structures[pdbcode_source] = (data, seq_map)

        with open(self.structures_cache, 'wb') as f:
            pickle.dump(structures, f)
        return structures

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        if self.predicted_structure:
            data, seq_map = copy.deepcopy(self.structures[entry['pdbcode']])
            _, seq_map = copy.deepcopy(self.real_structures[entry['pdbcode']])
        else:
            data, seq_map = copy.deepcopy(self.structures[entry['pdbcode']])

        keys = {'id', 'complex', 'mutstr', 'num_muts', 'pdbcode', 'ddG'}   # 'dG_wt', 'dG_mt'
        for k in keys:
             data[k] = entry[k]

        group_id = []
        for ch in data['chain_id']:
            if ch in entry['group_ligand']:
                group_id.append(1)   # ligand group id to 1
            elif ch in entry['group_receptor']:
                group_id.append(2)    # receptor group id to 2
            else:
                group_id.append(0)
        data['group_id'] = torch.LongTensor(group_id)

        aa_mut = data['aa'].clone()
        for mut in entry['mutations']:
            ch_rs_ic = (mut['chain'], mut['resseq'], mut['icode'])
            if ch_rs_ic not in seq_map:
                continue
            aa_mut[seq_map[ch_rs_ic]] = one_to_index(mut['mt'])

        data['aa_mut'] = aa_mut
        data['mut_flag'] = (data['aa'] != data['aa_mut'])
        assert data['mut_flag'].sum() > 0, 'Mutation type and wide type are the same!'

        if self.use_plm:
            data['plm_wt'] = copy.deepcopy(self.plm_feature[entry['pdbcode'][:4]])
            data['plm_mut'] = copy.deepcopy(self.plm_feature[entry['pdbcode'][:4] + entry['mutstr']])

        if self.transform is not None:
            data = self.transform(data)
        return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--skempi_csv_path', type=str, default='./data/SKEMPI_v2/skempi_v2.csv')
    parser.add_argument('--skempi_pdb_dir', type=str, default='./data/SKEMPI_v2/PDBs')
    parser.add_argument('--cache_dir', type=str, default='./data/SKEMPI_v2_cache')
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args()

    dataset = SkempiABbindDataset(skempi_csv_path=args.skempi_csv_path, skempi_pdb_dir=args.skempi_pdb_dir, cache_dir=args.cache_dir, split='val', num_cvfolds=5, cvfold_index=2,
                                  reset=args.reset, )
    print(dataset[0])
    print(len(dataset))
