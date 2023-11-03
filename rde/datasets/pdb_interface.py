import collections
import math
import os
import pickle
import random
from typing import Mapping, List, Dict, Optional

import lmdb
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException
from joblib import Parallel, delayed, cpu_count
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from rde.utils.protein.parsers import parse_biopython_structure

ClusterIdType, PdbCodeType = str, str


def load_interface_entries(csv_path, block_list=(), ):
    df = pd.read_csv(csv_path)

    entries = []
    for i, row in df.iterrows():
        pdbcode, pdb_path = row['pdbcode'], row['pdb_file']
        if pdbcode in block_list or not os.path.exists(pdb_path):
            continue

        # Determine ligand and receptor by chain lengths
        chain_length = eval(row['chain_length'])
        group1, group2 = chain_length.keys()
        idx_1, idx_2 = eval(row['rcut8.itface_ext'])
        if chain_length[group1] < chain_length[group2]:
            group_ligand, group_receptor = group1, group2
            idx_ligand, idx_receptor = idx_1, idx_2
        else:
            group_ligand, group_receptor = group2, group1
            idx_ligand, idx_receptor = idx_2, idx_1

        entry = {'id': i, 'complex': pdbcode + '_' + group1 + '_' + group2, 'cluster_id': row['cluster_id'], 'pdbcode': pdbcode, 'pdb_path': pdb_path,
                 'interface': {'ligand': [group_ligand, idx_ligand], 'receptor': [group_receptor, idx_receptor]}}
        entries.append(entry)
    return entries


def _process_structure(pdb_path, structure_id, interface) -> Optional[Dict]:
    parser = PDBParser(QUIET=True)
    try:
        model = parser.get_structure(structure_id, pdb_path)[0]
    except PDBConstructionException:
        print(f'[INFO] Failed to load structure using PDBParser: {pdb_path}.')
        return None

    parsed = {'id': structure_id, 'ligand': parse_biopython_structure(model[interface['ligand'][0]], interface_idx=interface['ligand'][1])[0],
              'receptor': parse_biopython_structure(model[interface['receptor'][0]], interface_idx=interface['receptor'][1])[0]}
    if parsed['ligand'] is None or parsed['receptor'] is None:
        print(f'[INFO] Failed to parse structure. Too few valid residues: {pdb_path}')
        return None
    return parsed


class PDBInterfaceDataset(Dataset):
    MAP_SIZE = 384 * (1024 * 1024 * 1024)  # 384GB

    def __init__(self, split, csv_path='/nfs_beijing/qijin/WorkSpace/003.affinity/data/gen_data/01.130k_interface/collects_231028/interfaces.withstat.csv',
                 processed_dir='./data/PDB_Interface_processed', num_preprocess_jobs=math.floor(cpu_count() * 0.8), transform=None, split_seed=2023, reset=False, ):
        super().__init__()
        self.csv_path = csv_path
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        self.num_preprocess_jobs = num_preprocess_jobs
        self.transform = transform

        # Load entries
        self.entries_full = None
        self.entries_cache = os.path.join(processed_dir, 'entries.pkl')
        self._load_entries(reset)

        # Structure cache
        self.db_conn = None
        self.db_keys: Optional[List[PdbCodeType]] = None
        self._preprocess_structures(reset)

        # Load and sanitize clusters
        if os.path.exists(self.sanitized_clusters_path) and not reset:
            with open(self.sanitized_clusters_path, 'rb') as f:
                self.clusters = pickle.load(f)
        else:
            self.clusters: Mapping[ClusterIdType, List[PdbCodeType]] = collections.defaultdict(list)
            self._load_clusters()
            self._sanitize_clusters()

        # Load splits
        self.splits = {}
        self.split_seed = split_seed
        self._load_splits()

        # Select clusters of the split
        self._clusters_of_split = [c for c in self.splits[split] if c in self.clusters]

    @property
    def lmdb_path(self):
        return os.path.join(self.processed_dir, 'structures.lmdb')

    @property
    def keys_path(self):
        return os.path.join(self.processed_dir, 'keys.pkl')

    @property
    def sanitized_clusters_path(self):
        return os.path.join(self.processed_dir, 'sanitized_clusters.pkl')

    def _load_entries(self, reset):
        if not os.path.exists(self.entries_cache) or reset:
            entries = load_interface_entries(self.csv_path, )
            with open(self.entries_cache, 'wb') as f:
                pickle.dump(entries, f)
            self.entries_full = entries
        else:
            with open(self.entries_cache, 'rb') as f:
                self.entries_full = pickle.load(f)

    def _load_clusters(self):
        for e in self.entries_full:
            self.clusters[e['cluster_id']].append(e['complex'])

    def _load_splits(self, split_ratio=0.1):
        cluster_id_list = sorted(self.clusters.keys())
        train_id, val_id = train_test_split(cluster_id_list, test_size=split_ratio, random_state=self.split_seed)
        self.splits['train'] = train_id
        self.splits['val'] = val_id

    def _preprocess_structures(self, reset):
        if os.path.exists(self.lmdb_path) and not reset:
            return
        tasks = []
        for e in self.entries_full:
            tasks.append(delayed(_process_structure)(e['pdb_path'], e['complex'], e['interface']))

        # Split data into chunks
        chunk_size = 8192
        task_chunks = [tasks[i * chunk_size:(i + 1) * chunk_size] for i in range(math.ceil(len(tasks) / chunk_size))]

        # Establish database connection
        db_conn = lmdb.open(self.lmdb_path, map_size=self.MAP_SIZE, create=True, subdir=False, readonly=False, )

        keys = []
        for i, task_chunk in enumerate(task_chunks):
            with db_conn.begin(write=True, buffers=True) as txn:
                processed = Parallel(n_jobs=self.num_preprocess_jobs)(task for task in tqdm(task_chunk, desc=f"Chunk {i + 1}/{len(task_chunks)}"))
                stored = 0
                for data in processed:
                    if data is not None:
                        key = data['id']
                        keys.append(key)
                        txn.put(key=key.encode(), value=pickle.dumps(data))
                        stored += 1
                print(f"[INFO] {stored} processed for chunk#{i + 1}")
        db_conn.close()

        with open(self.keys_path, 'wb') as f:
            pickle.dump(keys, f)

    def _connect_db(self):
        assert self.db_conn is None
        self.db_conn = lmdb.open(self.lmdb_path, map_size=self.MAP_SIZE, create=False, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, )
        with open(self.keys_path, 'rb') as f:
            self.db_keys = pickle.load(f)

    def _close_db(self):
        self.db_conn.close()
        self.db_conn = None
        self.db_keys = None

    def _get_from_db(self, pdb_chain):
        if self.db_conn is None:
            self._connect_db()
        data = pickle.loads(self.db_conn.begin().get(pdb_chain.encode()))  # Made a copy
        return data

    def _sanitize_clusters(self):
        # Step 1: Find structures that do not exist in PDBs
        pdb_removed = 0
        pdb_chains = set()
        clusters_raw = self.clusters
        self._connect_db()
        n = 0
        for _, pdb_chain_list in tqdm(self.clusters.items(), desc='Sanitize'):
            for pdb_chain in pdb_chain_list:
                n += 1
                if pdb_chain not in self.db_keys:
                    pdb_removed += 1
                    continue
                pdb_chains.add(pdb_chain)
        print(f'[INFO]  Total structures: {n}. Structures removed: {pdb_removed}, structures left: {len(pdb_chains)}.')

        # Step 2: Rebuild the clusters according to the allowed PDBs.
        clusters_sanitized = collections.defaultdict(list)
        for clust_name, pdb_chain_list in clusters_raw.items():
            for pdb_chain in pdb_chain_list:
                if pdb_chain in pdb_chains:
                    clusters_sanitized[clust_name].append(pdb_chain)
        print('[INFO] %d clusters after sanitization (from %d).' % (len(clusters_sanitized), len(clusters_raw)))

        with open(self.sanitized_clusters_path, 'wb') as f:
            pickle.dump(clusters_sanitized, f)
        self.clusters = clusters_sanitized

    def __len__(self):
        return len(self._clusters_of_split)

    def __getitem__(self, index):
        if isinstance(index, int):
            index = (index, None)

        # Select cluster
        clust = self._clusters_of_split[index[0]]
        pdb_chain_list = self.clusters[clust]

        # Select a pdb-interface from the cluster and retrieve the data point
        if index[1] is None:
            pdb_chain = random.choice(pdb_chain_list)
        else:
            pdb_chain = pdb_chain_list[index[1]]
        data = self._get_from_db(pdb_chain)  # Made a copy

        # group_id = []
        # for ch in data['chain_id']:
        #     if ch in entry['group_ligand']:
        #         group_id.append(1)   # ligand group id to 1
        #     elif ch in entry['group_receptor']:
        #         group_id.append(2)    # receptor group id to 2
        #     else:
        #         group_id.append(0)
        # data['group_id'] = torch.LongTensor(group_id)

        # # Focus on the chain
        # focus_flag = torch.zeros(len(data['chain_id']), dtype=torch.bool)
        # for i, ch in enumerate(data['chain_id']):
        #     if ch == chain: focus_flag[i] = True
        # data['focus_flag'] = focus_flag

        if self.transform is not None:
            data = self.transform(data)
        return data


def get_pdb_interface_dataset(split, cfg, ):
    from rde.utils.transforms import get_transform
    return PDBInterfaceDataset(split=split, csv_path=cfg.csv_path, processed_dir=cfg.processed_dir, transform=get_transform(cfg.transform),)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()

    dataset = PDBInterfaceDataset(args.split)
    for data in tqdm(dataset, desc='Iterating'):
        pass
    print(data)
    print(f'[INFO] {len(dataset.clusters)} clusters in the entire dataset.')
    print(f'[INFO] {len(dataset)} samples in the split.')
