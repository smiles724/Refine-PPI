import copy
import os
import pickle
import random

from Bio.PDB.PDBParser import PDBParser
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.utils.protein.parsers import parse_biopython_structure

md_entries = ['2b2x', '2noj', '2qja', '3mzg', '4uyq', '5e9d', '5f4e', ]


class MolecularDynamicsDataset(Dataset):

    def __init__(self, split, md_pdb_dir, cache_dir, transform=None, blocklist=frozenset({}), reset=False, ):
        super().__init__()
        prefix = 'md_'
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.md_pdb_dir = md_pdb_dir
        self.entries_cache = os.path.join(cache_dir, prefix + 'entries.pkl')
        self.rmsf_cache = os.path.join(cache_dir, prefix + 'rmsf.pkl')
        self.structures_cache = os.path.join(cache_dir, prefix + 'structures.pkl')

        self.blocklist = blocklist
        self.transform = transform

        self.rmsf = None
        self._load_rmsf(reset)

        self.structures = None
        self._load_structures(reset)
        self.entries = self.entries[split]

    def _load_rmsf(self, reset):
        if not os.path.exists(self.rmsf_cache) or reset:
            rmsf_dict = {}
            for pdbcode in md_entries:
                rmsf_dict[pdbcode] = {}
                with open(os.path.join(self.md_pdb_dir, f'rmsfvec-{pdbcode}.txt')) as f:
                    for line in f.readlines()[1:]:
                        print(line)
                        tmp = line.split()
                        if len(tmp) > 0:
                            chain, res_id, rmsf_x, rmsf_y, rmsf_z = tmp[0], tmp[1], float(tmp[3]), float(tmp[4]), float(tmp[5])
                            rmsf_dict[pdbcode][chain + res_id] = [rmsf_x, rmsf_y, rmsf_z]
            with open(self.rmsf_cache, 'wb') as f:
                pickle.dump(rmsf_dict, f)

        with open(self.rmsf_cache, 'rb') as f:
            self.rmsf = pickle.load(f)

    def _load_structures(self, reset):
        if not os.path.exists(self.structures_cache) or not os.path.exists(self.entries_cache) or reset:
            self.entries, self.structures = self._preprocess_structures()
        else:
            with open(self.entries_cache, 'rb') as f:
                self.entries = pickle.load(f)
            with open(self.structures_cache, 'rb') as f:
                self.structures = pickle.load(f)

    def _preprocess_structures(self):
        structures = {}
        entries = {'train': [], 'val': []}
        for pdbcode in tqdm(md_entries, desc='Structures'):

            file_path = os.path.join(self.md_pdb_dir, f'{pdbcode}-pdb')
            idx_tmp = []
            for file_name in os.listdir(file_path):
                parser = PDBParser(QUIET=True)
                idx = file_name.split('.')[0]
                pdb_path = os.path.join(file_path, file_name)

                model = parser.get_structure(None, pdb_path)[0]
                data, seq_map = parse_biopython_structure(model, name=file_name, rmsf=self.rmsf[pdbcode])
                structures[idx] = (data, seq_map)

                idx_tmp.append(idx)
            random.shuffle(idx_tmp)
            entries['train'] += idx_tmp[:int(len(idx_tmp) * 0.95)]
            entries['val'] += idx_tmp[int(len(idx_tmp) * 0.95):]

        with open(self.entries_cache, 'wb') as f:
            pickle.dump(entries, f)
        with open(self.structures_cache, 'wb') as f:
            pickle.dump(structures, f)
        return entries, structures

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        idx = self.entries[index]
        data, seq_map = copy.deepcopy(self.structures[idx])

        if self.transform is not None:
            data = self.transform(data)
        return data


def get_md_dataset(split, cfg):
    from src.utils.transforms import get_transform
    return MolecularDynamicsDataset(split=split, md_pdb_dir=cfg.md_pdb_dir, cache_dir=cfg.cache_dir, transform=get_transform(cfg.transform), reset=cfg.reset)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--md_pdb_dir', type=str, default='./data/SKEMPI_v2/skempi_v2.csv')
    parser.add_argument('--cache_dir', type=str, default='./data/SKEMPI_v2_cache')
    parser.add_argument('--reset', action='store_true', default=False)
    args = parser.parse_args()

    dataset = MolecularDynamicsDataset('val', md_pdb_dir=args.md_pdb_dir, cache_dir=args.cache_dir, reset=args.reset, )
    print(dataset[0])
    print(len(dataset))
