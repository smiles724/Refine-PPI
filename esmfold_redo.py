"""
-i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pytorch-lightning==1.8.4

pip install "fair-esm[esmfold]"
# OpenFold and its remaining dependency
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
"""
import os
import pickle

import esm
import lmdb
import torch
from tqdm import tqdm

from esm_embedding import get_seq

split = 'val'
lmdb_path = './data/PDB_REDO_processed_raw/structures.lmdb'
keys_path = './data/PDB_REDO_processed_raw/keys.pkl'
esmfold_path = './data/PDB_REDO_processed_raw/esmfold'
os.makedirs(esmfold_path, exist_ok=True)


MAP_SIZE = 384 * (1024 * 1024 * 1024)  # 384GB
db_conn = lmdb.open(lmdb_path, map_size=MAP_SIZE, create=False, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, )
with open(keys_path, 'rb') as f:
    db_keys = pickle.load(f)

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()
model.set_chunk_size(128)
for pdbcode in tqdm(db_keys[::-1]):  # [int(len(db_keys)/2):]

    if os.path.exists(os.path.join(esmfold_path, f"{pdbcode}_esmfold.pdb")):
        continue
    try:  # some do not have processed structure data
        data = pickle.loads(db_conn.begin().get(pdbcode.encode()))  # Made a copy
    except:
        continue

    try:
        # Multimer prediction can be done with chains separated by ':'
        sequence = get_seq(data['aa'].numpy().tolist(), data['chain_nb'].numpy().tolist())
        with torch.no_grad():
            output = model.infer_pdb(sequence)
        with open(os.path.join(esmfold_path, f"{pdbcode}_esmfold.pdb"), "w") as f:
            f.write(output)
    except:
        print(f'{pdbcode} fails due to OOM')
        continue
