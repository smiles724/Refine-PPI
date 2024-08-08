"""
-i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pytorch-lightning==1.8.4

pip install "fair-esm[esmfold]"
# OpenFold and its remaining dependency
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
"""
import os
import collections
import lmdb
import pickle
from tqdm import tqdm
from typing import Mapping, List, Tuple
from extract_embedding import get_seq
import torch
import esm

split = 'val'
clusters_path = './data/pdbredo_clusters.txt'
splits_path = './data/pdbredo_splits.txt'
lmdb_path = './data/PDB_REDO_processed_raw/structures.lmdb'
keys_path = './data/PDB_REDO_processed_raw/keys.pkl'
esmfold_path = './data/PDB_REDO_processed_raw/esmfold'
os.makedirs(esmfold_path, exist_ok=True)
ClusterIdType, PdbCodeType, ChainIdType = str, str, str

clusters: Mapping[ClusterIdType, List[Tuple[PdbCodeType, ChainIdType]]] = collections.defaultdict(list)
splits: Mapping[str, List[ClusterIdType]] = collections.defaultdict(list)

with open(clusters_path, 'r') as f:
    lines = f.readlines()
current_cluster = None
for line in lines:
    line = line.strip()
    if not line:
        continue
    for word in line.split():
        if word[0] == '[' and word[-1] == ']':
            current_cluster = word[1:-1]
        else:
            pdbcode, chain_id = word.split(':')
            clusters[current_cluster].append((pdbcode, chain_id))

with open(splits_path, 'r') as f:
    lines = f.readlines()
current_split = None
for line in lines:
    line = line.strip()
    if not line:
        continue
    for word in line.split():
        if word[0] == '[' and word[-1] == ']':
            current_split = word[1:-1]
        else:
            splits[current_split].append(word)

_clusters_of_split = [c for c in splits[split] if c in clusters]

MAP_SIZE = 384 * (1024 * 1024 * 1024)  # 384GB
db_conn = lmdb.open(lmdb_path, map_size=MAP_SIZE, create=False, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, )
with open(keys_path, 'rb') as f:
    db_keys = pickle.load(f)

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()
# Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
# Lower sizes will have lower memory requirements at the cost of increased speed.
model.set_chunk_size(128)

for i in tqdm(_clusters_of_split):
    pdbcode = i.split('_')[0].lower()
    if os.path.exists(os.path.join(esmfold_path, f"{pdbcode}_esmfold.pdb")):
        continue
    try:  # some do not have processed structure data
        data = pickle.loads(db_conn.begin().get(pdbcode.encode()))  # Made a copy
    except:
        continue

    # Multimer prediction can be done with chains separated by ':'
    sequence = get_seq(data['aa'].numpy().tolist())

    with torch.no_grad():
        output = model.infer_pdb(sequence)

    with open(os.path.join(esmfold_path, f"{pdbcode}_esmfold.pdb"), "w") as f:
        f.write(output)
