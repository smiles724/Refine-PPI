"""
-i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pytorch-lightning==1.8.4

pip install "fair-esm[esmfold]"
# OpenFold and its remaining dependency
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
"""
import argparse
import copy
import os
import pickle

import esm
import torch
from Bio.PDB.Polypeptide import one_to_index
from tqdm import tqdm

from extract_embedding import get_seq

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract per-token representations and model outputs for sequences in a FASTA file")  # noqa
    parser.add_argument("--abbind", action="store_true", help="Whether to process ABbind dataset.")
    parser.add_argument("--mt", action="store_true", help="Whether to generate mutant structures.")
    args = parser.parse_args()

    cache_dir = './data/SKEMPI_v2_cache'
    prefix = 'skempi_' if not args.abbind else 'skempi_abbind_'
    structures_cache = os.path.join(cache_dir, prefix + 'structures.pkl')
    entries_cache = os.path.join(cache_dir, prefix + 'entries.pkl')

    esmfold_path = './data/SKEMPI_v2_cache/esmfold'
    os.makedirs(esmfold_path, exist_ok=True)

    with open(structures_cache, 'rb') as f:
        structures = pickle.load(f)
    with open(entries_cache, 'rb') as f:
        entries_full = pickle.load(f)

    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()    # CPU error: RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'
    # Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
    # Lower sizes will have lower memory requirements at the cost of increased speed.
    model.set_chunk_size(1)

    ids, n_failed = [], 0
    for entry in tqdm(entries_full):
        idx = entry['pdbcode']

        if idx not in ids and not os.path.exists(os.path.join(esmfold_path, f"{idx}_wt_esmfold.pdb")):
            ids.append(idx)
            data, seq_map = copy.deepcopy(structures[idx])
            seq_wt = get_seq(data['aa'].numpy().tolist(), data['chain_nb'].numpy().tolist())
            try:
                torch.cuda.empty_cache()
                with torch.no_grad():
                    output = model.infer_pdb(seq_wt)
                with open(os.path.join(esmfold_path, f"{idx}_wt_esmfold.pdb"), "w") as f:
                    f.write(output)
            except:
                n_failed += 1
                print(f'{idx} fails due to OOM.\nSeq:{seq_wt}\n Use https://esmatlas.com/resources?action=fold to generate its structure.')   # 3VR6
                continue

        if args.mt:
            idx_mut = idx + entry['mutstr']
            if not os.path.exists(os.path.join(esmfold_path, f"{idx_mut}_mt_esmfold.pdb")):
                data, seq_map = copy.deepcopy(structures[idx])
                aa_mut = data['aa'].clone()
                for mut in entry['mutations']:
                    ch_rs_ic = (mut['chain'], mut['resseq'], mut['icode'])
                    if ch_rs_ic not in seq_map:
                        continue
                    aa_mut[seq_map[ch_rs_ic]] = one_to_index(mut['mt'])
                seq_mut = get_seq(aa_mut.numpy().tolist(), data['chain_nb'].numpy().tolist())
                try:
                    with torch.no_grad():
                        output = model.infer_pdb(seq_mut)

                    with open(os.path.join(esmfold_path, f"{idx_mut}_mt_esmfold.pdb"), "w") as f:
                        f.write(output)
                except:
                    print(f'{idx} fails due to OOM')

    print(f'{n_failed} not generated...')  # ['5DWU', '3BIW', '1QSF', '1KBH', '3VR6', '1QRN']  345 -> 339

