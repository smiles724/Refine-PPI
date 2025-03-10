"""
https://github.com/facebookresearch/esm/tree/main/examples/inverse_folding
"""
import torch
import os
import pandas as pd
import pickle
import esm.inverse_folding    # pip install biotite==0.41, pip install "numpy<2.0"
from src.utils.skempi import eval_skempi_three_modes

# Load the pre-trained ESM-IF model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.to(device)
model.eval()

cache_dir = './data/SKEMPI_v2_cache'
entries_cache = os.path.join(cache_dir, 'entries.pkl')
with open(entries_cache, 'rb') as f:
    entries_full = pickle.load(f)
complex_to_entries = {}
for e in entries_full:
    if e['complex'] not in complex_to_entries:
        complex_to_entries[e['complex']] = []
    complex_to_entries[e['complex']].append(e)
fold_path = './data/'
with open(os.path.join(fold_path, f'test_pdb.pkl'), 'rb') as file:
    complexes_this = pickle.load(file)
entries = []
for cplx in complexes_this:
    if cplx.startswith('1KBH'):  # not used by RDE
        continue
    entries += complex_to_entries[cplx]
print('#test: ', len(entries))

# ll_fullseq: the average log-likelihood averaged over all amino acids in a sequence.
# ll_withcoord: is averaged only over those amino acids with associated backbone coordinates in the input, i.e., excluding those with missing backbone coordinates.
results = []
for e in entries:
    pdb_path = f'./data/SKEMPI_v2/PDBs/{e["pdbcode"]}.pdb'
    chain_ids = e['group_ligand'] + e['group_receptor']
    try:
        structure = esm.inverse_folding.util.load_structure(pdb_path, chain_ids)
        coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
        pred = 0
        for mut in e['mutations']:
            target_chain_id = mut['chain']
            target_seq = native_seqs[target_chain_id]
            ll_fullseq, ll_withcoord = esm.inverse_folding.multichain_util.score_sequence_in_complex(model, alphabet, coords, target_chain_id, target_seq)
            assert target_seq[mut['resseq'] - 1] == mut['wt']
            target_seq_ = target_seq[:mut['resseq'] - 1] + mut['mt'] + target_seq[mut['resseq']:]
            ll_fullseq_, ll_withcoord_ = esm.inverse_folding.multichain_util.score_sequence_in_complex(model, alphabet, coords, target_chain_id, target_seq_)
            pred += ll_fullseq_ - ll_fullseq

        results.append({'complex': e['complex'], 'mutstr': e['mutstr'], 'num_muts': e['num_muts'], 'ddG': e['ddG'], 'ddG_pred': pred})
    except Exception as error:
        print(f'Fail for {e["id"] + "_" + e["complex"]} due to {error}')

results = pd.DataFrame(results)
print(results)
results['method'] = f'RSM-IF'
results.to_csv(f'esm_results.csv', index=False)

df_metrics = eval_skempi_three_modes(results)
print(df_metrics)
df_metrics.to_csv(f'esm_results_metrics.csv', index=False)