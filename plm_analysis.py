import copy
import os
import pickle
import sys
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

sim = sys.argv[1]
assert sim in ['cosine', 'dist']

prefix = 'skempi_'
cache_dir = './data/SKEMPI_v2_cache'
entries_cache = os.path.join(cache_dir, prefix + 'entries.pkl')
plm_feature = torch.load(os.path.join(cache_dir, 'esm2_embeddings.pt'))

with open(entries_cache, 'rb') as f:
    entries_full = pickle.load(f)

scores, ddGs = [], []
for entry in entries_full:
    idx, ddG = entry['pdbcode'], entry['ddG']
    plm_wt = copy.deepcopy(plm_feature[entry['pdbcode']])
    plm_mut = copy.deepcopy(plm_feature[entry['pdbcode'] + entry['mutstr']])

    plm_wt = torch.max(plm_wt, dim=0)[0]   # take average/max embeddings
    plm_mut = torch.max(plm_mut, dim=0)[0]

    if sim == 'cosine':
        score = F.cosine_similarity(plm_wt, plm_mut, dim=0).item()
    else:
        score = torch.cdist(plm_mut, plm_wt).item()
    scores.append(score)
    ddGs.append(ddG)
    print(score)
print(f'Number of samples: {len(ddGs)}.')

plt.scatter(scores, ddGs, s=3, c='purple')
plt.savefig('plm_analysis.png', bbox_inches='tight')
