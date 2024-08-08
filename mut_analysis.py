import argparse
import matplotlib.pyplot as plt
import torch

from src.datasets.skempi import SkempiABbindDataset
from src.utils.transforms import SelectAtom
from src.utils.transforms._base import _get_CB_positions

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='mut_knn', choices=['dg_dist', 'mut_knn'])
parser.add_argument('--skempi_csv_path', type=str, default='./data/SKEMPI_v2/skempi_v2.csv')
parser.add_argument('--skempi_pdb_dir', type=str, default='./data/SKEMPI_v2/PDBs')
parser.add_argument('--cache_dir', type=str, default='./data/SKEMPI_v2_cache')
parser.add_argument('--reset', action='store_true', default=False)
args = parser.parse_args()

dataset = SkempiABbindDataset(skempi_csv_path=args.skempi_csv_path, skempi_pdb_dir=args.skempi_pdb_dir, cache_dir=args.cache_dir, split='all',
                              transform=SelectAtom('backbone+CB'), reset=args.reset)
print(len(dataset))

dist_, ddG_, knn_ = [], [], {64: [], 128: [], 256: [], 512: []}
select_attr = 'mut_flag'
for data in dataset:
    select_flag = (data[select_attr] > 0)
    if torch.sum(select_flag) != 1:   # only single-mutated ab-ag
        continue
    pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])  # (L, 3)
    pos_sel = pos_CB[select_flag]  # (S, 3)

    if args.mode == 'dg_dist':
        ag_index = data['group_id'] == 2
        pos_ag = pos_CB[ag_index]
        dist_from_sel = torch.cdist(pos_sel, pos_ag).min(dim=1)[0]  # (L, )
        dist_.append(dist_from_sel.item())
        ddG_.append(data['ddG'])
    else:
        dist_from_sel = torch.cdist(pos_CB, pos_sel).min(dim=1)[0]  # (L, )
        for k in knn_.keys():
            patch_idx = torch.argsort(dist_from_sel)[:k]
            n_ag = torch.sum(data['group_id'][patch_idx] == 2)
            knn_[k].append(n_ag.item())

if args.mode == 'dg_dist':
    plt.scatter(dist_, ddG_, s=7)
    plt.xlabel('Minimum Distance to Antigen (A)')
    plt.ylabel('ddG (kcal/mol)')
    plt.savefig('./dg_dist.pdf')

    torch.save([dist_, ddG_], 'dg_dist.pt')
    print(sum(dist_) / len(dist_), sum([abs(i) for i in ddG_]) / len(ddG_))
else:
    torch.save(knn_, 'mut_knn.pt')







