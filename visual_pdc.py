"""
python visual_pdc.py  --ckpt=./logs_skempi/pdc_ddg_skempi-resume_2023_08_11__17_48_11/checkpoints/60000.pt
"""
import os
import argparse

import torch.utils.tensorboard

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from src.utils.train import *
from src.utils.skempi import SkempiDatasetManager
from src.models.pdc_ddg import DDG_PDC_Network

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='redo', choices=['skempi', 'redo'])
parser.add_argument('--backbone', type=str, default='egnn', choices=['egnn', 'ga'])
parser.add_argument('--ckpt', type=str)
parser.add_argument('-o', '--output', type=str, default='./pos/')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()
os.makedirs(args.output, exist_ok=True)

ckpt = torch.load(args.ckpt)
config = ckpt['config']
num_cvfolds = len(ckpt['model']['models'])
print('Loading datasets...')
dataset_mgr = SkempiDatasetManager(config, num_cvfolds=num_cvfolds, num_workers=args.num_workers, )

print('Building model...')
cv_mgr = CrossValidation(model_factory=DDG_PDC_Network, config=config, num_cvfolds=num_cvfolds).to(args.device)
print('Loading state dict...')
cv_mgr.load_state_dict(ckpt['model'])

id_, pos_, wt_aa, resseq, res_nb, chain_nb, vars_ = [], [], [], [], [], [], []
with torch.no_grad():
    for fold in range(num_cvfolds):
        model, _, _ = cv_mgr.get(fold)
        for i, batch in enumerate(dataset_mgr.get_val_loader(fold)):
            batch = recursive_to(batch, args.device)
            with torch.no_grad():
                _, coors_var = model.encode(batch)

            pos_.append(batch['pos_atoms'])
            wt_aa.append(batch['aa'])
            resseq.append(batch['resseq'])
            chain_nb.append(batch['chain_nb'])
            vars_.append(coors_var)

    pos_ = torch.cat(pos_).cpu().numpy()
    wt_aa = torch.cat(wt_aa).cpu().numpy()
    resseq = torch.cat(resseq).cpu().numpy()
    chain_nb = torch.cat(chain_nb).cpu().numpy()
    vars_ = torch.cat(vars_).cpu().numpy()

    import string
    atoms_ = ['N', 'CA', 'C', 'O', 'CB']
    symbols_ = ['N', 'C', 'C', 'O', 'C']
    chains_ = list(string.ascii_uppercase)
    template = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n"

    for n in range(len(pos_)):
        atom_number = 0
        rmsf = (vars_[n] ** 2).sum(axis=-1) ** 0.5
        with open(os.path.join(args.output, f'{n}.pdb'), 'w') as f:
            for i in range(len(pos_[n])):
                try:
                    resname = AA(wt_aa[n][i])._name_
                except:  # padding token break
                    break

                for j in range(5):
                    atom_number += 1
                    xyz = pos_[n][i][j].tolist()
                    f.write(
                        template.format("ATOM", atom_number, atoms_[j], '', resname, chains_[chain_nb[n][i]], resseq[n][i], '', xyz[0], xyz[1], xyz[2],
                                        1.00, rmsf[i], symbols_[j], ''))
    print('Finished.')


