import argparse

import pandas as pd
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os
import math
import pickle
import lmdb
from joblib import Parallel, delayed, cpu_count

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from rde.utils.misc import get_logger, load_config
from rde.utils.train import *
from rde.utils.data import PaddingCollate
from rde.utils.protein.constants import num_chi_angles
from rde.datasets.pdbredo_chain import get_pdbredo_chain_dataset, _process_structure


def _preprocess_structures(reset=False):
    num_preprocess_jobs = math.floor(cpu_count() * 0.8)
    MAP_SIZE = 384 * (1024 * 1024 * 1024)  # 384GB

    if os.path.exists(lmdb_path) and not reset:
        return
    tasks = []
    for file in os.listdir(esmfold_path):
        pdbcode = file.split('.')[0]
        cif_path = os.path.join(esmfold_path, file)
        if not os.path.exists(cif_path):
            print(f'[WARNING] CIF not found: {cif_path}.')
            continue
        tasks.append(delayed(_process_structure)(cif_path, pdbcode))

    # Split data into chunks
    chunk_size = 8192
    task_chunks = [tasks[i * chunk_size:(i + 1) * chunk_size] for i in range(math.ceil(len(tasks) / chunk_size))]

    # Establish database connection
    db_conn = lmdb.open(lmdb_path, map_size=MAP_SIZE, create=True, subdir=False, readonly=False, )
    keys = []
    for i, task_chunk in enumerate(task_chunks):
        with db_conn.begin(write=True, buffers=True) as txn:
            processed = Parallel(n_jobs=num_preprocess_jobs)(task for task in tqdm(task_chunk, desc=f"Chunk {i + 1}/{len(task_chunks)}"))
            stored = 0
            for data in processed:
                if data is None:
                    continue
                key = data['id']
                keys.append(key)
                txn.put(key=key.encode(), value=pickle.dumps(data))
                stored += 1
            print(f"[INFO] {stored} processed for chunk#{i + 1}")
    db_conn.close()

    with open(keys_path, 'wb') as f:
        pickle.dump(keys, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='pdc', choices=['pdc', 'rde', 'equiformer'])
    parser.add_argument('--task', type=str, default='cluster', choices=['cluster', 'chi_angles'])
    parser.add_argument('--ckpt', type=str, default='./trained_models/RDE.pt')
    parser.add_argument('--output', type=str, default='redo_results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--esmfold', action='store_true')

    args = parser.parse_args()
    logger = get_logger('test', None)
    ckpt = torch.load(args.ckpt)
    if args.backbone == 'rde':
        config, _ = load_config('configs/train/rde.yml')
    else:
        config = ckpt['config']

    logger.info('Loading datasets...')
    if args.esmfold:
        esmfold_path = './data/PDB_REDO_processed_raw/esmfold'
        lmdb_path = os.path.join(esmfold_path, 'structures.lmdb')
        keys_path = os.path.join(esmfold_path, 'keys.pkl')
        _preprocess_structures()
        config.data.processed_dir = esmfold_path
    dataset = get_pdbredo_chain_dataset('val', config.data, iter_='sample', use_plm=config.model.get('use_plm', None))
    logger.info('Data points: ', len(dataset))
    loader = DataLoader(dataset, batch_size=config.train.batch_size * 4, shuffle=False, collate_fn=PaddingCollate(), num_workers=args.num_workers)

    logger.info('Building model...')
    if args.backbone == 'pdc':
        from rde.models.pdc import PDC_Network
        model = PDC_Network(config.model).to(args.device)
    elif args.backbone == 'equiformer':
        from rde.models.equiformer import EquiformerNet
        model = EquiformerNet(config.model).to(args.device)
    else:
        from rde.models.rde import CircularSplineRotamerDensityEstimator
        model = CircularSplineRotamerDensityEstimator(ckpt['config'].model).to(args.device)
    logger.info('Loading state dict...')
    model.load_state_dict(ckpt['model'], strict=False)

    loss_dict = {}
    chis_logger = {'1': [], '2': [], '3': [], '4': []}
    xs_list, ys_list = [], []
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(tqdm(loader, desc='Validate', dynamic_ncols=True)):
            batch = recursive_to(batch, args.device)

            # Side-chain recovery
            if args.task == 'chi_angles':
                if args.backbone == 'rde':   # RDE
                    xs = model(batch)    # (N, L, 4)
                    xs = torch.cat(xs, dim=-1)
                    xs -= np.pi
                    n_chis_data = batch['chi_mask'].sum(-1)  # (N, L, 4) -> (N, L)
                    chi_complete = batch['chi_complete']  # (N, L), only consider complete chi-angles

                    for res_name, num_chis in num_chi_angles.items():
                        if num_chis < 1:
                            continue
                        loss_mask = torch.logical_and(chi_complete, batch['aa'] == res_name._value_)
                        for n_chis in range(1, num_chis + 1):
                            loss = torch.nn.functional.l1_loss(xs.squeeze(-1)[..., n_chis - 1][loss_mask], batch['chi_native'][..., n_chis - 1][loss_mask], reduction='none')
                            loss_dict[f'mae_{res_name._name_}_{n_chis}'] = loss
                else:
                    loss_dict = model(batch, mode='test')

                for key in loss_dict.keys():
                    if len(chis_logger[key[-1]]) == 0:
                        chis_logger[key[-1]] = loss_dict[key]
                    else:
                        chis_logger[key[-1]] = torch.cat([chis_logger[key[-1]], loss_dict[key]])

                    if key not in chis_logger:
                        chis_logger[key] = loss_dict[key]
                    else:
                        chis_logger[key] = torch.cat([chis_logger[key], loss_dict[key]])

            # Clustering evaluation
            else:
                xs = model.encode(batch).detach().max(dim=1)[0]   # (N, F)
                xs_list.append(xs)
                ys_list += batch['clust_name']

    if args.task == 'chi_angles':
        chis_logger = {k: v.mean().item() / (2 * np.pi) * 360 for k, v in chis_logger.items()}
        results = pd.DataFrame.from_dict(chis_logger, orient='index')
        results.to_csv(args.ckpt.split('.')[0] + args.output + '.csv', index=True)
    else:
        xs_list = torch.cat(xs_list).cpu().numpy()
        torch.save([xs_list, ys_list], f'{args.backbone}.pt')

        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        score = silhouette_score(xs_list, ys_list)
        db_index = davies_bouldin_score(xs_list, ys_list)
        ch_score = calinski_harabasz_score(xs_list, ys_list)

        print('Silhouette Score (best: 1, worse: -1):', score)
        print('Davies-Bouldin Index (the lower, the better, min: 0 ):', db_index)
        print('Calinski Haabasz Score (the higher, the better):', ch_score)




































