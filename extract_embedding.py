import argparse
import os
import pathlib
import pickle
import shutil
import copy

import lmdb
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB.Polypeptide import one_to_index
from esm import FastaBatchedDataset, pretrained, MSATransformer
from tqdm import tqdm

from src.utils.misc import seed_all
from src.utils.protein.constants import (AA, three_to_one, non_standard_residue_substitutions)


def run(args):
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model.eval()
    if isinstance(model, MSATransformer):
        raise ValueError("This script currently does not handle models with MSA input (MSA Transformer).")
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(fasta_cache)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length), batch_sampler=batches)
    print(f"Read {fasta_cache} with {len(dataset)} sequences")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in args.include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

    with torch.no_grad():
        tmp = tqdm(enumerate(data_loader))
        for batch_idx, (labels, strs, toks) in tmp:
            tmp.set_description(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(labels):
                args.output_file = args.output_dir / f"{label}.pt"
                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                truncate_len = min(args.truncation_seq_length, len(strs[i]))
                if len(strs[i]) > args.truncation_seq_length:
                    print(f'Warning: {label} is truncated from {len(strs[i])} to {args.truncation_seq_length}')
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                if "per_tok" in args.include:
                    result["representations"] = {layer: t[i, 1: truncate_len + 1].clone() for layer, t in representations.items()}
                if "mean" in args.include:
                    result["mean_representations"] = {layer: t[i, 1: truncate_len + 1].mean(0).clone() for layer, t in representations.items()}
                if "bos" in args.include:
                    result["bos_representations"] = {layer: t[i, 0].clone() for layer, t in representations.items()}
                if return_contacts:
                    result["contacts"] = contacts[i, : truncate_len, : truncate_len].clone()

                torch.save(result, args.output_file, )


def get_seq(aa_list):
    seq = ''
    for x in aa_list:
        if AA(x).name not in three_to_one.keys():
            seq += three_to_one[non_standard_residue_substitutions[AA(x).name]]
        else:
            seq += three_to_one[AA(x).name]
    return seq


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract per-token representations and model outputs for sequences in a FASTA file")  # noqa
    parser.add_argument('data', type=str, choices=['skempi', 'redo'])
    parser.add_argument("model_location", type=str, help="PyTorch model file OR name of pretrained model to download (see README for models)", )
    parser.add_argument("output_dir", type=pathlib.Path, help="output directory for extracted representations", )
    parser.add_argument("--abbind", action="store_true", help="Whether to process ABbind dataset.")

    parser.add_argument("--toks_per_batch", type=int, default=16394, help="maximum batch size")
    parser.add_argument("--repr_layers", type=int, default=[-1], nargs="+", help="layers indices from which to extract representations (0 to num_layers, inclusive)", )
    parser.add_argument("--include", type=str, nargs="+", choices=["mean", "per_tok", "bos", "contacts"], help="specify which representations to return", required=True, )
    parser.add_argument("--truncation_seq_length", type=int, default=16394, help="truncate sequences longer than the given value", )   # default 4096

    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    args = parser.parse_args()
    seed_all(2023)   # default seed

    # run train.py first to preprocess the dataset
    if args.data == 'skempi':
        cache_dir = './data/SKEMPI_v2_cache'
        prefix = 'skempi_' if not args.abbind else 'skempi_abbind_'
        structures_cache = os.path.join(cache_dir, prefix + 'structures.pkl')
        entries_cache = os.path.join(cache_dir, prefix + 'entries.pkl')
        fasta_cache = os.path.join(cache_dir, prefix + 'sequences.fasta')
    else:
        cache_dir = './data/PDB_REDO_processed_raw'
        sanitized_clusters_path = os.path.join(cache_dir, 'sanitized_clusters.pkl')
        lmdb_path = os.path.join(cache_dir, 'structures.lmdb')
        fasta_cache = os.path.join(cache_dir, 'sequences.fasta')
        MAP_SIZE = 384 * (1024 * 1024 * 1024)  # 384GB

    if not os.path.exists(fasta_cache):
        if args.data == 'skempi':
            with open(structures_cache, 'rb') as f:
                structures = pickle.load(f)
            with open(entries_cache, 'rb') as f:
                entries_full = pickle.load(f)

            seqs, ids = [], []
            for entry in entries_full:
                idx = entry['pdbcode']
                data, seq_map = copy.deepcopy(structures[idx])

                if idx not in ids:  # prevent dubplicate
                    seq_wt = get_seq(data['aa'].numpy().tolist())
                    seqs.append(seq_wt)
                    ids.append(idx)

                idx_mut = idx + entry['mutstr']
                if idx_mut not in ids:
                    aa_mut = data['aa'].clone()
                    for mut in entry['mutations']:
                        ch_rs_ic = (mut['chain'], mut['resseq'], mut['icode'])
                        if ch_rs_ic not in seq_map:
                            continue
                        aa_mut[seq_map[ch_rs_ic]] = one_to_index(mut['mt'])
                    seq_mut = get_seq(aa_mut.numpy().tolist())
                    seqs.append(seq_mut)
                    ids.append(idx_mut)

        else:
            with open(sanitized_clusters_path, 'rb') as f:
                clusters = pickle.load(f)
            pdbcodes = set()
            for _, pdbchain_list in clusters.items():
                for pdbcode, _ in pdbchain_list:
                    pdbcodes.add(pdbcode)

            db_conn = lmdb.open(lmdb_path, map_size=MAP_SIZE, create=False, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, )

            seqs, ids = [], []
            for pdbcode in tqdm(pdbcodes):
                data = pickle.loads(db_conn.begin().get(pdbcode.encode()))  # Made a copy
                if pdbcode not in ids:  # prevent dubplicate
                    seq = get_seq(data['aa'].numpy().tolist())
                    seqs.append(seq)
                    ids.append(pdbcode)

        # https://github.com/gcorso/DiffDock/blob/main/datasets/pdbbind_lm_embedding_preparation.py
        records = []
        for (index, seq) in zip(ids, seqs):
            record = SeqRecord(Seq(seq), str(index))
            record.description = ''
            records.append(record)
        SeqIO.write(records, fasta_cache, "fasta")
        print("End extracting FAST. Please continue to generate ESM embeddings.")

    run(args)
    if args.data == 'skempi':     # REDO is too large to save in one file
        output_path = os.path.join(cache_dir, 'esm2_embeddings.pt')
        dict = {}
        for filename in tqdm(os.listdir(args.output_dir)):
            dict[filename.split('.')[0]] = torch.load(os.path.join(args.output_dir, filename))['representations'][33]
        torch.save(dict, output_path)
        shutil.rmtree(args.output_dir)
