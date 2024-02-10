# Refine-PPI

## Install
### Environment

```bash
conda env create -f env.yml -n REFINE
conda activate REFINE
```

The default PyTorch version is 1.12.1 and cudatoolkit version is 11.3. They can be changed in [`env.yml`](./env.yml).

### Datasets

| Dataset   | Download Script                                    |
| --------- | -------------------------------------------------- |
| [PDB-REDO](https://pdb-redo.eu/)  | [`data/get_pdbredo.sh`](./data/get_pdbredo.sh)     |
| [SKEMPI v2](https://life.bsc.es/pid/skempi2) | [`data/get_skempi_v2.sh`](./data/get_skempi_v2.sh) |

## Usage
### Train RDE

```bash
python train_redo.py ./configs/train/pdc_redo.yml
```

### Train RDE-Network (DDG)

```bash
python train_skempi_abbind.py ./configs/train/pdc_ddg_refine.yml
```
