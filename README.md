# Refine-PPI

[//]: # (<img src="./assets/cover.png" alt="cover" style="width:50%;" />)

## Install
### Environment

```bash
conda env create -f env.yml -n REFINE
conda activate REFINE
```

The default PyTorch version is 1.12.0. They can be changed in [`env.yml`](./env.yml).

### Datasets

| Dataset   | Download Script                                    |
| --------- | -------------------------------------------------- |
| [PDB-REDO](https://pdb-redo.eu/)  | [`data/get_pdbredo.sh`](./data/get_pdbredo.sh)     |
| [SKEMPI v2](https://life.bsc.es/pid/skempi2) | [`data/get_skempi_v2.sh`](./data/get_skempi_v2.sh) |

## Usage
### Evaluate Refine-PPI

```bash
python test_skempi.py
```

### Train Model (DDG)

```bash

python train.py ./configs/pdc_ddg_refine.yml

```

### Pretrain 

```bash
python train_rde_network_skempi.py ./configs/pdc_redo_refine.yml
```

