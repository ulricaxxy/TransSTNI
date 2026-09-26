<div align="center">
  <h2><b> [SIGKDD'26 STUNet] Unified Spatio-Temporal Tokens are Bases for Generalizable Traffic Forecasting
 </b></h2>
</div>

![The framework of STUNet](./fig/model.png#pic_center)

---

This repo is the official implementation of our paper "Unified Spatio-Temporal Tokens are Bases for Generalizable Traffic Forecasting". 

## Introduction

**STUNet** is a generalizable spatio-temporal forecasting model that explicitly models spatial information from road network topology. By decoupling spatial representation from temporal dynamics, STUNet enables zero-shot transfer across different traffic networks. 

## Environment Requirements
1. `python==3.10.16`
2. `torch==2.6.0+cu1x.x`
3. `transformers==4.46.2`
4. `tensorboard==2.19.0`
5. `torch-geometric==2.6.1`
6. `lightning>=2.0.0`
7. `torchmetrics>=0.11.4`
8. `hydra-core==1.3.2`
9. `hydra-colorlog==1.2.0`
10. `hydra-optuna-sweeper==1.2.0`
11. `swanlab`
12. `wandb`
13. `rootutils`
14. `rich`
15. `pytest`
16. `sh`

## Project Architecture

```text
STUNet/
│
├─ configs/                     # configs
│
├─ notebooks/                   # Utils of data preprocessing
│
├─ src/
│  ├─ train.py                  # Unified training entry
│  ├─ eval.py                   # Evaluation entry
│  │
│  ├─ data/                     # Data pipeline
│  │
│  ├─ models/                   # Model implementations
│  │  ├─ STUNet/                # STUNet core components
│  │  │  ├─ core                # Network implementation
│  │  │  │  ├─ STUNet.py        # Core implementation
│  │  │  │  ├─ SpatialAE.py     # Spatial tokenizer pre-train
│  │  │  │  ├─ TemporalAE.py    # Temporal tokenizer pre-train
│  │  │  │  ├─ loss_fn.py       # Loss functions
│  │  │  │  └─ ...
│  │  │  ├─ pretrainer.py       # Trainer for tokenizer pre-training
│  │  │  ├─ backbonetrainer.py  # Trainer for backbone training
│  │  │  └─ ...
│  │  └─ ...
│  │
│  └─ utils/                    # General utilities
│
└─ README.md                    
```

## Data Preparation
We follow [LargeST](https://github.com/liuxu77/LargeST) to prepare data. Please follow the data preparation instructions of [LargeST](https://github.com/liuxu77/LargeST) strickly. Then, please refer to `notebooks/data_transform.ipynb` to obtain data for training.

## Experiments

STUNet employs a two-stage training paradigm. Please follow the instructions below to run STUNet.

### Setups

This code relies on config files in `configs/`, so the first step is to complete the config settings. Please check `configs` and fill in related paths. For **Quick Start**, please check `configs/paths/default.yaml` and replace them with your paths.

### Tokenizer pre-training

To pre-train spatial or temporal tokenizer, please refer to `configs/model/SpatialAE.yaml` or `configs/model/SpatialAE.yaml`. Note that the `save_path` term is required since the checkpoints will be used in the next stage. Then run the following command:

```
python src/train.py --config-name=spatial
python src/train.py --config-name=temporal
```

### Backbone training

Please refer to `configs/model/STUNet.yaml` to train the backbone of STUNet. Note that the `save_path` terms are required to be the same as in tokenizer pre-training. Then run the following command:

```
python src/train.py --config-name=train
```

### Evaluation

If you want to evaluate the model, please run the following command:

```
python src/eval.py --config-name=eval
```


## Citation

If you find our work helpful, please cite our research:

```bibtex
@inproceedings{chen2026stunet,
  title={Unified Spatio-Temporal Tokens are Bases for Generalizable Traffic Forecasting},
  author={Chen, Yujun and Tu, Shihao and Ding, Wenyue and Lu, Yicheng and Ren, Qingkai and Zheng, Yangjie and Yang, Yang},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2026},
  note={To appear}
}
```