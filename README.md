# Predicting the Validity of Set Data with Self-supervised Masked Transformer

## Introduction

This is the implementation of _Predicting the Validity of Set Data with Self-supervised Masked Transformer (SetMtr)_. We implement it (`setmtr`) based on a simple training toolkit [torchility](https://github.com/hitlic/torchility) and provide a data preprocessing toolbox (`./datasets`)

## Dependency

- `pytorch>=2.0`
- `pytorch-lightning>=2.0,<2.1`
- `torchmetrics>=0.11,<0.12`
- `torchility == 0.9`
    - `pip install torchility==0.9`


## Usage

- Data Prepare
    - run `datasets/original_data_process/<dataset_name>/data_gen.py` to generate data in the required format which will be saved in `datasets/txts` 
    - run ``datasets/process.py` to generate dataset for traning and evaluationg, which will bed saved in `datasets/pkls`.

- Configure
    - Configure model parameters and data sets in `setmtr/config.yaml`
- Train and evaluate
    - `python setmtr/train.py`
