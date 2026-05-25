#!/bin/bash

conda activate stage1_env

cd "$(dirname "$0")"

# CUB-200: stratified 70/10/20 split per class.
python split_dataset.py \
    --src ${DATA_ROOT}/NABirds/CUB_200_2011/images \
    --dst ${DATA_ROOT}/cub200_split \
    --train-size 0.7 \
    --val-size 0.1 \
    --test-size 0.2 \
    --seed 42 \
    --copy \
    --overwrite

# Caltech-256: stratified 70/10/20 split per class.
python split_dataset.py \
    --src ${DATA_ROOT}/Caltech_256 \
    --dst ${DATA_ROOT}/caltech256_split \
    --train-size 0.7 \
    --val-size 0.1 \
    --test-size 0.2 \
    --seed 42 \
    --copy \
    --overwrite
