#!/bin/bash

#SBATCH --partition=thin
#SBATCH --job-name=split_datasets
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/split_datasets_%A.out



cd ${SLURM_SUBMIT_DIR}

# CUB-200: stratified 80/20 split per class.
python scripts/split_dataset.py \
    --src /scratch-shared/scur0164/datasets/NABirds/CUB_200_2011/images \
    --dst /scratch-shared/scur0164/datasets/cub200_split \
    --test-size 0.2 \
    --seed 42 \
    --copy

# Caltech-256: stratified 80/20 split per class.
python scripts/split_dataset.py \
    --src /scratch-shared/scur0164/datasets/Caltech_256 \
    --dst /scratch-shared/scur0164/datasets/caltech256_split \
    --test-size 0.2 \
    --seed 42 \
    --copy
