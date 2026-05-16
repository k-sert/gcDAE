#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=split_datasets
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/split_datasets_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate stage1_env

cd ${SLURM_SUBMIT_DIR}

# CUB-200: stratified 70/10/20 split per class.
python split_dataset.py \
    --src /scratch-shared/scur0164/datasets/NABirds/CUB_200_2011/images \
    --dst /scratch-shared/scur0164/datasets/cub200_split \
    --train-size 0.7 \
    --val-size 0.1 \
    --test-size 0.2 \
    --seed 42 \
    --copy \
    --overwrite

# Caltech-256: stratified 70/10/20 split per class.
python split_dataset.py \
    --src /scratch-shared/scur0164/datasets/Caltech_256 \
    --dst /scratch-shared/scur0164/datasets/caltech256_split \
    --train-size 0.7 \
    --val-size 0.1 \
    --test-size 0.2 \
    --seed 42 \
    --copy \
    --overwrite
