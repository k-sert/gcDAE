#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=patch_ckpt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:10:00
#SBATCH --output=logs/patch_checkpoint_%A.out

module purge
module load 2025
module load Miniconda3/25.5.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/scur0164/conda_envs/stage1_env

# Set this to the checkpoint you want to patch
CKPT=/scratch-shared/scur0164/checkpoints/Stage1/cub200/2026-05-12T09-05-00_v1_cub200/checkpoints/epoch=000019.ckpt

python scripts/patch_early_stopping.py ${CKPT} --patience 30 --wait_count 0
