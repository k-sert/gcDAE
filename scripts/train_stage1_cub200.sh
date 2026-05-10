#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=stage1_cub200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=logs/stage1_cub200_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate stage1_env

cd ${SLURM_SUBMIT_DIR}/Stage1

# Path to the pretrained Stable Diffusion v2.1 checkpoint.
# Verify this path exists on Snellius before submitting:
#   ls /scratch-shared/scur0164/checkpoints/v2-1_512-ema-pruned.ckpt
SD_CKPT=/scratch-shared/scur0164/checkpoints/v2-1_512-ema-pruned.ckpt

python -u main.py \
    --base configs/v1_cub200.yaml \
    --logdir /scratch-shared/scur0164/checkpoints/Stage1/cub200 \
    --pretrained_model ${SD_CKPT} \
    --scale_lr False
