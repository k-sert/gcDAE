#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --job-name=stage1_cub200
#SBATCH --time=48:00:00
#SBATCH --output=logs/stage1_cub200_%A.out

module purge
module load 2025
module load Miniconda3/25.5.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/scur0164/conda_envs/stage1_env

cd ${SLURM_SUBMIT_DIR}/Stage1

# Path to the pretrained Stable Diffusion v2.1 checkpoint.
# Verify this path exists on Snellius before submitting:
#   ls /scratch-shared/scur0164/checkpoints/v2-1_512-ema-pruned.ckpt
SD_CKPT=/scratch-shared/scur0164/checkpoints/v2-1_512-ema-pruned.ckpt

python -u main.py \
    --base configs/v1_cub200.yaml \
    --logdir /scratch-shared/scur0164/checkpoints/Stage1/cub200 \
    --pretrained_model ${SD_CKPT} \
    --seed 23 \
    --scale_lr False