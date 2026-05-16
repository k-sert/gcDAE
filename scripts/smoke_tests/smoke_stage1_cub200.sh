#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --job-name=smoke_stage1_cub200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_stage1_cub200_%A.out

module purge
module load 2025
module load Miniconda3/25.5.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/scur0164/conda_envs/stage1_env


cd ${SLURM_SUBMIT_DIR}/Stage1

SD_CKPT=/scratch-shared/scur0164/checkpoints/v2-1_512-ema-pruned.ckpt

python -u main.py \
    --base configs/v1_cub200.yaml \
    --logdir /scratch-shared/scur0164/checkpoints/Stage1/cub200_smoke \
    --pretrained_model ${SD_CKPT} \
    --scale_lr False \
    --max_epochs 1 \
    --limit_train_batches 50 \
    --no-test True
