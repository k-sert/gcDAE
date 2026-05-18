#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=smk_s2_cub_euc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/smoke_stage2_cub200_euc_%A.out

module purge
module load 2025
module load Miniconda3/25.5.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/scur0164/conda_envs/stage2_env
export LD_LIBRARY_PATH=/home/scur0164/conda_envs/stage2_env/lib:$LD_LIBRARY_PATH

cd ${SLURM_SUBMIT_DIR}/Stage2

STAGE1_CKPT=/scratch-shared/scur0164/checkpoints/Stage1/cub200/2026-05-17T10-49-32_v1_cub200/checkpoints/epoch=000007.ckpt
ENCODER_VERSION=/scratch-shared/scur0164/checkpoints/clip-vit-large-patch14
DEST_DIR=/scratch-shared/scur0164/checkpoints/Stage2/smoke/cub200_euc_${SLURM_JOB_ID}
CC_PROJECT_NAME=smoke_stage2_cub200_euc

srun python scripts/train.py \
    --dataset_type=cub200_encode \
    --stage1_checkpoint_path=${STAGE1_CKPT} \
    --encoder_version=${ENCODER_VERSION} \
    --exp_dir=${DEST_DIR} \
    --seed 23 \
    --use_codecarbon \
    --codecarbon_project_name=${CC_PROJECT_NAME} \
    --geometry euclidean \
    --feature_size=512 \
    --transformer_layers=30 \
    --train_all \
    --learning_rate=0.001 \
    --step_size=2 \
    --gamma=0.5 \
    --optim_name=adamw \
    --batch_size=256 \
    --test_batch_size=4 \
    --workers=2 \
    --test_workers=2 \
    --max_steps=2 \
    --save_interval=2 \
    --board_interval=1 \
    --val_interval=2 \
    --early_stop_patience=0 \
    --geometry_lambda=1.0 \
    --reverse_lambda=0.1 \
    --cosine_lambda=0.1 \
    --ngpus-per-node 1 \
    --dist-url 'tcp://localhost:10100' \
    --multiprocessing-distributed \
    --dist-backend nccl \
    --world-size 1 \
    --rank 0
