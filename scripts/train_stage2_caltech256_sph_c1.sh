#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=s2_cal_sph_c1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/stage2_caltech256_sph_c1_%A.out

module purge
module load 2025
module load Miniconda3/25.5.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/scur0164/conda_envs/stage2_env
export LD_LIBRARY_PATH=/home/scur0164/conda_envs/stage2_env/lib:$LD_LIBRARY_PATH

cd ${SLURM_SUBMIT_DIR}/Stage2

# TODO: Set to best Stage 1 checkpoint after Stage 1 training completes
STAGE1_CKPT=/scratch-shared/scur0164/checkpoints/Stage1/caltech256/2026-05-17T10-49-13_v1_caltech256/checkpoints/epoch=000020.ckpt
ENCODER_VERSION=/scratch-shared/scur0164/checkpoints/clip-vit-large-patch14
DEST_DIR=/scratch-shared/scur0164/checkpoints/Stage2/caltech256_sph_c1
CC_PROJECT_NAME=train_stage2_caltech256_sph_c1

srun python scripts/train.py \
    --dataset_type=caltech256_encode \
    --stage1_checkpoint_path=${STAGE1_CKPT} \
    --encoder_version=${ENCODER_VERSION} \
    --exp_dir=${DEST_DIR} \
    --seed 23 \
    --use_codecarbon \
    --codecarbon_project_name=${CC_PROJECT_NAME} \
    --geometry spherical \
    --c=1 \
    --feature_size=512 \
    --transformer_layers=30 \
    --train_all \
    --learning_rate=0.001 \
    --step_size=5000 \
    --gamma=0.5 \
    --optim_name=adamw \
    --batch_size=256 \
    --test_batch_size=64 \
    --workers=8 \
    --test_workers=4 \
    --max_steps=100000 \
    --save_interval=5000 \
    --board_interval=50 \
    --val_interval=1000 \
    --early_stop_patience=20 \
    --early_stop_min_delta=0.0001 \
    --geometry_lambda=1.0 \
    --reverse_lambda=0.1 \
    --cosine_lambda=0.1 \
    --ngpus-per-node 1 \
    --dist-url 'tcp://localhost:10063' \
    --multiprocessing-distributed \
    --dist-backend nccl \
    --world-size 1 \
    --rank 0
