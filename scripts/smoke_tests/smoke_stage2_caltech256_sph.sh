#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=smk_s2_cal_sph
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/smoke_stage2_caltech256_sph_%A.out

module purge
module load 2025
module load Miniconda3/25.5.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/scur0164/conda_envs/stage2_env
export LD_LIBRARY_PATH=/home/scur0164/conda_envs/stage2_env/lib:$LD_LIBRARY_PATH

cd ${SLURM_SUBMIT_DIR}/Stage2

STAGE1_CKPT=/scratch-shared/scur0164/checkpoints/Stage1/caltech256_smoke/2026-05-14T14-12-29_v1_caltech256/checkpoints/epoch=000000.ckpt
EXP_DIR=/scratch-shared/scur0164/checkpoints/Stage2/smoke/caltech256_sph_${SLURM_JOB_ID}

python scripts/train.py \
    --dataset_type=caltech256_encode \
    --psp_checkpoint_path=${STAGE1_CKPT} \
    --encoder_version=/scratch-shared/scur0164/checkpoints/clip-vit-large-patch14 \
    --exp_dir=${EXP_DIR} \
    --seed 23 \
    --geometry spherical \
    --c=1 \
    --feature_size=512 \
    --transformer_layers=5 \
    --train_all \
    --learning_rate=0.0001 \
    --step_size=2 \
    --gamma=0.5 \
    --optim_name=adam \
    --batch_size=4 \
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
    --dist-url tcp://localhost:10105 \
    --multiprocessing-distributed \
    --dist-backend nccl \
    --world-size 1 \
    --rank 0
