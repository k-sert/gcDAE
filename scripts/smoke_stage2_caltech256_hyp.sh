#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=smoke_s2_cal_hyp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_stage2_caltech256_hyp_%A.out

module purge
module load 2025
module load Miniconda3/25.5.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/scur0164/conda_envs/stage2_env
export LD_LIBRARY_PATH=/home/scur0164/conda_envs/stage2_env/lib:$LD_LIBRARY_PATH

cd ${SLURM_SUBMIT_DIR}/Stage2

# TODO: Set to best Stage 1 checkpoint after Stage 1 training completes
# Find it with: ls /scratch-shared/scur0164/checkpoints/Stage1/caltech256/*/checkpoints/
STAGE1_CKPT=/scratch-shared/scur0164/checkpoints/Stage1/caltech256/2026-05-12T09-05-27_v1_caltech256/checkpoints/epoch=000004.ckpt

python scripts/train.py \
    --dataset_type=caltech256_encode \
    --psp_checkpoint_path=${STAGE1_CKPT} \
    --encoder_version=/scratch-shared/scur0164/checkpoints/clip-vit-large-patch14 \
    --exp_dir=/scratch-shared/scur0164/checkpoints/Stage2/caltech256_hyp_smoke \
    --hyperbolic \
    --c=0.1 \
    --feature_size=512 \
    --transformer_layers=5 \
    --train_all \
    --learning_rate=0.0001 \
    --step_size=5000 \
    --gamma=0.3 \
    --optim_name=adam \
    --batch_size=64 \
    --test_batch_size=32 \
    --workers=8 \
    --test_workers=4 \
    --max_steps=200 \
    --save_interval=200 \
    --board_interval=20 \
    --val_interval=999999 \
    --hyperbolic_lambda=0.3 \
    --reverse_lambda=5 \
    --cosine_lambda=0.1 \
    --ngpus-per-node 1 \
    --dist-url 'tcp://localhost:10031' \
    --multiprocessing-distributed \
    --dist-backend nccl \
    --world-size 1 \
    --rank 0
