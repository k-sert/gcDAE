#!/bin/bash
conda activate stage2_env
export LD_LIBRARY_PATH=$(conda info --base)/envs/stage2_env/lib:$LD_LIBRARY_PATH

cd "$(dirname "$0")/../../Stage2"

# TODO: Set to best Stage 1 checkpoint after Stage 1 training completes
STAGE1_CKPT=${CKPT_ROOT}/Stage1/cub200/2026-05-17T10-49-32_v1_cub200/checkpoints/epoch=000007.ckpt
ENCODER_VERSION=${CKPT_ROOT}/clip-vit-large-patch14
DEST_DIR=${CKPT_ROOT}/Stage2/cub200_sph_c10_$(date +%Y%m%d_%H%M%S)
CC_PROJECT_NAME=train_stage2_cub200_sph_c10

python scripts/train.py \
    --dataset_type=cub200_encode \
    --stage1_checkpoint_path=${STAGE1_CKPT} \
    --encoder_version=${ENCODER_VERSION} \
    --exp_dir=${DEST_DIR} \
    --seed 23 \
    --use_codecarbon \
    --codecarbon_project_name=${CC_PROJECT_NAME} \
    --geometry spherical \
    --c=10 \
    --feature_size=512 \
    --transformer_layers=30 \
    --train_all \
    --learning_rate=0.0001 \
    --step_size=1000 \
    --gamma=0.5 \
    --optim_name=adamw \
    --batch_size=128 \
    --test_batch_size=64 \
    --workers=8 \
    --test_workers=4 \
    --max_steps=100000 \
    --save_interval=5000 \
    --board_interval=50 \
    --val_interval=1000 \
    --early_stop_patience=5 \
    --early_stop_min_delta=0.0001 \
    --geometry_lambda=1.0 \
    --reverse_lambda=0.1 \
    --cosine_lambda=0.1 \
    --ngpus-per-node 1 \
    --dist-url 'tcp://localhost:10062' \
    --multiprocessing-distributed \
    --dist-backend nccl \
    --world-size 1 \
    --rank 0
