#!/bin/bash
# Required env vars: CKPT_ROOT (checkpoint root), DATA_ROOT (dataset root)
# Optional: STAGE1_CKPT (for Stage 2 scripts)


conda activate stage1_env

cd "$(dirname "$0")/../../Stage1"

# Path to the pretrained Stable Diffusion v2.1 checkpoint.
# Verify this path exists on Snellius before submitting:
#   ls ${CKPT_ROOT}/v2-1_512-ema-pruned.ckpt
SD_CKPT=${CKPT_ROOT}/v2-1_512-ema-pruned.ckpt

python -u main.py \
    --base configs/v1_cub200.yaml \
    --logdir ${CKPT_ROOT}/Stage1/cub200 \
    --pretrained_model ${SD_CKPT} \
    --seed 23 \
    --scale_lr False