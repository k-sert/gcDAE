import os

# Base paths — set these environment variables before running, or edit directly.
_DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/datasets")
_CKPT_ROOT = os.environ.get("CKPT_ROOT", "/path/to/checkpoints")

dataset_paths = {
    'cub200_train': f'{_DATA_ROOT}/cub200_split/train',
    'cub200_val':   f'{_DATA_ROOT}/cub200_split/val',
    'cub200_test':  f'{_DATA_ROOT}/cub200_split/test',
    'caltech256_train': f'{_DATA_ROOT}/caltech256_split/train',
    'caltech256_val':   f'{_DATA_ROOT}/caltech256_split/val',
    'caltech256_test':  f'{_DATA_ROOT}/caltech256_split/test',
}

# Optional registry for model/checkpoint paths.
# Active Stage 2 runs pass the Stage 1 checkpoint and CLIP encoder paths through
# launch scripts. Add entries here only for optional utilities or losses.
model_paths = {
    # 'example_checkpoint': f'{_CKPT_ROOT}/checkpoint.pt',
}
