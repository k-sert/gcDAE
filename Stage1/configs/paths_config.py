import os

# Base paths — set these environment variables before running, or edit directly.
_DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/datasets")
_CKPT_ROOT = os.environ.get("CKPT_ROOT", "/path/to/checkpoints")

dataset_paths = {
    'cub200_smoke_train': f'{_DATA_ROOT}/cub200_smoke/train',
    'cub200_smoke_test':  f'{_DATA_ROOT}/cub200_smoke/test',
    'caltech256_smoke_train': f'{_DATA_ROOT}/caltech256_smoke/train',
    'caltech256_smoke_test':  f'{_DATA_ROOT}/caltech256_smoke/test',
    'cub200_train': f'{_DATA_ROOT}/cub200_split/train',
    'cub200_val':   f'{_DATA_ROOT}/cub200_split/val',
    'cub200_test':  f'{_DATA_ROOT}/cub200_split/test',
    'caltech256_train': f'{_DATA_ROOT}/caltech256_split/train',
    'caltech256_val':   f'{_DATA_ROOT}/caltech256_split/val',
    'caltech256_test':  f'{_DATA_ROOT}/caltech256_split/test',
}

# Optional registry for model/checkpoint paths.
# Keep this as a template only; active Stage 1 runs currently pass checkpoint
# paths through config files or launch scripts.
model_paths = {
    # 'example_checkpoint': f'{_CKPT_ROOT}/checkpoint.ckpt',
}
