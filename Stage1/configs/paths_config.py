dataset_paths = {
    'cub200_smoke_train': '/scratch-shared/scur0164/datasets/cub200_smoke/train',
    'cub200_smoke_test': '/scratch-shared/scur0164/datasets/cub200_smoke/test',
    'caltech256_smoke_train': '/scratch-shared/scur0164/datasets/caltech256_smoke/train',
    'caltech256_smoke_test': '/scratch-shared/scur0164/datasets/caltech256_smoke/test',
    'cub200_train': '/scratch-shared/scur0164/datasets/cub200_split/train',
    'cub200_val': '/scratch-shared/scur0164/datasets/cub200_split/val',
    'cub200_test': '/scratch-shared/scur0164/datasets/cub200_split/test',
    'caltech256_train': '/scratch-shared/scur0164/datasets/caltech256_split/train',
    'caltech256_val': '/scratch-shared/scur0164/datasets/caltech256_split/val',
    'caltech256_test': '/scratch-shared/scur0164/datasets/caltech256_split/test',
}


# Optional registry for model/checkpoint paths.
# Keep this as a template only; active Stage 1 runs currently pass checkpoint
# paths through config files or launch scripts.
model_paths = {
    # 'example_checkpoint': '/path/to/checkpoint.ckpt',
}
