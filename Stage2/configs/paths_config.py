dataset_paths = {
    'cub200_train': '/scratch-shared/scur0164/datasets/cub200_split/train',
    'cub200_val': '/scratch-shared/scur0164/datasets/cub200_split/val',
    'cub200_test': '/scratch-shared/scur0164/datasets/cub200_split/test',
    'caltech256_train': '/scratch-shared/scur0164/datasets/caltech256_split/train',
    'caltech256_val': '/scratch-shared/scur0164/datasets/caltech256_split/val',
    'caltech256_test': '/scratch-shared/scur0164/datasets/caltech256_split/test',
}


# Optional registry for model/checkpoint paths.
# Active Stage 2 runs pass the Stage 1 checkpoint and CLIP encoder paths through
# launch scripts. Add entries here only for optional utilities or losses.
model_paths = {
    # 'example_checkpoint': '/path/to/checkpoint.pt',
}
