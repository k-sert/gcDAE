from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
    'cub200_smoke_encode': {
        'transforms': transforms_config.EncodeTransforms,
        'train_source_root': dataset_paths['cub200_smoke_train'],
        'train_target_root': dataset_paths['cub200_smoke_train'],
        'test_source_root': dataset_paths['cub200_smoke_test'],
        'test_target_root': dataset_paths['cub200_smoke_test'],
    },
    'caltech256_smoke_encode': {
        'transforms': transforms_config.EncodeTransforms,
        'train_source_root': dataset_paths['caltech256_smoke_train'],
        'train_target_root': dataset_paths['caltech256_smoke_train'],
        'test_source_root': dataset_paths['caltech256_smoke_test'],
        'test_target_root': dataset_paths['caltech256_smoke_test'],
    },
    'cub200_encode': {
        'transforms': transforms_config.EncodeTransforms,
        'train_source_root': dataset_paths['cub200_train'],
        'train_target_root': dataset_paths['cub200_train'],
        'val_source_root': dataset_paths['cub200_val'],
        'val_target_root': dataset_paths['cub200_val'],
        'test_source_root': dataset_paths['cub200_test'],
        'test_target_root': dataset_paths['cub200_test'],
    },
    'caltech256_encode': {
        'transforms': transforms_config.EncodeTransforms,
        'train_source_root': dataset_paths['caltech256_train'],
        'train_target_root': dataset_paths['caltech256_train'],
        'val_source_root': dataset_paths['caltech256_val'],
        'val_target_root': dataset_paths['caltech256_val'],
        'test_source_root': dataset_paths['caltech256_test'],
        'test_target_root': dataset_paths['caltech256_test'],
    },
}
