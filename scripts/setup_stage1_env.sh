#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=setup_stage1_env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup_stage1_env_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1
source activate stage1_env

# Install PyTorch for CUDA 12.1 (H100 compatible)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Stage 1 dependencies
pip install \
    pytorch-lightning==1.8.5.post0 \
    omegaconf \
    transformers \
    einops \
    kornia \
    albumentations \
    tqdm \
    open-clip-torch \
    invisible-watermark \
    pudb

echo "Done. Verifying torch install:"
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
