#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=upgrade_stage1_env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/upgrade_stage1_env_%A.out

module purge
module load 2025
module load Miniconda3/25.5.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/scur0164/conda_envs/stage1_env

# Upgrade PyTorch to version with H100 (sm_90) support, CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Upgrade PyTorch Lightning and torchmetrics to compatible versions
pip install pytorch-lightning==1.8.5.post0 torchmetrics==0.11.0

echo "Verifying:"
python -c "
import torch
import pytorch_lightning as pl
print('torch:', torch.__version__)
print('pytorch_lightning:', pl.__version__)
print('CUDA available:', torch.cuda.is_available())
print('H100 sm_90 supported:', torch.cuda.get_arch_list())
"
