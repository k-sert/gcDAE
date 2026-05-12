#!/bin/bash

#SBATCH --partition=staging
#SBATCH --job-name=upgrade_stage2_env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/upgrade_stage2_env_%A.out

module purge
module load 2025
module load Miniconda3/25.5.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/scur0164/conda_envs/stage2_env

# Install missing packages
pip install transformers==4.30.0
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

echo "Verifying:"
python -c "
import torch
import transformers
import geoopt
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('geoopt:', geoopt.__version__)
print('CUDA available:', torch.cuda.is_available())
print('H100 sm_90 supported:', torch.cuda.get_arch_list())
"
