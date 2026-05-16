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

# Make conda env's libstdc++ take precedence over system /lib64/libstdc++.so.6
export LD_LIBRARY_PATH=/home/scur0164/conda_envs/stage2_env/lib:$LD_LIBRARY_PATH

# Install missing packages (torch and transformers already present)
pip install geoopt==0.5.0
pip install albumentations==1.3.0
pip install codecarbon

echo "Verifying:"
python -c "
import torch
import transformers
import geoopt
import albumentations
import codecarbon
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('geoopt:', geoopt.__version__)
print('albumentations:', albumentations.__version__)
print('codecarbon:', codecarbon.__version__)
print('CUDA available:', torch.cuda.is_available())
print('H100 sm_90 supported:', torch.cuda.get_arch_list())
"
