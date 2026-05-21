#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=hd_cub_euc_val
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00
#SBATCH --output=logs/hypdiffusion_cub200_euc_val_%A.out

module purge
module load 2025
module load Miniconda3/25.5.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/scur0164/conda_envs/hypdiff_env
export LD_LIBRARY_PATH=/home/scur0164/conda_envs/hypdiff_env/lib:$LD_LIBRARY_PATH

cd ${SLURM_SUBMIT_DIR}/HypDiffusion

VAL_DIR=/scratch-shared/scur0164/datasets/cub200_split/val
CONFIG_PATH=configs/stable-diffusion/v2_inference_cub200_euc.yaml
SD_CKPT=/scratch-shared/scur0164/checkpoints/v2-1_512-ema-pruned.ckpt

DDIM_STEPS=${DDIM_STEPS:-50}
STRENGTH=${STRENGTH:-0.95}
SCALE=${SCALE:-1.3}
N_SELECTED=${N_SELECTED:-2}
N_CLASSES=${N_CLASSES:-5}
N_SAMPLES=${N_SAMPLES:-1}
N_PERTURB_SAMPLES=${N_PERTURB_SAMPLES:-0}
OUTDIR=${OUTDIR:-/scratch-shared/scur0164/outputs/HypDiffusion/cub200_euc_val_s${STRENGTH}_cfg${SCALE}_n${N_SELECTED}_c${N_CLASSES}_${SLURM_JOB_ID}}

echo "Running CUB200 Euclidean HypDiffusion validation"
echo "VAL_DIR=${VAL_DIR}"
echo "OUTDIR=${OUTDIR}"
echo "DDIM_STEPS=${DDIM_STEPS}, STRENGTH=${STRENGTH}, SCALE=${SCALE}"
echo "N_SELECTED=${N_SELECTED}, N_CLASSES=${N_CLASSES}, N_SAMPLES=${N_SAMPLES}, N_PERTURB_SAMPLES=${N_PERTURB_SAMPLES}"

srun python inference_for_eva.py \
    --image_folder ${VAL_DIR} \
    --outdir ${OUTDIR} \
    --config_path ${CONFIG_PATH} \
    --ckpt ${SD_CKPT} \
    --ddim_steps ${DDIM_STEPS} \
    --ddim_eta 0.0 \
    --strength ${STRENGTH} \
    --scale ${SCALE} \
    --n_samples ${N_SAMPLES} \
    --n_perturb_samples ${N_PERTURB_SAMPLES} \
    --n_selected ${N_SELECTED} \
    --n_classes ${N_CLASSES}
