#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=hd_cal_euc_val
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/hypdiffusion_caltech256_euc_val_%A.out

module purge
module load 2025
module load Miniconda3/25.5.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/scur0164/conda_envs/hypdiff_env
export LD_LIBRARY_PATH=/home/scur0164/conda_envs/hypdiff_env/lib:$LD_LIBRARY_PATH

cd ${SLURM_SUBMIT_DIR}/HypDiffusion

VAL_DIR=/scratch-shared/scur0164/datasets/caltech256_split/val
CONFIG_PATH=configs/stable-diffusion/v2_inference_caltech256_euc.yaml
MODEL_CKPT=/scratch-shared/scur0164/checkpoints/Stage1/caltech256/2026-05-17T10-49-13_v1_caltech256/checkpoints/epoch=000020.ckpt

OUTDIR=/scratch-shared/scur0164/outputs/HypDiffusion/caltech256_euc_val_s0.95_cfg1.3_n2_c5_${SLURM_JOB_ID}
CODECARBON_PROJECT_NAME=infer_hypdiffusion_caltech256_euc_val
CODECARBON_OUTPUT_DIR=/scratch-shared/scur0164/outputs/codecarbon/HypDiffusion/${SLURM_JOB_ID}

echo "Running Caltech256 Euclidean HypDiffusion validation"
echo "VAL_DIR=${VAL_DIR}"
echo "OUTDIR=${OUTDIR}"
echo "MODEL_CKPT=${MODEL_CKPT}"
echo "CODECARBON_OUTPUT_DIR=${CODECARBON_OUTPUT_DIR}"

srun python inference_for_eva.py \
    --image_folder ${VAL_DIR} \
    --outdir ${OUTDIR} \
    --config_path ${CONFIG_PATH} \
    --ckpt ${MODEL_CKPT} \
    --ddim_steps 50 \
    --ddim_eta 0.0 \
    --strength 0.95 \
    --scale 1.3 \
    --seed 23 \
    --n_samples 1 \
    --n_perturb_samples 0 \
    --n_selected 2 \
    --n_classes 5 \
    --use_codecarbon \
    --codecarbon_project_name ${CODECARBON_PROJECT_NAME} \
    --codecarbon_output_dir ${CODECARBON_OUTPUT_DIR}
