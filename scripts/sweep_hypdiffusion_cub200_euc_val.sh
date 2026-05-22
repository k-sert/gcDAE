#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=hd_cub_euc_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --array=0-5
#SBATCH --output=logs/hypdiffusion_cub200_euc_sweep_%A_%a.out

module purge
module load 2025
module load Miniconda3/25.5.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/scur0164/conda_envs/hypdiff_env
export LD_LIBRARY_PATH=/home/scur0164/conda_envs/hypdiff_env/lib:$LD_LIBRARY_PATH

cd ${SLURM_SUBMIT_DIR}/HypDiffusion

VAL_DIR=/scratch-shared/scur0164/datasets/cub200_split/val
CONFIG_PATH=configs/stable-diffusion/v2_inference_cub200_euc.yaml
MODEL_CKPT=/scratch-shared/scur0164/checkpoints/Stage1/cub200/2026-05-17T10-49-32_v1_cub200/checkpoints/epoch=000007.ckpt

case ${SLURM_ARRAY_TASK_ID} in
    0)
        STRENGTH=0.20
        SCALE=1.0
        ;;
    1)
        STRENGTH=0.50
        SCALE=1.0
        ;;
    2)
        STRENGTH=0.75
        SCALE=1.0
        ;;
    3)
        STRENGTH=0.85
        SCALE=1.0
        ;;
    4)
        STRENGTH=0.95
        SCALE=1.0
        ;;
    5)
        STRENGTH=0.95
        SCALE=1.3
        ;;
    *)
        echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
        exit 2
        ;;
esac

SETTING=s${STRENGTH}_cfg${SCALE}
OUTDIR=/scratch-shared/scur0164/outputs/HypDiffusion/cub200_euc_val_${SETTING}_n2_c5_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
CODECARBON_PROJECT_NAME=infer_hypdiffusion_cub200_euc_sweep_${SETTING}
CODECARBON_OUTPUT_DIR=/scratch-shared/scur0164/outputs/codecarbon/HypDiffusion/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

echo "Running CUB200 Euclidean HypDiffusion validation sweep"
echo "ARRAY_JOB=${SLURM_ARRAY_JOB_ID}, ARRAY_TASK=${SLURM_ARRAY_TASK_ID}"
echo "VAL_DIR=${VAL_DIR}"
echo "OUTDIR=${OUTDIR}"
echo "MODEL_CKPT=${MODEL_CKPT}"
echo "STRENGTH=${STRENGTH}, SCALE=${SCALE}"
echo "CODECARBON_OUTPUT_DIR=${CODECARBON_OUTPUT_DIR}"

srun python inference_for_eva.py \
    --image_folder ${VAL_DIR} \
    --outdir ${OUTDIR} \
    --config_path ${CONFIG_PATH} \
    --ckpt ${MODEL_CKPT} \
    --ddim_steps 50 \
    --ddim_eta 0.0 \
    --strength ${STRENGTH} \
    --scale ${SCALE} \
    --seed 23 \
    --n_samples 1 \
    --n_perturb_samples 0 \
    --n_selected 2 \
    --n_classes 5 \
    --use_codecarbon \
    --codecarbon_project_name ${CODECARBON_PROJECT_NAME} \
    --codecarbon_output_dir ${CODECARBON_OUTPUT_DIR}
