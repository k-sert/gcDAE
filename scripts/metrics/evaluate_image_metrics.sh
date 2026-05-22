#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=eval_metrics
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_metrics_%A.out

set -euo pipefail

module purge
module load 2025
module load Miniconda3/25.5.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/scur0164/conda_envs/hypdiff_env

export MPLCONFIGDIR=/scratch-local/${SLURM_JOB_ID}/matplotlib
mkdir -p "${MPLCONFIGDIR}"
FIDELITY_CACHE=/scratch-shared/scur0164/checkpoints/metric_cache/torch_fidelity
mkdir -p "${FIDELITY_CACHE}"

if [ "$#" -lt 3 ]; then
    echo "Usage: sbatch $0 <generated_dir> <reference_dir> <out_json>"
    exit 2
fi

GENERATED_DIR="$1"
REFERENCE_DIR="$2"
OUT_JSON="$3"
LPIPS_MODE=${LPIPS_MODE:-paired}
LPIPS_MAX_PAIRS_PER_CLASS=${LPIPS_MAX_PAIRS_PER_CLASS:-1000}
LPIPS_SEED=${LPIPS_SEED:-23}
RUN_CODECARBON=${RUN_CODECARBON:-0}
CODECARBON_PROJECT_NAME=${CODECARBON_PROJECT_NAME:-evaluate_image_metrics}
CODECARBON_OUTPUT_DIR=${CODECARBON_OUTPUT_DIR:-/scratch-shared/scur0164/outputs/codecarbon/metrics/${SLURM_JOB_ID}}

cd "${SLURM_SUBMIT_DIR}"

CMD=(python scripts/metrics/evaluate_image_metrics.py \
    --generated "${GENERATED_DIR}" \
    --reference "${REFERENCE_DIR}" \
    --out "${OUT_JSON}" \
    --batch_size 32 \
    --workers 8 \
    --cache_root "${FIDELITY_CACHE}" \
    --lpips_mode "${LPIPS_MODE}" \
    --lpips_max_pairs_per_class "${LPIPS_MAX_PAIRS_PER_CLASS}" \
    --lpips_seed "${LPIPS_SEED}")

if [ "${RUN_CODECARBON}" = "1" ]; then
    python scripts/run_with_codecarbon.py \
        --project-name "${CODECARBON_PROJECT_NAME}" \
        --output-dir "${CODECARBON_OUTPUT_DIR}" \
        -- "${CMD[@]}"
else
    "${CMD[@]}"
fi
