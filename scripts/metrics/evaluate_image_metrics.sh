#!/bin/bash
set -euo pipefail

conda activate hypdiff_env
PYTHON_BIN=${PYTHON_BIN:-python}

export MPLCONFIGDIR=${TMPDIR:-/tmp}/matplotlib
mkdir -p "${MPLCONFIGDIR}"
FIDELITY_CACHE=${FIDELITY_CACHE:-${OUTPUTS_ROOT:-outputs}/metric_cache/torch_fidelity}
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
SKIP_LPIPS=${SKIP_LPIPS:-0}
RUN_CODECARBON=${RUN_CODECARBON:-0}
CODECARBON_PROJECT_NAME=${CODECARBON_PROJECT_NAME:-evaluate_image_metrics}
CODECARBON_OUTPUT_DIR=${CODECARBON_OUTPUT_DIR:-${OUTPUTS_ROOT:-outputs}/codecarbon/metrics/${SLURM_JOB_ID}}

cd "$(dirname "$0")/../../"

echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" - <<'PY'
import sys
print("Python executable:", sys.executable)
PY

CMD=("${PYTHON_BIN}" scripts/metrics/evaluate_image_metrics.py \
    --generated "${GENERATED_DIR}" \
    --reference "${REFERENCE_DIR}" \
    --out "${OUT_JSON}" \
    --batch_size 32 \
    --workers 8 \
    --cache_root "${FIDELITY_CACHE}" \
    --lpips_mode "${LPIPS_MODE}" \
    --lpips_max_pairs_per_class "${LPIPS_MAX_PAIRS_PER_CLASS}" \
    --lpips_seed "${LPIPS_SEED}")

if [ "${SKIP_LPIPS}" = "1" ]; then
    CMD+=(--skip_lpips)
fi

if [ "${RUN_CODECARBON}" = "1" ]; then
    if [ -f scripts/run_with_codecarbon.py ]; then
        "${PYTHON_BIN}" scripts/run_with_codecarbon.py \
            --project-name "${CODECARBON_PROJECT_NAME}" \
            --output-dir "${CODECARBON_OUTPUT_DIR}" \
            -- "${CMD[@]}"
    else
        echo "RUN_CODECARBON=1 but scripts/run_with_codecarbon.py is missing; running metrics without CodeCarbon."
        "${CMD[@]}"
    fi
else
    "${CMD[@]}"
fi
