#!/bin/bash

set -euo pipefail

# Grid-search HypDiffusion sampling settings on a validation subset.
#
# This is intentionally separate from run_hypdiffusion_geometry_eval.sh:
# - use this script to choose a good global (strength, CFG scale) setting
# - then use run_hypdiffusion_geometry_eval.sh for the final fixed-setting run
#
# Usage:
#   bash scripts/grid_search_hypdiffusion_settings.sh dry-run
#   bash scripts/grid_search_hypdiffusion_settings.sh submit
#
# Defaults are deliberately small: 5 classes x 2 images/class on val.
# Useful overrides:
#   STRENGTH_GRID="0.20 0.50 0.75 0.85 0.95" SCALE_GRID="1.0 1.3" \
#     bash scripts/grid_search_hypdiffusion_settings.sh submit

MODE=${1:-dry-run}
if [ -n "${HYPDAE_REPO_ROOT:-}" ]; then
    REPO_ROOT="${HYPDAE_REPO_ROOT}"
else
    REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
SCRIPT_PATH="${REPO_ROOT}/scripts/grid_search_hypdiffusion_settings.sh"

SPLIT=${SPLIT:-val}
STRENGTH_GRID=${STRENGTH_GRID:-"0.20 0.50 0.75 0.85 0.95"}
SCALE_GRID=${SCALE_GRID:-"1.0 1.3"}
DDIM_STEPS=${DDIM_STEPS:-50}
DDIM_ETA=${DDIM_ETA:-0.0}
SEED=${SEED:-23}
N_SAMPLES=${N_SAMPLES:-1}
N_PERTURB_SAMPLES=${N_PERTURB_SAMPLES:-0}
N_SELECTED=${N_SELECTED:-2}
N_CLASSES=${N_CLASSES:-5}
WITH_METRICS=${WITH_METRICS:-1}
LPIPS_MODE=${LPIPS_MODE:-both}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}

OUTPUT_ROOT=${OUTPUT_ROOT:-${OUTPUT_ROOT}/HypDiffusion/setting_grid_${SPLIT}_${RUN_TAG}}
METRICS_ROOT=${METRICS_ROOT:-${OUTPUT_ROOT}/HypDiffusion/setting_grid_metrics_${SPLIT}_${RUN_TAG}}
CODECARBON_ROOT=${CODECARBON_ROOT:-${OUTPUT_ROOT}/codecarbon/HypDiffusion/setting_grid_${RUN_TAG}}

PARTITION=${PARTITION:-gpu_h100}
GEN_TIME=${GEN_TIME:-00:30:00}
METRICS_TIME=${METRICS_TIME:-01:00:00}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MAX_PARALLEL=${MAX_PARALLEL:-4}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs}

DATASETS=(cub200 cub200 cub200 caltech256 caltech256 caltech256)
GEOMETRIES=(euc hyp_c1 sph_c1 euc hyp_c1 sph_c1)
read -r -a STRENGTHS <<< "${STRENGTH_GRID}"
read -r -a SCALES <<< "${SCALE_GRID}"

NUM_MODELS=${#DATASETS[@]}
NUM_STRENGTHS=${#STRENGTHS[@]}
NUM_SCALES=${#SCALES[@]}
NUM_SETTINGS=$((NUM_STRENGTHS * NUM_SCALES))
NUM_RUNS=$((NUM_MODELS * NUM_SETTINGS))
ARRAY_MAX=$((NUM_RUNS - 1))

config_for() {
    echo "configs/stable-diffusion/v2_inference_${1}_${2}.yaml"
}

stage1_ckpt_for() {
    case "$1" in
        cub200)
            echo "${CKPT_ROOT}/Stage1/cub200/best_model.ckpt"
            ;;
        caltech256)
            echo "${CKPT_ROOT}/Stage1/caltech256/best_model.ckpt"
            ;;
        *)
            echo "Unknown dataset: $1" >&2
            exit 2
            ;;
    esac
}

dataset_dir_for() {
    echo "${DATA_ROOT}/${1}_split/${SPLIT}"
}

decode_task() {
    local task_id=$1
    MODEL_INDEX=$((task_id / NUM_SETTINGS))
    local setting_index=$((task_id % NUM_SETTINGS))
    local strength_index=$((setting_index / NUM_SCALES))
    local scale_index=$((setting_index % NUM_SCALES))

    DATASET=${DATASETS[$MODEL_INDEX]}
    GEOMETRY=${GEOMETRIES[$MODEL_INDEX]}
    STRENGTH=${STRENGTHS[$strength_index]}
    SCALE=${SCALES[$scale_index]}
    RUN_NAME="${DATASET}__${GEOMETRY}__s${STRENGTH}__cfg${SCALE}"
}

require_array_task() {
    if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
        echo "This mode must run as a SLURM array task." >&2
        exit 2
    fi
    if [ "${SLURM_ARRAY_TASK_ID}" -lt 0 ] || [ "${SLURM_ARRAY_TASK_ID}" -gt "${ARRAY_MAX}" ]; then
        echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}; expected 0-${ARRAY_MAX}" >&2
        exit 2
    fi
}

setup_modules() {
    
    
    
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate hypdiff_env
    export LD_LIBRARY_PATH=$(conda info --base)/envs/hypdiff_env/lib:${LD_LIBRARY_PATH:-}
}

print_plan() {
    echo "HypDiffusion setting grid-search plan"
    echo "  split=${SPLIT}"
    echo "  strengths=${STRENGTH_GRID}"
    echo "  scales=${SCALE_GRID}"
    echo "  models=${NUM_MODELS}, settings/model=${NUM_SETTINGS}, total_runs=${NUM_RUNS}"
    echo "  ddim_steps=${DDIM_STEPS}, ddim_eta=${DDIM_ETA}, seed=${SEED}"
    echo "  n_selected=${N_SELECTED}, n_classes=${N_CLASSES}, n_samples=${N_SAMPLES}, n_perturb_samples=${N_PERTURB_SAMPLES}"
    echo "  output_root=${OUTPUT_ROOT}"
    echo "  metrics_root=${METRICS_ROOT}"
    echo "  with_metrics=${WITH_METRICS}, lpips_mode=${LPIPS_MODE}"
    echo
    for ((task=0; task<NUM_RUNS; task++)); do
        decode_task "${task}"
        echo "  [${task}] ${RUN_NAME}"
        echo "      config=HypDiffusion/$(config_for "${DATASET}" "${GEOMETRY}")"
        echo "      generated=${OUTPUT_ROOT}/${RUN_NAME}"
        echo "      metrics=${METRICS_ROOT}/${RUN_NAME}.json"
    done
}

submit_jobs() {
    mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${METRICS_ROOT}" "${CODECARBON_ROOT}"
    print_plan

    local export_vars
    export_vars="ALL,HYPDAE_REPO_ROOT=${REPO_ROOT},SPLIT=${SPLIT},STRENGTH_GRID=${STRENGTH_GRID},SCALE_GRID=${SCALE_GRID},DDIM_STEPS=${DDIM_STEPS},DDIM_ETA=${DDIM_ETA},SEED=${SEED},N_SAMPLES=${N_SAMPLES},N_PERTURB_SAMPLES=${N_PERTURB_SAMPLES},N_SELECTED=${N_SELECTED},N_CLASSES=${N_CLASSES},WITH_METRICS=${WITH_METRICS},LPIPS_MODE=${LPIPS_MODE},RUN_TAG=${RUN_TAG},OUTPUT_ROOT=${OUTPUT_ROOT},METRICS_ROOT=${METRICS_ROOT},CODECARBON_ROOT=${CODECARBON_ROOT},PARTITION=${PARTITION},GEN_TIME=${GEN_TIME},METRICS_TIME=${METRICS_TIME},CPUS_PER_TASK=${CPUS_PER_TASK},MAX_PARALLEL=${MAX_PARALLEL},LOG_DIR=${LOG_DIR}"

    local gen_job
    gen_job=$(sbatch \
        --parsable \
        --partition="${PARTITION}" \
        --gpus=1 \
        --ntasks=1 \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --time="${GEN_TIME}" \
        --array="0-${ARRAY_MAX}%${MAX_PARALLEL}" \
        --job-name=hd_grid_gen \
        --output="${LOG_DIR}/hypdiffusion_setting_grid_gen_%A_%a.out" \
        --export="${export_vars}" \
        "${SCRIPT_PATH}" generate)
    echo "Submitted generation grid: ${gen_job}"

    if [ "${WITH_METRICS}" = "1" ]; then
        local metrics_job
        metrics_job=$(sbatch \
            --parsable \
            --partition="${PARTITION}" \
            --gpus=1 \
            --ntasks=1 \
            --cpus-per-task="${CPUS_PER_TASK}" \
            --time="${METRICS_TIME}" \
            --array="0-${ARRAY_MAX}%${MAX_PARALLEL}" \
            --dependency="afterok:${gen_job}" \
            --job-name=hd_grid_metrics \
            --output="${LOG_DIR}/hypdiffusion_setting_grid_metrics_%A_%a.out" \
            --export="${export_vars}" \
            "${SCRIPT_PATH}" metrics)
        echo "Submitted metrics grid: ${metrics_job} (afterok:${gen_job})"

        local aggregate_job
        aggregate_job=$(sbatch \
            --parsable \
            --partition="${PARTITION}" \
            --gpus=1 \
            --ntasks=1 \
            --cpus-per-task=1 \
            --time=00:10:00 \
            --dependency="afterok:${metrics_job}" \
            --job-name=hd_grid_aggregate \
            --output="${LOG_DIR}/hypdiffusion_setting_grid_aggregate_%A.out" \
            --export="${export_vars}" \
            "${SCRIPT_PATH}" aggregate)
        echo "Submitted aggregate job: ${aggregate_job} (afterok:${metrics_job})"
    fi
}

run_generation() {
    require_array_task
    decode_task "${SLURM_ARRAY_TASK_ID}"
    setup_modules

    local config_path
    local model_ckpt
    local image_dir
    local outdir
    config_path=$(config_for "${DATASET}" "${GEOMETRY}")
    model_ckpt=$(stage1_ckpt_for "${DATASET}")
    image_dir=$(dataset_dir_for "${DATASET}")
    outdir="${OUTPUT_ROOT}/${RUN_NAME}"

    mkdir -p "${outdir}" "${CODECARBON_ROOT}/${RUN_NAME}"
    cd "${REPO_ROOT}/HypDiffusion"

    echo "Running grid task ${SLURM_ARRAY_TASK_ID}/${ARRAY_MAX}: ${RUN_NAME}"
    echo "  image_dir=${image_dir}"
    echo "  config=${config_path}"
    echo "  ckpt=${model_ckpt}"
    echo "  strength=${STRENGTH}, scale=${SCALE}"
    echo "  outdir=${outdir}"

    python inference_for_eva.py \
        --image_folder "${image_dir}" \
        --outdir "${outdir}" \
        --config_path "${config_path}" \
        --ckpt "${model_ckpt}" \
        --ddim_steps "${DDIM_STEPS}" \
        --ddim_eta "${DDIM_ETA}" \
        --strength "${STRENGTH}" \
        --scale "${SCALE}" \
        --seed "${SEED}" \
        --n_samples "${N_SAMPLES}" \
        --n_perturb_samples "${N_PERTURB_SAMPLES}" \
        --n_selected "${N_SELECTED}" \
        --n_classes "${N_CLASSES}" \
        --use_codecarbon \
        --codecarbon_project_name "grid_hypdiffusion_${RUN_NAME}" \
        --codecarbon_output_dir "${CODECARBON_ROOT}/${RUN_NAME}"
}

run_metrics() {
    require_array_task
    decode_task "${SLURM_ARRAY_TASK_ID}"
    setup_modules

    local generated_dir="${OUTPUT_ROOT}/${RUN_NAME}"
    local reference_dir
    local out_json="${METRICS_ROOT}/${RUN_NAME}.json"
    reference_dir=$(dataset_dir_for "${DATASET}")

    mkdir -p "${METRICS_ROOT}"
    cd "${REPO_ROOT}"

    echo "Evaluating grid task ${SLURM_ARRAY_TASK_ID}/${ARRAY_MAX}: ${RUN_NAME}"
    echo "  generated=${generated_dir}"
    echo "  reference=${reference_dir}"
    echo "  out=${out_json}"

    LPIPS_MODE="${LPIPS_MODE}" \
    RUN_CODECARBON=1 \
    CODECARBON_PROJECT_NAME="metrics_grid_hypdiffusion_${RUN_NAME}" \
    CODECARBON_OUTPUT_DIR="${CODECARBON_ROOT}/metrics_${RUN_NAME}" \
        bash scripts/metrics/evaluate_image_metrics.sh \
            "${generated_dir}" \
            "${reference_dir}" \
            "${out_json}"
}

aggregate_metrics() {
    cd "${REPO_ROOT}"
    mkdir -p "${METRICS_ROOT}"

    python - <<'PY'
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

metrics_root = Path(os.environ["METRICS_ROOT"])
per_run_csv = metrics_root / "per_run_summary.csv"
setting_csv = metrics_root / "setting_summary.csv"
setting_md = metrics_root / "setting_summary.md"

def get_nested(data, *keys):
    cur = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur

def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

rows = []
for path in sorted(metrics_root.glob("*.json")):
    parts = path.stem.split("__")
    if len(parts) != 4:
        continue
    dataset, geometry, strength_token, scale_token = parts
    strength = strength_token.removeprefix("s")
    scale = scale_token.removeprefix("cfg")
    data = json.loads(path.read_text())
    fidelity = data.get("torch_fidelity", {})
    lpips_data = data.get("lpips")
    rows.append({
        "dataset": dataset,
        "geometry": geometry,
        "strength": strength,
        "scale": scale,
        "setting": f"s{strength}_cfg{scale}",
        "num_generated": data.get("num_generated"),
        "num_reference": data.get("num_reference"),
        "fid": as_float(fidelity.get("frechet_inception_distance")),
        "kid_mean": as_float(fidelity.get("kernel_inception_distance_mean")),
        "kid_std": as_float(fidelity.get("kernel_inception_distance_std")),
        "precision": as_float(fidelity.get("precision")),
        "recall": as_float(fidelity.get("recall")),
        "lpips_paired": as_float(get_nested(lpips_data, "paired", "mean")),
        "lpips_diversity": as_float(get_nested(lpips_data, "diversity", "mean")),
        "json": str(path),
    })

per_run_fields = [
    "dataset", "geometry", "strength", "scale", "setting",
    "num_generated", "num_reference", "fid", "kid_mean", "kid_std",
    "precision", "recall", "lpips_paired", "lpips_diversity", "json",
]
with per_run_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=per_run_fields)
    writer.writeheader()
    writer.writerows(rows)

def mean(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    return sum(values) / len(values) if values else None

by_setting = defaultdict(list)
for row in rows:
    by_setting[row["setting"]].append(row)

setting_rows = []
for setting, group in sorted(by_setting.items()):
    strength = group[0]["strength"]
    scale = group[0]["scale"]
    setting_rows.append({
        "setting": setting,
        "strength": strength,
        "scale": scale,
        "num_runs": len(group),
        "fid_mean": mean([r["fid"] for r in group]),
        "kid_mean": mean([r["kid_mean"] for r in group]),
        "precision_mean": mean([r["precision"] for r in group]),
        "recall_mean": mean([r["recall"] for r in group]),
        "lpips_paired_mean": mean([r["lpips_paired"] for r in group]),
        "lpips_diversity_mean": mean([r["lpips_diversity"] for r in group]),
    })

def add_rank(rows, key, rank_key, reverse=False):
    values = [(i, row[key]) for i, row in enumerate(rows) if row.get(key) is not None]
    values.sort(key=lambda item: item[1], reverse=reverse)
    for rank, (idx, _) in enumerate(values, start=1):
        rows[idx][rank_key] = rank

add_rank(setting_rows, "fid_mean", "rank_fid", reverse=False)
add_rank(setting_rows, "lpips_paired_mean", "rank_lpips_paired", reverse=False)
add_rank(setting_rows, "lpips_diversity_mean", "rank_lpips_diversity", reverse=True)
add_rank(setting_rows, "precision_mean", "rank_precision", reverse=True)
add_rank(setting_rows, "recall_mean", "rank_recall", reverse=True)

for row in setting_rows:
    ranks = [
        row.get("rank_fid"),
        row.get("rank_lpips_paired"),
        row.get("rank_lpips_diversity"),
        row.get("rank_precision"),
        row.get("rank_recall"),
    ]
    ranks = [r for r in ranks if r is not None]
    row["rank_mean"] = mean(ranks)

setting_rows.sort(key=lambda row: (row["rank_mean"] is None, row["rank_mean"] or 1e9))

setting_fields = [
    "setting", "strength", "scale", "num_runs",
    "fid_mean", "kid_mean", "precision_mean", "recall_mean",
    "lpips_paired_mean", "lpips_diversity_mean",
    "rank_fid", "rank_lpips_paired", "rank_lpips_diversity",
    "rank_precision", "rank_recall", "rank_mean",
]
with setting_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=setting_fields)
    writer.writeheader()
    writer.writerows(setting_rows)

def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

headers = [
    "Setting", "Strength", "CFG", "Runs",
    "FID mean ↓", "LPIPS paired ↓", "LPIPS diversity ↑",
    "Precision ↑", "Recall ↑", "Mean rank ↓",
]
lines = [
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
]
for row in setting_rows:
    lines.append("| " + " | ".join([
        row["setting"],
        row["strength"],
        row["scale"],
        fmt(row["num_runs"]),
        fmt(row["fid_mean"]),
        fmt(row["lpips_paired_mean"]),
        fmt(row["lpips_diversity_mean"]),
        fmt(row["precision_mean"]),
        fmt(row["recall_mean"]),
        fmt(row["rank_mean"]),
    ]) + " |")
setting_md.write_text("\n".join(lines) + "\n")

print(f"Wrote {per_run_csv}")
print(f"Wrote {setting_csv}")
print(f"Wrote {setting_md}")
PY
}

case "${MODE}" in
    dry-run)
        print_plan
        ;;
    submit)
        submit_jobs
        ;;
    generate)
        run_generation
        ;;
    metrics)
        run_metrics
        ;;
    aggregate)
        aggregate_metrics
        ;;
    *)
        echo "Usage: $0 {dry-run|submit|generate|metrics|aggregate}" >&2
        exit 2
        ;;
esac
