#!/bin/bash

set -euo pipefail

# Submit fixed-setting HypDiffusion generation and metric jobs for:
#   cub200/caltech256 x euclidean/hyperbolic_c1/spherical_c1
#
# Usage:
#   bash scripts/run_hypdiffusion_geometry_eval.sh dry-run
#   bash scripts/run_hypdiffusion_geometry_eval.sh submit
#
# Useful overrides:
#   SPLIT=test N_SELECTED=5 N_CLASSES=0 WITH_METRICS=1 \
#     bash scripts/run_hypdiffusion_geometry_eval.sh submit
#
# N_CLASSES=0 means "all classes" for HypDiffusion/inference_for_eva.py.

MODE=${1:-dry-run}
REPO_ROOT="${HYPDAE_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SCRIPT_PATH="${REPO_ROOT}/scripts/run_hypdiffusion_geometry_eval.sh"

SPLIT=${SPLIT:-test}
STRENGTH=${STRENGTH:-0.35}
SCALE=${SCALE:-1.0}
DDIM_STEPS=${DDIM_STEPS:-50}
DDIM_ETA=${DDIM_ETA:-0.0}
SEED=${SEED:-23}
N_SAMPLES=${N_SAMPLES:-1}
N_PERTURB_SAMPLES=${N_PERTURB_SAMPLES:-4}
N_SELECTED=${N_SELECTED:-4}
N_CLASSES=${N_CLASSES:-0}
WITH_METRICS=${WITH_METRICS:-1}
LPIPS_MODE=${LPIPS_MODE:-both}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
HYP_DISTANCE=${HYP_DISTANCE:-5.6}
SPH_DISTANCE=${SPH_DISTANCE:-0.5}
EUC_DISTANCE=${EUC_DISTANCE:-0.2}

OUTPUT_ROOT=${OUTPUT_ROOT:-${OUTPUTS_ROOT:-outputs}/HypDiffusion/geometry_eval_${SPLIT}_s${STRENGTH}_cfg${SCALE}_${RUN_TAG}}
METRICS_ROOT=${METRICS_ROOT:-${OUTPUTS_ROOT:-outputs}/HypDiffusion/geometry_eval_metrics_${SPLIT}_s${STRENGTH}_cfg${SCALE}_${RUN_TAG}}
CODECARBON_ROOT=${CODECARBON_ROOT:-${OUTPUT_ROOT}/codecarbon/HypDiffusion/geometry_eval_${RUN_TAG}}

PARTITION=${PARTITION:-gpu_h100}
GEN_TIME=${GEN_TIME:-06:00:00}
METRICS_TIME=${METRICS_TIME:-02:00:00}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs}

DATASETS=(cub200 cub200 cub200 caltech256 caltech256 caltech256)
GEOMETRIES=(euc hyp_c1 sph_c1 euc hyp_c1 sph_c1)

dataset_label() {
    case "$1" in
        cub200) echo "CUB200" ;;
        caltech256) echo "Caltech256" ;;
        *) echo "$1" ;;
    esac
}

geometry_label() {
    case "$1" in
        euc) echo "Euclidean" ;;
        hyp_c1) echo "Hyperbolic c=1" ;;
        sph_c1) echo "Spherical c=1" ;;
        *) echo "$1" ;;
    esac
}

config_for() {
    local dataset=$1
    local geometry=$2
    echo "configs/stable-diffusion/v2_inference_${dataset}_${geometry}.yaml"
}

stage1_ckpt_for() {
    local dataset=$1
    case "${dataset}" in
        cub200)
            echo "${CKPT_ROOT}/Stage1/cub200/best_model.ckpt"
            ;;
        caltech256)
            echo "${CKPT_ROOT}/Stage1/caltech256/best_model.ckpt"
            ;;
        *)
            echo "Unknown dataset: ${dataset}" >&2
            exit 2
            ;;
    esac
}

dataset_dir_for() {
    local dataset=$1
    echo "${DATA_ROOT}/${dataset}_split/${SPLIT}"
}

run_name_for() {
    local dataset=$1
    local geometry=$2
    echo "${dataset}_${geometry}_${SPLIT}_s${STRENGTH}_cfg${SCALE}_shots${N_SELECTED}_seed${SEED}"
}

require_array_task() {
    if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
        echo "This mode must run as a SLURM array task." >&2
        exit 2
    fi
    if [ "${SLURM_ARRAY_TASK_ID}" -lt 0 ] || [ "${SLURM_ARRAY_TASK_ID}" -ge "${#DATASETS[@]}" ]; then
        echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
    fi
}

setup_modules() {
    
    
    
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate hypdiff_env
    export LD_LIBRARY_PATH=$(conda info --base)/envs/hypdiff_env/lib:${LD_LIBRARY_PATH:-}
}

print_plan() {
    echo "HypDiffusion geometry evaluation plan"
    echo "  split=${SPLIT}"
    echo "  strength=${STRENGTH}, scale=${SCALE}, ddim_steps=${DDIM_STEPS}, ddim_eta=${DDIM_ETA}"
    echo "  seed=${SEED}, n_selected=${N_SELECTED}, n_classes=${N_CLASSES}, n_samples=${N_SAMPLES}, n_perturb_samples=${N_PERTURB_SAMPLES}"
    echo "  output_root=${OUTPUT_ROOT}"
    echo "  metrics_root=${METRICS_ROOT}"
    echo "  with_metrics=${WITH_METRICS}, lpips_mode=${LPIPS_MODE}"
    echo
    for idx in "${!DATASETS[@]}"; do
        local dataset=${DATASETS[$idx]}
        local geometry=${GEOMETRIES[$idx]}
        local run_name
        run_name=$(run_name_for "${dataset}" "${geometry}")
        echo "  [${idx}] ${run_name}"
        echo "      config=HypDiffusion/$(config_for "${dataset}" "${geometry}")"
        echo "      reference=$(dataset_dir_for "${dataset}")"
        echo "      generated=${OUTPUT_ROOT}/${run_name}"
        echo "      metrics=${METRICS_ROOT}/${run_name}.json"
    done
}

submit_jobs() {
    mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" "${METRICS_ROOT}" "${CODECARBON_ROOT}"
    print_plan

    local export_vars
    export_vars="ALL,SPLIT=${SPLIT},REPO_ROOT=${REPO_ROOT},STRENGTH=${STRENGTH},HYP_DISTANCE=${HYP_DISTANCE},SPH_DISTANCE=${SPH_DISTANCE},EUC_DISTANCE=${EUC_DISTANCE},SCALE=${SCALE},DDIM_STEPS=${DDIM_STEPS},DDIM_ETA=${DDIM_ETA},SEED=${SEED},N_SAMPLES=${N_SAMPLES},N_PERTURB_SAMPLES=${N_PERTURB_SAMPLES},N_SELECTED=${N_SELECTED},N_CLASSES=${N_CLASSES},WITH_METRICS=${WITH_METRICS},LPIPS_MODE=${LPIPS_MODE},RUN_TAG=${RUN_TAG},OUTPUT_ROOT=${OUTPUT_ROOT},METRICS_ROOT=${METRICS_ROOT},CODECARBON_ROOT=${CODECARBON_ROOT},PARTITION=${PARTITION},GEN_TIME=${GEN_TIME},METRICS_TIME=${METRICS_TIME},CPUS_PER_TASK=${CPUS_PER_TASK},LOG_DIR=${LOG_DIR}"

    local gen_job
    gen_job=$(sbatch \
        --parsable \
        --partition="${PARTITION}" \
        --gpus=1 \
        --ntasks=1 \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --time="${GEN_TIME}" \
        --array=0-5 \
        --job-name=hd_geom_gen \
        --output="${LOG_DIR}/hypdiffusion_geometry_eval_gen_%A_%a.out" \
        --export="${export_vars}" \
        "${SCRIPT_PATH}" generate)
    echo "Submitted generation array: ${gen_job}"

    if [ "${WITH_METRICS}" = "1" ]; then
        local metrics_job
        metrics_job=$(sbatch \
            --parsable \
            --partition="${PARTITION}" \
            --gpus=1 \
            --ntasks=1 \
            --cpus-per-task="${CPUS_PER_TASK}" \
            --time="${METRICS_TIME}" \
            --array=0-5 \
            --dependency="afterany:${gen_job}" \
            --job-name=hd_geom_metrics \
            --output="${LOG_DIR}/hypdiffusion_geometry_eval_metrics_%A_%a.out" \
            --export="${export_vars}" \
            "${SCRIPT_PATH}" metrics)
        echo "Submitted metrics array: ${metrics_job} (afterany:${gen_job})"

        local aggregate_job
        aggregate_job=$(sbatch \
            --parsable \
            --partition=rome \
            --ntasks=1 \
            --cpus-per-task=1 \
            --time=00:10:00 \
            --dependency="afterany:${metrics_job}" \
            --job-name=hd_geom_aggregate \
            --output="${LOG_DIR}/hypdiffusion_geometry_eval_aggregate_%A.out" \
            --export="${export_vars}" \
            "${SCRIPT_PATH}" aggregate)
        echo "Submitted aggregate job: ${aggregate_job} (afterany:${metrics_job})"
    fi
}

run_generation() {
    require_array_task
    setup_modules

    local dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
    local geometry=${GEOMETRIES[$SLURM_ARRAY_TASK_ID]}
    local run_name
    run_name=$(run_name_for "${dataset}" "${geometry}")
    local val_dir
    local config_path
    local model_ckpt
    local outdir
    val_dir=$(dataset_dir_for "${dataset}")
    config_path=$(config_for "${dataset}" "${geometry}")
    model_ckpt=$(stage1_ckpt_for "${dataset}")
    outdir="${OUTPUT_ROOT}/${run_name}"

    mkdir -p "${outdir}" "${CODECARBON_ROOT}/${run_name}"
    cd "${REPO_ROOT}/HypDiffusion"

    echo "Running ${run_name}"
    echo "  reference=${val_dir}"
    echo "  config=${config_path}"
    echo "  ckpt=${model_ckpt}"
    echo "  outdir=${outdir}"

    local extra_dist_args=()      
    case "${geometry}" in
    hyp_c1) extra_dist_args+=(--hyp_distance "${HYP_DISTANCE}") ;;
    sph_c1) extra_dist_args+=(--sph_distance "${SPH_DISTANCE}") ;;
    euc)    extra_dist_args+=(--euc_distance "${EUC_DISTANCE}") ;;
    *) echo "Unknown geometry: ${geometry}" >&2; exit 2 ;;
    esac

    python inference_for_eva.py \
        --image_folder "${val_dir}" \
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
        "${extra_dist_args[@]}" \
        --use_codecarbon \
        --codecarbon_project_name "infer_hypdiffusion_${run_name}" \
        --codecarbon_output_dir "${CODECARBON_ROOT}/${run_name}"
}
run_metrics() {
    require_array_task
    setup_modules

    local dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}
    local geometry=${GEOMETRIES[$SLURM_ARRAY_TASK_ID]}
    local run_name
    run_name=$(run_name_for "${dataset}" "${geometry}")

    local generated_dir="${OUTPUT_ROOT}/${run_name}"
    
    # Guard: check generated images exist
    local n_images
    n_images=$(find "${generated_dir}" -name "*.jpg" 2>/dev/null | wc -l)
    if [ "${n_images}" -eq 0 ]; then
        echo "No generated images found in ${generated_dir}, skipping metrics."
        exit 1
    fi
    echo "Found ${n_images} generated images, proceeding with metrics."

    local reference_dir
    local out_json="${METRICS_ROOT}/${run_name}.json"
    reference_dir=$(dataset_dir_for "${dataset}")

    mkdir -p "${METRICS_ROOT}"
    cd "${REPO_ROOT}"

    echo "Evaluating ${run_name}"
    echo "  generated=${generated_dir}"
    echo "  reference=${reference_dir}"
    echo "  out=${out_json}"

    LPIPS_MODE="${LPIPS_MODE}" \
    RUN_CODECARBON=1 \
    CODECARBON_PROJECT_NAME="metrics_hypdiffusion_${run_name}" \
    CODECARBON_OUTPUT_DIR="${CODECARBON_ROOT}/metrics_${run_name}" \
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
import os
from pathlib import Path

metrics_root = Path(os.environ["METRICS_ROOT"])
csv_path = metrics_root / "summary.csv"
md_path = metrics_root / "summary.md"

def get_nested(data, *keys):
    cur = data
    for key in keys:
        if cur is None:
            return None
        cur = cur.get(key) if isinstance(cur, dict) else None
    return cur

rows = []
for path in sorted(metrics_root.glob("*.json")):
    data = json.loads(path.read_text())
    name = path.stem
    parts = name.split("_")
    dataset = "caltech256" if name.startswith("caltech256_") else "cub200"
    if "_hyp_c1_" in name:
        geometry = "hyperbolic_c1"
    elif "_sph_c1_" in name:
        geometry = "spherical_c1"
    else:
        geometry = "euclidean"
    fidelity = data.get("torch_fidelity", {})
    lpips_data = data.get("lpips")
    rows.append({
        "dataset": dataset,
        "geometry": geometry,
        "num_generated": data.get("num_generated"),
        "num_reference": data.get("num_reference"),
        "fid": fidelity.get("frechet_inception_distance"),
        "kid_mean": fidelity.get("kernel_inception_distance_mean"),
        "kid_std": fidelity.get("kernel_inception_distance_std"),
        "precision": fidelity.get("precision_recall_precision"),
        "recall":    fidelity.get("precision_recall_recall"),
        "lpips_paired": get_nested(lpips_data, "paired", "mean"),
        "lpips_diversity": get_nested(lpips_data, "diversity", "mean"),
        "json": str(path),
    })

fieldnames = [
    "dataset", "geometry", "num_generated", "num_reference",
    "fid", "kid_mean", "kid_std", "precision", "recall",
    "lpips_paired", "lpips_diversity", "json",
]
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

headers = ["Dataset", "Geometry", "FID ↓", "LPIPS paired ↓", "LPIPS diversity ↑", "Precision ↑", "Recall ↑"]
lines = [
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
]
for row in rows:
    lines.append("| " + " | ".join([
        row["dataset"],
        row["geometry"],
        fmt(row["fid"]),
        fmt(row["lpips_paired"]),
        fmt(row["lpips_diversity"]),
        fmt(row["precision"]),
        fmt(row["recall"]),
    ]) + " |")
md_path.write_text("\n".join(lines) + "\n")

print(f"Wrote {csv_path}")
print(f"Wrote {md_path}")
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
