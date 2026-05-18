# HypDAE Geometry Comparison Project Guide

This document describes the intended experiment, the project decisions that should remain stable, and the work that needs to be completed before full training and evaluation.

## Goal

Compare whether Hyperbolic, Euclidean, and Spherical Stage 2 encoders represent dataset structure differently when the dataset has strong inherent hierarchy versus flatter heterogeneous classes.

The core experimental control is:

> Stage 1 is fixed per dataset; geometry varies only in Stage 2.

This keeps the diffusion decoder/generator constant within a dataset, so observed differences are attributable primarily to the latent geometry used by Stage 2.

## Main Experimental Principle

Each dataset should have one dataset-specific Stage 1 model. That Stage 1 model is then reused for all Stage 2 geometry variants.

For each dataset:

1. Train or select one fixed Stage 1 checkpoint.
2. Train one Stage 2 encoder for each geometry:
   - Hyperbolic
   - Euclidean
   - Spherical
3. Run inference/evaluation with the matching Stage 1 checkpoint and Stage 2 checkpoint.
4. Compare geometry effects within the same dataset.

This avoids mixing dataset effects, Stage 1 effects, and geometry effects.

## Dataset Choice

Use two datasets with different expected structure:

| Role | Dataset | Root | Rationale |
| --- | --- | --- | --- |
| Hierarchical / fine-grained | CUB-200 inside the NABirds directory | `/scratch-shared/scur0164/datasets/NABirds/CUB_200_2011/images` | Fine-grained bird species classes have strong taxonomic and visual hierarchy. |
| Flatter / heterogeneous | Caltech-256 | `/scratch-shared/scur0164/datasets/Caltech_256` | Broad object categories are more independent and less dominated by a single biological hierarchy. |

Do not use Animal Faces as the main flat/uniform contrast dataset. It likely contains meaningful latent hierarchy, for example animal type, species, breed, and visual family clusters.

Important naming note: the current bird dataset path is CUB-200-style data stored under the `NABirds` scratch directory. Treat it as `cub200_encode` in new experiment configs unless full NABirds is explicitly added later.

The existing `v2_inference_nabirds.yaml` config should not be used directly for the main CUB-200 experiment. Its checkpoints correspond to the full NABirds setup, while the current bird data is CUB-200.

## Experiment Matrix

Each dataset gets one fixed Stage 1 model and three Stage 2 geometry variants.

| Dataset | Stage | Geometry | Dataset type | Checkpoint/output directory | Purpose |
| --- | --- | --- | --- | --- | --- |
| CUB-200 | Stage 1 | N/A | `cub200_encode` | `/scratch-shared/scur0164/checkpoints/HypDAE/cub200/stage1` | Train/fix the dataset-specific diffusion backbone. |
| CUB-200 | Stage 2 | Hyperbolic | `cub200_encode` | `/scratch-shared/scur0164/checkpoints/HypDAE/cub200/stage2_hyperbolic` | Hierarchy-aligned latent baseline. |
| CUB-200 | Stage 2 | Euclidean | `cub200_encode` | `/scratch-shared/scur0164/checkpoints/HypDAE/cub200/stage2_euclidean` | Flat latent comparison. |
| CUB-200 | Stage 2 | Spherical | `cub200_encode` | `/scratch-shared/scur0164/checkpoints/HypDAE/cub200/stage2_spherical` | Angular/unit-norm latent comparison. |
| Caltech-256 | Stage 1 | N/A | `caltech256_encode` | `/scratch-shared/scur0164/checkpoints/HypDAE/caltech256/stage1` | Train/fix the dataset-specific diffusion backbone. |
| Caltech-256 | Stage 2 | Hyperbolic | `caltech256_encode` | `/scratch-shared/scur0164/checkpoints/HypDAE/caltech256/stage2_hyperbolic` | Test hyperbolic geometry on flatter classes. |
| Caltech-256 | Stage 2 | Euclidean | `caltech256_encode` | `/scratch-shared/scur0164/checkpoints/HypDAE/caltech256/stage2_euclidean` | Flat latent baseline. |
| Caltech-256 | Stage 2 | Spherical | `caltech256_encode` | `/scratch-shared/scur0164/checkpoints/HypDAE/caltech256/stage2_spherical` | Angular/unit-norm latent comparison. |

## Dataset Preparation

Before full training, both datasets need an explicit and reproducible preprocessing plan.

Required checks:

- Confirm that CUB-200 matches the expected class-folder image layout.
- Inspect the Caltech-256 extracted folder layout.
- Create deterministic train/validation/test splits for both datasets.
- Store split files or materialized split folders under scratch-shared.
- Use the same split strategy for all geometry variants within a dataset.

Suggested dataset identifiers:

- `cub200_encode`
- `caltech256_encode`

## Baseline Smoke Test

The initial HypDAE smoke test succeeded on a tiny CUB/NABirds-style subset.

This was inference only, not training.

Working paths:

| Item | Path |
| --- | --- |
| Smoke input subset | `/scratch-shared/scur0164/datasets/nabirds_smoke/images` |
| Smoke outputs | `/scratch-shared/scur0164/hypdae_outputs/nabirds_smoke` |
| NABirds/CUB checkpoint directory | `/scratch-shared/scur0164/checkpoints/HypDAE/nabirds` |
| Local CLIP checkpoint | `/scratch-shared/scur0164/checkpoints/clip-vit-large-patch14` |
| Stage 1 NABirds checkpoint | `/scratch-shared/scur0164/checkpoints/HypDAE/nabirds/stage-1-nabirds-epoch=000083.ckpt` |
| Stage 2 NABirds hyperbolic checkpoint | `/scratch-shared/scur0164/checkpoints/HypDAE/nabirds/stage-2-nabirds-iteration_162500.pt` |

The smoke run verified:

- Snellius GPU allocation works through Slurm.
- `hypdiff_env` can run CUDA with `torch==1.12.1+cu113`.
- HypDiffusion loads the pretrained Stage 1 checkpoint.
- The local CLIP checkpoint loads.
- The Stage 2 hyperbolic encoder checkpoint loads.
- Images are read from the tiny dataset subset and generated outputs are saved.

Use smoke tests as the first validation step for new configs, dataset entries, and geometry implementations.

## Inference Config Requirements

HypDAE inference requires a config file for the specific experiment being run.

For pretrained or trained models, the important fields are:

| Config field | Meaning |
| --- | --- |
| `checkpoint_path` | Stage 2 encoder checkpoint |
| `stage1_checkpoint_path` | Stage 1 checkpoint |
| `encoder_version` | CLIP/local encoder checkpoint path |
| `dataset_type` | Dataset identifier used by the data loader |
| `hyperbolic` or geometry-specific option | Selects the intended latent geometry behavior |

For the final experiment scope, expect:

- 2 Stage 1 checkpoint paths: one for CUB-200 and one for Caltech-256.
- 6 Stage 2 checkpoint paths: 2 datasets x 3 geometries.

## Implementation Tasks

### Dataset And Path Configuration

Add or verify dataset definitions:

- `cub200_encode`
- `caltech256_encode`

Add or verify path entries:

- `cub200`
- `caltech256`
- optional split-specific roots if train/validation/test folders are materialized.

### Config Templates

Create config templates for:

- CUB-200 Stage 1 training.
- CUB-200 Stage 2 training for Hyperbolic, Euclidean, and Spherical.
- CUB-200 inference for each geometry.
- Caltech-256 Stage 1 training.
- Caltech-256 Stage 2 training for Hyperbolic, Euclidean, and Spherical.
- Caltech-256 inference for each geometry.

### Euclidean Stage 2

Inspect existing Euclidean support before long training:

- Check whether `hyperbolic: false` fully bypasses hyperbolic layers/losses.
- Verify argparse/config boolean handling does not accidentally treat `"False"` as true.
- Run a tiny Euclidean Stage 2 smoke-training job before committing to full training.

### Spherical Stage 2

Add minimal Spherical Stage 2 support:

- Normalize latent features to unit norm.
- Add a spherical classifier or angular-distance objective.
- Decode the spherical feature back to the same 1024-dimensional conditioning feature used by HypDiffusion.
- Keep the Stage 1 interface unchanged.

### Snellius Scripts

Prepare Snellius scripts only after config and split decisions are stable:

- Interactive smoke command.
- Batch Stage 1 training command.
- Batch Stage 2 training commands.
- Batch inference/evaluation commands.

## Evaluation Protocol

Use the same evaluation protocol for all geometry variants within each dataset.

Primary checks:

- Reconstruction or conditioning fidelity on held-out images.
- Generated image quality and diversity.
- Class consistency against the input class.
- Latent geometry diagnostics:
  - nearest-neighbor class purity
  - hierarchy-aware distance correlation for CUB-200 if taxonomy metadata is available
  - radial distribution for Hyperbolic
  - norm stability for Euclidean
  - angular separation for Spherical

For smoke tests, use only small subsets and image output inspection. For real results, run the full held-out split.

## Project Checklist

- [ ] Confirm final dataset pair: CUB-200 and Caltech-256.
- [ ] Inspect Caltech-256 folder layout.
- [ ] Verify CUB-200 folder layout.
- [ ] Decide exact train/validation/test split strategy.
- [ ] Create reproducible split files or split folders.
- [ ] Create or verify dataset entries for `cub200_encode` and `caltech256_encode`.
- [ ] Confirm Stage 1 is fixed per dataset and Stage 2 is the only geometry-varying component.
- [ ] Create Stage 1 config templates for both datasets.
- [ ] Create Stage 2 config templates for all geometry variants.
- [ ] Inspect Euclidean mode and list required fixes.
- [ ] Implement or verify Euclidean Stage 2 behavior.
- [ ] Design minimal Spherical Stage 2 behavior.
- [ ] Implement Spherical Stage 2 support.
- [ ] Run smoke tests for new dataset/config combinations.
- [ ] Prepare Snellius batch scripts after smoke tests succeed.
- [ ] Run full training.
- [ ] Run inference/evaluation on held-out splits.
- [ ] Summarize results using the same metrics across all geometry variants.

## Open Risks

- Caltech-256 folder layout may need preprocessing before it matches the expected class-folder image layout.
- The current CUB-200 path is not full NABirds; results should be described as CUB-200 unless full NABirds is added.
- Spherical Stage 2 is not currently implemented and needs careful loss/classifier design.
- Stage 1 training is expensive; run tiny smoke jobs before launching full training.
- Environment stability matters: install packages carefully when preserving `torch==1.12.1+cu113`.
- Hardcoded Snellius paths should be kept private or parameterized before any public release.
