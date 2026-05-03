# HypDAE Geometry Comparison Experiment Design

Date planned: Monday, May 4, 2026

## Goal

Compare whether Hyperbolic, Euclidean, and Spherical Stage2 encoders represent dataset structure differently when the dataset has strong inherent hierarchy versus flatter heterogeneous classes.

The core experimental control is:

> Stage1 is fixed per dataset; geometry varies only in Stage2.

This keeps the diffusion decoder/generator constant within a dataset, so observed differences are attributable primarily to the latent geometry used by Stage2.

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
| Stage1 NABirds checkpoint | `/scratch-shared/scur0164/checkpoints/HypDAE/nabirds/stage-1-nabirds-epoch=000083.ckpt` |
| Stage2 NABirds hyperbolic checkpoint | `/scratch-shared/scur0164/checkpoints/HypDAE/nabirds/stage-2-nabirds-iteration_162500.pt` |

The smoke run verified:

- Snellius GPU allocation works through Slurm.
- `hypdiff_env` can run CUDA with `torch==1.12.1+cu113`.
- HypDiffusion loads the pretrained Stage1 checkpoint.
- The local CLIP checkpoint loads.
- The Stage2 hyperbolic encoder checkpoint loads.
- Images are read from the tiny dataset subset and generated outputs are saved.

## Dataset Decision

Use two datasets:

| Role | Dataset | Root | Rationale |
| --- | --- | --- | --- |
| Hierarchical / fine-grained | CUB-200 inside the NABirds directory | `/scratch-shared/scur0164/datasets/NABirds/CUB_200_2011/images` | Fine-grained bird species classes have strong taxonomic and visual hierarchy. |
| Flatter / heterogeneous | Caltech-256 | `/scratch-shared/scur0164/datasets/Caltech_256` | Broad object categories are more independent and less dominated by a single biological hierarchy. |

Do not use Animal Faces as the main flat/uniform contrast dataset. It likely contains meaningful latent hierarchy, for example animal type, species, breed, and visual family clusters.

Important naming note: the current bird dataset path is CUB-200-style data stored under the `NABirds` scratch directory. Treat it as `cub200_encode` in new experiment configs unless full NABirds is explicitly added later.

## Experiment Matrix

Each dataset gets one fixed Stage1 model and three Stage2 geometry variants.

| Dataset | Stage | Geometry | Dataset type | Checkpoint/output directory | Purpose |
| --- | --- | --- | --- | --- | --- |
| CUB-200 | Stage1 | N/A | `cub200_encode` | `/scratch-shared/scur0164/checkpoints/HypDAE/cub200/stage1` | Train/fix the dataset-specific diffusion backbone. |
| CUB-200 | Stage2 | Hyperbolic | `cub200_encode` | `/scratch-shared/scur0164/checkpoints/HypDAE/cub200/stage2_hyperbolic` | Hierarchy-aligned latent baseline. |
| CUB-200 | Stage2 | Euclidean | `cub200_encode` | `/scratch-shared/scur0164/checkpoints/HypDAE/cub200/stage2_euclidean` | Flat latent comparison. |
| CUB-200 | Stage2 | Spherical | `cub200_encode` | `/scratch-shared/scur0164/checkpoints/HypDAE/cub200/stage2_spherical` | Angular/unit-norm latent comparison. |
| Caltech-256 | Stage1 | N/A | `caltech256_encode` | `/scratch-shared/scur0164/checkpoints/HypDAE/caltech256/stage1` | Train/fix the dataset-specific diffusion backbone. |
| Caltech-256 | Stage2 | Hyperbolic | `caltech256_encode` | `/scratch-shared/scur0164/checkpoints/HypDAE/caltech256/stage2_hyperbolic` | Test hyperbolic geometry on flatter classes. |
| Caltech-256 | Stage2 | Euclidean | `caltech256_encode` | `/scratch-shared/scur0164/checkpoints/HypDAE/caltech256/stage2_euclidean` | Flat latent baseline. |
| Caltech-256 | Stage2 | Spherical | `caltech256_encode` | `/scratch-shared/scur0164/checkpoints/HypDAE/caltech256/stage2_spherical` | Angular/unit-norm latent comparison. |

Splits still need to be made explicit before training:

- CUB-200: use a deterministic train/val/test split from the class-folder image tree.
- Caltech-256: inspect the extracted folder layout, then create a deterministic train/val/test split.
- Keep split files or generated split folders under scratch so training and evaluation are reproducible.

## Evaluation Metrics

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

## Implementation Tasks

1. Add dataset definitions:
   - `cub200_encode`
   - `caltech256_encode`

2. Add path entries:
   - `cub200`
   - `caltech256`
   - optional split-specific roots if train/val/test folders are materialized.

3. Create config templates:
   - CUB-200 Stage1 training config.
   - CUB-200 Stage2 training configs for Hyperbolic, Euclidean, Spherical.
   - CUB-200 inference configs for each geometry.
   - Caltech-256 equivalents.

4. Inspect existing Euclidean support:
   - Check whether `hyperbolic: false` fully bypasses hyperbolic layers/losses.
   - Verify argparse/config bool handling does not accidentally treat `"False"` as true.
   - Run a tiny Euclidean Stage2 smoke training job before committing to long training.

5. Add minimal Spherical Stage2 support:
   - Normalize latent features to unit norm.
   - Add spherical classifier or angular-distance objective.
   - Decode the spherical feature back to the same 1024-dimensional conditioning feature used by HypDiffusion.
   - Keep the Stage1 interface unchanged.

6. Prepare Snellius scripts only after the above config and split decisions are stable:
   - Interactive smoke command.
   - Batch Stage1 training command.
   - Batch Stage2 training commands.
   - Batch inference/evaluation commands.

## Open Risks

- Caltech-256 folder layout may need preprocessing before it matches the expected class-folder image layout.
- The current CUB-200 path is not full NABirds; results should be described as CUB-200 unless full NABirds is added.
- Spherical Stage2 is not currently implemented and needs careful loss/classifier design.
- Stage1 training is expensive; first run tiny smoke jobs before launching full training.
- Environment stability matters: install packages with `--no-deps` when preserving `torch==1.12.1+cu113`.

## Tomorrow Acceptance Checklist

- [ ] Confirm final dataset pair: CUB-200 and Caltech-256.
- [ ] Inspect Caltech-256 folder layout.
- [ ] Decide exact train/val/test split strategy.
- [ ] Create or plan dataset entries for `cub200_encode` and `caltech256_encode`.
- [ ] Confirm Stage1 is fixed per dataset and Stage2 is the only geometry-varying component.
- [ ] Inspect Euclidean mode and list required fixes.
- [ ] Sketch the minimal Spherical Stage2 implementation.
- [ ] Prepare the first small config/code change list for the next work session.
