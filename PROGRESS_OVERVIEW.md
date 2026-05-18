# Recent Progress Overview

Date: Monday, May 4, 2026

This document summarizes the practical setup work completed over the past days, the current experimental direction, and the main items colleagues should keep in mind before launching training or inference on Snellius.

## 1. Snellius Data And Environment Setup

The project datasets were copied to the Snellius HPC node under scratch-shared storage. The relevant conda environments for Stage 1, Stage 2, and HypDAE were also prepared there.

This setup took about 4 hours, mostly because of installation and dependency resolution waiting time.

Important environment fixes:

- The original environment files were not directly usable because of local editable install entries such as `-e .`. Those local editable entries were removed from the current YAML files.
- Stage 2 dependency resolution from the authors' original environment was not compatible with conda and caused an infinite solving loop. The current Stage 2 environment is documented in `Stage2/environment/psp_env.yaml`.
- HypDAE inference required the `geoopt` library, which was missing initially and has now been installed.

Fixed environment files in the repository:

| Component | Fixed environment file |
| --- | --- |
| Stage 1 | `Stage1/environment.yaml` |
| Stage 2 | `Stage2/environment/psp_env.yaml` |
| HypDAE / HypDiffusion | `HypDiffusion/tf_icon_env_no_editable.yaml` |

When working on Snellius, use the datasets and conda environments from the HPC node/scratch-shared location, not from the home directory. At the start of each SSH session, load Miniconda and activate the correct environment:

```bash
module load 2025
module load Miniconda3/25.5.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stage1_env   # or hypdiff_env or stage2_env
```

## 2. Repository Privacy

The public GitHub repository was deleted and replaced with a private copy.

Reason: during active development on Snellius, it is too easy to accidentally commit or expose private Snellius paths and directory names. The repository should stay private until the project is finished and the code has been cleaned so it is safe for public release.

Before making the repository public again, we should remove or parameterize private paths, check configs and scripts for hardcoded scratch directories, and prepare public-safe documentation.

## 3. HypDAE Smoke Test

A smoke-test script was created to verify whether HypDAE inference works and to estimate what is needed for future inference runs.

The smoke test produced a mini-dataset on the HPC node with 2 classes and 5 images per class. This can also be reused as a small test dataset for other parts of the research, for example Stage 1 or Stage 2 training smoke tests.

Current smoke-test paths documented in `EXPERIMENT_DESIGN.md`:

| Item | Path |
| --- | --- |
| Smoke input subset | `/scratch-shared/scur0164/datasets/nabirds_smoke/images` |
| Smoke outputs | `/scratch-shared/scur0164/hypdae_outputs/nabirds_smoke` |
| NABirds/CUB checkpoint directory | `/scratch-shared/scur0164/checkpoints/HypDAE/nabirds` |
| Local CLIP checkpoint | `/scratch-shared/scur0164/checkpoints/clip-vit-large-patch14` |

The smoke test verified that:

- Slurm GPU allocation works on Snellius.
- `hypdiff_env` can run CUDA with the required PyTorch/CUDA combination.
- HypDiffusion can load the pretrained Stage 1 checkpoint.
- The local CLIP checkpoint loads correctly.
- The Stage 2 hyperbolic encoder checkpoint loads correctly.
- Images are read from the small dataset and generated outputs are saved.

## 4. Inference Config Requirements

HypDAE inference requires an experiment-specific config file. For pretrained models, the config must point to the correct checkpoints and encoder version:

| Config field | Meaning |
| --- | --- |
| `checkpoint_path` | Stage 2 encoder checkpoint |
| `stage1_checkpoint_path` | Stage 1 checkpoint |
| `encoder_version` | CLIP/local encoder checkpoint path |

The smoke inference config is available at:

```text
HypDiffusion/configs/stable-diffusion/v2_inference_nabirds_smoke.yaml
```

For our intended experiment scope, we should expect:

- 2 Stage 1 checkpoint paths: one for CUB-200 and one for Caltech-256.
- 6 Stage 2 checkpoint paths: 2 datasets x 3 geometries.

Stage 1 is fixed per dataset and shared across geometries. Stage 2 is the part that changes between Hyperbolic, Euclidean, and Spherical geometry.

## 5. Dataset And Config Decisions

The existing `v2_inference_nabirds.yaml` config should not be used directly for our main bird experiment. Its checkpoints were trained for the full NABirds setup, while our current bird dataset is CUB-200 stored under the NABirds scratch directory. Results should therefore be described as CUB-200 unless full NABirds is explicitly added later.

Animal Faces should not be used as the flat/uniform comparison dataset. It likely contains more inherent hierarchy than desired, for example animal type, species, breed, and visual family structure.

The current dataset plan is:

| Role | Dataset | Reason |
| --- | --- | --- |
| Hierarchical / fine-grained | CUB-200 | Bird species have taxonomic and visual hierarchy. |
| Flatter / heterogeneous | Caltech-256 | Broad object classes are more independent and less dominated by one biological hierarchy. |

## 6. Experiment Design Direction

The main scientific control is described in `EXPERIMENT_DESIGN.md`:

> Stage 1 is fixed per dataset; geometry varies only in Stage 2.

The main implementation work is therefore to adapt Stage 2 for the additional geometries:

- Hyperbolic: existing/pretrained baseline.
- Euclidean: expected to be simpler; inspect whether `hyperbolic: false` already bypasses all hyperbolic-specific logic correctly.
- Spherical: requires extra implementation, likely including unit-norm latent features and an angular or spherical objective.

Once the required Stage 2 logic exists, the datasets can be trained under the three geometry settings.

## 7. Preprocessing Still Needed

Before launching full training, dataset preprocessing must be checked.

- Caltech-256 has not yet been preprocessed.
- CUB-200 preprocessing status is uncertain and should be verified.
- Deterministic train/validation/test splits should be created or documented before training.
- Split files or materialized split folders should live under scratch-shared so runs are reproducible on Snellius.

## 8. Recommended Next Steps

1. Confirm final dataset pair: CUB-200 and Caltech-256.
2. Inspect the Caltech-256 extracted folder layout and decide the preprocessing strategy.
3. Verify whether CUB-200 already matches the expected class-folder image layout.
4. Create deterministic train/validation/test splits for both datasets.
5. Add or verify dataset entries for `cub200_encode` and `caltech256_encode`.
6. Inspect Stage 2 Euclidean mode and run a tiny smoke-training job.
7. Design and implement minimal Spherical Stage 2 support.
8. Create dataset-specific Stage 1 and Stage 2 config templates.
9. Only after the above is stable, prepare full Snellius batch scripts for training and inference.

## 9. Practical Remarks

- Always run small smoke jobs before full training. Stage 1 training is expensive, so config/path mistakes should be caught early.
- Keep all Snellius-specific paths private until public cleanup.
- Prefer scratch-shared for large datasets, checkpoints, and generated outputs.
- Keep environment stability in mind when installing additional packages. For HypDAE, preserving the working PyTorch/CUDA combination is more important than letting conda or pip freely upgrade dependencies.
- Document every successful smoke command and Slurm command once it works, so the next person can reproduce the result without rediscovering the setup.
