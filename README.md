# Geometry-Aware Semantic Latent Spaces for Diffusion-Based Image Generation

<div align="center">

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-GhostK030-yellow)](https://huggingface.co/k-sert)

**[Kagan Sert](https://github.com/k-sert)** · **[Elyanne Oey](https://github.com/ElyanneOey)** · **[Daniël Amidirad](https://github.com/damidirad)** · **[Arthur van Campenhout](https://github.com/Arthurvc-uva)** . **[Casper Schuttrups](https://github.com/CasperWS)**

*University of Amsterdam - Deep Learning 2, 2026*

</div>

---

<div align="center">
  <img src="assets/Hierarchy.png" width="90%">
</div>

> **Abstract:** This project studies how the geometry of semantic latent spaces affects diffusion-based image generation. Starting from a two-stage HypDAE-style pipeline, we compare Euclidean, hyperbolic, and spherical Stage 2 representations for class-conditional image generation on hierarchical visual datasets. The goal is to understand whether geometry-aware latent spaces can provide better semantic structure, controllable diversity, and more meaningful perturbations than a purely Euclidean representation. In particular, the HypDiffusion generation code combines a Stage 1 diffusion checkpoint with a Stage 2 semantic encoder checkpoint to generate images while varying the geometry of the learned latent space.

<div align="center">
  <img src="assets/Poincare.png" width="45%"> <img src="assets/Pipeline.png" width="45%">
</div>

## 🛠️ Setup

Our codebase is built on [HypDAE](https://github.com/lingxiao-li/HypDAE). We adapt the original pipeline to compare Euclidean, hyperbolic, and spherical semantic latent spaces for CUB-200-2011 and Caltech-256.

### Environment

```bash
# Clone the repository
git clone https://github.com/k-sert/HypDAE-private.git
cd HypDAE-private
```

Create the environment that matches the part of the project you want to run:

```bash
# Stage 1 training
conda env create -f Stage1/environment.yaml
conda activate stage1_env

# Stage 2 training
conda env create -f Stage2/environment.yaml
conda activate stage2_env

# HypDiffusion generation
conda env create -f HypDiffusion/hypdae_env.yaml
conda activate hypdae
```

### Model Weights

Pre-trained Stage 1 and Stage 2 checkpoints will be available from [Hugging Face](https://huggingface.co/k-sert). Place downloaded checkpoints under a local `checkpoints/` directory, or update the paths in the relevant YAML files under `HypDiffusion/configs/stable-diffusion/`.

## 📂 Data Preparation

We use the following datasets for training and evaluation. Please download and unzip them into the `data/` folder.

| Dataset | Link | Description |
| :--- | :--- | :--- |
| **CUB-200-2011** | [Download](https://dl.allaboutbirds.org/nabirds) | Fine-grained bird image dataset |
| **Caltech-256** | [Download](https://data.caltech.edu/records/nyy15-4j048) | Object recognition dataset |

**Recommended Directory Structure:**

```text
HypDAE-private/
├── HypDiffusion/
├── Stage1/
├── Stage2/
├── data/
│   ├── cub/
│   └── caltech/
└── ...
```

## 🚀 Inference

Image generation is handled through `HypDiffusion/inference_for_eva.py`. Choose the YAML that matches the dataset and geometry you want to evaluate, then point the script to your Stage 1 diffusion checkpoint and image folder.

```bash
cd HypDiffusion

python inference_for_eva.py \
  --config_path configs/stable-diffusion/v2_inference_[dataset]_[geometry]_[curvature].yaml \
  --ckpt /path/to/stage1_diffusion.ckpt \
  --image_folder /path/to/data/cub/test \
  --outdir /path/to/output_folder
```

Available CUB/Caltech configs include Euclidean, hyperbolic, and spherical variants under `HypDiffusion/configs/stable-diffusion/`.

## 🏋️‍♂️ Training

The project is organized as a two-stage pipeline.

**Stage 1** trains the diffusion autoencoder backbone for the target dataset:

```bash
bash scripts/train_stage1_cub200.sh
bash scripts/train_stage1_caltech256.sh
```

**Stage 2** trains the semantic latent-space component on top of a Stage 1 checkpoint. Separate launch scripts are provided for Euclidean, hyperbolic, and spherical variants:

```bash
bash scripts/train_stage2_cub200_euc.sh
bash scripts/train_stage2_cub200_hyp_c1.sh
bash scripts/train_stage2_cub200_hyp_c5.sh
bash scripts/train_stage2_cub200_hyp_c10.sh
bash scripts/train_stage2_cub200_sph_c1.sh
bash scripts/train_stage2_cub200_sph_c5.sh
bash scripts/train_stage2_cub200_sph_c10.sh

bash scripts/train_stage2_caltech256_euc.sh
bash scripts/train_stage2_caltech256_hyp_c1.sh
bash scripts/train_stage2_caltech256_hyp_c5.sh
bash scripts/train_stage2_caltech256_hyp_c10.sh
bash scripts/train_stage2_caltech256_sph_c1.sh
bash scripts/train_stage2_caltech256_sph_c5.sh
bash scripts/train_stage2_caltech256_sph_c10.sh
```

Before running the scripts, adjust dataset paths, checkpoint paths, and output directories to match your machine or cluster environment.

## 🙏 Acknowledgments

This project is based on [HypDAE](https://github.com/lingxiao-li/HypDAE) and [Hyperbolic-Flow-Matching](https://github.com/federicavaleau/Hyperbolic-Flow-Matching). We thank the authors for their open-source contributions.
