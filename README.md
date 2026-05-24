# GeoDiff: Geometry-Conditioned Latent Spaces for Diffusion-Based Generation

<div align="center">

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-yellow)](https://huggingface.co/k-sert/geodiff-checkpoints/tree/main)

**[Kagan Sert](https://github.com/k-sert)** · **[Elyanne Oey](https://github.com/ElyanneOey)** · **[Daniël Amidirad](https://github.com/damidirad)** · **[Arthur van Campenhout](https://github.com/Arthurvc-uva)** · **[Casper Schuttrups](https://github.com/CasperWS)**

*University of Amsterdam - Deep Learning 2, 2026*

</div>

> **Abstract:** This project investigates how latent-space geometry influences diffusion-based image generation. Building upon a unified two-stage HypDAE-style framework, we systematically compare Euclidean, hyperbolic, and spherical conditioned latent spaces while keeping the underlying diffusion backbone fixed. We evaluate whether geometry-aware representations improve latent organization, controllable diversity, and latent perturbation behavior compared to conventional Euclidean embeddings, particularly on datasets with differing degrees of hierarchical structure. The HypDiffusion pipeline combines a Stage 1 diffusion backbone with geometry-specific Stage 2 encoders, enabling controlled image generation under varying latent curvature assumptions.

## 🛠️ Setup

Our codebase is built on [HypDAE](https://github.com/lingxiao-li/HypDAE). We adapt the original pipeline to compare Euclidean, hyperbolic, and spherical conditioned latent spaces for CUB-200-2011 and Caltech-256.

### Environment

```bash
# Clone the repository
git clone https://github.com/k-sert/GeoDiff.git
cd GeoDiff
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
conda env create -f HypDiffusion/environment.yaml
conda activate hypdiff_env
```

### Model Weights

Pre-trained Stage 1 and Stage 2 checkpoints will be available from [Hugging Face](https://huggingface.co/k-sert). Place downloaded checkpoints under a local `checkpoints/` directory, or update the paths in the relevant YAML files under `HypDiffusion/configs/stable-diffusion/`.

## 📂 Data Preparation

We use the following datasets for training and evaluation. Please download and unzip them into a `data/` folder.

| Dataset | Link | Description |
| :--- | :--- | :--- |
| **CUB-200-2011** | [Download](https://www.vision.caltech.edu/datasets/cub_200_2011) | Fine-grained bird image dataset |
| **Caltech-256** | [Download](https://data.caltech.edu/records/nyy15-4j048) | Object recognition dataset |

**Recommended Directory Structure:**

```text
GeoDiff/
├── HypDiffusion/
├── Stage1/
├── Stage2/
├── scripts/
├── checkpoints/
├── datasets/
│   ├── cub200/
│   └── caltech256/
└── ...
```

## 🚀 Inference

Image generation is handled through `HypDiffusion/inference_for_eva.py`. Choose the YAML that matches the dataset and geometry you want to evaluate, then point the script to your Stage 1 diffusion checkpoint and image folder.

```bash
cd HypDiffusion

python inference_for_eva.py \
  --config_path configs/stable-diffusion/v2_inference_[dataset]_[geometry]_[curvature].yaml \
  --ckpt /path/to/stage1_diffusion.ckpt \
  --image_folder /path/to/data_folder \
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

**Stage 2** trains the latent-space conditioned component on top of a Stage 1 checkpoint. Separate launch scripts are provided for Euclidean, hyperbolic, and spherical variants:

```bash
bash scripts/train_stage2_cub200_euc.sh
bash scripts/train_stage2_cub200_hyp_c1.sh
bash scripts/train_stage2_cub200_sph_c1.sh

bash scripts/train_stage2_caltech256_euc.sh
bash scripts/train_stage2_caltech256_hyp_c1.sh
bash scripts/train_stage2_caltech256_sph_c1.sh
```

Before running the scripts, adjust dataset paths, checkpoint paths, and output directories to match your machine or cluster environment.

## 🙏 Acknowledgments

This project is based on [HypDAE](https://github.com/lingxiao-li/HypDAE) and [Hyperbolic-Flow-Matching](https://github.com/federicavaleau/Hyperbolic-Flow-Matching). We thank the authors for their open-source contributions.

## 📜 License

GeoDiff builds on Stable Diffusion 2.1 and follows the CreativeML Open RAIL++-M license terms for the diffusion backbone and derived checkpoints. The Stage 2 encoder uses OpenAI CLIP ViT-L/14 features; the OpenAI CLIP repository is released under the MIT License. Users should also comply with the dataset terms for CUB-200-2011 and Caltech-256.

Please also refer to:
- Stable Diffusion 2.1: https://huggingface.co/stabilityai/stable-diffusion-2-1-base
- OpenAI CLIP: https://github.com/openai/CLIP
