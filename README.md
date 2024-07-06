---
title: Open Remove Background Model (ormbg)
license: apache-2.0
tags:
  - segmentation
  - remove background
  - background
  - background-removal
  - Pytorch
pretty_name: Open Remove Background Model
models:
  - schirrmacher/ormbg
datasets:
  - schirrmacher/humans
emoji: 💻
colorFrom: red
colorTo: red
sdk: gradio
sdk_version: 4.29.0
app_file: hf_space/app.py
pinned: false
---

# Open Remove Background Model (ormbg)

[>>> DEMO <<<](https://huggingface.co/spaces/schirrmacher/ormbg)

Join our [Research Discord Group](https://discord.gg/YYZ3D66t)!

![](examples/image/image01_no_background.png)

This model is a **fully open-source background remover** optimized for images with humans. It is based on [Highly Accurate Dichotomous Image Segmentation research](https://github.com/xuebinqin/DIS). The model was trained with the synthetic [Human Segmentation Dataset](https://huggingface.co/datasets/schirrmacher/humans), [P3M-10k](https://paperswithcode.com/dataset/p3m-10k), [PPM-100](https://github.com/ZHKKKe/PPM) and [AIM-500](https://paperswithcode.com/dataset/aim-500).

This model is similar to [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4), but with open training data/process and commercially free to use.

## Inference

```
python ormbg/inference.py
```

## Training

Install dependencies:

```
conda env create -f environment.yaml
conda activate ormbg
```

Replace dummy dataset with [training dataset](https://huggingface.co/datasets/schirrmacher/humans).

```
python3 ormbg/train_model.py
```

# Research

I started training the model with synthetic images of the [Human Segmentation Dataset](https://huggingface.co/datasets/schirrmacher/humans) crafted with [LayerDiffuse](https://github.com/layerdiffusion/LayerDiffuse). However, I noticed that the model struggles to perform well on real images.

Synthetic datasets have limitations for achieving great segmentation results. This is because artificial lighting, occlusion, scale or backgrounds create a gap between synthetic and real images. A "model trained solely on synthetic data generated with naïve domain randomization struggles to generalize on the real domain", see [PEOPLESANSPEOPLE: A Synthetic Data Generator for Human-Centric Computer Vision (2022)](https://arxiv.org/pdf/2112.09290).

Latest changes (05/07/2024):

- Added [P3M-10K](https://paperswithcode.com/dataset/p3m-10k) dataset for training and validation
- Added [AIM-500](https://paperswithcode.com/dataset/aim-500) dataset for training and validation
- Added [PPM-100](https://github.com/ZHKKKe/PPM) dataset for training and validation
- Applied [Grid Dropout](https://albumentations.ai/docs/api_reference/augmentations/dropout/grid_dropout/) to make the model smarter

Next steps:

- Expand dataset with synthetic and real images
- Research on multi-step segmentation/matting by incorporating [ViTMatte](https://github.com/hustvl/ViTMatte)
