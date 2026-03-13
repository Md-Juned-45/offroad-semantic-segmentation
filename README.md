# Offroad Autonomy Semantic Segmentation

Hackathon submission for the Duality AI Offroad Autonomy Segmentation Challenge.

This project builds a semantic segmentation pipeline for off-road autonomous driving scenes using synthetic data from Duality AI. The model predicts a class label for every pixel in an image, helping identify terrain and environmental elements such as trees, bushes, rocks, landscape, sky, logs, and flowers.

## Overview

- Challenge: segment off-road scenes for autonomy perception
- Team: Team Chaos
- Framework: PyTorch
- Model: DeepLabV3+ with MiT-B2 backbone
- Classes: 11 semantic classes
- Training environment: Kaggle Notebook GPU
- Deliverables: training code, inference script, exported weights, and a Flask demo UI

## Problem Statement

Off-road autonomy requires a detailed understanding of terrain and obstacles at the pixel level. Unlike standard road-scene segmentation, this setting includes unstructured environments with vegetation, rocks, clutter, and uneven ground. Our goal was to train a robust model that generalizes well to unseen off-road scenes while remaining efficient enough for practical inference.

## Approach

We used `segmentation_models_pytorch` to build a DeepLabV3+ network with a MiT-B2 encoder. This combination gave us strong multi-scale context capture with a relatively lightweight backbone.

To improve generalization, we applied aggressive augmentations with `albumentations`, including flips, rotations, grid distortion, elastic transforms, color jitter, grayscale conversion, and blur. To address class imbalance, we combined weighted cross-entropy with Dice loss so that rare classes such as logs, flowers, and dry grass were not ignored during training.

## Training Environment

Model training was performed in a **Kaggle Notebook** using GPU acceleration. This repository contains the training code used for the experiment, along with the local scripts for inference and the Flask-based demo UI. The trained checkpoint, `best_model.pth`, was exported from the Kaggle training run for local testing and visualization.

## Results

- Best validation mIoU: `0.5245`
- Inference speed: typically under `50 ms` per image in the target setup
- Output: colorized segmentation masks for visual inspection and a browser-based interactive demo

## Semantic Classes

The model predicts the following classes:

`Background`, `Trees`, `Lush Bushes`, `Dry Grass`, `Dry Bushes`, `Ground Clutter`, `Flowers`, `Logs`, `Rocks`, `Landscape`, `Sky`

## Repository Structure

- `train.py` - training pipeline used for the Kaggle experiment
- `test.py` - batch inference on unseen images
- `app.py` - Flask app for interactive testing
- `requirements.txt` - Python dependencies
- `Hackathon_Report.md` - short technical write-up
- `details.pdf` - supporting report / submission material
- `best_model.pth` - exported trained model weights if included locally

## Setup

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Layout

Expected dataset structure:

```text
data/
  train/
    Color_Images/
    Segmentation/
  val/
    Color_Images/
    Segmentation/
  testImages/
    Color_Images/
    Segmentation/
```

## Training

The training workflow for the hackathon was executed in Kaggle Notebook.

If you want to reproduce the experiment, run:

```bash
python train.py
```

What the training script does:

- loads training and validation data
- applies augmentation and normalization
- trains DeepLabV3+ with weighted CE + Dice loss
- evaluates per-class IoU and mean IoU
- saves the best checkpoint as `best_model.pth`

## Testing

Run:

```bash
python test.py
```

This will:

- load `best_model.pth`
- run inference on images in `data/testImages/Color_Images`
- save predicted color masks to `runs/test_outputs/`

## Demo UI

To launch the interactive web demo:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The UI supports:

- image upload
- split-view comparison
- overlay mask visualization
- class legend display
- quick qualitative testing of predictions

## Key Design Choices

- DeepLabV3+ for strong dense prediction performance
- MiT-B2 encoder for a good accuracy-speed balance
- heavy augmentation to reduce overfitting to synthetic textures
- weighted loss to handle rare and thin classes
- cosine annealing warm restarts and AdamW optimization
- mixed precision training for faster GPU training in Kaggle

## Limitations

- performance can still drop on rare classes and hard shadows
- current training script is primarily tuned for GPU usage
- the included demo is intended for qualitative testing, not production deployment

## Future Improvements

- stronger domain adaptation from synthetic to real-world off-road scenes
- model ensembling for higher final IoU
- self-supervised pretraining for better feature extraction
- test-time augmentation and post-processing for cleaner masks

## Notes For GitHub

If you are publishing this repository, it is usually better not to upload large generated outputs such as `runs/` or raw datasets. Keep the code, documentation, and a few sample predictions instead.
