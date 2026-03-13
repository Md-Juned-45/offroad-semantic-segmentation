# Duality AI Offroad Semantic Scene Segmentation Report

## 1. Title and Summary

**Team Name:** Team Chaos  
**Project Name:** Offroad Autonomy Semantic Segmentation  
**Track:** Duality AI Offroad Semantic Scene Segmentation Challenge

This project tackles pixel-wise semantic segmentation for off-road autonomy using synthetic desert scenes generated with Duality AI Falcon. The goal is to help an autonomous ground vehicle understand terrain structure and obstacles by assigning a semantic label to every pixel in an image.

Our team trained and iteratively improved multiple segmentation models in a **Kaggle Notebook GPU environment**, then exported the best checkpoint for local inference and demo usage. The final solution achieved a **best validation mIoU of 0.5283** on the provided validation split and includes:

- training code
- batch inference code
- exported model weights
- a Flask-based interactive demo UI
- documentation for reproducibility and presentation

## 2. Challenge Context

Duality AI's challenge focuses on training robust semantic segmentation models using synthetic desert environments and then evaluating how well those models generalize to unseen but related desert scenes. This setting is highly relevant for off-road autonomous navigation, where vehicles must distinguish terrain, vegetation, obstacles, and clutter under changing visual conditions.

Semantic segmentation is especially important for off-road autonomy because path planning decisions depend on fine-grained scene understanding rather than simple object detection alone.

## 3. Objective

The core objectives of our work were:

- train a robust semantic segmentation model using only the provided synthetic dataset
- improve generalization to unseen desert environments
- optimize the training setup through architecture, augmentation, and loss design
- document the workflow, experiments, metrics, and failure cases clearly for judges

## 4. Dataset Overview

The dataset contains RGB images and paired segmentation masks for desert scenes. The data is organized into `train`, `val`, and `testImages` splits. We strictly kept the test images separate from training and validation throughout the project.

### Semantic Classes

| Raw ID | Class Name |
|---|---|
| 0 | Background |
| 100 | Trees |
| 200 | Lush Bushes |
| 300 | Dry Grass |
| 500 | Dry Bushes |
| 550 | Ground Clutter |
| 600 | Flowers |
| 700 | Logs |
| 800 | Rocks |
| 7100 | Landscape |
| 10000 | Sky |

### Dataset Compliance

To remain compliant with the challenge rules:

- we trained only on the provided training split
- we used the provided validation split for model selection
- we did not use the designated test images for training
- the final checkpoint was selected based on validation performance only

## 5. Training Environment

The main training workflow was executed in **Kaggle Notebook** using GPU acceleration. Kaggle was chosen because it provided a practical environment for experimentation, model downloads, iterative retraining, and checkpoint export during the hackathon.

Our local repository contains the code used for the final approach, the exported checkpoint, and local scripts for testing and the demo UI.

## 6. Methodology

## 6.1 Overall Workflow

```mermaid
flowchart LR
    A["Synthetic RGB + Segmentation Masks"] --> B["Mask ID Remapping"]
    B --> C["Train / Validation Split"]
    C --> D["Augmentation + Normalization"]
    D --> E["Model Training in Kaggle"]
    E --> F["Validation mIoU Tracking"]
    F --> G["Export best_model.pth from Kaggle"]
    G --> H["Download Locally for Batch Inference & Flask Demo"]
```

## 6.2 Experiment Progression

We did not jump directly to the final model. Instead, we iteratively improved the pipeline based on validation performance and observed class-level weaknesses.

| Version | Model | Key Characteristics | Best Val IoU / mIoU |
|---|---|---|---|
| V1 | DINOv2 ViT-S/14 + custom ConvNeXt-style segmentation head | Kaggle-optimized prototype, 10 classes, no flower class mapping | 0.3991 |
| V2 | DeepLabV3+ with EfficientNet-B3 encoder | Combined CE + Dice loss, stronger augmentation, better dense prediction baseline | 0.4672 |
| V3 | DeepLabV3+ with MiT-B2 encoder | Added missing flower class, Albumentations pipeline, class weights, warm restarts, mixed precision | **0.5283** |

### Why the Earlier Versions Were Important

**V1: DINOv2 Prototype**
- helped us establish the first realistic benchmark quickly in Kaggle
- showed that the dataset was learnable
- revealed that the segmentation head alone was not enough to reach competitive performance
- also exposed an important issue: the original class mapping omitted **Flowers (ID 600)**, reducing class coverage to 10 classes

**V2: DeepLabV3+ EfficientNet-B3**
- significantly improved spatial segmentation quality over the DINOv2 prototype
- raised validation IoU from `0.3991` to `0.4672`
- confirmed that a dedicated segmentation architecture worked better than the first custom setup

**V3: DeepLabV3+ MiT-B2 Final Model**
- restored the missing flower class and expanded the task back to the required 11 classes
- switched to stronger `albumentations` transforms
- used better class weighting for rare classes and stronger optimization
- produced the best final score: `0.5283` mIoU

## 6.3 Final Model Configuration

The final and best-performing configuration used:

- **Architecture:** DeepLabV3+
- **Encoder:** MiT-B2
- **Framework:** `segmentation_models_pytorch`
- **Input resolution:** `512 x 512`
- **Batch size:** `8`
- **Epochs:** `60`
- **Optimizer:** AdamW
- **Learning rates:** lower LR for encoder, higher LR for decoder and head
- **Scheduler:** CosineAnnealingWarmRestarts
- **Precision:** mixed precision with `torch.amp`

## 6.4 Preprocessing and Augmentation

For the final model, we used `albumentations` because it gave us better control over paired image-mask transforms and richer augmentation options than our earlier torchvision-only setup.

Final training augmentations included:

- resize to `512 x 512`
- horizontal flip
- vertical flip
- random 90-degree rotation
- affine transforms with scale, translation, and rotation
- grid distortion
- elastic transform
- color jitter
- gaussian blur
- grayscale conversion
- ImageNet normalization

Validation used only resize and normalization to keep evaluation stable.

## 6.5 Mask Mapping Fix

One of the most important corrections we made was restoring the missing **Flowers** class.

Early experiments used a `value_map` with only 10 classes:
- `0, 100, 200, 300, 500, 550, 700, 800, 7100, 10000`

In the final pipeline, we corrected the mapping to include:
- `600 -> Flowers`

This ensured that the final model aligned with the official challenge class list and learned all required categories.

## 6.6 Loss Function and Optimization Strategy

The final training objective combined:

- **weighted cross-entropy loss**
- **Dice loss**

This combination worked well because:
- cross-entropy stabilized class prediction learning
- Dice loss directly rewarded better mask overlap
- class weights improved learning for smaller and rarer categories such as flowers, logs, rocks, and ground clutter

We also used:
- gradient clipping for training stability
- differential learning rates for pretrained encoder vs decoder/head
- cosine warm restarts to encourage better convergence over longer training

## 7. Results and Performance Metrics

## 7.1 Final Best Score

The final model achieved:

- **Best validation mIoU:** `0.5283`
- **Training duration:** `60 epochs`
- **Best checkpoint:** exported from Kaggle as `best_model.pth`

## 7.2 Training Progress Summary

The final run showed steady improvement over time:

| Epoch | Validation mIoU | Avg Train Loss |
|---|---:|---:|
| 1 | 0.2976 | 1.8731 |
| 10 | 0.4804 | 1.1045 |
| 20 | 0.4981 | 1.0641 |
| 30 | 0.5139 | 1.0275 |
| 40 | 0.5169 | 1.0147 |
| 50 | 0.5124 | 1.0150 |
| 57 | **0.5283** | 0.9945 |
| 60 | 0.5242 | 0.9908 |

This trend shows that the model converged steadily and benefited from the final augmentation and optimization setup.

## 7.3 Final Per-Class Validation IoU at Best Checkpoint

The best recorded run reached the following approximate class-level IoU values:

| Class | IoU |
|---|---:|
| Background | 0.0000 |
| Trees | 0.6979 |
| Lush Bushes | 0.6015 |
| Dry Grass | 0.6558 |
| Dry Bushes | 0.4770 |
| Ground Clutter | 0.3621 |
| Flowers | 0.6201 |
| Logs | 0.3211 |
| Rocks | 0.4418 |
| Landscape | 0.6514 |
| Sky | 0.9827 |

## 7.4 Interpretation of Results

Key observations from the final results:

- **Sky** and broad structural classes were segmented very reliably
- **Trees**, **Dry Grass**, **Landscape**, and **Lush Bushes** achieved strong IoU
- **Flowers** improved substantially after fixing the class mapping and using stronger weighting
- **Logs**, **Ground Clutter**, and **Rocks** remained harder because they are smaller, thinner, and visually ambiguous
- **Background** remained near zero, suggesting either very low presence in validation or strong confusion with nearby terrain classes such as landscape

## 8. Challenges and Solutions

## 8.1 Missing Class in Early Pipeline

**Problem:** Our early training pipeline did not include class `600` for **Flowers**, so the model was effectively solving a reduced 10-class problem.

**Fix:** We corrected the mask remapping logic and updated the model to train on the full 11-class challenge specification.

**Impact:** This improved task correctness and gave the model the opportunity to learn flower regions explicitly.

## 8.2 Low Performance in the First Architecture

**Problem:** The first DINOv2-based prototype plateaued at `0.3991` IoU and did not capture fine segmentation details strongly enough.

**Fix:** We moved to DeepLabV3+, which is designed directly for dense pixel prediction.

**Impact:** The switch increased validation IoU to `0.4672`, proving the value of a stronger segmentation-specific architecture.

## 8.3 Generalization to Unseen Scenes

**Problem:** Synthetic datasets can encourage overfitting to color, lighting, or scene style.

**Fix:** We introduced richer augmentations, including affine transforms, distortion, blur, grayscale, and color jitter using `albumentations`.

**Impact:** The final model generalized more robustly and reached `0.5283` mIoU.

## 8.4 Rare and Thin Classes

**Problem:** Logs, rocks, and ground clutter occupy small or irregular regions, making them hard to segment reliably.

**Fix:** We increased class weighting and used Dice loss to improve overlap for small structures.

**Impact:** These classes improved, though they are still the main weakness of the system.

## 9. Failure Case Analysis

Even with the final model, several failure modes remain:

## 9.1 Background vs Landscape Confusion

The model often predicts **Landscape** for broad ground regions, while **Background** remains nearly unused. This suggests that the class boundary between generic ground and background is either weak in the data or visually difficult to separate.

## 9.2 Logs and Small Obstacles

**Logs** remain one of the hardest classes. They are thin, often partially occluded, and visually similar to nearby clutter or dry bushes.

## 9.3 Rocks vs Ground Clutter

Rocks and clutter can share texture and color patterns, especially in shadowed regions. This leads to partial confusion and lower IoU for both classes.

## 9.4 Small Flower Regions

Although the flower class improved significantly after the class-mapping fix, tiny flower patches still disappear in complex scenes or become absorbed into surrounding vegetation.

## 10. Optimizations Used

The following optimizations had the strongest impact on final performance:

- correcting the class mapping to include all 11 challenge classes
- switching from the early DINOv2 prototype to DeepLabV3+
- replacing simple torchvision transforms with `albumentations`
- using combined weighted cross-entropy and Dice loss
- applying stronger class weights for underrepresented categories
- using AdamW with differential learning rates
- using cosine warm restarts for longer training
- training with mixed precision on Kaggle GPU

## 11. Deliverables Included in the Project

Our submission package contains or references the following components:

- `train.py`: The exact training code ran on Kaggle for our final model
- `test.py`: Local batch inference script for unseen images
- `app.py`: Flask application for interactive local demo testing
- `best_model.pth`: Exported model weights (Hosted on Kaggle, to be downloaded and placed locally)
- `README.md`: Run instructions
- this markdown report for methodology, results, and analysis

## 12. Conclusion

We built a complete end-to-end semantic segmentation solution for off-road desert environments using synthetic data from Duality AI Falcon. The project evolved through three main model stages, with each iteration addressing a concrete weakness from the previous one.

The final system combined:
- DeepLabV3+
- MiT-B2 encoder
- full 11-class mask mapping
- rich Albumentations augmentation
- weighted CE + Dice loss
- mixed precision Kaggle training

This final setup achieved a **best validation mIoU of 0.5283**, outperforming our earlier DINOv2 and EfficientNet-based baselines and producing a practical model for demo inference and qualitative review.

## 13. Future Work

The most promising next improvements are:

- deeper class rebalancing or focal-style loss for logs and clutter
- stronger boundary-aware objectives for thin structures
- test-time augmentation for more stable predictions
- ensembling multiple backbones for higher IoU
- domain adaptation from synthetic scenes to real-world off-road imagery
- structured latency benchmarking on deployment hardware

## 14. Short Judge Pitch

We started with an early Kaggle prototype, then systematically improved the model through architecture changes, data augmentation, class remapping, and loss redesign. Our final DeepLabV3+ MiT-B2 model trained on the provided synthetic dataset reached **0.5283 validation mIoU**, supports all 11 required classes, and is packaged with both batch inference and an interactive demo for qualitative testing.
