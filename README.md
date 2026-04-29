# Inside the Arthritic Joint
## Finger Joint Bone Segmentation in Hand X-ray ROI Images

Medical image segmentation project for **MET CS 790: Computer Vision in AI** at **Boston University Metropolitan College**.

## Overview

This project focuses on **multiclass semantic segmentation** of finger-joint bone regions in hand X-ray ROI images.  
The goal is to segment the **upper bone** and **lower bone** around each joint so that the output can support structural assessment and downstream anatomical analysis.

The project compares three deep learning segmentation models:

- **U-Net**
- **Attention U-Net**
- **DeepLabV3**

In addition to standard segmentation overlap metrics such as **Dice** and **IoU**, the project also evaluates anatomical quality using **center Joint Space Width (center-JSW)** error.

---

## Problem Statement

Hand X-ray joint analysis is important for structural assessment, but manual outlining is slow and inconsistent.  
This project aims to automate segmentation of the upper and lower bones in ROI images centered on finger joints.

Each ROI image is paired with a multiclass mask:

- `0` = background
- `1` = upper bone
- `2` = lower bone

The dataset includes:

- **DIP** joints
- **PIP** joints
- **MCP** joints

The project also considers **KL-grade-aware evaluation**.

---

## Project Goals

- Manually label masks for the finger-joint ROI images.
- Build a complete segmentation pipeline for finger-joint ROI images
- Train and compare multiple segmentation architectures
- Generate predicted masks and overlays for qualitative analysis
- Evaluate segmentation performance using Dice and IoU
- Evaluate anatomical quality using center-JSW
- Support KL-wise analysis of segmentation performance

---

## Repository Structure

```text
finger_joint_bone_segmentation/
├── data/
│   ├── raw/
│   │   ├── images/
│   │   └── Hand.csv
│   ├── segmentation_seed/
│   │   ├── images/
│   │   └── masks/
│   └── segmentation_seed_1/
│       ├── images/
│       └── masks/
│
├── notebooks/
│   └── test_dataset_loader.py
│
├── outputs/
│   ├── checkpoints/
│   ├── data_splits/
│   ├── training_logs/
│   ├── segmentation_eval_by_kl/
│   ├── jsw_results/
│   ├── jsw_results_expanded/
│   ├── jsw_results_test/
│   ├── jsw_results_test_center/
│   ├── segmentation_inference_*/
│   └── experiment_results.csv
│
├── reports/
│   └── model_plan.md
│
├── scripts/
│   ├── train_unet.py
│   ├── train_attention_unet.py
│   ├── train_deeplabv3.py
│   ├── run_inference_unet.py
│   ├── run_inference_attention_unet.py
│   ├── run_inference_deeplabv3.py
│   ├── evaluate_segmentation.py
│   ├── evaluate_jsw.py
│   ├── jsw_center_evaluation.py
│   ├── evaluate_classifier.py
│   ├── classical_segmentation_demo.py
│   ├── dataset_summary.py
│   ├── save_experiment_results.py
│   └── ...
│
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   └── utils/
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Dataset

The dataset consists of manually curated ROI image-mask pairs.

### Joint Types
- DIP
- PIP
- MCP

### Mask Format
Each mask uses:
- background
- upper bone
- lower bone

### KL Grades
The project considers:
- KL0
- KL1
- KL2
- KL3
- KL4

### Input / Output
- **Input:** grayscale ROI X-ray image
- **Output:** multiclass segmentation mask

---

## Pipeline

The overall workflow is:

**ROI Image → Manual Multi-class Mask → Model Training → Predicted Mask → Overlay & Evaluation**

### Detailed Steps
1. Collect ROI X-ray images centered on finger joints  
2. Create manual multiclass masks  
3. Organize image-mask pairs into training-ready structure  
4. Perform train / validation / test split  
5. Train segmentation models  
6. Run inference and generate overlays  
7. Evaluate using:
   - Dice
   - IoU
   - center-JSW

---

## Models

### U-Net
Encoder-decoder segmentation model with skip connections; used as the main baseline.

### Attention U-Net
U-Net with attention gates; used to test better focus on joint regions.

### DeepLabV3
Segmentation model with multi-scale context; used to compare against the U-Net-based models.

---

## Training and Inference

### Training Scripts
- `scripts/train_unet.py`
- `scripts/train_attention_unet.py`
- `scripts/train_deeplabv3.py`

Each training script:
- loads the curated dataset
- performs train / validation / test splitting
- trains the model
- evaluates on the validation set
- saves:
  - best checkpoint
  - last checkpoint
  - training history CSV
  - training plot

### Inference Scripts
- `scripts/run_inference_unet.py`
- `scripts/run_inference_attention_unet.py`
- `scripts/run_inference_deeplabv3.py`

Each inference script saves:
- original ROI image
- ground-truth mask
- predicted mask
- ground-truth overlay
- predicted overlay

---

## Evaluation

### Segmentation Metrics
- **Dice Score**
- **Intersection over Union (IoU)**

### Anatomical Metric
- **Center Joint Space Width (center-JSW) error**

### Additional Analysis
- KL-wise segmentation evaluation
- broader JSW comparison across models
- qualitative comparison using overlays

---

## Scripts Summary

### Training
- `train_unet.py`
- `train_attention_unet.py`
- `train_deeplabv3.py`

### Inference
- `run_inference_unet.py`
- `run_inference_attention_unet.py`
- `run_inference_deeplabv3.py`

### Evaluation
- `evaluate_segmentation.py`
- `evaluate_jsw.py`
- `jsw_center_evaluation.py`
- `evaluate_classifier.py`

### Utilities / Analysis
- `dataset_summary.py`
- `save_experiment_results.py`
- `classical_segmentation_demo.py`

---

## Python Version Required

- Python version 3.10.00 or latest

## Example Run Commands

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Train
```bash
python scripts/train_unet.py
python scripts/train_attention_unet.py
python scripts/train_deeplabv3.py
```

### Inference
```bash
python scripts/run_inference_unet.py
python scripts/run_inference_attention_unet.py
python scripts/run_inference_deeplabv3.py
```

### Evaluation
```bash
python scripts/evaluate_segmentation.py
python scripts/evaluate_jsw.py
python scripts/jsw_center_evaluation.py
```

---

## Key Findings

- U-Net and Attention U-Net achieved the strongest overlap-based segmentation performance
- DeepLabV3 showed stronger center-JSW behavior in broader anatomical evaluation
- Overlap quality and anatomical gap preservation were not always identical
- Dataset quality and mask quality strongly affected downstream model behavior
- KL3 / KL4 coverage required additional curation and careful handling

---

## Challenges

- Pseudo-mask quality was unstable
- Manual mask curation was time-consuming
- KL-wise data balance was limited
- Dataset versioning and split consistency needed careful tracking
- Anatomical evaluation required more than overlap metrics alone

---

## Future Work

- Build a larger and more KL-balanced dataset
- Improve consistency across train / validation / test splits
- Refine anatomical evaluation beyond overlap metrics
- Improve reproducibility and experiment reporting
- Extend analysis toward publication-grade study

---

## Team

**Team-1**
- Michael Yeh
- Sujan Sunil Gowda
- Zihan Wang

**Boston University Metropolitan College**  
**Department of Computer Science**

---

## License

See LICENSE.txt in project root

---

## Acknowledgements

This work was completed as part of **MET CS 790: Computer Vision in AI** at Boston University.  
We acknowledge the course guidance, project feedback, and team collaboration that supported this project.