# Deepfake Image Detection and Explainable Analysis

## Overview

This repository contains the experimental code and resources used in my master's thesis:

**"Comparative Analysis and Explainability of Deepfake Image Detection: XceptionNet, EfficientNet-B0, and ViT"**

Author: **Yujin Jeon**

The main objective of this study is to compare different state-of-the-art deep learning architectures for deepfake image detection and analyze their decision-making processes using explainable AI (XAI) techniques.

---

## Objectives

- Evaluate and compare the performance of three models:
  - XceptionNet
  - EfficientNet-B0
  - Vision Transformer (ViT)

- Investigate explainability using Grad-CAM for CNN-based models and Attention Rollout for ViT.

- Analyze model robustness and focus areas on manipulated image regions.

---

## Models and Techniques

- **XceptionNet**: Depthwise separable convolutions, strong baseline for image classification and manipulation detection.
- **EfficientNet-B0**: Scalable and efficient model with high performance.
- **ViT (Vision Transformer)**: Attention-based approach, strong at capturing global contexts.

### Explainable AI (XAI)

- **Grad-CAM**: Applied to CNN-based models (Xception, EfficientNet-B0) for visualizing important regions.
- **Attention Rollout**: Used with ViT to interpret self-attention weights.

---

## Dataset

- A balanced deepfake image dataset prepared for this study.
- Dataset details, preprocessing methods, and splitting strategy are described in the thesis report.
---

## Code

The full experimental code is included in this repository as a Jupyter Notebook.  
Please refer to the notebook file for detailed implementation and experiments.

---

## Results

Detailed results, performance metrics, and analysis are described in the [report.pdf](./report.pdf).

---

## Report

The full thesis report is included:

- ([deepfake-image-detection-thesis.pdf](./deepfake-image-detection-thesis.pdf)).

---

## Repository Structure

- `/Code/01_initial_version.ipynb` ~ `/Code/07_final_version.ipynb`: Jupyter Notebooks containing all versions of the source code, showing the step-by-step research process. Version `07` is the final version.
- `deepfake-image-detection-thesis.pdf`: Full thesis report.
- `images/`: Example result images (if included).
- `models/`: Saved model weights (if included).

---

## Frameworks and Libraries Used

### Main deep learning and model libraries
- PyTorch
- timm
- torchvision
- EfficientNet-PyTorch
- Transformers (Hugging Face)

### Explainable AI (XAI)
- Grad-CAM (pytorch-grad-cam)

### Evaluation and analysis
- scikit-learn

### Image processing
- PIL (Python Imaging Library)

### Visualization
- Matplotlib
- Seaborn

### Utilities and others
- NumPy
- Pandas
- Joblib


---

## Contact

For any questions, feel free to contact:

- **Yujin Jeon**
- Email: yujin.jeon.developer@gmail.com
