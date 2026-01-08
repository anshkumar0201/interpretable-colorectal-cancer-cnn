# Interpretable Deep Learning for Colorectal Cancer Histopathology

## Motivation
Deep learning models achieve high accuracy in histopathology classification, but
their black-box nature limits trust and clinical interpretability. In medical
contexts, understanding *why* a model makes a prediction is as important as the
prediction itself.

This project focuses on **interpretable deep learning** for colorectal cancer
histopathology by integrating **Grad-CAM–based error analysis** into a CNN trained
using transfer learning.

---

## Project Overview
This repository contains an end-to-end pipeline for:
- Classifying colorectal tissue types from histopathology images
- Visualizing model attention using Grad-CAM
- Systematically analyzing **misclassifications**
- Comparing incorrect predictions with correct reference samples

Rather than evaluating performance only through accuracy, the project
investigates *failure modes* and tissue-level ambiguity.

---

## Dataset
- **Type:** Histopathology image patches
- **Classes:** adipose, complex, debris, empty, lympho, mucosa, stroma, tumor
- **Input Size:** 224 × 224 RGB
- **Preprocessing:** normalization, resizing, categorical encoding

> This project is intended for educational and research exploration only and is
> not a clinical diagnostic system.

---

## Model Architecture
- **Base model:** ResNet50 (ImageNet pretrained)
- **Transfer learning strategy:**
  - Frozen convolutional backbone
  - Dense layer (128 units, ReLU)
  - Dropout (0.5)
  - Softmax output (8 tissue classes)

---

## Interpretability & Error Analysis
- Grad-CAM is computed using the final convolutional block (`conv5_block3_out`)
- The pipeline automatically:
  - Detects misclassified samples
  - Visualizes Grad-CAM heatmaps for errors
  - Compares them against correctly classified examples from:
    - the predicted class
    - the true class

This enables qualitative investigation of morphological similarities that lead
to classification confusion.

---

## Limitations
- Patch-level inference (not whole-slide images)
- Dataset size and annotation constraints
- Interpretability results are qualitative

---

## Future Work
- Whole-slide inference
- Quantitative Grad-CAM similarity analysis
- Integration with clinical metadata
- Deployment as an interactive web application

---

## Technologies Used
TensorFlow • Keras • ResNet50 • Grad-CAM • NumPy • OpenCV • Matplotlib • Streamlit

## Data Directory

This folder is intended to store training and testing data.

Due to size and licensing constraints, datasets are not included in this
repository. Users should place preprocessed image arrays and labels here.

Example expected variables:
- X_train, y_train
- X_test, y_test
