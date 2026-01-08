# Interpretable Deep Learning for Colorectal Cancer Histopathology

## Motivation
Deep learning models achieve strong performance in histopathology classification,
but their lack of interpretability limits clinical trust. This project focuses on
making convolutional neural networks interpretable by analyzing model errors using
Grad-CAM visual explanations.

---

## Project Overview
This repository contains an interpretable deep-learning pipeline for classifying
colorectal cancer tissue types from histopathology images. The system integrates:

- Transfer learning with ResNet50
- Grad-CAM–based interpretability
- Error-driven analysis of misclassifications
- A deployable Streamlit interface

Rather than evaluating only accuracy, the project investigates *why* the model
confuses certain tissue types.

---

## Dataset
- Histopathology image patches
- 8 classes: adipose, complex, debris, empty, lympho, mucosa, stroma, tumor
- Input size: 224 × 224 RGB

This project is for educational and research exploration only.

---

## Model Architecture
- ResNet50 (ImageNet pretrained)
- Custom dense layer (128 units, ReLU)
- Dropout (0.5)
- Softmax output (8 classes)

---

## Interpretability
Grad-CAM is applied using the final convolutional block of ResNet50
(`conv5_block3_out`) to visualize spatial attention for predictions and errors.

## Data Directory

Place the following files here:
- X_train.npy
- y_train.npy
- X_test.npy
- y_test.npy

Data is excluded from the repository due to size and licensing constraints.

---

## Quick Start

```bash
git clone https://github.com/anshkumar0201/interpretable-colorectal-cancer-cnn.git
cd interpretable-colorectal-cancer-cnn
pip install -r requirements.txt
python run_training.py
streamlit run app/streamlit_app.py

