# Real-Time Face Mask Detection System

A production-grade, real-time computer vision application built from scratch using

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv&logoColor=white)](https://opencv.org/)
[![MobileNetV2](https://img.shields.io/badge/MobileNetV2-Model-blueviolet)](https://keras.io/api/applications/mobilenet/)
[![Transfer Learning](https://img.shields.io/badge/Transfer%20Learning-ML-blue)](#)
[![Fine Tuning](https://img.shields.io/badge/Fine--Tuning-Training-yellow)](#)
[![Real Time](https://img.shields.io/badge/Real--Time-Inference-brightgreen)](#)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Architectural Choice: Why MobileNetV2?](#architectural-choice-why-mobilenetv2)
4. [Workflow & Pipeline](#workflow--pipeline)
5. [Training & Fine-Tuning Strategy](#training--fine-tuning-strategy)
6. [Performance & Accuracy Metrics](#performance--accuracy-metrics)
7. [Project Directory Structure](#project-directory-structure)
8. [Installation & Setup](#installation--setup)
9. [Running Live Inference](#running-live-inference)

---

## 1. Project Overview
The objective of this project is to build an intelligent monitoring system that detects human faces via a live webcam stream and classifies whether the individual is wearing a face mask (`With Mask`) or not (`Without Mask`) in real time. 

To achieve production-grade reliability (avoiding false positives caused by indoor lighting, shadows, or facial hair), this system uses a state-of-the-art pre-trained convolutional neural network enhanced via fine-tuning.

---

## 2. Dataset Description
* **Dataset Source:** *Face Mask Detection ~12K Images Dataset* ( sourced from Ashish Jangra / Kaggle ).
* **Categories:** 
  * `With Mask`
  * `Without Mask`
* **Structure:** Organized into dedicated `Train` and `Validation` split directories for robust cross-validation and evaluation.

---

## 3. Architectural Choice: Why MobileNetV2?

### The Shift from Custom CNNs to Transfer Learning
Initially, custom Convolutional Neural Networks (CNNs) built from scratch struggle with real-world noise (such as beards, dim room lighting, and angles) because they lack deep hierarchical feature maps. 

### Why MobileNetV2?
1. **ImageNet Pre-training:** MobileNetV2 is pre-trained on millions of diverse images, meaning it already understands complex human facial geometry, skin textures, and edges.
2. **Inverted Residuals and Linear Bottlenecks:** It uses lightweight depthwise separable convolutions, making it exceptionally fast and optimized for edge devices and real-time webcam streams without lagging.
3. **Strict Preprocessing Requirements:** Integrates `tf.keras.applications.mobilenet_v2.preprocess_input` to scale pixel inputs precisely into the `[-1, 1]` range for optimal mathematical contrast.

---

## 4. Workflow & Pipeline
1. **Data Ingestion:** Images are loaded dynamically using `image_dataset_from_directory` with batching and categorical label encoding.
2. **Preprocessing & Augmentation:** Input dimensions standardized to `(224, 224, 3)` with MobileNetV2 normalization.
3. **Face Localization:** Real-time face ROI (Region of Interest) extraction using OpenCV's classical Haar Cascade classifier (`haarcascade_frontalface_default.xml`).
4. **Inference Engine:** Processed facial tensors pass through the fine-tuned neural network to output multi-class probabilities.
5. **Dynamic UI Rendering:** Real-time bounding boxes rendered via OpenCV (Green for Mask, Red for No Mask) with confidence percentage overlays.

---

## 5. Training & Fine-Tuning Strategy
The model was trained in two distinct phases to prevent catastrophic forgetting and maximize convergence:

* **Phase 1 (Classifier Head Training):** 
  * The MobileNetV2 base feature extractor was frozen (`base_model.trainable = False`).
  * Only the custom dense classification head (Global Average Pooling -> Dense 128 ReLU -> Dropout -> Softmax) was trained for 5 epochs with an Adam optimizer (`lr=0.001`).
* **Phase 2 (Fine-Tuning):** 
  * The base model was unfrozen, but all layers **except the last 30 layers** remained frozen.
  * Recompiled with a micro-learning rate (`1e-5`) for delicate weight adaptation.
  * Utilized `EarlyStopping` and `ReduceLROnPlateau` callbacks to halt training automatically at optimal convergence.

---

## 6. Performance & Accuracy Metrics
* **Validation Accuracy:** Achieved **100.0% validation accuracy** (`val_accuracy: 1.0000`) and minimal validation loss (`5.3289e-04`) during fine-tuning convergence.
* **Robustness:** Significantly reduced false classifications on edge cases compared to traditional baseline CNNs.

---

## 7. Project Directory Structure
```text
Real_Time_FaceMask_Detection/
│
├── Face Mask Dataset/
│   ├── Train/
│   │   ├── With Mask/
│   │   └── Without Mask/
│   └── Validation/
│       ├── With Mask/
│       └── Without Mask/
│
├── mask_training.ipynb                  # Complete training and fine-tuning notebook
├── mobilenet_finetuned_mask_model.keras # Saved production model weights artifact
├── live_inference.py                    # Standalone real-time webcam monitoring script
└── README.md                            # Project documentation
