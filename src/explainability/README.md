# Explainable AI (XAI) & Visualization Modules

## Overview
This directory contains a suite of scripts designed to interpret and visualize the decision-making process of the YOLOv11 models. Since deep learning models are often "black boxes," these tools help identify *where* the model is looking and *which* features it finds most important for detection.

The techniques implemented range from gradient-based methods (Grad-CAM) to perturbation-based methods (D-RISE) and feature extraction analysis.

## Explainability Techniques Implemented

### 1. Perturbation-Based (Black Box)
* **Script:** `d-rise.py`
* **Method:** **D-RISE (Detection Randomized Input Sampling for Explanation)**.
* **How it works:** It generates thousands of random masks, overlays them on the image, and observes how the detection score changes.
* **Pros:** Highly accurate; does not require access to internal model weights; specific to object detection.
* **Cons:** Slower than other methods due to repeated inference.

### 2. Principal Component Analysis (Unsupervised)
* **Script:** `eigenCAM.py`
* **Method:** **EigenCAM**.
* **How it works:** Computes the Principal Components (PCA) of the 2D feature maps from the convolutional layers.
* **Pros:** extremely fast; produces very clean, object-specific heatmaps; does not rely on backpropagation (gradients).
* **Best For:** Visualizing exactly which features the backbone CNN is extracting.

### 3. Gradient-Based
* **Script:** `gradcam.py` & `yolo_integrated_grad_cam.py`
* **Method:** **Grad-CAM (Gradient-weighted Class Activation Mapping)**.
* **How it works:** Uses the gradients of the target class flowing into the final convolutional layer to produce a coarse localization map.
* **Status:** *Experimental for YOLO.* (YOLO's complex head structure makes standard Grad-CAM noisy; EigenCAM or D-RISE are often preferred).

### 4. Feature Extraction & Hybrid Classification
* **Script:** `yolo_integrated_yolo.py`
* **Method:** **Backbone Feature Extraction**.
* **How it works:** Hooks into the penultimate layer of YOLO to extract raw feature vectors. These features are then used to train classical classifiers (SVM, Random Forest, XGBoost).
* **Purpose:** To validate if the features learned by YOLO are linearly separable and robust enough for simpler classifiers.

---

## File Manifest

| File | Description |
| :--- | :--- |
| **`eigenCAM.py`** | **[Recommended]** Generates EigenCAM heatmaps. Uses layers from 70-95% network depth to visualize object features. |
| **`d-rise.py`** | **[High Quality]** Generates D-RISE saliency maps. Requires no gradients but takes longer to run. |
| **`gradcam.py`** | Standalone script providing three visualization methods: Grad-CAM, Activation Maps, and Standard Detection. |
| **`yolo_integrated_grad_cam.py`** | Advanced script that integrates Grad-CAM with the `URPC2020` dataset pipeline. Includes experimental code for training downstream classifiers on Grad-CAM features. |
| **`yolo_integrated_yolo.py`** | Extracts features from YOLO and trains SVM/RF/XGBoost classifiers on the `URPC2020` dataset. Includes feature map overlay visualization. |
| **`heatmap.py`** | A simple utility that paints confidence scores onto the image canvas (basic "where is the box" visualization). |

---

## Dependencies
These scripts require specific XAI libraries. Install them via pip:

```bash
pip install ultralytics opencv-python matplotlib scikit-learn xgboost pytorch-grad-cam
```

## How to Run

### 1. Running EigenCAM (Recommended Visualization)
This script targets yolo11l.pt by default. You can edit the script to point to your custom trained weights.
```bash
python eigenCAM.py
```

*Output*: Saves _eigencam_baseline.png showing the heatmap overlay.

### 2. Running D-RISE (High-Fidelity Explanation)
Best for generating figures for papers/reports.

```python
python d-rise.py
```

*Output*: Creates a d_rise_output/ folder with heatmaps for every detected class.

### 3. Feature Extraction & Classifier Training
This script requires the URPC2020 dataset path to be correctly set in the code.
```bash
python yolo_integrated_yolo.py
```

*Output*: * Visualizes feature map overlays for individual channels.

* Trains SVM, RF, and XGBoost models.
* Saves classification reports to `Results/`.
