# Inference & Hybrid Classification Pipelines

## Overview
This directory contains scripts for running inference using your trained models. It supports two distinct modes of operation:
1.  **Hybrid Pipeline (`test_pipeline.py`):** A sophisticated two-stage approach where YOLO localizes objects, but the final classification is performed by traditional ML classifiers (SVM, Random Forest, XGBoost) using features extracted from the YOLO backbone.
2.  **Video Inference (`video_inference.py`):** Standard real-time object detection on video files using the YOLOv11 end-to-end architecture.

## Scripts

### 1. Hybrid Testing Pipeline (`test_pipeline.py`)
This script validates the "Integrated YOLO + Classifier" hypothesis. It demonstrates that features learned by the YOLO backbone are robust enough to drive classical classifiers.

**Workflow:**
1.  **Detection:** YOLO scans the image and proposes bounding boxes.
2.  **Feature Extraction:** For each detected crop, the script hooks into the **penultimate convolutional layer** of the YOLO backbone to extract a deep feature vector.
3.  **Classification:** This feature vector is normalized (using a pre-fitted `StandardScaler`) and passed to an external classifier (SVM, RF, XGBoost, or KNN) for the final class prediction.
4.  **Visualization:** Draws the bounding box with the label predicted by the *classical classifier*, not the YOLO head.

**Key Dependencies:**
* Requires the `Models/` directory to contain:
    * `yolo11_integrated_yolo_best.pt`
    * `svm_yolo11_normal.pkl`
    * `random_forest_yolo11_normal.pkl`, etc.
    * `X.npy` (used to recreate the `StandardScaler`).

### 2. Video Inference (`video_inference.py`)
A straightforward utility for running your best YOLO model on video data (e.g., MP4 files from the URPC dataset). It saves the annotated video to the `runs/` directory.

---

## How to Run

### 1. Running the Hybrid Pipeline
This script automatically picks 5 random images from your test set and runs them through a randomly selected classifier (e.g., Image 1 via SVM, Image 2 via XGBoost).

```bash
python test_pipeline.py
```

*Output*:

* Console logs showing which classifier was chosen for which image.
* Annotated images saved to `test_pipeline_results/`.
  * Filename format: image_name_ClassifierName_result.png

### 2. Running Video Inference
Note: You must edit the file `video_inference.py` to point to your specific video path and model weight path before running.

```bash
python video_inference.py
```

## Directory Structure
The scripts expect the following layout:
```text
.
├── test_pipeline.py
├── video_inference.py
├── Models/                 # Directory containing all .pt and .pkl files
└── URPC2020/               # Dataset directory
```
