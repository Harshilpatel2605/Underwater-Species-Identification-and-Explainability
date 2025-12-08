# YOLOv11 Custom Architecture Experiments

## Overview
This directory contains a suite of custom YOLOv11 model configurations (`.yaml`). The goal of these experiments is to improve detection performance—specifically for **small objects** and **feature extraction efficiency**—by modifying the standard YOLOv11 architecture.

The modifications are categorized into three experimental streams:
1.  **Head Modifications:** Adding high-resolution detection heads (P2) for small objects.
2.  **Attention Mechanisms:** Integrating C2PSA attention blocks at various backbone stages to improve feature focus.
3.  **Backbone Redesign:** Replacing standard C3k2 blocks with custom modules (CSMB, LKSP) for enhanced receptive fields.

## Model Configuration Breakdown

### 1. Baseline & Reference
* **`yolo11-backbone.yaml`**
    * **Description:** The standard YOLOv11n architecture reference.
    * **Purpose:** Serves as the baseline for benchmarking performance improvements achieved by the custom models below.

### 2. Multi-Scale Head Experiments (Small Object Focus)
* **`yolo11-add-head.yaml`**
    * **Modification:** Adds a **4th detection head (P2)** operating at a higher resolution (160x160 for 640p input).
    * **Hypothesis:** The standard P3-P5 heads often lose detail for tiny objects. Adding a P2 head propagates high-resolution low-level features directly to the detection layer.

### 3. Attention Mechanism Experiments
* **`yolo11-attention.yaml` (Light Attention)**
    * **Modification:** Adds C2PSA attention blocks after **P4** and before the Neck.
    * **Goal:** To test if late-stage attention improves classification accuracy without significantly increasing computational cost.
* **`yolo11-two-attention.yaml` (Medium Attention)**
    * **Modification:** Extends attention to the **P3** stage (P3, P4, and Neck).
    * **Goal:** To determine if applying attention earlier in the backbone helps preserve spatial information for smaller objects.

### 4. Hybrid Architectures (The "Best of Both" Approach)
* **`yolo11-head-attention.yaml`**
    * **Modification:** Combines the **4-Head structure** (from `add-head`) with **Enhanced Attention** (C2PSA after P3 & P4).
    * **Logic:** This is the most robust model, attempting to capture small objects via the P2 head while using attention to filter noise from the feature maps.

### 5. Experimental Backbone Modules
* **`yolo11-head-backbone.yaml`**
    * **Modification:** * Replaces standard `C3k2` blocks with **CSMB** (Custom Separable convolutions).
        * Replaces `SPPF` with **LKSP** (Large Kernel Spatial Pyramid).
    * **Goal:** To test if changing the fundamental building blocks of the network provides better feature extraction than the standard Ultralytics modules.

---

## How to Run

### Prerequisites
Ensure you have the Ultralytics package installed:
```bash
pip install ultralytics
```

### 1. Training a Custom Model
To train one of these specific configurations, point the `model` argument to the YAML file path.

Using Python:
```python
from ultralytics import YOLO

# Load the custom configuration (e.g., the hybrid head+attention model)
model = YOLO("configs/yolo11-head-attention.yaml") 

# Train the model
results = model.train(
    data="coco8.yaml",  # Replace with your dataset YAML
    epochs=100,
    imgsz=640,
    batch=16,
    name="yolo11_head_attn_run"
)
```

Using CLI:
```bash
yolo task=detect mode=train model=configs/yolo11-head-attention.yaml data=your_dataset.yaml epochs=100 imgsz=640
```

### 2. Validating/Testing
To validate a trained model (after training is complete and weights are saved):
```bash
yolo task=detect mode=val model=runs/detect/yolo11_head_attn_run/weights/best.pt data=your_dataset.yaml
```

### 3. Comparison Strategy
For a fair comparison in the final report, ensure all models are trained with identical hyperparameters:

* Same seed (for reproducibility)
* Same epochs and batch size
* Same imgsz (Input image size)
