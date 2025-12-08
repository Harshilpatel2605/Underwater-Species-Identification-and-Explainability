# Data Preprocessing & Enhancement

## Overview
This directory contains the pipeline for preprocessing the **URPC2020** dataset. The primary goal of this script is to apply the custom **Hybrid Transformer-CNN Enhancement Model** to the raw underwater images before they are fed into the YOLO object detector.

By enhancing the images first, we aim to correct color casts, improve contrast, and reduce turbidity, thereby increasing the detection accuracy of the downstream YOLO model.

## Script: `enhance_dataset.py`

### Functionality
The script automates the following processing chain for the `train`, `valid`, and `test` splits:

1.  **Model Loading:** Initializes the custom `Generator` (from `net.Ushape_Trans`) and loads the pre-trained weights (`generator_795.pth`).
2.  **Resizing (Input):** Resizes raw images to **256x256** (the native resolution of the enhancement model).
3.  **Inference:** Passes the image through the Transformer-CNN generator to produce an enhanced version.
4.  **Resizing (Output):** Upscales the enhanced result to **1024x1024** to provide high-resolution input for the YOLO detector.
5.  **Label Handling:** Automatically copies the corresponding YOLO `.txt` label files to the new directory.

### Directory Structure Requirements
The script expects the following directory layout relative to itself:

```text
.
├── enhance_dataset.py        # This script
├── URPC2020/                 # Raw Dataset
│   ├── train/
│   ├── valid/
│   └── test/
├── saved_models/
│   └── G/
│       └── generator_795.pth # Pre-trained Generator Weights
└── net/                      # Architecture definitions (Ushape_Trans.py)
```

### Outputs
The script creates new "enhanced" directories alongside the original splits.

* `URPC2020/train` -> `URPC2020/train_enh`
* `URPC2020/valid` -> `URPC2020/valid_enh`
* `URPC2020/test` -> `URPC2020/test_enh`

## How to run

### 1. Prerequisites

Ensure you have the required Python libraries installed:
```python
pip install torch torchvision opencv-python tqdm
```

### 2. Execution

Run the script directly from the terminal. It will automatically detect the GPU (cuda) if available.
```bash
python enhance_dataset.py
```

### 3. Verification
After the script completes, check the `URPC2020/train_enh/images` folder. You should see images that look clearer and more color-balanced compared to the original `URPC2020/train/images`.
