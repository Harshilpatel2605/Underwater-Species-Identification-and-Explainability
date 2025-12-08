# Model Training Pipeline

## Overview
This directory contains the Python scripts used to train the various custom YOLOv11 architectures defined in the `configs/` directory.

Each script follows a standardized **Transfer Learning** pipeline. Instead of training from scratch, we load the official pre-trained YOLOv11 Large (`yolo11l.pt`) weights into our custom architectures. This allows the model to retain learned features for the standard layers while learning weights for our new custom modules (e.g., Attention blocks, P2 heads) from scratch.

## Training Logic
Each script (`yolo11_*.py`) performs the following automated steps:

1.  **Architecture Initialization:** Builds the model architecture using the specific YAML configuration (e.g., `yolo11-two-attention.yaml`).
2.  **Weight Transfer:** Loads the pre-trained `yolo11l.pt` weights.
    * *Note:* Layers with matching shapes are transferred. New layers (like `C2PSA` or `P2` heads) are initialized randomly.
3.  **Training:** Executes the training loop on the **URPC2020** dataset (Original + Enhanced images).
4.  **Metric Logging:** Automatically evaluates the model on the Test set immediately after training.
5.  **Artifact Management:** * Saves the full run logs to the `yolov11_*` project folder.
    * Extracts and copies the best weights (`best.pt`) to a centralized `Models/` directory.
    * Writes a summary text file (`test_metrics.txt`) containing Precision, Recall, and mAP scores.

## Hyperparameters & Augmentation
All models are trained with a consistent set of hyperparameters to ensure a fair comparison:

| Parameter | Value | Note |
| :--- | :--- | :--- |
| **Epochs** | 100 | Sufficient for convergence on this dataset size. |
| **Image Size** | 640 | Standard YOLO input resolution. |
| **Batch Size** | 8 | Adjusted for GPU memory constraints. |
| **Optimizer** | AdamW | Robust optimizer for Transformer-based architectures. |
| **Warmup** | 10 Epochs | Stabilizes gradients for the new random layers. |

### Underwater Augmentations
To improve robustness against underwater turbidity, the following specific augmentations are applied during training:
* **HSV-Hue:** `0.015`
* **HSV-Saturation:** `0.6` (Heavy saturation jitter)
* **HSV-Value:** `0.4` (Lighting variation)
* **Mosaic:** `0.5`
* **Erasing:** `0.2` (Random erasing to occlusion robustness)

## How to Run
The scripts are designed to be run directly. Ensure the `configs/` folder is accessible relative to the script.

**Example: Training the Two-Attention Model**
```bash
python train_attention.py
```

*Note*: You may need to update the `dataset_path` variable inside the scripts if running on a different machine.

### Output Structure
After a successful run, the following outputs are generated:
```text
.
├── Models/
│   └── yolov11_two_attention.pt      # The final deployment-ready weights
└── yolov11_two_attention/            # Full training log directory
    └── custom-fish-detector/
        ├── weights/                  # Checkpoints
        ├── results.csv               # Loss curves
        └── results/
            └── test_metrics.txt      # Final validation summary
```
