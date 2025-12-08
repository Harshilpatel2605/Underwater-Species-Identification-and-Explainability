# Model Validation & Evaluation

## Overview
This directory contains the standalone validation script used to assess the performance of trained YOLO models. While the training scripts automatically run a validation pass at the end of training, this script allows for **on-demand evaluation** of any specific model weight file (`.pt`) against any dataset split.

It provides a detailed breakdown of the following key metrics:
* **Precision (P):** The accuracy of positive predictions.
* **Recall (R):** The ability of the model to find all positive instances.
* **mAP@0.5:** Mean Average Precision at an Intersection over Union (IoU) threshold of 0.5.
* **mAP@0.5:0.95:** The average mAP over multiple IoU thresholds (the primary COCO metric).

### Functionality
The script uses the `ultralytics` library to load a trained model and run it against the validation or test set defined in your `data.yaml`. It generates a JSON report and prints a concise summary of the box metrics to the console.

### Arguments
The script uses `argparse` for flexible configuration.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | `str` | *(User specific path)* | Path to the trained weights file (`.pt`). |
| `--data` | `str` | *(User specific path)* | Path to the dataset configuration file (`data.yaml`). |
| `--imgsz` | `int` | `640` | The image resolution to use for inference. |
| `--batch` | `int` | `16` | Batch size for validation. |
| `--split` | `str` | `'val'` | The dataset split to evaluate (`val`, `test`, or `train`). |

> **Note:** The script currently contains hardcoded default paths for the user `harshil`. When running on a different machine, you **must** explicitly provide the `--model` and `--data` arguments.

---

## How to Run

### 1. Basic Usage
To run validation using the default settings (if paths match your system):
```bash
python validate.py
```

Here is the README for the folder containing your validation script. I have assumed the script is named validate.py.

validation/README.md
Markdown

# Model Validation & Evaluation

## Overview
This directory contains the standalone validation script used to assess the performance of trained YOLO models. While the training scripts automatically run a validation pass at the end of training, this script allows for **on-demand evaluation** of any specific model weight file (`.pt`) against any dataset split.

It provides a detailed breakdown of the following key metrics:
* **Precision (P):** The accuracy of positive predictions.
* **Recall (R):** The ability of the model to find all positive instances.
* **mAP@0.5:** Mean Average Precision at an Intersection over Union (IoU) threshold of 0.5.
* **mAP@0.5:0.95:** The average mAP over multiple IoU thresholds (the primary COCO metric).

## Script: `validate.py`

### Functionality
The script uses the `ultralytics` library to load a trained model and run it against the validation or test set defined in your `data.yaml`. It generates a JSON report and prints a concise summary of the box metrics to the console.

### Arguments
The script uses `argparse` for flexible configuration.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | `str` | *(User specific path)* | Path to the trained weights file (`.pt`). |
| `--data` | `str` | *(User specific path)* | Path to the dataset configuration file (`data.yaml`). |
| `--imgsz` | `int` | `640` | The image resolution to use for inference. |
| `--batch` | `int` | `16` | Batch size for validation. |
| `--split` | `str` | `'val'` | The dataset split to evaluate (`val`, `test`, or `train`). |

> **Note:** The script currently contains hardcoded default paths for the user `harshil`. When running on a different machine, you **must** explicitly provide the `--model` and `--data` arguments.

---

## How to Run

### 1. Basic Usage
To run validation using the default settings (if paths match your system):
```bash
python validate.py
```

### 2. Custom Evaluation (Recommended)
To evaluate a specific model on the test set:

```bash
python validate.py \
  --model ../Models/yolov11_two_attention.pt \
  --data ../URPC2020/data.yaml \
  --split test \
  --imgsz 640 \
  --batch 32
```

### 3. Output
The script will print a summary table to the terminal:
```text
==============================
📈 VALIDATION METRICS SUMMARY
==============================
   Precision (P):     0.8540
   Recall (R):        0.7920
   mAP@0.5:           0.8210
   mAP@0.5-0.95:      0.6450

Validation complete. Results and plots saved to 'runs/detect/val'.
```

Detailed plots (Confusion Matrix, F1 Curve, etc.) are automatically saved by YOLO to the `runs/detect/val` directory.
