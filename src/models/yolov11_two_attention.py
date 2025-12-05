## MODIFIED MODEL (Attention) ---
## Train YOLO v11 large (Modified) on Original + Enhanced images.

import os
import shutil
import torch
from ultralytics import YOLO

custom_model_yaml = "yolo11-two-attention.yaml" 
base_weights = "yolo11l.pt"


# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Available GPUs: {torch.cuda.device_count()}")

# --- Dataset paths ---
dataset_path = "/home/harshil/URPC2020/"

# --- MODIFICATION: Initialize YOLO from your custom YAML ---
# 1. Load the 'l' scaled version of your custom attention
print(f"Loading custom 'l' scale model from: {custom_model_yaml}")
yolo_model = YOLO(custom_model_yaml, task='detect') # Explicitly set task

# 2. Load (transfer) weights from the pre-trained standard yolo11l.pt
# New attention layers will be randomly initialized.
print(f"Transferring weights from '{base_weights}'")
yolo_model = yolo_model.load(base_weights)
print(f"✅ Loaded custom architecture from '{custom_model_yaml}'")
print(f"✅ Transferred weights from '{base_weights}'")


# --- MODIFICATION: Update project name for new model ---
project_name = "yolov11_two_attention" # New project folder name
run_name = "custom-fish-detector"

# --- Train YOLO using the data.yaml file ---
yolo_model.train(
    data=os.path.join(dataset_path, "data.yaml"),
    epochs=100,
    
    # --- Underwater augmentation ---
    hsv_h=0.015,
    hsv_s=0.6,
    hsv_v=0.4,
    translate=0.1,
    scale=0.3,
    fliplr=0.5,
    mosaic=0.5,
    erasing=0.2,
    
    # --- Training setup ---
    imgsz=640,
    batch=8, 
    device="6,7", # Correct for running directly
    optimizer="AdamW",
    
    # --- Learning rate & scheduling ---
    cos_lr=True,
    lr0=0.001,
    warmup_epochs=10, # <--- RECOMMENDED FIX
    
    # --- Mixed precision training (AMP) ---
    amp=True, # Keep this, but warmup is the real fix
    
    # --- Output organization ---
    project=project_name, 
    name=run_name,
    exist_ok=True,
    
    # --- Misc ---
    verbose=True
)

# --- Create directory to store trained model ---
models_dir = "Models"
os.makedirs(models_dir, exist_ok=True)

# --- MODIFICATION: Define paths for the new trained model ---
trained_model_path = f"{project_name}/{run_name}/weights/best.pt"
final_model_path = os.path.join(models_dir, f"{project_name}.pt") # e.g., Models/yolov11_attention_4head.pt

# --- Copy best trained model to Models directory ---
if os.path.exists(trained_model_path):
    shutil.copy(trained_model_path, final_model_path)
    print(f"✅ Best Attention model copied to: {final_model_path}")
else:
    print(f"⚠️ Warning: {trained_model_path} not found — skipping model copy.")

# --- Load YOLO model on correct device ---
model = YOLO(final_model_path)
print(f"✅ Loaded final {project_name} model successfully.")

# --- Run validation and compute metrics ---
# The .val() method will automatically use the validation set defined in your data.yaml
print("📈 Running validation on the original test set...")
metrics = model.val(device=0, batch=64, split='test') # Run validation on a single GPU

# --- MODIFICATION: Save validation metrics to the new results directory ---
results_dir = f"{project_name}/{run_name}/results"
os.makedirs(results_dir, exist_ok=True)
metrics_file = os.path.join(results_dir, "test_metrics.txt")

with open(metrics_file, "w") as f:
    f.write("==== YOLO (two attention) Model Test Metrics (Original + Enhanced) ====\n")
    f.write(f"Precision:      {metrics.box.p.mean():.4f}\n")
    f.write(f"Recall:         {metrics.box.r.mean():.4f}\n")
    f.write(f"mAP@0.5:        {metrics.box.map50:.4f}\n")
    f.write(f"mAP@0.5:0.95:   {metrics.box.map:.4f}\n")
    f.write("===============================\n")

print(f"📁 Metrics saved to: {metrics_file}")