import os
import shutil
import torch
from ultralytics import YOLO

# --- Prerequisite Check ---
base_weights = "yolo11l.pt" # This is now the MODEL we will fine-tune

# --- MODIFIED: Check if the base model file exists ---
if not os.path.exists(base_weights):
    print(f"❌ Error: Base model file not found: {base_weights}")
    print(f"Please make sure '{base_weights}' is in the same directory.")
    # You might want to download it first if it's missing
    # e.g., YOLO("yolo11l.pt") might download it if it's a known model
    exit()

# --- Device ---
device_count = torch.cuda.device_count()
print(f"Available GPUs: {device_count}")

# --- Dataset paths ---
dataset_path = "/home/harshil/URPC2020/"

# --- MODIFICATION: Initialize YOLO by loading the .pt file directly ---
# This loads the standard model architecture and weights from 'yolo11l.pt'
yolo_model = YOLO(base_weights)
print(f"✅ Loaded baseline model directly from '{base_weights}'")


# --- MODIFICATION: Update project name for the baseline model ---
project_name = "yolov11l_baseline_enhanced" # New project folder for baseline
run_name = "baseline-enhanced-fish-detector"

# --- Train YOLO using the data.yaml file ---
yolo_model.train(
    data=os.path.join(dataset_path, "data.yaml"),
    epochs=100,
    # --- Training setup ---
    imgsz=640,
    batch=16,
    device=None, # Use the device list we determined
    optimizer="AdamW",
    # --- Learning rate & scheduling ---
    cos_lr=True,
    # --- Mixed precision training (AMP) ---
    amp=True,
    # --- Output organization ---
    project=project_name, # Updated
    name=run_name,
    exist_ok=True,
    # --- Misc ---
    verbose=False
)

# --- Create directory to store trained model ---
models_dir = "Models"
os.makedirs(models_dir, exist_ok=True)

# --- MODIFICATION: Define paths for the new trained model ---
trained_model_path = f"{project_name}/{run_name}/weights/best.pt"
final_model_path = os.path.join(models_dir, f"{project_name}.pt") # e.g., Models/yolov11l_baseline.pt

# --- Copy best trained model to Models directory ---
if os.path.exists(trained_model_path):
    shutil.copy(trained_model_path, final_model_path)
    print(f"✅ Best model copied to: {final_model_path}")
else:
    print(f"⚠️ Warning: {trained_model_path} not found — skipping model copy.")

# --- Load YOLO model on correct device ---
model = YOLO(final_model_path)
print("✅ Loaded final Baseline model successfully.")

# --- Run validation and compute metrics ---
# The .val() method will automatically use the validation set defined in your data.yaml
print("📈 Running validation on the original validation set...")
# Use device 0 for validation as in your original script
metrics = model.val(device=None, batch=8, split='val') 

# --- MODIFICATION: Save validation metrics to the new results directory ---
results_dir = f"{project_name}/{run_name}/results"
os.makedirs(results_dir, exist_ok=True)
metrics_file = os.path.join(results_dir, "val_metrics.txt")

with open(metrics_file, "w") as f:
    # --- MODIFIED: Updated title for baseline metrics ---
    f.write(f"==== YOLOv11l ({base_weights}) Baseline + enhanced Model Validation Metrics ====\n")
    f.write(f"Precision:      {metrics.box.p.mean():.4f}\n")
    f.write(f"Recall:         {metrics.box.r.mean():.4f}\n")
    f.write(f"mAP@0.5:        {metrics.box.map50:.4f}\n")
    f.write(f"mAP@0.5:0.95:   {metrics.box.map:.4f}\n")
    f.write("==============================================================\n")

print(f"📁 Metrics saved to: {metrics_file}")