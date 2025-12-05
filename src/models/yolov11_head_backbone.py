import os
import shutil
import torch
from ultralytics import YOLO

# --- Prerequisite Check ---
custom_model_yaml = "yolo11-head-backbone.yaml"
base_weights = "yolo11l.pt"

if not os.path.exists(custom_model_yaml):
    print(f"❌ Error: Custom model file not found: {custom_model_yaml}")
    print("Please create 'yolo11-head-backbone.yaml' as described.")
    exit()
    
if not os.path.exists(base_weights):
    print(f"⚠️ Warning: Base weights '{base_weights}' not found.")
    print("Model will be trained from scratch (random weights).")
    # You might want to download it first if it's missing
    # YOLO("yolo11l.pt")  # This would download it

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Available GPUs: {torch.cuda.device_count()}")

# --- Dataset paths ---
dataset_path = "/home/harshil/URPC2020/"

# --- MODIFICATION: Initialize YOLO from your custom YAML ---
# 1. Load the 'l' scaled version of your custom 4-head architecture
yolo_model = YOLO(custom_model_yaml, task='detect')

# 2. Load (transfer) weights from the pre-trained 3-head yolo11l.pt
# This will load all matching layers (backbone, etc.) and
# randomly initialize the new P2 head layers.
yolo_model = yolo_model.load(base_weights) 
print(f"✅ Loaded custom architecture from '{custom_model_yaml}'")
print(f"✅ Transferred weights from '{base_weights}'")


# --- MODIFICATION: Update project name for new model ---
project_name = "yolov11_head_backbone" # New project folder
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
    scale=0.5,
    fliplr=0.5,
    mosaic=0.8,
    erasing=0.3,
    # --- Training setup ---
    imgsz=640,
    batch=32,
    device="4,5,6,7",
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
    verbose=True
)

# --- Create directory to store trained model ---
models_dir = "Models"
os.makedirs(models_dir, exist_ok=True)

# --- MODIFICATION: Define paths for the new trained model ---
trained_model_path = f"{project_name}/{run_name}/weights/best.pt"
final_model_path = os.path.join(models_dir, f"{project_name}.pt") # e.g., Models/yolov11_p2_head_l.pt

# --- Copy best trained model to Models directory ---
if os.path.exists(trained_model_path):
    shutil.copy(trained_model_path, final_model_path)
    print(f"✅ Best model copied to: {final_model_path}")
else:
    print(f"⚠️ Warning: {trained_model_path} not found — skipping model copy.")

# --- Load YOLO model on correct device ---
model = YOLO(final_model_path)
print("✅ Loaded final P2-head+backbone model successfully.")

# --- Run validation and compute metrics ---
# The .val() method will automatically use the validation set defined in your data.yaml
print("📈 Running validation on the original validation set...")
metrics = model.val(device=0, batch=64, split='val')

# --- MODIFICATION: Save validation metrics to the new results directory ---
results_dir = f"{project_name}/{run_name}/results"
os.makedirs(results_dir, exist_ok=True)
metrics_file = os.path.join(results_dir, "val_metrics.txt")

with open(metrics_file, "w") as f:
    f.write("==== YOLO (P2-Head+Backbone) Validation Metrics ====\n") # Updated title
    f.write(f"Precision:      {metrics.box.p[0]:.4f}\n")
    f.write(f"Recall:         {metrics.box.r[0]:.4f}\n")
    f.write(f"mAP@0.5:        {metrics.box.map50:.4f}\n")
    f.write(f"mAP@0.5:0.95:   {metrics.box.map:.4f}\n")
    f.write("=======================================\n")

print(f"📁 Metrics saved to: {metrics_file}")