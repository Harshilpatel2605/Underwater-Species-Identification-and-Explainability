import os
import shutil
import torch
from ultralytics import YOLO

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"Using device(s): {[i for i in range(torch.cuda.device_count())]}")

# --- Dataset paths ---
dataset_path = "/home/harshil/URPC2020/"

# --- Initialize YOLO ---
yolo_model = YOLO("yolo11l.pt") # updated

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
    batch=64,
    device="0,1,2,3,4,5,6,7",
    optimizer="AdamW",
    # --- Learning rate & scheduling ---
    cos_lr=True,
    # --- Mixed precision training (AMP) ---
    amp=True,
    # --- Output organization ---
    project="yolov11l_backbone_change",
    name="custom-fish-detector",
    exist_ok=True,
    # --- Misc ---
    verbose=True,
    augment=True, # Set to True to use YOLO's built-in augmentations on the combined dataset
)

# --- Create directory to store trained model ---
models_dir = "Models"
os.makedirs(models_dir, exist_ok=True)

# --- Define trained YOLO weight path ---
trained_model_path = "yolov11l_backbone_change/custom-fish-detector/weights/best.pt"
final_model_path = os.path.join(models_dir, "yolov11l_backbone_change.pt")

# --- Copy best trained model to Models directory ---
if os.path.exists(trained_model_path):
    shutil.copy(trained_model_path, final_model_path)
    print(f"✅ Best model copied to: {final_model_path}")
else:
    print("⚠️ Warning: best.pt not found — skipping model copy.")

# --- Load YOLO model on correct device ---
model = YOLO(final_model_path)
print("✅ Loaded final model successfully.")

# --- Run validation and compute metrics ---
# The .val() method will automatically use the validation set defined in your data.yaml
print("📈 Running validation on the original validation set...")
metrics = model.val(device=0, batch=64, split='val')

# --- Save validation metrics to file ---
results_dir = "yolov11l_backbone_change/custom-fish-detector/results"
os.makedirs(results_dir, exist_ok=True)
metrics_file = os.path.join(results_dir, "val_metrics.txt")

with open(metrics_file, "w") as f:
    f.write("==== YOLO Validation Metrics ====\n")
    f.write(f"Precision:      {metrics.box.p[0]:.4f}\n")
    f.write(f"Recall:         {metrics.box.r[0]:.4f}\n")
    f.write(f"mAP@0.5:        {metrics.box.map50:.4f}\n")
    f.write(f"mAP@0.5:0.95:   {metrics.box.map:.4f}\n")
    f.write(f"Per-class mAPs: {metrics.box.maps}\n")
    f.write("===============================\n")

print(f"📁 Metrics saved to: {metrics_file}")