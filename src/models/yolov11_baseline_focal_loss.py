import os
import shutil
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import FocalLoss

# --- NEW: Define the Custom Trainer for Focal Loss ---
class FocalLossTrainer(DetectionTrainer):
    """
    A custom trainer that uses Focal Loss for the classification loss.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Check if 'cls' loss function exists in the criterion
        if 'cls' in self.criterion.loss_functions:
            # Get the original BCE loss
            bce_loss = self.criterion.loss_functions['cls'].loss
            
            # --- This is the key part ---
            # Override the classification loss with FocalLoss
            # You can tune gamma and alpha here
            self.criterion.loss_functions['cls'].loss = FocalLoss(bce_loss, gamma=2.0, alpha=0.25)
            
            print("\n✅ Custom FocalLossTrainer initialized: Replaced 'cls' loss with FocalLoss (gamma=2.0, alpha=0.25)\n")
        else:
            print("\n⚠️ Warning: 'cls' loss function not found in criterion. FocalLoss not applied.\n")

# --- Prerequisite Check ---
base_weights = "yolo11l.pt" # This is the MODEL we will fine-tune

# --- Check if the base model file exists ---
if not os.path.exists(base_weights):
    print(f"❌ Error: Base model file not found: {base_weights}")
    print(f"Please make sure '{base_weights}' is in the same directory.")
    exit()

# --- Device ---
device_count = torch.cuda.device_count()
print(f"Available GPUs: {device_count}")

# --- Dataset paths ---
dataset_path = "/home/harshil/URPC2020/"

# --- Initialize YOLO by loading the .pt file directly ---
yolo_model = YOLO(base_weights)
print(f"✅ Loaded baseline model directly from '{base_weights}'")


# --- MODIFIED: Update project/run name for the new Focal Loss model ---
project_name = "yolov11_baseline_focal_loss" # New project folder for comparison
run_name = "custom-fish-detector"

# --- NEW: Assign the custom FocalLossTrainer to the model ---
# This MUST be done before calling .train()
yolo_model.trainer = FocalLossTrainer
print(f"✅ Assigned custom FocalLossTrainer to the model.")

# --- Train YOLO using the data.yaml file ---
print(f"\n🚀 Starting training for {project_name}/{run_name}...")
yolo_model.train(
    data=os.path.join(dataset_path, "data.yaml"),
    epochs=100,
    # --- Training setup ---
    imgsz=640,
    batch=2,
    device=None, # Use the device list we determined
    workers=2,
    optimizer="AdamW",
    # --- Learning rate & scheduling ---
    cos_lr=True,
    # --- Mixed precision training (AMP) ---
    amp=False,   # .
    # --- Output organization ---
    project=project_name, # Updated to focalloss project
    name=run_name,
    exist_ok=True,
    # --- Misc ---
    verbose=False
)
print(f"\n🏁 Finished training for {project_name}.")

# --- Create directory to store trained model ---
models_dir = "Models"
os.makedirs(models_dir, exist_ok=True)

# --- Define paths for the new trained model ---
trained_model_path = f"{project_name}/{run_name}/weights/best.pt"
final_model_path = os.path.join(models_dir, f"{project_name}.pt") # e.g., Models/yolov11l_focalloss.pt

# --- Copy best trained model to Models directory ---
if os.path.exists(trained_model_path):
    shutil.copy(trained_model_path, final_model_path)
    print(f"✅ Best model copied to: {final_model_path}")
else:
    print(f"⚠️ Warning: {trained_model_path} not found — skipping model copy.")

# --- Load YOLO model on correct device ---
print(f"Loading final FocalLoss model from {final_model_path}...")
model = YOLO(final_model_path)
print("✅ Loaded final FocalLoss model successfully.")

# --- Run validation and compute metrics ---
# The .val() method will automatically use the validation set defined in your data.yaml
print("📈 Running validation on the original validation set...")
# Use device 0 for validation
metrics = model.val(device=None, batch=8, split='val') 

# --- Save validation metrics to the new results directory ---
results_dir = f"{project_name}/{run_name}/results"
os.makedirs(results_dir, exist_ok=True)
metrics_file = os.path.join(results_dir, "val_metrics.txt")

with open(metrics_file, "w") as f:
    # --- MODIFIED: Updated title for FocalLoss metrics ---
    f.write(f"==== YOLOv11l ({base_weights}) + FocalLoss Model Validation Metrics ====\n")
    f.write(f"Precision:      {metrics.box.p.mean():.4f}\n")
    f.write(f"Recall:         {metrics.box.r.mean():.4f}\n")
    f.write(f"mAP@0.5:        {metrics.box.map50:.4f}\n")
    f.write(f"mAP@0.5:0.95:   {metrics.box.map:.4f}\n")
    f.write("==============================================================\n")

print(f"📁 Metrics saved to: {metrics_file}")
print("\n🎉 All steps complete. You can now compare the results in 'yolov11l_baseline' and 'yolov11l_focalloss'.")