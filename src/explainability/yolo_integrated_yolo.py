# #!/usr/bin/env python

# # Check GPU availability

import os
import cv2
from sklearn.discriminant_analysis import StandardScaler
import torch
import shutil
import joblib
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import Dataset, DataLoader
# --- U-shape Transformer imports ---
from net.Ushape_Trans import Generator
import torch.nn.functional as F
from tqdm import tqdm

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"Using device(s): {[i for i in range(torch.cuda.device_count())]}")


# --- Initialize enhancement model ---
generator = Generator().to(device)
generator.load_state_dict(torch.load("./saved_models/G/generator_795.pth", map_location=device))
generator.eval()

# --- Enhancement function ---
# @torch.no_grad()
# def enhance_image_tensor(img_tensor):
#     """
#     img_tensor: [B, C, H, W] in [0,1]
#     returns enhanced image tensor [B, C, H, W]
#     """
#     return generator(img_tensor)[3]

# --- Custom Dataset with on-the-fly enhancement ---
# class EnhancedYOLODataset(Dataset):
#     def __init__(self, img_dir, label_dir, img_size=640):
#         self.img_dir = img_dir
#         self.label_dir = label_dir
#         self.img_list = sorted(os.listdir(img_dir))
#         self.img_size = img_size

#     def __len__(self):
#         return len(self.img_list)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_list[idx])
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(cv2.resize(img, (self.img_size, self.img_size)), cv2.COLOR_BGR2RGB)

#         # Convert to tensor [1, C, H, W] in [0,1]
#         img_tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float() / 255.0
#         img_tensor = img_tensor.to(device, non_blocking=True)

#         # Enhance image on GPU
#         enhanced_tensor = enhance_image_tensor(img_tensor).squeeze(0)  # [C,H,W]

#         # Load labels
#         label_path = os.path.join(self.label_dir, self.img_list[idx].replace(".jpg", ".txt"))
#         if os.path.exists(label_path):
#             labels = torch.tensor(
#                 [list(map(float, line.strip().split())) for line in open(label_path)],
#                 dtype=torch.float32,
#                 device=device
#             )
#         else:
#             labels = torch.zeros((0,5), dtype=torch.float32, device=device)

#         return enhanced_tensor, labels

# --- Dataset paths ---
dataset_path = "/home/harshil/URPC2020/"
# train_dataset = EnhancedYOLODataset(os.path.join(dataset_path, "train/images"),
#                                 os.path.join(dataset_path, "train/labels"))
# val_dataset = EnhancedYOLODataset(os.path.join(dataset_path, "valid/images"),
#                                 os.path.join(dataset_path, "valid/labels"))
# test_dataset = EnhancedYOLODataset(os.path.join(dataset_path, "test/images"),
#                                 os.path.join(dataset_path, "test/labels"))

# --- DataLoaders with multiple workers ---
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
#                         num_workers=8, pin_memory=True, prefetch_factor=2)
# val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False,
#                         num_workers=8, pin_memory=True, prefetch_factor=2)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
#                         num_workers=8, pin_memory=True, prefetch_factor=2)


# --- Initialize YOLO ---
# yolo_model = YOLO("yolo11n.pt")

# --- Train YOLO using Dataloader ---
# yolo_model.train(
#     data=os.path.join(dataset_path, "data.yaml"),
#     epochs=100,
#     imgsz=640,
#     batch=128,
#     device="0,1,2,3,4,5,6,7",       # multiple GPUs
#     optimizer="AdamW",
#     project="yolo11_integrated_yolo",
#     name="custom-fish-detector",
#     exist_ok=True,
#     verbose=True,
#     augment=False           # don't use YOLO augment
# )

# --- Save trained model ---
models_dir = "Models"
# os.makedirs(models_dir, exist_ok=True)
# trained_model_path = "yolo11_integrated_yolo/custom-fish-detector/weights/best.pt"
# if os.path.exists(trained_model_path):
#     shutil.copy(trained_model_path, os.path.join(models_dir, "yolo11_integrated_yolo_best.pt"))

# Load YOLO model on correct device
model = YOLO(os.path.join(models_dir, "yolo11_integrated_yolo_best.pt"))
if torch.cuda.is_available():
    model.model.to(device)

metrics = model.val(device=0, batch=64)
model.model.eval()
print("✅ Loaded YOLO model successfully.")


# metrics is a dict containing key performance values
# print("\n YOLO v11 Model Evaluation Metrics (Original Data):")
# print(f"Precision:       {metrics.box.p.mean():.4f}")
# print(f"Recall:          {metrics.box.r.mean():.4f}")
# print(f"mAP@0.5:         {metrics.box.map50:.4f}")
# print(f"mAP@0.5:0.95:    {metrics.box.map:.4f}")

# Optional: print per-class metrics (if dataset has multiple classes)
# try:
#     class_names = model.names
#     per_class_data = {
#         "Class": class_names,
#         "Precision": metrics.box.p.tolist(),
#         "Recall": metrics.box.r.tolist(),
#         "mAP@0.5": metrics.box.map50_per_class.tolist(),
#         "mAP@0.5:0.95": metrics.box.map_per_class.tolist()
#     }

#     df = pd.DataFrame(per_class_data)
#     print("\n📈 Per-Class Metrics:\n")
#     print(df)
# except Exception as e:
#     print("\nSkipping per-class metrics (single class or incompatible metrics).")



# Choose a feature extraction layer (usually penultimate or backbone output)
feature_layer = model.model.model[-3]   # last conv before detection head

# Hook to capture features
features_list = []
def hook_fn(module, input, output):
    output = output.to("cpu")
    features_list.append(output.detach())

hook_handle = feature_layer.register_forward_hook(hook_fn)

def extract_yolo_features(img_path, visualize=False, save_vis=False):
    features_list.clear()
    
    # Load and preprocess image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    # Forward pass to trigger hook
    _ = model.model(img_tensor)

    # Get feature map from hook
    features = features_list[0]  # [B, C, H, W] as tensor
    features_flat = torch.mean(features, dim=(2, 3)).squeeze(0).cpu().numpy()  # Global average pooling -> [C]

    # Optional: visualize some feature maps
    if visualize:
        num_maps = min(16, features.shape[1])
        plt.figure(figsize=(12, 6))
        for i in range(num_maps):
            plt.subplot(2, 8, i + 1)
            fm = features[0, i].cpu().numpy()
            fm_norm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
            plt.imshow(fm_norm, cmap="viridis")
            plt.axis("off")
            plt.title(f"Ch {i}")
        plt.suptitle(f"Feature maps for {os.path.basename(img_path)}")

        if save_vis:
            save_path = os.path.join("feature_maps", f"{os.path.basename(img_path)}_features.png")
            os.makedirs("feature_maps", exist_ok=True)
            plt.savefig(save_path)
            print(f"✅ Saved feature visualization at {save_path}")
        else:
            plt.show(block=True)

    return features_flat, features

# Example usage for one image.
img_path = "/home/harshil/URPC2020/test/images/000004.jpg"
feat, features = extract_yolo_features(img_path, visualize=True)
print("Feature shape:", feat.shape)
print("Feature map shape (full):", features.shape)

save_path = os.path.join("feature_maps", f"{os.path.basename(img_path)}_overlay.png")
def visualize_feature_overlay(img_path, features, channel_idx=12, alpha=0.5, save_path=save_path):
    """
    Overlays a single feature map channel on the original image.
    
    Args:
        img_path (str): Path to the original image.
        features (torch.Tensor): Feature tensor from model hook [1, C, H, W].
        channel_idx (int): Which feature map channel to visualize.
        alpha (float): Transparency for overlay (0=no overlay, 1=only heatmap).
        save_path (str): Optional path to save the visualization.
    """
    # Load original image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (640, 640))

    # Extract and normalize selected feature map
    fm = features[0, channel_idx].cpu().numpy()
    fm_norm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)

    # Resize feature map to image size
    fm_resized = cv2.resize(fm_norm, (img_rgb.shape[1], img_rgb.shape[0]))

    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * fm_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay heatmap on image
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)

    # Plot result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(fm_resized, cmap='viridis')
    plt.title(f"Feature Map (Ch {channel_idx})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Overlay (Ch {channel_idx})")
    plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ Saved overlay at: {save_path}")
    else:
        plt.show(block=True)

visualize_feature_overlay(img_path, features, channel_idx=12, alpha=0.5)


img_dir = "/home/harshil/URPC2020/train/images"
label_dir = "/home/harshil/URPC2020/train/labels"
X_file = "X.npy"
y_file = "y.npy"
# Store numerical feature vectors, labels
if os.path.exists(X_file) and os.path.exists(y_file):
    # Load precomputed features
    print("✅ Loading precomputed features...")
    X = np.load(X_file, allow_pickle=True)
    y = np.load(y_file, allow_pickle=True)
else:
    # Compute features
    print("⚡ Computing features from images...")
    X = []
    y = []
    for img_name in tqdm(sorted(os.listdir(img_dir))):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(img_dir, img_name)
        features_flat, features = extract_yolo_features(img_path, visualize=False)
        features_flat = (
            features_flat.cpu().numpy() if torch.is_tensor(features_flat) else features_flat
        )
        X.append(features_flat)

        # Load label
        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt"))
        if os.path.exists(label_path):
            with open(label_path) as f:
                labels = f.readlines()
            class_id = int(labels[0].split()[0]) if len(labels) > 0 else -1
        else:
            class_id = -1

        y.append(class_id)
        del features
        torch.cuda.empty_cache()

    # Filter out -1 labels
    mask = np.array(y) != -1
    X = np.array(X)[mask]
    y = np.array(y)[mask]

    # Save for future use
    np.save(X_file, X)
    np.save(y_file, y)
    print("✅ Features computed and saved.")

print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("✅ Final check before training:")
print("X shape:", X.shape, " | y shape:", y.shape)
assert X.shape[0] == y.shape[0], "❌ Mismatch between X and y length!"

# SVM
class_names = ['holothurian', 'echinus', 'scallop', 'starfish']
svm = SVC(kernel="linear", probability=True, random_state=42, verbose=False)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)
print("=== SVM Results ===")
print(classification_report(y_test, y_pred_svm, target_names=class_names))

# # Save SVM model
joblib.dump(svm, os.path.join(models_dir, "svm_yolo11_normal.pkl"))


# Random Forest
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1,random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("=== Random Forest Results ===")
print(classification_report(y_test, y_pred_rf, target_names=class_names))

# # Save Random Forest model
joblib.dump(rf, os.path.join(models_dir, "random_forest_yolo11_normal.pkl"))



# # XGBoost
xgb = XGBClassifier(
    tree_method='hist',
    device="cuda",         
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    n_estimators=200,
    max_depth=4
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("=== XGBoost Results ===")
print(classification_report(y_test, y_pred_xgb, target_names=class_names))

# # Save XGBoost model
joblib.dump(xgb, os.path.join(models_dir, "xgboost_yolo11_normal.pkl"))

# # K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("=== KNN Results ===")
print(classification_report(y_test, y_pred_knn, target_names=class_names))

# Save KNN model
joblib.dump(knn, os.path.join(models_dir, "knn_yolo11_normal.pkl"))

# # List of models and their predictions
models = {
    "Random Forest": y_pred_rf,
    "SVM": y_pred_svm,
    "XGBoost": y_pred_xgb,
    "KNN": y_pred_knn
}

# for name, y_pred in models.items():
#     print(f"\n===== {name} =====")
#     print(classification_report(y_test, y_pred, target_names=class_names))
#     acc = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {acc:.4f}")

#     # Confusion matrix heatmap
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(8,6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=class_names, yticklabels=class_names)
#     plt.title(f"{name} Confusion Matrix")
#     plt.xlabel("Predicted Class")
#     plt.ylabel("Actual Class")
#     plt.show()

test_img_dir = os.path.join(dataset_path, "test/images")
test_files = [f for f in os.listdir(test_img_dir) if f.lower().endswith((".jpg",".png",".jpeg"))]

#Saving results in result folder
results_dir = os.path.join(os.getcwd(), "Results")
os.makedirs(results_dir, exist_ok=True)

# Save classification reports
with open(os.path.join(results_dir, "random_forest_yolo_report.txt"), "w") as f:
    f.write(classification_report(y_test, y_pred_rf, target_names=class_names))
with open(os.path.join(results_dir, "svm_yolo_report.txt"), "w") as f:
    f.write(classification_report(y_test, y_pred_svm, target_names=class_names))
with open(os.path.join(results_dir, "xgboost_yolo_report.txt"), "w") as f:
    f.write(classification_report(y_test, y_pred_xgb, target_names=class_names))
with open(os.path.join(results_dir, "knn_yolo_report.txt"), "w") as f:
    f.write(classification_report(y_test, y_pred_knn, target_names=class_names))

# Save prediction images for only the first 5 test images
for file in test_files[:5]:
    img_path = os.path.join(test_img_dir, file)
    results_list = model.predict(img_path, conf=0.25, save=False)

    for idx, result in enumerate(results_list):
        img_with_boxes = result.plot()
        out_path = os.path.join(results_dir, f"{os.path.splitext(file)[0]}_result_yolo11_integrated_yolo{idx}.png")
        cv2.imwrite(out_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
