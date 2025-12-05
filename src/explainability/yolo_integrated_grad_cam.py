# #!/usr/bin/env python

# # Check GPU availability
# import torch
# print(f"Available GPUs: {torch.cuda.device_count()}")
# print(f"Using device(s): {[i for i in range(torch.cuda.device_count())]}")

import os
import cv2
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
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image
import torch.nn.functional as F
from matplotlib import cm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import torch.nn as nn
# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Initialize enhancement model ---
# generator = Generator().to(device)
# generator.load_state_dict(torch.load("./saved_models/G/generator_795.pth", map_location=device))
# generator.eval()

# --- Enhancement function ---
# @torch.no_grad()
# def enhance_image_tensor(img_tensor):
#     """
#     img_tensor: [B, C, H, W] in [0,1]
#     returns enhanced image tensor [B, C, H, W]
#     """
#     return generator(img_tensor)[3]

# # --- Custom Dataset with on-the-fly enhancement ---
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
#                                     os.path.join(dataset_path, "train/labels"))
# val_dataset = EnhancedYOLODataset(os.path.join(dataset_path, "valid/images"),
#                                   os.path.join(dataset_path, "valid/labels"))
# test_dataset = EnhancedYOLODataset(os.path.join(dataset_path, "test/images"),
#                                 os.path.join(dataset_path, "test/labels"))

# # --- DataLoaders with multiple workers ---
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
#                           num_workers=8, pin_memory=True, prefetch_factor=2)
# val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False,
#                           num_workers=8, pin_memory=True, prefetch_factor=2)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
#                         num_workers=8, pin_memory=True, prefetch_factor=2)

# --- Initialize YOLO ---
# yolo_model = YOLO("yolo11n.pt")

# # --- Train YOLO using Dataloader ---
# yolo_model.train(
#     data=os.path.join(dataset_path, "data.yaml"),
#     epochs=50,
#     imgsz=640,
#     batch=128,
#     device="0,1,2,3",       # multiple GPUs
#     optimizer="AdamW",
#     project="yolo11_integrated_grad_cam",
#     name="custom-fish-detector",
#     exist_ok=True,
#     verbose=True,
#     augment=False           # don't use YOLO augment
# )

# --- Save trained model ---
models_dir = "Models"
# os.makedirs(models_dir, exist_ok=True)
# trained_model_path = "yolo11_integrated_grad_cam/custom-fish-detector/weights/best.pt"
# if os.path.exists(trained_model_path):
#     shutil.copy(trained_model_path, os.path.join(models_dir, "yolo11_integrated_grad_cam_best.pt"))

# Load YOLO model on correct device
model = YOLO(os.path.join(models_dir, "yolo11_integrated_grad_cam_best.pt"))
if torch.cuda.is_available():
    model.model.to(device)

# metrics = model.val(device=0, batch=64)

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
# --- Wrap YOLO so it returns a single tensor for Grad-CAM ---
class YOLOGradCAMWrapper(nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        self.model = yolo_model.model  # internal YOLO model (backbone+neck+head)

    def forward(self, x):
        outputs = []
        for i, m in enumerate(self.model.model):
            # YOLO models often concatenate feature maps from previous layers
            if m.f != -1:  # if it uses previous layer outputs
                x = [x if j == -1 else outputs[j] for j in m.f]
                x = m(x)
            else:
                x = m(x)
            outputs.append(x)
            # stop before detection head
            if i == len(self.model.model) - 2:
                break

        # return only the tensor (some layers output tuples)
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x
wrapped_model = YOLOGradCAMWrapper(model).to(device)
target_layers = [wrapped_model.model.model[-2]]  # second last conv block
cam = GradCAM(model=wrapped_model, target_layers=target_layers)

# --- Safe resize helper for YOLO (divisible by 32) ---
def resize_to_multiple_of_32(img):
    h, w = img.shape[:2]
    new_h = int(np.ceil(h / 32) * 32)
    new_w = int(np.ceil(w / 32) * 32)
    return cv2.resize(img, (new_w, new_h))

# --- Grad-CAM feature extraction ---
def get_object_crops_with_gradcam(img_path):
    results = model.predict(img_path, conf=0.25, verbose=False)
    crops, labels, heatmaps = [], [], []
    img = cv2.imread(img_path)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue  # skip invalid crop

            # Resize to multiple of 32
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            rgb_crop = resize_to_multiple_of_32(rgb_crop)
            rgb_crop_float = np.float32(rgb_crop) / 255.0

            input_tensor = preprocess_image(rgb_crop_float, mean=[0,0,0], std=[1,1,1]).to(device)

            try:
                grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            except Exception as e:
                print(f"⚠️ Grad-CAM failed on crop: {e}")
                continue

            heatmap = cv2.resize(grayscale_cam, (64, 64))
            heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            feature_vector = heatmap_norm.flatten()

            crops.append(crop)
            labels.append(cls)
            heatmaps.append(feature_vector)

    return crops, labels, heatmaps


features_path = "cached_gradcam_features.joblib"
labels_path = "cached_gradcam_labels.joblib"

# --- Feature Extraction with Caching ---
if os.path.exists(features_path) and os.path.exists(labels_path):
    print("📂 Loading cached Grad-CAM features...")
    all_features = joblib.load(features_path)
    all_labels = joblib.load(labels_path)
else:
    print("⚙️ Extracting Grad-CAM features (first run, may take time)...")
    all_features, all_labels = [], []
    train_img_dir = os.path.join(dataset_path, "train/images")

    for fname in os.listdir(train_img_dir):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        path = os.path.join(train_img_dir, fname)
        crops, labels, features = get_object_crops_with_gradcam(path)
        all_features.extend(features)
        all_labels.extend(labels)

    # Cache for future runs
    joblib.dump(all_features, features_path)
    joblib.dump(all_labels, labels_path)
    print(f"✅ Saved features to {features_path} and labels to {labels_path}")

# --- Visualization function ---
def visualize_gradcam_results(num_samples=5, save_dir="gradcam_viz"):
    os.makedirs(save_dir, exist_ok=True)
    img_dir = os.path.join(dataset_path, "train/images")
    sample_files = random.sample(
        [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))],
        num_samples
    )

    for fname in sample_files:
        img_path = os.path.join(img_dir, fname)
        crops, labels, heatmaps = get_object_crops_with_gradcam(img_path)
        original = cv2.imread(img_path)
        vis_img = original.copy()

        print(f"\n🖼️ Visualizing: {fname}")

        for i, (crop, label, feature_vector) in enumerate(zip(crops, labels, heatmaps)):
            # Convert back from flattened heatmap to 64x64
            heatmap_2d = feature_vector.reshape(64, 64)
            heatmap_resized = cv2.resize(heatmap_2d, (crop.shape[1], crop.shape[0]))
            heatmap_color = cm.jet(heatmap_resized)[:, :, :3]  # apply color map
            heatmap_overlay = (heatmap_color * 255).astype(np.uint8)

            # Overlay Grad-CAM on crop
            blended = cv2.addWeighted(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), 0.6, heatmap_overlay, 0.4, 0)

            # Paste back into main image
            results = model.predict(img_path, conf=0.25, verbose=False)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vis_img[y1:y2, x1:x2] = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

            # Save crop visualization
            save_path = os.path.join(save_dir, f"{os.path.splitext(fname)[0]}_obj{i}_label{label}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

        # Show full image with Grad-CAM overlays
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Grad-CAM Visualization: {fname}")
        plt.axis("off")
        plt.show()

# --- Run visualization for random images ---
visualize_gradcam_results(num_samples=5)

# --- Pad and Split ---
# L = max(len(f) for f in all_features)
# X = np.array([np.pad(f, (0, L - len(f)), constant_values=0) for f in all_features])
# y = np.array(all_labels)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# # Random Forest
# rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
# rf.fit(X_train, y_train)
# y_pred_rf = rf.predict(X_test)
# print("=== Random Forest Results ===")
# print(classification_report(y_test, y_pred_rf, target_names=class_names))

# # Save Random Forest model
# import joblib
# joblib.dump(rf, os.path.join(models_dir, "random_forest_yolo11_normal.pkl"))


# # SVM
# svm = SVC(kernel="linear", probability=True, random_state=42)
# svm.fit(X_train, y_train)
# y_pred_svm = svm.predict(X_test)
# print("=== SVM Results ===")
# print(classification_report(y_test, y_pred_svm, target_names=class_names))

# # Save SVM model
# joblib.dump(svm, os.path.join(models_dir, "svm_yolo11_normal.pkl"))

# # XGBoost
# xgb = XGBClassifier(
#     use_label_encoder=False,
#     eval_metric='mlogloss',
#     random_state=42,
#     n_jobs=-1
# )
# xgb.fit(X_train, y_train)
# y_pred_xgb = xgb.predict(X_test)
# print("=== XGBoost Results ===")
# print(classification_report(y_test, y_pred_xgb, target_names=class_names))

# # Save XGBoost model
# joblib.dump(xgb, os.path.join(models_dir, "xgboost_yolo11_normal.pkl"))

# # K-Nearest Neighbors
# knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
# knn.fit(X_train, y_train)
# y_pred_knn = knn.predict(X_test)
# print("=== KNN Results ===")
# print(classification_report(y_test, y_pred_knn, target_names=class_names))

# # Save KNN model
# joblib.dump(knn, os.path.join(models_dir, "knn_yolo11_normal.pkl"))

# # List of models and their predictions
# models = {
#     "Random Forest": y_pred_rf,
#     "SVM": y_pred_svm,
#     "XGBoost": y_pred_xgb,
#     "KNN": y_pred_knn
# }

# for name, y_pred in models.items():
#     print(f"\n===== {name} =====")
#     print(classification_report(y_test, y_pred, target_names=class_names))
#     acc = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {acc:.4f}")

#     # Confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(10,8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=class_names, yticklabels=class_names)
#     plt.title(f"{name} Confusion Matrix")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.xticks(rotation=45)
#     plt.yticks(rotation=0)
#     plt.show()

# test_img_dir = os.path.join(dataset_path, "test/images")

# test_files = [f for f in os.listdir(test_img_dir) if f.lower().endswith((".jpg",".png",".jpeg"))]
# for file in test_files:
#     img_path = os.path.join(test_img_dir, file)
#     results_list = model.predict(img_path, conf=0.25, save=False)

#     for result in results_list:
#         img_with_boxes = result.plot()
#         plt.figure(figsize=(6,6))
#         plt.imshow(img_with_boxes)
#         plt.axis("off")
#         plt.show()

# #Saving results in result folder
# results_dir = os.path.join(os.path.dirname(__file__), "..", "Results")
# os.makedirs(results_dir, exist_ok=True)

# # Save classification reports
# with open(os.path.join(results_dir, "random_forest_yolo11_normal_report.txt"), "w") as f:
#     f.write(classification_report(y_test, y_pred_rf, target_names=class_names))
# with open(os.path.join(results_dir, "svm_yolo11_normal_report.txt"), "w") as f:
#     f.write(classification_report(y_test, y_pred_svm, target_names=class_names))
# with open(os.path.join(results_dir, "xgboost_yolo11_normal_report.txt"), "w") as f:
#     f.write(classification_report(y_test, y_pred_xgb, target_names=class_names))
# with open(os.path.join(results_dir, "knn_yolo11_normal_report.txt"), "w") as f:
#     f.write(classification_report(y_test, y_pred_knn, target_names=class_names))

# # Save prediction images for only the first 5 test images
# for file in test_files[:5]:
#     img_path = os.path.join(test_img_dir, file)
#     results_list = model.predict(img_path, conf=0.25, save=False)

#     for idx, result in enumerate(results_list):
#         img_with_boxes = result.plot()
#         out_path = os.path.join(results_dir, f"{os.path.splitext(file)[0]}_result_yolo11_normal_{idx}.png")
#         cv2.imwrite(out_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
