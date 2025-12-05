#!/usr/bin/env python

# Check GPU availability
import torch
print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"Using device(s): {[i for i in range(torch.cuda.device_count())]}")

import os
import cv2
import torch
import shutil
import joblib
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

print("FOR PRE-PROCESSED DATA")

dataset_path = "/home/harshil/URPC2020/"
train_img_dir = os.path.join(dataset_path, "train/images")
valid_img_dir = os.path.join(dataset_path, "valid/images")
test_img_dir  = os.path.join(dataset_path, "test/images")

class_names = ['holothurian', 'echinus', 'scallop', 'starfish']

yolo_model = YOLO("yolo11l.pt")

yolo_model.train(
    data="/home/harshil/URPC2020/data.yaml",
    epochs=100,
    imgsz=640,
    batch=64,                     
    device="0,1,2,3,4,5,6,7",            
    optimizer="AdamW",
    project="yolo11_processed",
    name="custom-fish-detector",
    exist_ok=True,
    verbose=True
)

# --- Save trained model ---
models_dir = "Models"
os.makedirs(models_dir, exist_ok=True)
trained_model_path = "yolo11_processed/custom-fish-detector/weights/best.pt"
if os.path.exists(trained_model_path):
    shutil.copy(trained_model_path, os.path.join(models_dir, "yolo11_processed_best.pt"))

# Load YOLO model on correct device
model = YOLO(os.path.join(models_dir, "yolo11_processed_best.pt"))
if torch.cuda.is_available():
    model.model.to(device)

metrics = model.val(device=1, batch=64)


# metrics is a dict containing key performance values
print("\n📊 YOLO v11 Model Evaluation Metrics (Processed Data):")
print(f"Precision:       {metrics.box.p.mean():.4f}")
print(f"Recall:          {metrics.box.r.mean():.4f}")
print(f"mAP@0.5:         {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95:    {metrics.box.map:.4f}")

# Optional: print per-class metrics (if dataset has multiple classes)
try:
    class_names = model.names
    per_class_data = {
        "Class": class_names,
        "Precision": metrics.box.p.tolist(),
        "Recall": metrics.box.r.tolist(),
        "mAP@0.5": metrics.box.map50_per_class.tolist(),
        "mAP@0.5:0.95": metrics.box.map_per_class.tolist()
    }

    df = pd.DataFrame(per_class_data)
    print("\n📈 Per-Class Metrics:\n")
    print(df)
except Exception as e:
    print("\nSkipping per-class metrics (single class or incompatible metrics).")


# orb = cv2.ORB_create(nfeatures=2000)

# def get_object_crops(img_path):
#     results = model.predict(img_path, conf=0.25, verbose=False)
#     crops, labels = [], []
#     img = cv2.imread(img_path)
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             crop = img[y1:y2, x1:x2]
#             cls = int(box.cls[0])
#             crops.append(crop)
#             labels.append(cls)
#     return crops, labels

# all_features, all_labels = [], []

# for fname in os.listdir(train_img_dir):
#     if not fname.lower().endswith((".jpg",".png",".jpeg")):
#         continue
#     path = os.path.join(train_img_dir, fname)
#     crops, labels = get_object_crops(path)
#     for crop, label in zip(crops, labels):
#         gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#         kps, desc = orb.detectAndCompute(gray, None)
#         if desc is not None:
#             all_features.append(desc.flatten())
#             all_labels.append(label)

# # --- Pad features and split dataset ---
# L = max(len(f) for f in all_features)
# X = np.array([np.pad(f, (0, L - len(f)), constant_values=0) for f in all_features])
# y = np.array(all_labels)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# Random Forest
# rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
# rf.fit(X_train, y_train)
# y_pred_rf = rf.predict(X_test)
# print("=== Random Forest Results ===")
# print(classification_report(y_test, y_pred_rf, target_names=class_names))

# Save Random Forest model
# import joblib
# joblib.dump(rf, os.path.join(models_dir, "random_forest_yolo11_normal.pkl"))


# SVM
# svm = SVC(kernel="linear", probability=True, random_state=42)
# svm.fit(X_train, y_train)
# y_pred_svm = svm.predict(X_test)
# print("=== SVM Results ===")
# print(classification_report(y_test, y_pred_svm, target_names=class_names))

# Save SVM model
# joblib.dump(svm, os.path.join(models_dir, "svm_yolo11_normal.pkl"))

# XGBoost
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

# Save XGBoost model
# joblib.dump(xgb, os.path.join(models_dir, "xgboost_yolo11_normal.pkl"))

# K-Nearest Neighbors
# knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
# knn.fit(X_train, y_train)
# y_pred_knn = knn.predict(X_test)
# print("=== KNN Results ===")
# print(classification_report(y_test, y_pred_knn, target_names=class_names))

# Save KNN model
# joblib.dump(knn, os.path.join(models_dir, "knn_yolo11_normal.pkl"))

# List of models and their predictions
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

# Save classification reports
# with open(os.path.join(results_dir, "random_forest_yolo11_normal_report.txt"), "w") as f:
#     f.write(classification_report(y_test, y_pred_rf, target_names=class_names))
# with open(os.path.join(results_dir, "svm_yolo11_normal_report.txt"), "w") as f:
#     f.write(classification_report(y_test, y_pred_svm, target_names=class_names))
# with open(os.path.join(results_dir, "xgboost_yolo11_normal_report.txt"), "w") as f:
#     f.write(classification_report(y_test, y_pred_xgb, target_names=class_names))
# with open(os.path.join(results_dir, "knn_yolo11_normal_report.txt"), "w") as f:
#     f.write(classification_report(y_test, y_pred_knn, target_names=class_names))

# Save prediction images for only the first 5 test images
# for file in test_files[:5]:
#     img_path = os.path.join(test_img_dir, file)
#     results_list = model.predict(img_path, conf=0.25, save=False)

#     for idx, result in enumerate(results_list):
#         img_with_boxes = result.plot()
#         out_path = os.path.join(results_dir, f"{os.path.splitext(file)[0]}_result_yolo11_normal_{idx}.png")
#         cv2.imwrite(out_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
