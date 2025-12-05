#!/usr/bin/env python

import os
import cv2
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.discriminant_analysis import StandardScaler
import random # <--- Import the random module

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = "Models"
DATASET_PATH = "/home/harshil/URPC2020/"
TEST_IMG_DIR = os.path.join(DATASET_PATH, "test/images")
TEST_LABEL_DIR = os.path.join(DATASET_PATH, "test/labels")
OUTPUT_DIR = "test_pipeline_results"
CLASS_NAMES = ['holothurian', 'echinus', 'scallop', 'starfish']

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

# --- 1. Load All Trained Models ---

# Load YOLOv8 model for feature extraction
print("Loading YOLO model...")
yolo_model = YOLO(os.path.join(MODELS_DIR, "yolo11_integrated_yolo_best.pt"))
yolo_model.model.to(DEVICE).eval()
print("✅ YOLO model loaded.")

# Load the traditional classifiers and the scaler
print("Loading classifiers and scaler...")
try:
    svm = joblib.load(os.path.join(MODELS_DIR, "svm_yolo11_normal.pkl"))
    rf = joblib.load(os.path.join(MODELS_DIR, "random_forest_yolo11_normal.pkl"))
    xgb = joblib.load(os.path.join(MODELS_DIR, "xgboost_yolo11_normal.pkl"))
    knn = joblib.load(os.path.join(MODELS_DIR, "knn_yolo11_normal.pkl"))

    # IMPORTANT: The scaler must be loaded from the training phase.
    # We will create a dummy one here, but you should save and load your actual scaler.
    # Assuming X.npy exists from your training script to fit the scaler.
    X_train_full = np.load("X.npy", allow_pickle=True)
    scaler = StandardScaler().fit(X_train_full)
    
    classifiers = {
        "SVM": svm,
        "RandomForest": rf,
        "XGBoost": xgb,
        "KNN": knn
    }
    print("✅ Classifiers and scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error loading models: {e}. Please ensure all model files and X.npy exist.")
    exit()


# --- 2. Set up Feature Extraction Hook ---
features_list = []
def hook_fn(module, input, output):
    """Hook to capture the output of a specific layer."""
    output = output.to("cpu")
    features_list.append(output.detach())

# Attach the hook to the desired layer (e.g., the last layer before the detection head)
feature_layer = yolo_model.model.model[-3]
hook_handle = feature_layer.register_forward_hook(hook_fn)
print(f"✅ Hook attached to layer: {feature_layer.__class__.__name__}")


# --- 3. Define the Core Pipeline Functions ---

def extract_features_from_crop(image_crop):
    """
    Takes an image crop (numpy array), preprocesses it, and extracts features using the YOLO backbone.
    """
    features_list.clear() # Clear list for each new extraction

    # Preprocess the crop: resize, convert to tensor, normalize
    img_resized = cv2.resize(image_crop, (640, 640))
    if img_resized.ndim == 2: # Handle grayscale if it occurs
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0

    # Forward pass through the model to trigger the hook
    with torch.no_grad():
        _ = yolo_model.model(img_tensor)

    if not features_list:
        return None

    # Process captured features: global average pooling
    features = features_list[0]
    features_flat = torch.mean(features, dim=(2, 3)).squeeze(0).cpu().numpy()
    return features_flat

def get_ground_truth_labels(image_filename):
    """Reads the YOLO label file to get ground truth boxes and classes."""
    label_path = os.path.join(TEST_LABEL_DIR, image_filename.replace(".jpg", ".txt").replace(".png", ".txt"))
    ground_truths = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                # YOLO format [class, x_center, y_center, width, height]
                ground_truths.append({'class_id': class_id, 'box_yolo': [float(p) for p in parts[1:]]})
    return ground_truths


# --- 4. Main Testing and Visualization Loop ---

def run_testing_pipeline(image_path, classifier_name="RandomForest"):
    """
    Processes a single image: detects objects, classifies them, and visualizes the result.
    """
    if classifier_name not in classifiers:
        print(f"❌ Classifier '{classifier_name}' not found. Available: {list(classifiers.keys())}")
        return

    classifier = classifiers[classifier_name]
    
    # Load the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Could not read image: {image_path}")
        return
    
    vis_image = original_image.copy() # Create a copy for drawing
    H, W, _ = vis_image.shape

    # --- Step 1: Get object detections from YOLO ---
    results = yolo_model.predict(image_path, verbose=False)
    
    # --- Step 2: Get ground truth labels ---
    gt_labels = get_ground_truth_labels(os.path.basename(image_path))
    
    # --- Step 3: Process each detected box ---
    for box in results[0].boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
        
        # --- Step 3a: Crop the detected object ---
        cropped_object = original_image[y1:y2, x1:x2]

        if cropped_object.size == 0:
            continue

        # --- Step 3b: Extract features from the crop ---
        features = extract_features_from_crop(cropped_object)
        
        if features is None:
            continue
            
        # --- Step 3c: Scale features and predict with classifier ---
        scaled_features = scaler.transform(features.reshape(1, -1))
        predicted_class_id = classifier.predict(scaled_features)[0]
        predicted_label = CLASS_NAMES[predicted_class_id]

        # --- Step 3d: Find the corresponding ground truth label ---
        real_label = "N/A"
        if gt_labels:
            # This is a simplification. A real implementation would use IoU to match the detected box to a GT box.
            real_label = CLASS_NAMES[gt_labels[0]['class_id']]

        # --- Step 4: Draw results on the image ---
        label_text = f"Pred: {predicted_label} | Real: {real_label}"
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save the final image
    output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(image_path))[0]}_{classifier_name}_result.png")
    cv2.imwrite(output_path, vis_image)
    print(f"✅ Result saved to {output_path}")

# ------------------------------------------------------------------
#  MODIFIED SECTION
# ------------------------------------------------------------------
if __name__ == "__main__":
    test_images = [os.path.join(TEST_IMG_DIR, f) for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Get a list of the names of your available classifiers
    classifier_names = list(classifiers.keys())

    print(f"\n🚀 Starting randomized testing on the first 5 images...")
    
    # Run the pipeline on the first 5 test images
    for img_path in test_images[:5]:
        # For each image, pick a classifier at random! 🎲
        chosen_classifier = random.choice(classifier_names)
        
        print("-" * 30)
        print(f"Processing {os.path.basename(img_path)} with ==> {chosen_classifier}")
        
        run_testing_pipeline(img_path, classifier_name=chosen_classifier)

    # Don't forget to remove the hook when you're done to avoid memory leaks
    hook_handle.remove()
    print("-" * 30)
    print("\n✅ Testing complete.")