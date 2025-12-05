import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
import os  # <-- Make sure os is imported

from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics.utils.checks import check_imgsz


class YOLOCAMWrapper(nn.Module):
    """Wrap a YOLO DetectionModel so CAM receives a tensor output."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):  # type: ignore[override]
        outputs = self.model(x)
        if isinstance(outputs, (list, tuple)):
            # The first element is the [batch, 84, 8400] output tensor
            return outputs[0]
        return outputs

# --- NEW HELPER FUNCTION ---
def get_ground_truth_boxes(image_path, class_names_dict):
    """Finds and parses the YOLO label file for a given image path."""
    # 1. Get the base image filename and extension
    base_name = os.path.basename(image_path)
    extension = os.path.splitext(base_name)[1] # e.g., ".jpg"
    
    # 2. Construct the label path
    # Assumes a standard YOLO structure: .../images/img.jpg -> .../labels/img.txt
    label_path = image_path.replace("/images/", "/labels/").replace(extension, ".txt")
    
    # 3. Check if the label file exists
    if not os.path.exists(label_path):
        print(f"--- Warning: No ground truth label file found at: {label_path}")
        return []
        
    # 4. Read and parse the label file
    boxes_norm = []
    try:
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"--- Warning: Skipping malformed line in {label_path}: {line.strip()}")
                    continue # Skip malformed lines
                
                class_id = int(parts[0])
                x_center_norm = float(parts[1])
                y_center_norm = float(parts[2])
                width_norm = float(parts[3])
                height_norm = float(parts[4])
                
                class_name = class_names_dict.get(class_id, "Unknown")
                boxes_norm.append((class_name, x_center_norm, y_center_norm, width_norm, height_norm))
    except Exception as e:
        print(f"--- Error reading label file {label_path}: {e}")
        return []
        
    return boxes_norm
# --- END OF NEW FUNCTION ---


def get_yolo_heatmaps(image_path: str,
                      model_path='yolo11l.pt',
                      conf_threshold=0.3):
    """
    Generates and displays EigenCAM heatmaps for all detections in an image
    using a YOLOv11 model.
    """
    
    # 1. Load the YOLO model
    print(f"--- Loading model from: {model_path} ---")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please ensure the model_path is correct.")
        return
        
    # Get the internal PyTorch model (DetectionModel)
    internal_model = model.model
    
    # 2. Find the best convolutional layers for CAM (Your 70-95% strategy)
    conv_layers = []
    for name, module in internal_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            conv_layers.append((name, module))
    
    if not conv_layers:
        print("Could not find any Conv2d layers for CAM.")
        return
    
    print(f"--- Found {len(conv_layers)} Conv2d layers ---")
    
    # Strategy: Use layers from 70-95% through the network
    total_layers = len(conv_layers)
    start_idx = int(total_layers * 0.70)
    end_idx = int(total_layers * 0.95)
    
    if end_idx - start_idx >= 3:
        step = (end_idx - start_idx) // 3
        selected_indices = [start_idx, start_idx + step, end_idx - 1]
    elif end_idx - start_idx >= 2:
        selected_indices = [start_idx, end_idx - 1]
    else:
        selected_indices = [start_idx] if start_idx < total_layers else [total_layers - 1]
    
    selected_indices = sorted(list(set([min(i, total_layers - 1) for i in selected_indices])))
    
    target_layers = [conv_layers[i][1] for i in selected_indices]
    layer_names = [conv_layers[i][0] for i in selected_indices]
    
    print(f"--- Using {len(selected_indices)} optimized layers (70-95% depth): {layer_names} ---")

    # 3. Run standard YOLO prediction once
    print(f"--- Running prediction on: {image_path} ---")
    results = model.predict(image_path,
                            conf=conf_threshold,
                            save=False,
                            verbose=False)
    if not results:
        print("No predictions returned by YOLO model.")
        return
    detection_result = results[0] # Get the first (and only) result object

    # 4. Load the image and prepare tensors
    try:
        rgb_img = np.array(Image.open(image_path).convert('RGB'))
    except FileNotFoundError:
        print(f"Error: Image not found at path: {image_path}")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # --- MODIFICATION: Get image dimensions and GT boxes ---
    img_h, img_w = rgb_img.shape[:2] # Get image dimensions for denormalization
        
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    # 4b. Load Ground Truth Labels
    print(f"--- Loading ground truth labels for: {image_path} ---")
    gt_boxes_norm = get_ground_truth_boxes(image_path, model.names)
    print(f"--- Found {len(gt_boxes_norm)} ground truth objects ---")
    # --- END OF MODIFICATION ---

    predictor = model.predictor
    if predictor is None:
        raise RuntimeError("YOLO predictor is not initialized.")

    if getattr(predictor, "imgsz", None) is None:
        predictor.imgsz = check_imgsz(predictor.args.imgsz, stride=predictor.model.stride, min_dim=2)

    device = predictor.device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor.device = device
    internal_model.to(device)
    internal_model.eval()

    cam_model = YOLOCAMWrapper(internal_model)

    input_tensor = predictor.preprocess([bgr_img])
    input_tensor = input_tensor.to(device)

    # 5. Get detection information first
    boxes = detection_result.boxes
    confs = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    valid_indices = np.where(confs > conf_threshold)[0]
    
    if len(valid_indices) == 0:
        print(f"No objects detected with confidence > {conf_threshold}.")
    else:
        print(f"--- Found {len(valid_indices)} objects with confidence > {conf_threshold} ---")
    
    # 6. Initialize EigenCAM (gradient-free)
    cam = EigenCAM(model=cam_model,
                   target_layers=target_layers)

    # 7. Generate EigenCAM heatmap (EigenCAM works without targets)
    print("--- Generating EigenCAM heatmap from best layers ---")
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    
    # Resize to original image resolution
    grayscale_cam_resized = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))
    
    # 8. THE CRITICAL STEP: Create focused heatmaps (your local normalization)
    focused_cam = np.zeros_like(grayscale_cam_resized, dtype=np.float32)
    
    # Create mask for *all* boxes (both GT and Pred) for better blending
    all_box_mask = np.zeros_like(grayscale_cam_resized, dtype=bool)

    # --- MODIFICATION: Use *all* boxes for heatmap focusing ---
    
    # Get prediction boxes (xyxy)
    pred_boxes_xyxy = [map(int, boxes.xyxy.cpu().numpy()[i]) for i in valid_indices]
    
    # Get GT boxes (convert from normalized)
    gt_boxes_xyxy = []
    for _, x_c, y_c, w_n, h_n in gt_boxes_norm:
        x_center = x_c * img_w
        y_center = y_c * img_h
        box_width = w_n * img_w
        box_height = h_n * img_h
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)
        gt_boxes_xyxy.append((x1, y1, x2, y2))
        
    # Combine all boxes to create the focus mask
    for x1, y1, x2, y2 in pred_boxes_xyxy + gt_boxes_xyxy:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(x2, img_w), min(y2, img_h)
        if x1 < x2 and y1 < y2:
            all_box_mask[y1:y2, x1:x2] = True
            
            box_cam = grayscale_cam_resized[y1:y2, x1:x2].copy()
            if box_cam.size > 0:
                if np.max(box_cam) > np.min(box_cam):
                    box_cam_normalized = (box_cam - np.min(box_cam)) / (np.max(box_cam) - np.min(box_cam))
                else:
                    box_cam_normalized = np.zeros_like(box_cam)
                focused_cam[y1:y2, x1:x2] = np.maximum(focused_cam[y1:y2, x1:x2], box_cam_normalized)
    
    # 9. Blend the focused heatmap with the background heatmap
    final_cam = np.zeros_like(grayscale_cam_resized, dtype=np.float32)
    grayscale_cam_normalized = (grayscale_cam_resized - np.min(grayscale_cam_resized)) / (np.max(grayscale_cam_resized) - np.min(grayscale_cam_resized))

    # Inside boxes, use the focused map. Outside, use a faded general map.
    final_cam[all_box_mask] = focused_cam[all_box_mask]
    final_cam[~all_box_mask] = grayscale_cam_normalized[~all_box_mask] * 0.4 # Faded background
    
    # 10. Overlay the final heatmap onto the original image
    cam_image = show_cam_on_image(rgb_img / 255.0, final_cam, use_rgb=True)

    # 11. Draw bounding boxes, labels, and save the image
    final_image = cam_image.copy()
    
    # --- DRAW PREDICTIONS (Green) ---
    pred_color = (0, 255, 0) # Green in BGR
    for i in valid_indices:
        x1, y1, x2, y2 = map(int, boxes.xyxy.cpu().numpy()[i])
        class_name = model.names[class_ids[i]] 
        label = f"{class_name}: {confs[i]:.2f}"
        
        cv2.rectangle(final_image, (x1, y1), (x2, y2), pred_color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(final_image, (x1, y1 - h - 5), (x1 + w, y1), pred_color, -1)
        cv2.putText(final_image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2) # Black text

    # --- DRAW GROUND TRUTH (Red) ---
    gt_color = (0, 0, 255) # Red in BGR
    text_color = (255, 255, 255) # White text
    for i, (x1, y1, x2, y2) in enumerate(gt_boxes_xyxy):
        class_name = gt_boxes_norm[i][0]
        label = f"GT: {class_name}"
        
        cv2.rectangle(final_image, (x1, y1), (x2, y2), gt_color, 2)
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Place GT label *below* the box to avoid clashes with pred labels
        cv2.rectangle(final_image, (x1, y2 + 5), (x1 + w_text, y2 + 5 + h_text), gt_color, -1)
        cv2.putText(final_image, label, (x1, y2 + h_text + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
    
    # --- END OF DRAWING MODIFICATIONS ---

    # Display the result
    plt.figure(figsize=(12, 8))
    plt.imshow(final_image)
    plt.title(f"YOLO EigenCAM (Model: {os.path.basename(model_path)}) - Green=Pred, Blue=GT")
    plt.axis('off')
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the base filename from the input image path
    base_name = os.path.basename(image_path)
    
    # Get the filename without the extension
    file_name_without_ext = os.path.splitext(base_name)[0]
    
    # Create the output filename
    # --- MODIFIED: Removed 'attention' suffix for clarity ---
    output_filename = f"{file_name_without_ext}_eigencam_attention.png" 
    
    # Join the script's directory with the new filename to get the full save path
    output_path = os.path.join(script_dir, output_filename)
    
    plt.savefig(output_path)  # Save to the new, full path
    print(f"--- Saved heatmap to: {output_path} ---") # Print the full path
    plt.show()

# --- RUN THE FUNCTION ---

# 1. Set the path to your image
my_custom_image_path = "/home/harshil/URPC2020/test/images/000797.jpg"

# 2. Set the path to your CUSTOM-TRAINED model
my_model_path = "/home/harshil/Models/yolov11_attention.pt"

# 3. Generate heatmaps
get_yolo_heatmaps(image_path=my_custom_image_path, 
                  model_path=my_model_path, 
                  conf_threshold=0.3)