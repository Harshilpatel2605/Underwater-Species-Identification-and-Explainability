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

# NOTE: No 'ObjectDetectionTarget' or 'HiResCAM' imports are needed.

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
        
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

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
        return
    
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
    
    for i in valid_indices:
        x1, y1, x2, y2 = map(int, boxes.xyxy.cpu().numpy()[i])
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, rgb_img.shape[1])
        y2 = min(y2, rgb_img.shape[0])
        
        if x1 >= x2 or y1 >= y2:
            continue
            
        box_cam = grayscale_cam_resized[y1:y2, x1:x2].copy()
        
        if box_cam.size > 0:
            if np.max(box_cam) > np.min(box_cam):
                box_cam_normalized = (box_cam - np.min(box_cam)) / (np.max(box_cam) - np.min(box_cam))
            else:
                box_cam_normalized = np.zeros_like(box_cam)
            
            focused_cam[y1:y2, x1:x2] = np.maximum(focused_cam[y1:y2, x1:x2], box_cam_normalized)
    
    # 9. Blend the focused heatmap with the background heatmap
    final_cam = np.zeros_like(grayscale_cam_resized, dtype=np.float32)
    
    box_mask = np.zeros_like(grayscale_cam_resized, dtype=bool)
    for i in valid_indices:
        x1, y1, x2, y2 = map(int, boxes.xyxy.cpu().numpy()[i])
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, rgb_img.shape[1])
        y2 = min(y2, rgb_img.shape[0])
        if x1 < x2 and y1 < y2:
            box_mask[y1:y2, x1:x2] = True
            
    grayscale_cam_normalized = (grayscale_cam_resized - np.min(grayscale_cam_resized)) / (np.max(grayscale_cam_resized) - np.min(grayscale_cam_resized))

    final_cam[box_mask] = focused_cam[box_mask]
    final_cam[~box_mask] = grayscale_cam_normalized[~box_mask] * 0.4
    
    # 10. Overlay the final heatmap onto the original image
    cam_image = show_cam_on_image(rgb_img / 255.0, final_cam, use_rgb=True)

    # 11. Draw bounding boxes, labels, and save the image
    final_image = cam_image.copy()
    for i in valid_indices:
        x1, y1, x2, y2 = map(int, boxes.xyxy.cpu().numpy()[i])
        
        class_name = model.names[class_ids[i]] 
        label = f"{class_name}: {confs[i]:.2f}"
        
        cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(final_image, (x1, y1 - h - 5), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(final_image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Display the result
    plt.figure(figsize=(12, 8))
    plt.imshow(final_image)
    plt.title(f"YOLO EigenCAM Heatmaps (Model: yolov11 baseline)")
    plt.axis('off')
    
    # --- MODIFIED SAVE PATH LOGIC ---
    
    # Get the directory where this script is located
    # __file__ is a special variable that holds the path to the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the base filename from the input image path
    base_name = os.path.basename(image_path)
    
    # Get the filename without the extension
    file_name_without_ext = os.path.splitext(base_name)[0]
    
    # Create the output filename
    output_filename = f"{file_name_without_ext}_eigencam_baseline.png"
    
    # Join the script's directory with the new filename to get the full save path
    output_path = os.path.join(script_dir, output_filename)
    
    # --- END OF MODIFICATION ---
    
    plt.savefig(output_path)  # Save to the new, full path
    print(f"--- Saved heatmap to: {output_path} ---") # Print the full path
    plt.show()

# --- RUN THE FUNCTION ---

# 1. Set the path to your image
my_custom_image_path = "/home/harshil/bus.jpg" 

# 2. Set the path to your CUSTOM-TRAINED model
my_model_path = "/home/harshil/yolo11l.pt"

# 3. Generate heatmaps
get_yolo_heatmaps(image_path=my_custom_image_path, 
                  model_path=my_model_path, 
                  conf_threshold=0.3)