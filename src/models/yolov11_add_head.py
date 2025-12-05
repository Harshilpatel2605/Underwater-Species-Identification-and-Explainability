import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

def get_yolo_heatmaps(model_path='yolo11n.pt', 
                      image_path='bus.jpg', 
                      conf_threshold=0.5):
    """
    Generates and displays EigenCAM heatmaps for all detections in an image
    using a YOLOv11 model.
    """
    
    # 1. Load the YOLOv11 model
    # This will download the model if not already present
    model = YOLO(model_path)
    
    # Get the internal PyTorch model
    # We need to access the underlying model architecture
    internal_model = model.model.model
    
    # 2. Find the target convolutional layer
    # We inspect the model's layers to find the last 'C2f' or 'Conv' block
    # in the backbone, which is a good target for heatmaps.
    target_layers = None
    target_layer_index = 15 # A common target layer index

    # --- START OF FIX ---
    # The previous loop was unreliable. We will try the direct index access first.
    
    if not target_layers:
        print(f"Could not find a suitable target layer by name. Using the default index '{target_layer_index}'.")
        # Fallback for a common YOLOv8/v11 backbone layer
        try:
            # FIX: Access by index 'internal_model[...]'
            # NOT 'internal_model.model[...]'
            target_layers = [internal_model[target_layer_index]] 
        except Exception as e:
            print(f"Error finding target layer at index {target_layer_index}: {e}")
            print("Please inspect `model.model.model` and set the target_layers manually.")
            return
    # --- END OF FIX ---

    print(f"--- Using Target Layer: {target_layers[0].__class__.__name__} ---")

    # 3. Load and preprocess the image
    rgb_img = np.array(Image.open(image_path).convert('RGB'))
    
    # --- ADDED A FIX HERE TOO for better results ---
    # Preprocess for YOLO and resize image to match
    input_tensor = model.preprocess(rgb_img, imgsz=640)[0] 
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).float()
    input_size = (input_tensor.shape[3], input_tensor.shape[2]) # (width, height)
    rgb_img = cv2.resize(rgb_img, input_size)
    
    # 4. Run inference to get detections
    # We need the model's raw output, not the pretty 'results' object
    # The 'augment=False' and 'profile=False' are standard for inference
    outputs = internal_model(input_tensor, augment=False, profile=False)
    
    # 5. Initialize EigenCAM
    cam = EigenCAM(model=internal_model,
                   target_layers=target_layers,
                   use_cuda=torch.cuda.is_available())
    
    # We target the raw output tensor
    targets = None

    # Generate the base heatmap
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets)[0, :]

    # 6. Post-process detections and display heatmaps
    # We use the standard 'model.postprocess()' to get clean bounding boxes
    results = model.postprocess(outputs, input_tensor, rgb_img.shape[:2])[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)

    # Filter by confidence threshold
    valid_indices = np.where(confs > conf_threshold)[0]
    
    if len(valid_indices) == 0:
        print(f"No objects detected with confidence > {conf_threshold}")
        # Show the raw heatmap for the whole image
        cam_image = show_cam_on_image(rgb_img / 255.0, grayscale_cam, use_rgb=True)
        plt.imshow(cam_image)
        plt.title("No detections - Raw EigenCAM")
        plt.axis('off')
        plt.show()
        return

    print(f"--- Found {len(valid_indices)} objects ---")

    # 7. Create a specific heatmap for each detected object
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    
    # This loop isolates the heatmap to just the detected bounding boxes
    for i in valid_indices:
        x1, y1, x2, y2 = map(int, boxes[i])
        # --- ADDED FIX: Boundary check ---
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
        
        # Take the heatmap data only from inside the bounding box
        box_heatmap = grayscale_cam[y1:y2, x1:x2]
        
        # Renormalize this specific area to be between 0 and 1
        # --- ADDED FIX: Safe normalization ---
        if box_heatmap.size > 0 and np.max(box_heatmap) > 0:
            box_heatmap = (box_heatmap - np.min(box_heatmap)) / (np.max(box_heatmap) + 1e-10)
            # Place it back into our blank image
            renormalized_cam[y1:y2, x1:x2] = box_heatmap

    # Overlay the box-specific heatmaps onto the original image
    cam_image = show_cam_on_image(rgb_img / 255.0, renormalized_cam, use_rgb=True)

    # 8. Draw the final bounding boxes and labels
    final_image = cam_image.copy()
    for i in valid_indices:
        x1, y1, x2, y2 = map(int, boxes[i])
        class_name = model.names[class_ids[i]]
        label = f"{class_name}: {confs[i]:.2f}"
        
        # Draw bounding box
        cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(final_image, (x1, y1 - h - 5), (x1 + w, y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(final_image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Display the result
    plt.figure(figsize=(12, 8))
    plt.imshow(final_image)
    plt.title("YOLO11 EigenCAM Heatmaps")
    plt.axis('off')
    plt.show()

# --- RUN THE FUNCTION ---

# Download a sample image (bus.jpg)
try:
    torch.hub.download_url_to_file('https://ultralytics.com/images/bus.jpg', 'bus.jpg')
except Exception as e:
    print(f"Could not download sample image: {e}")

# Generate heatmaps
# You can change 'yolo11n.pt' to 'yolo11s.pt', 'yolo11m.pt', etc.
get_yolo_heatmaps(model_path='yolo11n.pt', image_path='bus.jpg', conf_threshold=0.3)

# To run on your own 5 test images:
# my_images = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg"]
# for img_path in my_images:
#     print(f"\n--- Processing {img_path} ---")
#     get_yolo_heatmaps(model_path='yolo11n.pt', image_path=img_path, conf_threshold=0.3)