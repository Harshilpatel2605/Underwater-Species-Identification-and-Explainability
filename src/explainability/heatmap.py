import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

# Load YOLOv11 model (or any YOLOv8+ model)
model = YOLO("yolo11l.pt")  # or yolov8l.pt, yolov9l.pt, etc.

# Load image
image_path = "bus.jpg"  # Replace with your image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run inference
results = model(image, verbose=False)[0]

# COCO class names (80 classes)
class_names = model.names

# Initialize a blank heatmap for each class
heatmaps = {class_id: np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            for class_id in class_names}

# Extract detections and populate heatmaps
for det in results.boxes:
    x1, y1, x2, y2, conf, class_id = det.xyxy[0].tolist() + [det.conf[0].item()] + [int(det.cls[0].item())]
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    # Update heatmap for the detected class
    heatmaps[class_id][y1:y2, x1:x2] = np.maximum(
        heatmaps[class_id][y1:y2, x1:x2],
        conf  # confidence score
    )

# Normalize and apply colormap to each heatmap
colored_heatmaps = {}
for class_id in heatmaps:
    heatmap = heatmaps[class_id]
    if heatmap.max() > 0:  # Only process if detections exist
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
        colored_heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        colored_heatmaps[class_id] = colored_heatmap

# Overlay heatmaps on the original image
overlay = image_rgb.copy()
for class_id in colored_heatmaps:
    overlay = cv2.addWeighted(overlay, 0.7, colored_heatmaps[class_id], 0.3, 0)

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(overlay)
plt.title("Confidence Heatmaps (All Classes)")
plt.axis("off")

plt.tight_layout()
plt.show()

# Optionally, save the heatmap overlay
cv2.imwrite("heatmap_overlay.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))