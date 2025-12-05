from ultralytics import YOLO

# 1. Load your pre-trained model
# Replace 'path/to/best.pt' with the actual path to your weight file
model = YOLO('path/to/your_trained_weights.pt') 

# 2. Run inference on the video
# source: path to your URPC2020 video file
# save=True: saves the video with bounding boxes drawn
# conf=0.5: acts as a filter, only showing detections with >50% confidence
results = model.predict(
    source='path/to/your_video.mp4', 
    save=True, 
    conf=0.25,  # URPC data can be blurry; you might need to lower confidence
    imgsz=640
)

print("Inference complete. Check the 'runs' folder.")