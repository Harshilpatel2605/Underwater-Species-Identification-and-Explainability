import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

def validate_model(model_path, data_yaml, img_size, batch_size, split):
    """
    Runs YOLO validation and returns the specified metrics.
    
    Args:
        model_path (str): Path to the trained YOLO model weights (.pt).
        data_yaml (str): Path to the dataset's data.yaml file.
        img_size (int): Image size to use for validation.
        batch_size (int): Batch size for validation.
        split (str): Dataset split to use (e.g., 'val' or 'test').
    """
    
    # --- 1. Load the Model ---
    try:
        model = YOLO(model_path)
        print(f"✅ Successfully loaded model from: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # --- 2. Run Validation ---
    print(f"🚀 Starting validation on '{data_yaml}' (split: '{split}')...")
    print(f"   Image Size: {img_size}px, Batch Size: {batch_size}")
    
    try:
        # The model.val() method runs validation and returns a metrics object
        metrics = model.val(
            data=data_yaml,
            imgsz=img_size,
            batch=batch_size,
            split=split,
            save_json=True,  # Recommended for mAP calculation
            verbose=True     # Set to False if you want less console output
        )
        return metrics
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        sys.exit(1)
def main():
    """
    Main function to parse arguments and run validation.
    """
    parser = argparse.ArgumentParser(description="Run YOLO validation and print key metrics.")
    
    # --- MODIFIED ARGUMENTS ---
    
    parser.add_argument(
        "--model", 
        type=str, 
        # MODIFIED: Use 'default' to set the path
        default="/home/harshil/Fish-Detection-Kaggle/yolov11_attention/custom-fish-detector/weights/best.pt",
        # MODIFIED: 'help' should be a description
        help="Path to the YOLO model weights file."
    )
    parser.add_argument(
        "--data", 
        type=str, 
        # MODIFIED: Use 'default' to set the path
        default="/home/harshil/Fish-Detection-Kaggle/Fish-Detection-Dataset/Fish-Detection-Kaggle/data.yaml",
        # MODIFIED: 'help' should be a description
        help="Path to the dataset's data.yaml file."
    )
    
    # --- END OF MODIFICATIONS ---

    # Optional Arguments (unchanged)
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=640, 
        help="Image size for validation (default: 640)."
    )
    parser.add_argument(
        "--batch", 
        type=int, 
        default=16, 
        help="Batch size for validation (default: 16)."
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="val", 
        help="The dataset split to validate on (e.g., 'val', 'test'). Default: 'val'."
    )
    
    args = parser.parse_args()

    # Run the validation function
    metrics = validate_model(args.model, args.data, args.imgsz, args.batch, args.split)

    # ... (rest of your code is fine) ...
    # --- 3. Print the Final Metrics ---
    if metrics:
        print("\n" + "="*30)
        print("📈 VALIDATION METRICS SUMMARY")
        print("="*30)
        
        # The 'metrics' object has a 'box' attribute for detection metrics
        box_metrics = metrics.box
        
        print(f"   Precision (P):     {box_metrics.precision:.4f}")
        print(f"   Recall (R):        {box_metrics.recall:.4f}")
        print(f"   mAP@0.5:           {box_metrics.map50:.4f}")
        print(f"   mAP@0.5-0.95:      {box_metrics.map:.4f}")
        
        print("\nValidation complete. Results and plots saved to 'runs/detect/val'.")
    else:
        print("Validation did not return any metrics.")
if __name__ == "__main__":
    main()