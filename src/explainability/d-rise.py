import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
MODEL_PATH = 'yolo11l.pt'
IMAGE_PATH = 'bus.jpg'
OUTPUT_DIR = 'd_rise_output'

# D-RISE Parameters
N_SAMPLES = 200  # Number of random masks (balance between quality and speed)
SPARSITY = 0.2   # Percentage of pixels kept in each mask (0.2 = 20%)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONF_THRESHOLD = 0.3  # Minimum confidence to consider a detection
# ---------------------

class YOLODRISE:
    """
    Robust D-RISE (Detection RISE) implementation for YOLO models.
    Handles all edge cases and ensures compatibility with latest Ultralytics versions.
    """
    def __init__(self, model_path, device=DEVICE):
        self.device = torch.device(device)
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Get model input size
        self.input_size = self.model.overrides['imgsz']
        if isinstance(self.input_size, int):
            self.input_size = (self.input_size, self.input_size)

    def _preprocess_image(self, image_path):
        """Load and preprocess image safely"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read image at {image_path}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img, img_rgb
        except Exception as e:
            raise RuntimeError(f"Image preprocessing failed: {str(e)}")

    def _generate_mask(self, shape, sparsity):
        """Generate random binary mask with given sparsity"""
        try:
            mask = np.random.choice([0, 1], size=shape, p=[1-sparsity, sparsity])
            return torch.from_numpy(mask).float().to(self.device)
        except Exception as e:
            raise RuntimeError(f"Mask generation failed: {str(e)}")

    def _get_prediction_score(self, img_tensor, target_class_id):
        """Get maximum confidence score for target class"""
        try:
            with torch.no_grad():
                # Ensure tensor is in correct format [B, C, H, W]
                if len(img_tensor.shape) == 3:
                    img_tensor = img_tensor.unsqueeze(0)

                results = self.model(img_tensor, verbose=False)

                if not hasattr(results[0], 'boxes') or results[0].boxes is None:
                    return torch.tensor(0.0).to(self.device)

                detections = results[0].boxes
                target_boxes = detections[detections.cls == target_class_id]

                if len(target_boxes) == 0:
                    return torch.tensor(0.0).to(self.device)

                return target_boxes.conf.max().to(self.device)

        except Exception as e:
            print(f"Warning: Prediction score calculation failed: {str(e)}")
            return torch.tensor(0.0).to(self.device)

    def _get_baseline_results(self, image_path):
        """Get baseline detection results"""
        try:
            results = self.model(image_path, verbose=False)

            # Filter detections by confidence threshold
            detections = results[0].boxes
            if detections is not None:
                detections = detections[detections.conf >= CONF_THRESHOLD]

            detected_classes = []
            if detections is not None and len(detections) > 0:
                detected_classes = torch.unique(detections.cls).cpu().numpy()

            return results, detected_classes
        except Exception as e:
            raise RuntimeError(f"Failed to get baseline results: {str(e)}")

    def visualize(self, image_path, n_samples=N_SAMPLES, sparsity=SPARSITY):
        """
        Generates D-RISE heatmaps for all detected classes.
        """
        try:
            print(f"Starting D-RISE analysis on {image_path} with {n_samples} samples...")

            # 1. Load and preprocess image
            img_bgr, img_rgb = self._preprocess_image(image_path)
            H, W = img_rgb.shape[:2]

            # 2. Get baseline results
            base_results, detected_classes = self._get_baseline_results(image_path)
            if len(detected_classes) == 0:
                print("No detections found above confidence threshold. Exiting.")
                return

            class_names = self.model.names

            # 3. Get original image tensor from results
            # Convert numpy array to tensor and preprocess
            orig_img = base_results[0].orig_img
            if isinstance(orig_img, np.ndarray):
                orig_img = torch.from_numpy(orig_img).permute(2, 0, 1).float() / 255.0
            orig_img_tensor = orig_img.to(self.device)

            # 4. Calculate baseline scores
            class_scores_base = {}
            for cls_id in detected_classes:
                class_scores_base[cls_id] = self._get_prediction_score(orig_img_tensor.unsqueeze(0), cls_id)

            # 5. Initialize importance maps
            feature_map_h, feature_map_w = self.input_size
            importance_map = torch.zeros((feature_map_h, feature_map_w)).to(self.device)
            coverage_map = torch.zeros((feature_map_h, feature_map_w)).to(self.device)

            # 6. Generate masks and calculate perturbed scores
            print("Processing masks...")
            for _ in tqdm(range(n_samples), desc="Processing"):
                try:
                    # Generate mask
                    mask = self._generate_mask((feature_map_h, feature_map_w), sparsity)

                    # Apply mask to the input tensor
                    masked_tensor = orig_img_tensor.unsqueeze(0) * mask.unsqueeze(0).unsqueeze(0)

                    # Calculate scores for all classes under this mask
                    for cls_id in detected_classes:
                        score_masked = self._get_prediction_score(masked_tensor, cls_id)
                        importance = class_scores_base[cls_id] - score_masked
                        importance_map += importance * mask
                        coverage_map += mask
                except Exception as e:
                    print(f"Warning: Error processing mask: {str(e)}")
                    continue

            # 7. Final map calculation
            coverage_map = torch.clamp(coverage_map, min=1e-6)
            d_rise_map = importance_map / coverage_map

            # Resize to original image dimensions
            d_rise_map_np = d_rise_map.cpu().numpy()
            d_rise_resized = cv2.resize(d_rise_map_np, (W, H))

            # Normalize for visualization
            d_rise_norm = (d_rise_resized - d_rise_resized.min()) / (d_rise_resized.max() - d_rise_resized.min() + 1e-8)

            # 8. Visualization
            self._plot_results(img_rgb, d_rise_norm, base_results, detected_classes, OUTPUT_DIR)

            print(f"\n✅ D-RISE analysis completed successfully! Results saved to {OUTPUT_DIR}")

        except Exception as e:
            print(f"\n❌ D-RISE Execution Failed: {str(e)}")
            raise

    def _plot_results(self, original_img_rgb, heatmap_norm, results, detected_classes, output_path):
        """Create and save visualizations"""
        try:
            output_dir = Path(output_path)
            output_dir.mkdir(exist_ok=True, parents=True)

            # Prepare the base image with detections
            annotated_img_bgr = results[0].plot()
            annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)

            # Create figure
            fig, axes = plt.subplots(1, len(detected_classes) + 1,
                                   figsize=(4 * (len(detected_classes) + 1), 8))

            if len(detected_classes) == 0:
                axes = [axes]
            else:
                axes = axes.flatten()

            # Plot 1: Baseline Detections
            axes[0].imshow(annotated_img_rgb)
            axes[0].set_title("Original Detections", fontsize=10)
            axes[0].axis('off')

            # Plot heatmaps for each detected class
            for idx, cls_id in enumerate(detected_classes):
                class_name = self.model.names[cls_id]
                cls_boxes = results[0].boxes[results[0].boxes.cls == cls_id]

                # Create overlay
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                overlay = cv2.addWeighted(original_img_rgb, 0.5, heatmap_colored, 0.5, 0)

                # Draw bounding boxes
                for box in cls_boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    overlay = cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Plot
                ax_idx = idx + 1
                axes[ax_idx].imshow(overlay)
                axes[ax_idx].set_title(f"{class_name} Heatmap", fontsize=9)
                axes[ax_idx].axis('off')

                # Save individual heatmap
                plt.tight_layout()
                plt.savefig(output_dir / f"d_rise_{class_name}.jpg", dpi=150, bbox_inches='tight')

            # Save combined visualization
            plt.tight_layout()
            plt.suptitle(f"D-RISE Explanation ({N_SAMPLES} Samples)", fontsize=14)
            plt.savefig(output_dir / 'd_rise_combined.jpg', dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Warning: Error during visualization: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        # Check if files exist
        if not Path(IMAGE_PATH).exists():
            raise FileNotFoundError(f"Image file not found: {IMAGE_PATH}")
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        # Create output directory
        Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

        # Run D-RISE analysis
        drise_analyzer = YOLODRISE(MODEL_PATH)
        drise_analyzer.visualize(IMAGE_PATH)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPlease ensure:")
        print("1. You have bus.jpg and yolo11l.pt in the current directory")
        print("2. All required packages are installed (ultralytics, torch, opencv-python, matplotlib, tqdm)")
        print("3. Your GPU drivers are up to date if using CUDA")