"""
GradCAM Visualization for YOLO Models
Generates heatmaps showing which regions the model focuses on for detections
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class YOLOGradCAM:
    """
    GradCAM implementation for YOLO models
    """
    
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize YOLO GradCAM
        
        Args:
            model_path: Path to YOLO model (.pt file)
            device: Device to run model on
        """
        self.device = device
        self.model = YOLO(model_path)
        self.model.model.to(self.device)
        self.model.model.eval()
        
        # Storage for gradients and activations
        self.gradients = None
        self.activations = None
        
        # Find the target layer (last Conv layer before detection head)
        self.target_layer = self._find_target_layer()
        
        # Register hooks
        self._register_hooks()
    
    def _find_target_layer(self):
        """
        Find the last convolutional layer before detection heads
        Usually layer 9 or 10 in YOLO11
        """
        try:
            # For YOLO11, typically the last conv layer before heads
            # You can adjust this based on model architecture
            target_layer = self.model.model.model[-3]  # -3 is usually the last conv layer
            print(f"Target layer: {target_layer}")
            return target_layer
        except Exception as e:
            print(f"Error finding target layer: {e}")
            # Fallback to model[9] which is common
            return self.model.model.model[9]
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_heatmap(self, image_path, conf_threshold=0.25):
        """
        Generate GradCAM heatmap for detections
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            
        Returns:
            heatmaps: List of heatmaps for each detection
            results: YOLO detection results
        """
        # Read and preprocess image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_img = img_rgb.copy()
        
        # Get predictions with gradient tracking
        self.model.model.train()  # Set to train mode to enable gradients
        
        # Prepare input
        results = self.model(image_path, verbose=False)
        
        # Get detections
        detections = results[0].boxes
        
        heatmaps = []
        detection_info = []
        
        if len(detections) == 0:
            print("No detections found!")
            return heatmaps, detection_info, original_img
        
        # Process each detection
        for idx, det in enumerate(detections):
            conf = float(det.conf[0])
            if conf < conf_threshold:
                continue
            
            cls = int(det.cls[0])
            class_name = self.model.names[cls]
            
            # Get detection score for this class
            # We need to backpropagate from the detection score
            try:
                # Forward pass with gradient
                with torch.enable_grad():
                    # Re-run inference with gradients
                    pred = self.model.model(results[0].orig_img)
                    
                    # Get the maximum score for this detection
                    # This is a simplified approach
                    if self.activations is not None:
                        # Create a pseudo-loss from the activation map
                        score = self.activations.mean()
                        score.backward()
                        
                        if self.gradients is not None:
                            # Generate heatmap
                            weights = self.gradients.mean(dim=[2, 3], keepdim=True)
                            cam = (weights * self.activations).sum(dim=1, keepdim=True)
                            cam = torch.relu(cam)
                            
                            # Normalize
                            cam = cam - cam.min()
                            if cam.max() > 0:
                                cam = cam / cam.max()
                            
                            # Convert to numpy and resize
                            cam_np = cam.squeeze().cpu().numpy()
                            cam_resized = cv2.resize(cam_np, (original_img.shape[1], original_img.shape[0]))
                            
                            heatmaps.append(cam_resized)
                            detection_info.append({
                                'class': class_name,
                                'confidence': conf,
                                'bbox': det.xyxy[0].cpu().numpy()
                            })
                        
                        # Clear gradients
                        self.model.model.zero_grad()
            
            except Exception as e:
                print(f"Error generating heatmap for detection {idx}: {e}")
                continue
        
        return heatmaps, detection_info, original_img
    
    def visualize(self, image_path, output_path='output', conf_threshold=0.25):
        """
        Create and save GradCAM visualizations
        
        Args:
            image_path: Path to input image
            output_path: Directory to save outputs
            conf_threshold: Confidence threshold
        """
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        print(f"Processing {image_path}...")
        
        # Generate heatmaps
        heatmaps, detections, original_img = self.generate_heatmap(image_path, conf_threshold)
        
        if len(heatmaps) == 0:
            print("No heatmaps generated. Running regular detection...")
            # Fallback to regular detection visualization
            results = self.model(image_path)
            results[0].save(str(output_dir / 'detection_only.jpg'))
            return
        
        # Create visualizations
        fig, axes = plt.subplots(2, len(heatmaps) + 1, figsize=(5 * (len(heatmaps) + 1), 10))
        if len(heatmaps) == 0:
            axes = axes.reshape(2, -1)
        
        # Original image with all detections
        results = self.model(image_path, verbose=False)
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        if len(heatmaps) > 0:
            axes[0, 0].imshow(annotated)
            axes[0, 0].set_title('All Detections', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            axes[1, 0].imshow(original_img)
            axes[1, 0].set_title('Original Image', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
        
        # Individual heatmaps
        for idx, (heatmap, det_info) in enumerate(zip(heatmaps, detections)):
            col = idx + 1
            
            # Heatmap overlay
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Blend
            overlay = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
            
            # Draw bounding box
            bbox = det_info['bbox'].astype(int)
            overlay = cv2.rectangle(overlay.copy(), 
                                   (bbox[0], bbox[1]), 
                                   (bbox[2], bbox[3]), 
                                   (0, 255, 0), 2)
            
            axes[0, col].imshow(overlay)
            axes[0, col].set_title(f"{det_info['class']}\nConf: {det_info['confidence']:.2f}", 
                                  fontsize=10, fontweight='bold')
            axes[0, col].axis('off')
            
            # Pure heatmap
            axes[1, col].imshow(heatmap, cmap='jet')
            axes[1, col].set_title(f"Heatmap - {det_info['class']}", fontsize=10)
            axes[1, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'gradcam_visualization.jpg', dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_dir / 'gradcam_visualization.jpg'}")
        plt.close()
        
        # Save individual heatmaps
        for idx, (heatmap, det_info) in enumerate(zip(heatmaps, detections)):
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(original_img, 0.5, 
                                     cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB), 
                                     0.5, 0)
            
            bbox = det_info['bbox'].astype(int)
            overlay = cv2.rectangle(overlay.copy(), 
                                   (bbox[0], bbox[1]), 
                                   (bbox[2], bbox[3]), 
                                   (0, 255, 0), 3)
            
            # Add text
            text = f"{det_info['class']}: {det_info['confidence']:.2f}"
            overlay = cv2.putText(overlay, text, (bbox[0], bbox[1] - 10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            output_file = output_dir / f"gradcam_{det_info['class']}_{idx}.jpg"
            cv2.imwrite(str(output_file), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f"Saved {output_file}")


# Alternative: Simplified Approach using Activation Maps
class YOLOActivationMap:
    """
    Simplified activation map visualization for YOLO
    More stable than GradCAM for object detection
    """
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.activations = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture activations"""
        
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # Register hooks on multiple layers
        # Adjust layer numbers based on your model
        for i, layer in enumerate(self.model.model.model):
            if hasattr(layer, 'conv'):  # Convolutional layers
                layer.register_forward_hook(get_activation(f'layer_{i}'))
    
    def visualize(self, image_path, output_path='output_activation'):
        """Generate activation map visualizations"""
        
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        # Run inference
        results = self.model(image_path, verbose=False)
        
        # Read original image
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get the last activation map
        last_layer = max(self.activations.keys())
        activation = self.activations[last_layer]
        
        # Average across channels
        activation_map = activation.mean(dim=1).squeeze().cpu().numpy()
        
        # Normalize
        activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
        
        # Resize to original image size
        activation_resized = cv2.resize(activation_map, (img_rgb.shape[1], img_rgb.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * activation_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = cv2.addWeighted(img_rgb, 0.6, heatmap, 0.4, 0)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(activation_resized, cmap='jet')
        axes[1].set_title('Activation Map')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'activation_map.jpg', dpi=150, bbox_inches='tight')
        print(f"Saved activation map to {output_dir / 'activation_map.jpg'}")
        plt.close()


def main():
    """
    Main function to run GradCAM visualization
    """
    
    # Configuration
    MODEL_PATH = 'yolo11l.pt'  # YOLO model path
    IMAGE_PATH = 'bus.jpg'     # Input image
    OUTPUT_DIR = 'gradcam_output'
    CONF_THRESHOLD = 0.25
    
    # Check if files exist
    if not Path(IMAGE_PATH).exists():
        print(f"Error: Image file '{IMAGE_PATH}' not found!")
        print("Please ensure 'bus.jpg' is in the same directory")
        return
    
    print("=" * 60)
    print("YOLO GradCAM Visualization")
    print("=" * 60)
    
    try:
        # Method 1: GradCAM (More complex, may be unstable)
        print("\n[Method 1] GradCAM Approach")
        print("-" * 60)
        gradcam = YOLOGradCAM(MODEL_PATH)
        gradcam.visualize(IMAGE_PATH, OUTPUT_DIR, CONF_THRESHOLD)
        
    except Exception as e:
        print(f"GradCAM method failed: {e}")
        print("This is expected - GradCAM is tricky with YOLO architecture")
    
    try:
        # Method 2: Activation Maps (Simpler, more stable)
        print("\n[Method 2] Activation Map Approach (Recommended)")
        print("-" * 60)
        activation_viz = YOLOActivationMap(MODEL_PATH)
        activation_viz.visualize(IMAGE_PATH, OUTPUT_DIR + '_activation')
        
    except Exception as e:
        print(f"Activation map method failed: {e}")
    
    # Method 3: Built-in YOLO visualization
    print("\n[Method 3] Standard YOLO Detection (Baseline)")
    print("-" * 60)
    model = YOLO(MODEL_PATH)
    results = model(IMAGE_PATH)
    results[0].save(f"{OUTPUT_DIR}_baseline.jpg")
    print(f"Saved baseline detection to {OUTPUT_DIR}_baseline.jpg")
    
    print("\n" + "=" * 60)
    print("✓ Visualization complete!")
    print("=" * 60)
    
    # Print analysis
    print("\n📊 EXPLAINABILITY ANALYSIS:")
    print("-" * 60)
    print("""
    ❓ Is GradCAM the correct approach for YOLO?
    
    ANSWER: Partially - with caveats
    
    ✅ PROS:
    • Shows which regions the model focuses on
    • Provides visual interpretation of model decisions
    • Can help debug model behavior
    
    ⚠️ CONS:
    • GradCAM was designed for classification, not detection
    • YOLO has multiple detection heads, making gradient attribution complex
    • May not accurately reflect bbox regression attention
    • Can be unstable with YOLO architecture
    
    💡 BETTER ALTERNATIVES FOR YOLO:
    1. Activation Maps (simpler, more stable)
    2. Attention mechanisms (if model has them)
    3. YOLO-specific tools: ultralytics built-in visualization
    4. D-RISE (Detection RISE) - designed for object detectors
    5. LIME for object detection
    6. Feature map visualization
    
    📝 RECOMMENDATION:
    For YOLO explainability, consider:
    • Activation maps (included in Method 2 above)
    • Ultralytics' built-in visualization tools
    • Custom analysis of detection confidence scores
    • Ablation studies
    
    For this specific use case (bus + persons):
    • The activation maps will show general attention
    • Compare multiple methods for robust interpretation
    • Use ensemble of explanation techniques
    """)


if __name__ == "__main__":
    main()