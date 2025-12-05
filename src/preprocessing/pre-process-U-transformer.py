import os
import sys
import shutil
import cv2
import torch
from tqdm import tqdm

# Get the absolute path of the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# --- U-shape Transformer imports ---
from net.Ushape_Trans import Generator

print("🚀 Initializing Pre-processing...")

# --- Device Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Initialize Enhancement Model ---
model_weights_path = os.path.join(script_dir, "saved_models/G/generator_795.pth")
try:
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(model_weights_path, map_location=device))
    generator.eval()
    print("✅ Enhancement model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Could not find model weights at '{model_weights_path}'.")
    exit()

# --- Enhancement Function (Corrected) ---
@torch.no_grad()
def enhance_and_save(img_path, output_path, model_input_size=(256, 256), final_output_size=(1024, 1024)):
    """
    Loads an image, resizes for the model, enhances, resizes for YOLO, and saves.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image {img_path}. Skipping.")
        return

    # STEP 1: Resize the input image to what the enhancement model expects (256x256)
    img_resized_for_model = cv2.resize(img, model_input_size, interpolation=cv2.INTER_AREA)

    # The model expects RGB, but OpenCV loads as BGR
    img_rgb = cv2.cvtColor(img_resized_for_model, cv2.COLOR_BGR2RGB)
    
    # Preprocess image into a tensor [1, C, H, W] in [0, 1] range
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Run the enhancement model
    enhanced_tensor = generator(img_tensor)[3].squeeze(0)
    
    # Convert tensor back to a numpy image
    enhanced_img_np = enhanced_tensor.cpu().numpy().transpose(1, 2, 0) * 255
    enhanced_img_np = enhanced_img_np.clip(0, 255).astype('uint8')
    
    # STEP 2: Resize the enhanced output image to the final size for YOLO (1024x1024)
    enhanced_img_final = cv2.resize(enhanced_img_np, final_output_size, interpolation=cv2.INTER_CUBIC)
    
    # Convert back to BGR for saving with OpenCV
    enhanced_img_bgr = cv2.cvtColor(enhanced_img_final, cv2.COLOR_RGB2BGR) # <-- CORRECTED a typo here (was COLOR_RGB_BGR)
    
    cv2.imwrite(output_path, enhanced_img_bgr)


# --- Dataset Processing ---
dataset_base_path = os.path.join(script_dir, "URPC2020")
splits = ["train", "valid", "test"]


def process_split(split_name: str):
    print(f"\n🔄 Processing split: {split_name}")

    source_img_dir = os.path.join(dataset_base_path, split_name, "images")
    source_label_dir = os.path.join(dataset_base_path, split_name, "labels")

    if not os.path.isdir(source_img_dir):
        print(f"⚠️  Skipping '{split_name}': images directory not found at {source_img_dir}")
        return

    dest_split_dir = os.path.join(dataset_base_path, f"{split_name}_enh")
    if os.path.isdir(dest_split_dir):
        print(f"ℹ️  Skipping '{split_name}': destination '{dest_split_dir}' already exists.")
        return

    dest_img_dir = os.path.join(dest_split_dir, "images")
    dest_label_dir = os.path.join(dest_split_dir, "labels")

    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)

    image_filenames = [
        fname
        for fname in sorted(os.listdir(source_img_dir))
        if os.path.isfile(os.path.join(source_img_dir, fname))
    ]
    print(f"Found {len(image_filenames)} images in '{split_name}'.")

    for filename in tqdm(image_filenames, desc=f"{split_name} images", leave=False):
        base_name, extension = os.path.splitext(filename)
        label_filename = f"{base_name}.txt"

        source_img_path = os.path.join(source_img_dir, filename)
        source_label_path = os.path.join(source_label_dir, label_filename)

        dest_img_path = os.path.join(dest_img_dir, filename)
        enhance_and_save(source_img_path, dest_img_path)

        if os.path.exists(source_label_path):
            shutil.copy2(source_label_path, os.path.join(dest_label_dir, label_filename))


for split in splits:
    process_split(split)

print("\n🎉 All splits processed successfully!")