# Neural Network Architecture Modules (`net/`)

## Overview
This directory contains the PyTorch implementation of a **Hybrid Transformer-CNN GAN** for image-to-image translation. The architecture is designed to leverage the local feature extraction capabilities of Convolutional Neural Networks (CNNs) while utilizing Transformers to capture long-range global dependencies.

The core design is based on a **U-Net** structure, enhanced with two specific types of Transformer attention mechanisms:
1.  **Spatial-wise Attention (SGFMT):** Captures global spatial context at the bottleneck.
2.  **Channel-wise Attention (CMSFFT):** Captures inter-channel dependencies at multiple scales in the encoder.

## Key Components

### 1. Generator (`Ushape_Trans.py`)
The Generator is the core of the project. It follows a **MSG-U-Net (Multi-Scale Gradient U-Net)** design:
* **Encoder:** Standard convolutional downsampling blocks (`conv_block`) augmented with a **Channel Transformer (`mtc`)**.
* **Bottleneck:** A **Spatial Transformer** that flattens the feature map to sequence data (patches) to learn global context before upsampling.
* **Decoder:** Upsampling blocks (`up_conv`) with **CCA (Channel Channel Attention)** skip connections that fuse encoder features with decoder features.
* **Multi-Scale I/O:** The model takes input and produces output at multiple resolutions ($32^2, 64^2, 128^2, 256^2$), facilitating stable gradient flow.

### 2. Discriminator (`Ushape_Trans.py`)
A multi-scale discriminator designed to critique images at different resolutions.
* It accepts the input image and the generated/target image concatenated channel-wise.
* It uses **Equalized Convolution (`_equalized_conv2d`)** and **Pixelwise Normalization**, techniques often used in ProGAN/StyleGAN to stabilize training.

### 3. Attention Modules
* **`SGFMT.py` (Spatial-Gated Feature Merging Transformer):**
    * Implements the "Bottleneck Transformer."
    * Uses standard Self-Attention to relate every patch of the image to every other patch.
* **`CMSFFT.py` (Cross-Scale Multi-Stage Feature Fusion Transformer):**
    * Implements the "Channel Transformer."
    * applied to the encoder features (`e1, e2, e3, e4`) to allow information flow between different channel depths before they pass to the bottleneck.

### 4. Building Blocks (`block.py`)
Contains the fundamental custom layers used throughout the network:
* **`_equalized_conv2d`**: Convolutions with dynamic weight scaling for equalized learning rates.
* **`PixelwiseNorm`**: Normalization layer that normalizes the feature vector in each pixel to unit length.
* **`CCA` (Channel Channel Attention)**: Used in skip connections to re-weight feature channels adaptively.

---

## File Manifest

| File | Description |
| :--- | :--- |
| **`Ushape_Trans.py`** | **[Main]** Contains the `Generator` and `Discriminator` classes. |
| `SGFMT.py` | Implementation of the Spatial Transformer (Self-Attention) block. |
| `CMSFFT.py` | Implementation of the Channel-wise Transformer block. |
| `block.py` | Custom layers: Equalized Conv, PixelNorm, ResBlocks, CCA. |
| `PositionalEncoding.py` | Fixed (Sinusoidal) and Learned positional encodings for the Transformers. |
| `utils.py` | Helpers: VGG19 Perceptual Loss, PSNR calculation, Data Augmentation. |
| `IntmdSequential.py` | Utility to extract intermediate layer outputs (used in Transformers). |

---

## How to Run

### 1. Instantiating the Models
You can import the Generator and Discriminator directly into your training script.

```python
import torch
from net.Ushape_Trans import Generator, Discriminator

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Initialize Generator ---
# img_dim: Input image size (e.g., 256x256)
# embedding_dim: Dimension of the Transformer embeddings
netG = Generator(
    img_dim=256,
    patch_dim=16,
    embedding_dim=512,
    num_heads=8,
    num_layers=4,
    hidden_dim=2048
).to(device)

# --- Initialize Discriminator ---
netD = Discriminator(in_channels=3).to(device)

# Print parameter count (Optional)
print(f"Generator Parameters: {sum(p.numel() for p in netG.parameters())}")
```

### 2. Forward Pass Example
Since this is a multi-scale model, the input and output handling is specific.
```python
# Create dummy input (Batch Size=1, Channels=3, H=256, W=256)
dummy_input = torch.randn(1, 3, 256, 256).to(device)

# 1. Generator Forward Pass
# Returns a list of outputs at different scales: [256px, 128px, 64px, 32px]
fake_imgs = netG(dummy_input)

print(f"Output scales: {[img.shape for img in fake_imgs]}")
# Expected: [ (1,3,32,32), (1,3,64,64), (1,3,128,128), (1,3,256,256) ]


# 2. Discriminator Forward Pass
# The Discriminator expects a list of inputs matching the multi-scale outputs
# For training, you typically pass [Input_Image] and [Generated_Images_List]

# We need to downsample the real input to match the generator's multi-scale output
from torch.nn import functional as F
real_imgs_multiscale = [
    F.interpolate(dummy_input, size=(256, 256)), # Scale 0
    F.interpolate(dummy_input, size=(128, 128)), # Scale 1
    F.interpolate(dummy_input, size=(64, 64)),   # Scale 2
    F.interpolate(dummy_input, size=(32, 32))    # Scale 3
]
# Note: The Discriminator logic in your code reverses the list order (Smallest -> Largest)
# You may need to verify the order based on your specific training loop.

# Pass the largest real image + the list of multi-scale inputs
# (Logic depends on your specific training loop implementation in train.py)
validity = netD(real_imgs_multiscale, fake_imgs)
print(f"Discriminator Output Shape: {validity.shape}")
```

### 3. Using the Perceptual Loss
The utils.py file includes a VGG19-based perceptual loss which is crucial for high-quality image reconstruction.

```python
from net.utils import VGG19_PercepLoss

# Initialize Loss
criterion_perceptual = VGG19_PercepLoss().to(device)

# Calculate Loss
# Assuming 'fake_imgs[-1]' is the full-resolution 256x256 output
loss = criterion_perceptual(fake_imgs[-1], dummy_input)
print(f"Perceptual Loss: {loss.item()}")
```
