# Underwater Species Identification and Explainability using Machine Learning

A Bachelor's Technology Project implementing deep learning techniques for automated underwater species identification with integrated model explainability features. This project leverages YOLOv11 and YOLOv12 architectures with attention mechanisms to detect and classify marine organisms in real-time while providing transparent, interpretable predictions through explainable AI techniques.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Explainability](#explainability)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Results](#results)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Overview

Manually analyzing vast amounts of underwater video footage is slow, highly prone to human error, and impractical for large-scale marine ecosystem studies. This Bachelor's Technology Project addresses the critical need for automated marine monitoring by developing a real-time deep learning system for fish detection and classification. The system goes beyond standard accuracy metrics by focusing on explainability, providing clear and reliable insights into model predictions, making it trustworthy for researchers and conservationists.

The project addresses key domain-specific challenges including the scarcity of large, diverse underwater datasets and intrinsic image degradation caused by light absorption, scattering, color distortion, and marine particles. To ensure robustness, we utilize advanced architectures like YOLOv11 and YOLOv12, implement comprehensive preprocessing pipelines, and incorporate architectural modifications such as attention modules for enhanced feature extraction in low-visibility conditions.

## Features

- **Real-Time Species Detection**: Automated fish detection and classification using YOLOv11/YOLOv12
- **Attention-Enhanced Architecture**: Custom C2PSA attention blocks for improved feature extraction in challenging underwater conditions
- **Advanced Image Preprocessing**: U-shaped transformer-based enhancement module integrated with computer vision techniques
- **Explainable AI Integration**: EigenCAM and Grad-CAM methods for transparent, interpretable predictions
- **Multi-Dataset Training**: Robust performance across diverse underwater imaging conditions
- **Architectural Modifications**: Three custom architecture variants including extra attention modules, additional detection heads, and CSMB-Darknet blocks
- **HPC Support**: SLURM batch script integration for high-performance computing environments

## Project Structure

```
Underwater-Species-Identification-and-Explainability/
├── configs/              # YAML configuration files for models and training
├── net/                  # Neural network architectures and model definitions
├── src/                  # Source code for training, evaluation, and inference
├── utility/              # Utility functions and helper modules
├── requirements.txt      # Python dependencies
├── slurm.sh             # SLURM batch script for HPC environments
├── .gitignore           # Git ignore file
└── README.md            # Project documentation
```

### Directory Details

- **configs/**: YAML configuration files defining YOLOv11/YOLOv12 architectures, hyperparameters, and training settings including backbone, neck, and head configurations
- **net/**: Implementation of YOLO architectures with custom modifications (attention modules, CSMB-Darknet blocks, extra detection heads)
- **src/**: Training scripts, data loaders, inference pipelines, preprocessing modules, and explainability implementations
- **utility/**: Helper functions for data augmentation (Albumentations), visualization, evaluation metrics, and preprocessing pipelines

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Conda or virtualenv (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Harshilpatel2605/Underwater-Species-Identification-and-Explainability.git
cd Underwater-Species-Identification-and-Explainability
```

2. Create a virtual environment:
```bash
# Using conda
conda create -n underwater_species python=3.8
conda activate underwater_species

# Or using virtualenv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a model using the provided configuration:

```bash
python src/train.py --config configs/yolov11_attention.yaml
```

### Inference

Run inference on new images:

```bash
python src/inference.py --model path/to/model.pth --image path/to/image.jpg
```

### Explainability Analysis

Generate explainability visualizations using EigenCAM:

```bash
python src/explain.py --model path/to/model.pth --image path/to/image.jpg --method eigencam
```

### SLURM Execution

For HPC environments:

```bash
sbatch slurm.sh
```

## Model Architecture

This project implements YOLOv11 and YOLOv12 with custom attention-enhanced architectures designed specifically for underwater species detection. The architecture addresses challenges such as color distortion, low contrast, varying lighting conditions, and marine snow.

### YOLOv11 Architecture Components

**Backbone (Feature Extraction)**
- Progressive downsampling from P1 (stride 2) to P5 (stride 32)
- C3k2 blocks for efficient feature processing at multiple scales
- Strategic C2PSA attention modules after P4 and before the neck for enhanced spatial feature learning
- SPPF (Spatial Pyramid Pooling Fast) at P5 for multi-scale receptive fields

**Neck (Feature Fusion)**
- Feature Pyramid Network (FPN) structure with top-down pathway
- Path Aggregation Network (PAN) with bottom-up pathway
- Concatenation of multi-level features from backbone
- C3k2 blocks for feature refinement at P3, P4, and P5 scales

**Head (Detection)**
- Three-scale detection heads (P3, P4, P5) for objects of varying sizes
- Simultaneous prediction of bounding boxes and class probabilities
- Single-stage detection for real-time performance

### Architectural Modifications Implemented

Three architectural variants were developed and evaluated:

1. **Extra Detection Head**: Added an additional prediction layer in the neck for detecting smaller and finer underwater objects at a different feature scale

2. **Extra Attention Module**: Inserted an additional C2PSA attention module in the backbone to improve feature refinement in low-visibility conditions, helping the network focus on more informative regions

3. **Head + CSMB Backbone**: Combined approach adding an extra detection head and replacing standard C3K2 modules with Cross Stage Multi-Branch (CSMB)-Darknet blocks for improved feature flow and enhanced detection of partially occluded objects

### YOLOv11 vs YOLOv12

- **YOLOv11 (Late 2024)**: Refines CNN architecture for maximum versatility and feature extraction
- **YOLOv12 (Early 2025)**: Shifts to an "Attention-Centric" framework to maximize efficiency and speed

### Best Performing Model

The YOLOv11 model with Extra Attention Module consistently achieved the highest mAP scores across both datasets, demonstrating that attention mechanisms are critical for accurate detection under low visibility and high turbidity conditions.

## Explainability

Understanding model predictions is essential for scientific validation and building trust in automated species identification systems. This project implements explainable AI techniques to ensure transparency and reliability.

### Implemented Methods

**EigenCAM (Primary Method)**
- **Mechanism**: Gradient-free method that finds main directions of activation using Principal Component Analysis
- **Speed**: Real-time efficiency, generates explanations instantly without backpropagation, matching YOLO's high-speed inference
- **Localization**: Precise object localization with heatmaps that tightly contour species shape (fins, scales, spines)
- **Validation**: Proves the model identifies physical traits and texture rather than relying on color biases or background noise
- **Robustness**: Filters out visual noise common in underwater footage (turbidity, scattering)

**Grad-CAM (Comparative Analysis)**
- **Mechanism**: Gradient-based method using backpropagation, adapted with patch-based approach for YOLO multi-output structure
- **Application**: Extract localized patches for each detected instance to generate granular, class-specific heatmaps
- **Focus**: Confirms model focuses on discriminative morphological features (e.g., protruding spikes of Echinus)
- **Limitation**: Slower than EigenCAM and can produce noisier heatmaps

### Validation Results

EigenCAM heatmaps on the URPC2020 dataset demonstrated that the model focuses on correct morphological features:
- **Starfish**: High activation concentrated on spines and texture
- **Echinus**: Focus on sharp, protruding spikes
- **Physical traits**: Scales, fins, shells rather than environmental background

This transparency validates structural integrity and makes the system trustworthy for marine biologists and conservationists.

## Configuration

Model parameters are configured through YAML files in the `configs/` directory. The configuration defines the complete YOLOv11 architecture with attention mechanisms.

### Example Configuration Structure

```yaml
# YOLOv11 Model Configuration (Light Attention)
# Ultralytics AGPL-3.0 License
nc: 4  # number of classes

# Backbone (Enhanced Attention)
backbone:
  # [from, repeats, module, args(out_channels, kernel_size, stride)]
  - [-1, 1, Conv, [64, 3, 2]]          # 0 - P1/2
  - [-1, 1, Conv, [128, 3, 2]]         # 1 - P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2 - P2 feature block
  - [-1, 1, Conv, [256, 3, 2]]         # 3 - P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4 - P3 feature block
  - [-1, 1, Conv, [512, 3, 2]]         # 5 - P4/16
  - [-1, 2, C3k2, [512, True]]         # 6 - P4 feature block
  - [-1, 1, C2PSA, [512]]              # 7 - Attention after P4
  - [-1, 1, Conv, [1024, 3, 2]]        # 8 - P5/32
  - [-1, 2, C3k2, [1024, True]]        # 9 - P5 feature block
  - [-1, 1, SPPF, [1024, 5]]           # 10 - Spatial Pyramid Pooling Fast
  - [-1, 2, C2PSA, [1024]]             # 11 - Final Attention before neck

# Head (Standard 3-Head: P3, P4, P5)
head:
  # Top-Down Path
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 12 - Upsample from P5
  - [[-1, 7], 1, Concat, [1]]                   # 13 - Concat with P4 attention
  - [-1, 2, C3k2, [512, False]]                 # 14 - Head P4 processing
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 15 - Upsample from Head P4
  - [[-1, 4], 1, Concat, [1]]                   # 16 - Concat with P3 features
  - [-1, 2, C3k2, [256, False]]                 # 17 - Head P3 processing
  
  # Bottom-Up Path
  - [-1, 1, Conv, [256, 3, 2]]                  # 18 - Downsample from Head P3
  - [[-1, 14], 1, Concat, [1]]                  # 19 - Concat with Head P4
  - [-1, 2, C3k2, [512, False]]                 # 20 - Head P4 processing
  - [-1, 1, Conv, [512, 3, 2]]                  # 21 - Downsample from Head P4
  - [[-1, 11], 1, Concat, [1]]                  # 22 - Concat with P5 attention
  - [-1, 2, C3k2, [1024, True]]                 # 23 - Head P5 processing
  
  # Detection Layer
  - [[17, 20, 23], 1, Detect, [nc]]             # 24 - Detect(P3, P4, P5)
```

## Datasets

### Fish-Detection Dataset (Kaggle)
- **Classes**: 13 species (AngelFish, BlueTang, ButterflyFish, ClownFish, GoldFish, Gourami, MorishIdol, PlatyFish, RibbonedSweetlips, ThreeStripedDamselfish, YellowCichlid, YellowTang, ZebraFish)
- **Train**: 8,448 images
- **Test**: 407 images
- **Validation**: 798 images
- **Characteristics**: Relatively clear underwater images, suitable for baseline training

### URPC2020 Dataset
- **Classes**: 4 species (Holothurian, Echinus, Scallop, Starfish)
- **Train**: 5,543 images
- **Test**: 800 images
- **Validation**: 1,200 images
- **Characteristics**: Challenging real-world conditions with low visibility, blur, light scattering, and color cast

### Data Augmentation
Implemented using Albumentations library with transformations including:
- Horizontal and vertical flips
- Random brightness and contrast adjustments
- Rotation
- Standard geometric transformations

## Results

### Best Model Performance

**YOLOv11 + Extra Attention Module on URPC2020 (Original Data)**
- Precision: 0.799
- Recall: 0.709
- mAP@0.5: 0.775
- mAP@0.5:0.95: 0.446

**YOLOv11 + Extra Attention Module on Fish-Detection (Original Data)**
- Precision: 0.916
- Recall: 0.887
- mAP@0.5: 0.937
- mAP@0.5:0.95: 0.732

### Key Findings

1. The Extra Attention Module consistently achieved the highest mAP scores across both datasets
2. Computer vision preprocessing pipelines (Pipeline 3 with 30 dB PSNR) degraded performance on already-clear images
3. U-shaped transformer image enhancement improved robustness when combined with original images
4. YOLO classifier outperformed traditional ML classifiers (SVM, KNN, XGBoost, Random Forest) on extracted features
5. EigenCAM provided superior explainability with real-time performance compared to Grad-CAM

## Documentation

### Project Report
Comprehensive project report detailing methodology, experiments, and results:
[View Report](https://drive.google.com/file/d/1kz_tL8lt3DYGpTCoCmKxAVlOiezUXrk3/view)

### Presentation
Bachelor's Technology Project presentation:
[View Presentation](https://drive.google.com/file/d/1qt_N5f5_a472lNSduZDhAzOzpzwZ5K9v/view)

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add NewFeature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## Authors

**Harshil N Patel** - Roll Number: 2203123 - [@Harshilpatel2605](https://github.com/Harshilpatel2605)

**Md Hamza Z Sabugar** - Roll Number: 2203126 - [@md-hzs-22](https://github.com/md-hzs-22)

Department of Computer Science and Engineering  
Indian Institute of Technology Goa

## Acknowledgments

We express our sincere gratitude to our supervisors **Dr. Clint P. George** and **Dr. Satyanath Bhat**, Department of Computer Science and Engineering, for their continuous guidance, valuable insights, and constant encouragement throughout this project.

We also extend our heartfelt thanks to our co-supervisor **Dr. Shitala Prasad**, Department of Computer Science and Engineering, for his constructive feedback, support, and motivation that greatly contributed to the successful completion of this work.

## License

This project is submitted towards partial fulfillment of the requirements for the Bachelor of Technology degree in Computer Science and Engineering at the Indian Institute of Technology Goa. Available for academic and research purposes.

## Contact

For questions or collaborations, please open an issue on GitHub or contact the authors directly.
