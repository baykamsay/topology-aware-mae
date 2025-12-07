# Self-Supervised Pre-training for Topologically Accurate Medical Imaging

This repository contains the implementation code for the master's thesis "Self-Supervised Pre-training for Topologically Accurate Medical Imaging" by Baykam Say.

## Overview

This project investigates whether incorporating topological constraints through Betti Matching loss into Masked Autoencoder (MAE) pretraining can improve downstream medical image segmentation tasks. The approach focuses on preserving connectivity of network-like structures such as blood vessels, neural pathways, and bronchial trees.

We combine:
- **Masked Autoencoders (MAE)** for self-supervised pre-training
- **Betti Matching loss** for topological constraint enforcement
- A novel composite image evaluation strategy that reconciles MAE's patch-wise reconstruction with Betti Matching's holistic topological assessment

## Features

- **Three Architecture Implementations:**
  - Vision Transformer MAE (ViT-MAE)
  - Convolutional MAE (CNN-MAE)
  - ConvNeXt-V2 based MAE
  
- **Optimized Betti Matching Implementation:**
  - ~10x speedup through vectorized batch processing
  - Custom architectural improvements
  
- **Comprehensive Experimental Framework:**
  - Baseline evaluations
  - Ablation studies (loss weights, scheduling, filtration configurations)
  - Transfer learning with encoder freezing strategies

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/baykamsay/topology-aware-mae.git
   cd topo-conv-mae
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

Place your dataset in the `data/` directory. For pretraining, use the ImageFolder format. For segmentation use "train" and "val" subfolders each with "images" and "masks" subfolders.

### Pretraining

To train the MAE model, run:

```bash
python scripts/pretrain.py --config configs/pretrain/vitmae_test.yaml
```

### Segmentation

For segmentation:

```bash
python scripts/segmentation.py --config configs/segmentation/bm_test.yaml
```

## Project Structure

```
topo-conv-mae/
├── configs/            # Configuration files
├── data/               # Dataset storage
├── data_processing/    # Dataset preprocessing scripts
├── datasets/           # Pytorch dataset definitions
├── losses/             # Loss definitions
├── models/             # Model definitions (Encoder, Decoder, MAE)
├── scripts/            # Training and segmentation scripts
├── utils/              # Helper functions and topological tools
├── README.md
└── requirements.txt         # Python dependencies
```

## Related Work

This project builds upon:
- **Betti Matching**: Stucki et al., "Topologically Faithful Image Segmentation via Induced Matching of Persistence Barcodes"
- **Masked Autoencoders**: He et al., "Masked Autoencoders Are Scalable Vision Learners"
- **ConvNeXt V2**: Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"

## License

This project is licensed under the MIT License - see the LICENSE file for details.
