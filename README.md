# DC-ViT: Declutter Vision Transformers for GPR Clutter Removal
This repository contains the official PyTorch implementation of the paper "A Vision Transformer Based Approach to Clutter Removal in GPR: DC-ViT".

## Overview
Ground Penetrating Radar (GPR) systems often suffer from clutter caused by surface reflections and antenna crosstalk, which obscures subsurface targets. While Convolutional Neural Networks (CNNs) have been used for clutter removal, they are limited by their local receptive fields.

DC-ViT (Declutter Vision Transformer) introduces a Transformer-based architecture to capture long-range dependencies in GPR B-Scans. The model effectively separates the target signal from clutter using:

### Dense Connections: 
To facilitate information flow between Transformer blocks and prevent feature loss.

### Locally-enhanced Feed-Forward Network (LeFF): 
Replaces the standard MLP in Transformers with a block containing Depthwise Convolutions to better capture local spatial details .

## Architecture
The model consists of three main stages:

### Patching & Embedding: 
The input B-Scan is divided into patches and linearly projected.

### Dense Transformer Block: 
A sequence of Transformer Encoder Groups connected via dense concatenation. This block utilizes LeFF to capture both global (via Self-Attention) and local (via LeFF) features.


### Reconstruction Block: 
The deep features are reshaped and upsampled (using Depth-to-Space/PixelShuffle operations) to reconstruct the clutter-free target image.

# Implementation Notes
The original experimental results presented in the paper were obtained using a Keras/TensorFlow implementation. The code provided in this repository is a PyTorch revision of the model, optimized for flexibility and modern research workflows.

# Dataset
The model was trained using the Hybrid Dataset (synthetic and real data) proposed by Sun et al. If you use this code or dataset, please cite the original dataset paper:

H. -H. Sun, W. Cheng and Z. Fan, "Learning to Remove Clutter in Real-World GPR Images Using Hybrid Data," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-14, 2022, Art no. 5113714, doi: 10.1109/TGRS.2022.3176029.

Citation
If you find this code useful for your research, please cite our paper:
```bibtex
@ARTICLE{10493036,
  author={Kayacan, Yavuz Emre and Erer, Isin},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={A Vision-Transformer-Based Approach to Clutter Removal in GPR: DC-ViT}, 
  year={2024},
  volume={21},
  number={},
  pages={1-5},
  keywords={Transformers;Clutter;Image reconstruction;Sparse matrices;Feeds;Convolution;Vectors;Clutter removal;deep learning;ground-penetrating radar (GPR);vision transformers (ViTs)},
  doi={10.1109/LGRS.2024.3385694}}# DC-ViT: Declutter Vision Transformers for GPR Clutter Removal
