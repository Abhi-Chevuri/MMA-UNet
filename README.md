# \# MMA-UNet: A Hybrid MedNeXt-Mamba-CBAM U-Net for Brain Tumor Segmentation

# 

# !\[Python](https://img.shields.io/badge/Python-3.10-blue.svg)

# !\[PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)

# !\[License](https://img.shields.io/badge/License-MIT-green.svg)

# 

# Official PyTorch implementation of our Springer Nature manuscript:

# 

# > \*\*MMA-UNet for Brain Tumor Segmentation in Magnetic Resonance Imaging\*\*

# 

# \---

# 

# \# Overview

# 

# Brain tumor segmentation is a critical step in computer-aided diagnosis and treatment planning. This repository presents \*\*MMA-UNet\*\*, a hybrid encoder-decoder architecture that combines:

# 

# \- EfficientNet-B5 encoder

# \- MedNeXt bridge blocks

# \- Mamba (State Space Model) bottleneck

# \- CBAM attention decoder

# \- Deformable refinement module

# 

# The proposed architecture is designed to capture both local spatial information and long-range contextual dependencies while preserving fine boundary information for accurate tumor segmentation.

# 

# \---

# 

# \# Architecture

# 

# The proposed MMA-UNet consists of five major components:

# 

# ```

# Input MRI

# &#x20;     │

# &#x20;     ▼

# EfficientNet-B5 Encoder

# &#x20;     │

# &#x20;     ▼

# MedNeXt Bridge

# &#x20;     │

# &#x20;     ▼

# Mamba Bottleneck

# &#x20;     │

# &#x20;     ▼

# CBAM Decoder

# &#x20;     │

# &#x20;     ▼

# Deformable Refinement

# &#x20;     │

# &#x20;     ▼

# Segmentation Mask

# ```

# 

# 

# \---

# 

# \# Repository Structure

# 

# ```

# MMA-UNet/

# │

# ├── datasets/

# │   └── dataset.py

# │

# ├── models/

# │   ├── mma\_unet.py

# │   ├── mednext.py

# │   ├── mamba.py

# │   ├── cbam.py

# │   ├── decoder.py

# │   └── deformable.py

# │

# ├── losses/

# │   └── losses.py

# │

# ├── utils/

# │   ├── metrics.py

# │   ├── visualization.py

# │   ├── gradcam.py

# │   ├── config.py

# │   └── training\_utils.py

# │

# ├── configs/

# │   └── config.yaml

# │

# ├── scripts/

# │   ├── train.py

# │   ├── test.py

# │   └── predict.py

# │

# ├── requirements.txt

# ├── LICENSE

# └── README.md

# ```

# 

# \---

# 

# \# Dataset

# 

# Experiments were performed on the publicly available \*\*FigShare Brain Tumor MRI Segmentation Dataset\*\*.

# 

# The dataset contains three tumor classes:

# 

# \- Meningioma

# \- Glioma

# \- Pituitary

# 

# Please download the dataset from the official source and organize it as:

# 

# ```

# dataset/

# │

# ├── images/

# │

# └── masks/

# ```

# 

# \---

# 

# \# Installation

# 

# Clone the repository

# 

# ```bash

# git clone https://github.com/Abhi-Chevuri/MMA-UNet.git

# 

# cd MMA-UNet

# ```

# 

# Install dependencies

# 

# ```bash

# pip install -r requirements.txt

# ```

# 

# \---

# 

# \# Training

# 

# Modify the parameters in

# 

# ```

# configs/config.yaml

# ```

# 

# Start training

# 

# ```bash

# python scripts/train.py

# ```

# 

# \---

# 

# \# Testing

# 

# Evaluate a trained checkpoint

# 

# ```bash

# python scripts/test.py

# ```

# 

# \---

# 

# \# Prediction

# 

# Segment a new MRI image

# 

# ```bash

# python scripts/predict.py --image path/to/image.png

# ```

# 

# \---

# 

# \# Results

# 

# Performance of the proposed MMA-UNet on the held-out test set.

# 

# | Metric | Value |

# |----------|--------|

# | Dice | 0.9063 |

# | IoU | 0.8304 |

# | Precision | 0.9043 |

# | Recall | 0.9092 |

# | Specificity | 0.9982 |

# 

# 

# \---

# 

# \# Visualization

# 

# The repository also includes utilities for

# 

# \- Grad-CAM visualization

# \- Qualitative segmentation comparison and Failure case analysis

# \- Metric visualization

# 

# Example outputs include:

# 

# \- MRI segmentation Ground truth vs prediction

# \- Grad-CAM attention maps

# \- Failure case plots

# 

# \---

# 

# \# Reproducibility

# 

# The implementation uses

# 

# \- Fixed random seed

# \- Identical train/validation/test split

# \- PyTorch implementation

# \- Held-out test evaluation

# 

# to ensure reproducibility of all reported experiments.

# 

# \---

# 

# \# Citation

# 

# If you use this work in your research, please cite:

# 

# ```bibtex

# @article{MMAUNet2026,

# &#x20; title={MMAUNet for Brain Tumor Segmentation in Magnetic Resonance Imaging},

# &#x20; author={Author(s)},

# &#x20; journal={Springer Nature Journal},

# &#x20; year={2026}

# }

# ```

# 

# (Update this after publication.)

# 

# \---

# 

# \# License

# 

# This project is released under the MIT License.

# 

# \---

# 

# \# Acknowledgement

# 

# If you use this implementation in your work, please consider citing our paper.

