# X-Ray Image Classifier Using ResNet-18 (PyTorch)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red)
![ResNet18](https://img.shields.io/badge/Model-ResNet18-critical)
![Computer Vision](https://img.shields.io/badge/Application-Medical%20Imaging-9cf)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

> Fine-tuning ResNet-18 for medical X-ray image classification with augmentation pipeline, AdamW optimization, cosine LR scheduling, and GPU-accelerated training.

---

## 🚀 Overview

This project fine-tunes **ResNet-18** for X-ray image classification:

- Custom classification head  
- Data augmentation: flips, transforms, normalization  
- AdamW optimizer  
- Cosine annealing LR scheduler  
- GPU-accelerated training loop  
- Robust evaluation pipeline  

Demonstrates practical applied computer vision engineering.

---

## ▶️ Quickstart

```bash
pip install -r requirements.txt
python train_resnet18_xray.py

train_resnet18_xray.py
requirements.txt
```

---

## 🧱 Architecture Overview

At a high level, the training system looks like this:

```text
Input images (H x W x 3)
        |
    torchvision.transforms
    - Resize to 224x224
    - Data augmentation
    - Normalization
        |
        v
  Pretrained ResNet-18 backbone
        |
        +--> Convolution + Residual Blocks
        |
        v
 Global Average Pooling
        |
        v
  Fully-connected classification head
  (replaced with num_classes = 2)
        |
        v
  Softmax (via CrossEntropyLoss)
```

---

## 📜 License
This project is licensed under the MIT License — see the LICENSE file for details.

