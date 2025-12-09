# X-Ray Image Classifier Using ResNet-18 (PyTorch)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Domain](https://img.shields.io/badge/Domain-Medical--AI-blue)
![Model](https://img.shields.io/badge/Model-ResNet18-purple)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> A medical imaging classifier built by fine-tuning a ResNet-18 backbone on X-ray data. Includes GPU-optimized training, augmentation pipelines, transfer learning, and evaluation scripts useful for healthcare AI prototyping.

---

## 🚀 Quickstart Demo (For Reviewers)

Run a single-image inference demo:

```bash
pip install -r requirements.txt
python run_demo.py
```

This demonstrates preprocessing, feature extraction, and final classification.

---

## 📦 Full Training Run

Train the full ResNet-18 classifier:

```bash
python train_resnet18_xray.py
```

Includes:

- Frozen → unfrozen training phases  
- Augmentations (resize, normalize, flips)  
- ImageFolder dataset loading  
- Accuracy reporting  
- Transfer learning best practices  

---

## 📁 Files

```text
train_resnet18_xray.py   # Full transfer-learning training script
run_demo.py              # Example inference on a sample X-ray
requirements.txt         # Dependencies
```

---

## 🏗 Overview

The pipeline uses:

- Pretrained ResNet-18 backbone  
- Custom classification head  
- Cross-entropy loss  
- Optional mixed precision  
- Data augmentations for robustness  

This repo demonstrates applied computer vision engineering for medical imaging tasks.

---

## 📂 Project Structure

```text
.
├── train_resnet18_xray.py
├── run_demo.py
├── requirements.txt
├── CONTRIBUTING.md
└── SECURITY.md
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

## 🤝 Contributing
See `CONTRIBUTING.md` for contribution guidelines.

---

## 📄 License
MIT License. See `LICENSE` for details.
