# X-Ray Image Classifier Using ResNet-18 (PyTorch)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Domain](https://img.shields.io/badge/Domain-Medical--AI-blue)
![Model](https://img.shields.io/badge/Model-ResNet18-purple)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> A medical imaging classifier built by fine-tuning a ResNet-18 backbone on X-ray data. Includes GPU-optimized training, augmentation pipelines, transfer learning, and evaluation scripts useful for healthcare AI prototyping.

---

## 🚀 Quickstart Demo

```bash
pip install -r requirements.txt
python run_demo.py
```

Runs preprocessing + a forward pass on a sample X-ray.

---

## 📁 Files

```text
train_resnet18_xray.py   # Full training script using transfer learning
run_demo.py              # Single-image inference demo
requirements.txt         # Dependencies
```

---

## 🧠 Model Overview

- Pretrained ResNet-18 convolutional backbone  
- Frozen → partially unfrozen fine-tuning schedule  
- Data augmentations for robustness  
- Linear classifier head for target labels  
- Cross-entropy loss + accuracy evaluation  

Well-suited for demonstrating applied deep learning on real-world imaging tasks.

---

## 📂 Project Structure

```text
.
├── train_resnet18_xray.py
├── run_demo.py
├── requirements.txt
└── CONTRIBUTING.md
```

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
See CONTRIBUTING.md for branch workflow, issue guidelines, and PR instructions.

---

## 📄 License
MIT License. See `LICENSE` for details.
