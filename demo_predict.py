"""
Demo script for running inference with the fine-tuned ResNet-18 X-ray classifier.

Usage:
    python demo_predict.py path/to/sample_xray.jpg

If you have a fine-tuned checkpoint saved at `checkpoints/resnet18_xray_best.pt`,
the script will load it. Otherwise, it will run with random weights.
"""

import sys
from pathlib import Path

import torch
from torch import nn
from PIL import Image
from torchvision import models, transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(num_classes: int = 2) -> nn.Module:
    """Create a ResNet-18 model with a custom classification head."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_image(image_path: Path) -> torch.Tensor:
    """Load and preprocess an image for ResNet-18."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # [1, 3, 224, 224]


def main():
    if len(sys.argv) != 2:
        print("Usage: python demo_predict.py path/to/sample_xray.jpg")
        raise SystemExit(1)

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        print(f"File not found: {image_path}")
        raise SystemExit(1)

    num_classes = 2
    model = build_model(num_classes=num_classes)

    checkpoint_path = Path("checkpoints/resnet18_xray_best.pt")
    if checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: no checkpoint found at {checkpoint_path}, using random weights.")

    model.to(DEVICE)
    model.eval()

    x = load_image(image_path).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()

    print(f"Predicted class index: {pred_class}")
    print("Probabilities:", probs.cpu().numpy())


if __name__ == "__main__":
    main()
