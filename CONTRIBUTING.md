# Contributing to X-Ray ResNet-18 Classifier

Thank you for contributing to this medical imaging classification project using a fine-tuned ResNet-18 built with PyTorch.

---

## 1. Fork the Repository

Use the **Fork** button on GitHub to create your own working copy.

---

## 2. Clone Your Fork & Create a Branch

```bash
git clone https://github.com/<your-username>/xray-resnet-classifier.git
cd xray-resnet-classifier
git checkout -b feature/your-feature-name
```

---

## 3. Make Your Changes

- Maintain clarity and consistency—medical AI requires careful engineering.
- Augmentation changes should preserve diagnostic signal.
- Avoid altering normalization pipelines unless required.
- Test on a small subset before proposing large modifications.

---

## 4. Run Basic Checks

### Syntax Validation

```bash
python -m compileall .
```

### Quick Training Smoke Test

```bash
python train_resnet18_xray.py --epochs 1 --tiny
```

### Optional: Run Tests (if applicable)

```bash
pytest
```

---

## 5. Open a Pull Request

- Clearly describe the change, motivation, and expected impact.
- Include metrics if training or model improvements are involved.
- For image-processing PRs, note any medical-imaging conventions considered.

---

## Code Style Guidelines

- Use descriptive names like `features`, `logits`, `images_norm`.
- Add docstrings summarizing function responsibilities.
- Ensure dataloaders remain performant on GPU.
- Document shape transformations clearly.

---

## Thank You

Your contributions help advance reliable, transparent medical imaging AI.
