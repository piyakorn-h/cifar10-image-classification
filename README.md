# CIFAR-10 Image Classification with CNN

My first end-to-end Computer Vision project — building a custom CNN from scratch to classify images across 10 categories using the CIFAR-10 dataset.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat)

---

## Overview

This project covers the complete deep learning pipeline for image classification:

- **Dataset** — CIFAR-10 (60,000 color images, 10 classes)
- **Model** — Custom 3-layer CNN built with PyTorch
- **Regularization** — Dropout, Weight Decay, Data Augmentation, Early Stopping
- **Evaluation** — Learning Curves, Confusion Matrix, ROC Curves, Per-Class Accuracy

---

## Classes

| Label | Class | Label | Class |
|-------|-------|-------|-------|
| 0 | ✈️ Airplane | 5 | 🐕 Dog |
| 1 | 🚗 Automobile | 6 | 🐸 Frog |
| 2 | 🐦 Bird | 7 | 🐴 Horse |
| 3 | 🐱 Cat | 8 | 🚢 Ship |
| 4 | 🦌 Deer | 9 | 🚚 Truck |

---

## Model Architecture

```
Input: (3, 32, 32)
 │
 ├── Conv2d(3 → 32, 3×3)  →  ReLU  →  MaxPool2d(2×2)   # 32×32 → 16×16
 ├── Conv2d(32 → 64, 3×3) →  ReLU  →  MaxPool2d(2×2)   # 16×16 → 8×8
 ├── Conv2d(64 → 128, 3×3)→  ReLU  →  MaxPool2d(2×2)   # 8×8   → 4×4
 │
 ├── Flatten: 128 × 4 × 4 = 2,048
 ├── Linear(2048 → 256)  →  ReLU  →  Dropout(0.5)
 └── Linear(256 → 10)    →  Output

Total Parameters: ~1.07M
```

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Weight Decay | 1e-4 |
| Batch Size | 64 |
| Max Epochs | 20 |
| Dropout | 0.5 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early Stopping | patience=5 |

---

## Data Augmentation

Applied only to training set to improve generalization:

```python
transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])
```

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 72–75% |
| Average AUC | 0.85–0.90 |
| Best Val Loss | saved via `best_model.pth` |
| Training Time (GPU T4) | ~10–15 min |

**Evaluation outputs:**

```
results/
├── learning_curves.png         # Train vs Val Loss & Accuracy
├── confusion_matrix.png        # Per-class prediction breakdown
├── per_class_accuracy.png      # Bar chart of accuracy by class
├── misclassified_examples.png  # 16 examples the model got wrong
└── roc_curves.png              # ROC + AUC for all 10 classes
```

---

## Project Structure

```
cifar10-image-classification/
├── CIFAR_10_Complete_Project.ipynb   # Main notebook
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── models/
│   ├── best_model.pth                # Best checkpoint during training
│   └── cifar10_final_model.pth       # Final model + training history
└── results/                          # Generated visualizations
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/piyakorn-h/cifar10-image-classification.git
cd cifar10-image-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

**Google Colab (recommended):**

1. Upload `CIFAR_10_Complete_Project.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Enable GPU: `Runtime → Change runtime type → GPU (T4)`
3. Run all cells

**Local:**

```bash
jupyter notebook CIFAR_10_Complete_Project.ipynb
```

---

## What I Learned

**Overfitting is the biggest challenge.**
My first model hit 98% train accuracy and 62% validation accuracy — a clear overfit. Learning to diagnose this through learning curves, and address it through regularization techniques, was the most valuable part of this project.

**Evaluation tells the real story.**
Accuracy alone isn't enough. The confusion matrix revealed that my model confused `cat ↔ dog` and `automobile ↔ truck` most often — which makes intuitive sense. Per-class accuracy and ROC curves gave a much more complete picture of where the model succeeds and fails.

**Data augmentation works.**
Adding RandomFlip and RandomCrop alone narrowed the train/val accuracy gap significantly. Simple augmentations have a surprisingly large effect on generalization.

---

## Troubleshooting

**CUDA out of memory**
```python
BATCH_SIZE = 32  # Reduce from 64
```

**Slow training without GPU**
Enable GPU on Colab: `Runtime → Change runtime type → T4 GPU`

**Module not found**
```bash
pip install --upgrade -r requirements.txt
```

---

## Future Improvements

- [ ] Add Batch Normalization
- [ ] Implement K-Fold Cross-Validation
- [ ] Try ResNet / EfficientNet via Transfer Learning
- [ ] Add GradCAM visualization
- [ ] Experiment with SGD + momentum optimizer
- [ ] Mixed precision training (AMP)

---

## References

- [CIFAR-10 Dataset — Krizhevsky, 2009](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*First Computer Vision project — built to learn, not just to run.*
