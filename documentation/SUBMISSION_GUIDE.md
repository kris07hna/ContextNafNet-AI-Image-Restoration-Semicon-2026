# KLA Problem Statement – Official Submission Guide 🏆
**SEMICON Hackathon 2026**

This document provides complete instructions for executing, evaluating, and benchmarking the **`ContextNAFNet`** semiconductor wafer restoration solution.

---

## ⚡ 1. Quick Start Execution Command

To restore a directory of low-resolution noisy wafer `.npy` files:

```bash
python run.py <input-dir> <output-dir>
```

### Example:
```bash
python run.py Test_NoisyLR/NoisyLR outputs
```

- **Input Directory**: Path containing low-resolution noisy `.npy` wafer files.
- **Output Directory**: Target directory where restored `.npy` wafer files will be saved.
- **Automatic Output Format**: 2D grayscale float32 arrays with shape `(256, 256)` (2x target super-resolution), clamped within `[0.0, 1.0]` with zero `NaN` or `Inf` values.

---

## 📦 2. Environment Setup

```bash
# Install required dependencies with PyTorch CUDA GPU support
pip install -r requirements.txt
```

### Dependencies:
- `torch >= 2.0.0` (with CUDA 12.1 GPU support)
- `torchvision >= 0.15.0`
- `numpy >= 1.24.0`
- `scipy >= 1.10.0`
- `PyYAML >= 6.0.0`
- `tqdm >= 4.65.0`
- `matplotlib >= 3.7.0`

---

## 🏋️ 3. Training & Evaluation Scripts

```bash
# 1. Dataset-Wide Model Evaluation (8-Fold TTA)
python scripts/evaluate.py --checkpoint models/best.pt --tta 8

# 2. Single Wafer Visual & SNR Comparison
python scripts/compare.py --file 002060.npy --checkpoint models/best.pt

# 3. Model Training Entrypoint
python scripts/train.py --config configs/train.yaml
```

---

## 📁 4. Project Repository Structure

```
team_name/
├── run.py                 # Primary entry script (python run.py <input-dir> <output-dir>)
├── requirements.txt       # Python dependencies with CUDA PyTorch index
├── README.md              # Project documentation
├── LICENSE                # MIT License
├── models/
│   └── best.pt            # Pre-trained model weights (29.15 dB PSNR, 132.1 MB)
├── notebooks/
│   └── KLA_Semicon2026_Restoration_Colab.ipynb  # End-to-end Google Colab GPU notebook
├── documentation/
│   ├── IEEE_Paper_Architecture_Description.md   # Technical paper architecture writeup
│   └── SUBMISSION_GUIDE.md                      # Official submission guide
├── reports/
│   ├── wafer_restoration_demo.gif  # Live 1 fps animated restoration demo GIF
│   ├── architexture.png            # 3D Architecture diagram
│   └── github_readme_hero.png      # Real sample image input flow & visual comparisons
└── outputs/               # Restored test set predictions (.npy files)
```
