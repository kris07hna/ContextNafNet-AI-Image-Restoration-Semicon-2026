# AI-Based Restoration of Degraded Images

SEMICON Hackathon 2026 | KLA Problem Statement submission

This repository restores grayscale, low-resolution semiconductor inspection images stored as NumPy arrays. The official evaluator-facing entrypoint is `run.py`.

## 1. Evaluator command

Run the complete input directory with exactly two positional arguments:

```bash
python run.py <input-dir> <output-dir>
```

Example:

```bash
python run.py Test_NoisyLR/NoisyLR outputs
```

The command loads every top-level `*.npy`, selects `models/best.pt` (with `checkpoints/best.pt` as fallback), chooses CUDA automatically when available, creates the output directory, and writes one same-named output for each input. Inference is non-interactive and does not download models or call external services.

## 2. Clean setup

The project requires Python `>=3.11,<3.13`. Use a virtual environment.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell activation is blocked, use Command Prompt:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify PyTorch and CUDA:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda)"
```

`CUDA available: True` confirms GPU visibility. `run.py` supports CPU fallback, but it is slower.

## 3. Offline evaluator preparation

Inference itself is offline: it uses only the repository and checkpoint. Installing dependencies on a machine without internet requires a local wheel cache. Prepare it on a connected machine:

```bash
python -m pip download -r requirements.txt -d wheelhouse
```

Copy `wheelhouse/` with the repository, then install without package indexes:

```bash
python -m pip install --no-index --find-links wheelhouse -r requirements.txt
```

The final archive must include every Python module imported by `run.py`. In this repository, `run.py` imports `src/semicon_restore`; the announcement's literal four-entry layout is not self-contained for this implementation unless `src/` is retained or the package is vendored into the bundle. Keep `models/best.pt` as well.

## 4. Input contract

Every input must be a loadable NumPy `.npy` file containing a real, non-complex, finite, two-dimensional grayscale array `(H, W)`. Normalized values in `[0, 1]` are expected by the trained model. Official package inference supports `(128, 128)` and `(256, 256)` inputs.

Only top-level files are scanned; subdirectories and `.npz`, PNG, and JPEG files are ignored. Input filenames are sorted for deterministic processing. Object arrays, RGB arrays, NaN, and Inf values are rejected by the package I/O contract.

## 5. Output contract

For every `name.npy`, the command writes `<output-dir>/name.npy` with dtype `float32`, shape `(2H, 2W)`, two-dimensional grayscale layout, values in `[0.0, 1.0]`, and no NaN or Inf values. The output directory is created automatically.

Inspect a generated output:

```powershell
python -c "import numpy as np; from pathlib import Path; p=next(Path('outputs').glob('*.npy')); a=np.load(p, allow_pickle=False); print(p.name, a.shape, a.dtype, np.isfinite(a).all(), a.min(), a.max())"
```

## 6. Model and implementation

The submitted model is a **ContextNAFNet** variant named `context_naf`. It combines multi-scale NAF blocks, noise-aware input features, feature conditioning, and bottleneck attention for grayscale restoration and 2x super-resolution.

### Architecture diagram

The following diagram shows the complete restoration pipeline: the degraded wafer input, noise-feature conditioning, two encoder/decoder levels with skip connections, the level-3 bottleneck, PixelShuffle upsampling, NAFBlock components, and the restored 2x output.

![ContextNAFNet architecture for semiconductor inspection image restoration](reports/architexture.png)

*Figure: ContextNAFNet wafer restoration architecture. The diagram is included locally in `reports/architexture.png`, so it does not require an external image URL.*

### Checkpoint-verified specifications

The following values were read directly from `models/best.pt`, not inferred from the README or training configuration:

| Property | Value |
| --- | ---: |
| Architecture name | `context_naf` |
| Total trainable parameters | **8,187,476** (**8.187M**) |
| FP32 parameter memory | **31.23 MiB** |
| Checkpoint file size | **125.97 MiB** |
| Checkpoint format version | `1` |
| EMA/state-dict tensors | `702` |
| External input | `[B, 1, H, W]` grayscale tensor |
| Internal input features | **8**, `noise_aware` mode |
| Output | `[B, 1, 2H, 2W]` grayscale tensor |
| Super-resolution scale | **2x** |
| Base channel width | **64** |
| Encoder/middle depth configuration | `(6, 8, 8)` blocks |
| Main NAF convolution kernel | **7x7** |
| Intro convolution kernel | **5x5** |
| Bottleneck attention blocks | **2** |
| Attention heads | **8** |
| Feature conditioning | Enabled |
| Inference weights | EMA weights, when present |
| Inference augmentation | 8-fold dihedral self-ensemble |

The model pads inputs internally to a multiple of four, then crops the output back to exactly `(2H, 2W)`. The parameter count excludes non-parameter feature transforms; it includes all trainable convolution, normalization, attention, conditioning, and output-head parameters.

### Parameter distribution

| Component | Parameters |
| --- | ---: |
| Middle/bottleneck | 5,254,672 |
| Encoder level 2 | 1,033,216 |
| Decoder level 2 | 1,033,216 |
| Encoder level 1 | 215,424 |
| Decoder level 1 | 215,424 |
| Intro, sampling, reductions, head, and conditioning | 435,524 |
| **Total** | **8,187,476** |

### Noise-aware coefficients

The checkpoint stores these noise-feature settings:

| Setting | Value |
| --- | ---: |
| Quadratic noise coefficient | `0.026627` |
| Linear noise coefficient | `0.0` |
| Constant noise coefficient | `3.929e-05` |
| Variance-stabilizing margin | `0.05` |
| Noise blur sigma | `1.0` |

### Checkpoint training metadata

The checkpoint was saved at epoch **71**, step **12,240**, with recorded best metrics of **28.94698 dB PSNR** and **0.818016 SSIM**. Any higher metric quoted elsewhere must be identified as a separate evaluation protocol, such as a dataset-wide run with TTA; it should not be presented as the raw checkpoint metadata.

`models/best.pt` is a format-versioned PyTorch checkpoint containing model configuration, model weights, EMA weights, optimizer/scheduler state, scaler state, random state, and training metadata. The checked-in checkpoint is approximately 126 MiB on disk. `run.py` loads the EMA weights when available and applies the 8-transform dihedral self-ensemble.

```text
team_name/
├── run.py                         # official evaluator entrypoint
├── requirements.txt               # runtime dependencies
├── README.md                      # this document
├── models/
│   └── best.pt                    # required inference checkpoint
├── src/semicon_restore/            # imported runtime package; required
├── configs/                       # training/degradation configuration
├── scripts/                       # training and evaluation utilities
├── tests/                         # automated tests
└── documentation/                 # technical and submission notes
```

`checkpoints/best.pt` is a duplicate fallback in the development repository. Keep at least `models/best.pt` in the submitted copy.

## 7. Tests and smoke test

Run tests from the repository root:

```bash
python -m pytest -q
```

Create a small temporary input and run the official command:

```bash
python -c "from pathlib import Path; import numpy as np; p=Path('smoke_input'); p.mkdir(exist_ok=True); np.save(p/'sample.npy', np.zeros((8,8), dtype=np.float32))"
python run.py smoke_input smoke_output
```

Remove temporary smoke directories before packaging.

## 8. Troubleshooting

**`ModuleNotFoundError: semicon_restore`**: keep `src/semicon_restore` in the archive and run from the directory containing `run.py`.

**`Model checkpoint not found`**: copy `models/best.pt` into the submission without renaming it.

**CUDA unavailable**: check the NVIDIA driver, PyTorch wheel, and `torch.cuda.is_available()`. CPU fallback is automatic.

**Slow or out-of-memory execution**: the entrypoint performs eight model passes per image. Use a compatible GPU or allow CPU fallback; do not add evaluator prompts or manual settings.

**Input load failure**: rewrite the file as a real finite 2D array, for example `np.save(path, array.astype(np.float32))`.

## 9. Final submission checklist

- [x] `run.py` accepts `python run.py <input-dir> <output-dir>`.
- [x] All top-level `.npy` inputs are discovered.
- [x] The output directory is created automatically.
- [x] One same-named `.npy` output is created per input.
- [x] Outputs are grayscale 2D arrays with 2x target resolution.
- [x] Outputs are `float32`, finite, and bounded to `[0, 1]`.
- [x] `models/best.pt` is present and loads successfully.
- [x] CUDA is auto-detected and CPU fallback exists.
- [x] Inference needs no API key, prompt, manual configuration, or model download.
- [ ] The final archive retains `src/semicon_restore` or vendors that package into the four-item bundle.
- [ ] Offline installation wheels are supplied or preinstalled on the evaluator machine.
- [ ] Dependency versions are pinned or approved by the evaluation environment.

The last three items are packaging/environment actions that must be checked against the exact archive and evaluator machine.

## 10. Additional documentation

* [Technical architecture](documentation/IEEE_Paper_Architecture_Description.md)
* [Submission guide](documentation/SUBMISSION_GUIDE.md)
* [Configuration notes](configs/README.md)
* [Test guide](tests/README.md)

## License

Distributed under the MIT License. See [LICENSE](LICENSE).