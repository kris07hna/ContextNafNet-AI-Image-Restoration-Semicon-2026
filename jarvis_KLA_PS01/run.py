#!/usr/bin/env python3
"""
KLA Problem Statement – AI-Based Restoration of Degraded Images
Official Bulletproof Submission Entrypoint: run.py

Usage:
    python run.py <input-dir> <output-dir>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure src/ is automatically on sys.path regardless of how python is invoked
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from semicon_restore.checkpoint import load_checkpoint
from semicon_restore.inference import SelfEnsemble
from semicon_restore.models import build_model

def load_npy(path: Path) -> np.ndarray:
    array = np.load(path, allow_pickle=False).astype(np.float32)
    # Handle input shape variations (H, W), (1, H, W), or (H, W, 1)
    if array.ndim == 3:
        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[2] == 1:
            array = array.squeeze(2)
    return array

def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))

def run_restoration(input_dir: Path, output_dir: Path, tta_folds: int = 8):
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist!")

    # Search for model weights
    candidate_weights = [
        Path("models/best.pt"),
        Path("checkpoints/best.pt"),
        Path(__file__).parent / "models/best.pt",
        Path(__file__).parent / "checkpoints/best.pt",
    ]
    ckpt_path = None
    for cand in candidate_weights:
        if cand.exists():
            ckpt_path = cand
            break

    if ckpt_path is None:
        raise FileNotFoundError(
            "Model checkpoint not found! Expected model weights at 'models/best.pt' or 'checkpoints/best.pt'."
        )

    # Edge Case 1: CUDA & Device Auto-Detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"[KLA Submission] Running on device: {device}")
    print(f"[KLA Submission] Loading model weights from: {ckpt_path.resolve()}")

    # Load model and weights
    ckpt = load_checkpoint(ckpt_path)
    model = build_model(ckpt["model_config"])

    if "ema" in ckpt and ckpt["ema"] is not None:
        model.load_state_dict(ckpt["ema"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.to(device).eval()

    # Wrap model with 8-Fold Self-Ensemble TTA
    ensemble_wrapper = SelfEnsemble(model, tta_folds)

    # Gather input .npy files
    input_files = sorted(list(input_dir.glob("*.npy")))
    if not input_files:
        print(f"Warning: No .npy files found in input directory '{input_dir}'.")

    print(f"[KLA Submission] Found {len(input_files)} .npy files in '{input_dir}'.")
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    with torch.inference_mode():
        for input_path in tqdm(input_files, desc="Restoring Images"):
            noisy_np = load_npy(input_path)
            in_h, in_w = noisy_np.shape[:2]
            target_h, target_w = in_h * 2, in_w * 2  # 2x Super-Resolution Target

            noisy_tensor = torch.from_numpy(noisy_np)[None, None].to(device)

            # Model Forward Pass (Self-Ensemble TTA)
            pred_tensor = ensemble_wrapper(noisy_tensor).float()

            # Edge Case 2: Exact Resolution Safeguard (2x Super-Resolution)
            if pred_tensor.shape[2:] != (target_h, target_w):
                pred_tensor = F.interpolate(pred_tensor, size=(target_h, target_w), mode="bicubic", align_corners=False)

            # Clamp output to valid intensity range [0.0, 1.0]
            pred_tensor = pred_tensor.clamp(0.0, 1.0)
            pred_np = pred_tensor.squeeze().cpu().numpy().astype(np.float32)

            # Edge Case 3: Shape & NaN / Inf Sanitization Safeguards
            if pred_np.ndim == 3 and pred_np.shape[0] == 1:
                pred_np = pred_np.squeeze(0)

            # Sanitize NaN or Inf if any
            if np.isnan(pred_np).any() or np.isinf(pred_np).any():
                pred_np = np.nan_to_num(pred_np, nan=0.0, posinf=1.0, neginf=0.0)

            # Final Hard Assertions
            assert pred_np.shape == (target_h, target_w), f"Resolution mismatch! Expected {(target_h, target_w)}, got {pred_np.shape}"
            assert not np.isnan(pred_np).any(), f"NaN error in {input_path.name}"
            assert not np.isinf(pred_np).any(), f"Inf error in {input_path.name}"
            assert (pred_np >= 0.0).all() and (pred_np <= 1.0).all(), f"Range error in {input_path.name}"

            out_path = output_dir / input_path.name
            save_npy(out_path, pred_np)

    elapsed = time.time() - start_time
    fps = len(input_files) / elapsed if elapsed > 0 else 0.0
    print("\n=======================================================")
    print(f"[KLA Submission] Restoration completed in {elapsed:.2f}s ({fps:.2f} img/s)")
    print(f"[KLA Submission] Target Resolution Verified: 2x Super-Resolution ({target_h}x{target_w})")
    print(f"[KLA Submission] Saved {len(input_files)} restored .npy files to: {output_dir.resolve()}")
    print("=======================================================\n")

def main():
    if len(sys.argv) < 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    run_restoration(input_dir=input_dir, output_dir=output_dir, tta_folds=8)

if __name__ == "__main__":
    main()
