from __future__ import annotations

import argparse
import time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from semicon_restore.checkpoint import load_checkpoint
from semicon_restore.inference import SelfEnsemble
from semicon_restore.models import build_model

def load_npy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False).astype(np.float32)

def save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))

def run_test_inference(
    ckpt_path: Path,
    test_dir: Path,
    output_dir: Path,
    tta_folds: int = 8,
    device: torch.device | None = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Error: Checkpoint file {ckpt_path} not found!")

    candidates = [
        test_dir,
        Path("Test_NoisyLR/NoisyLR"),
        Path("Test_NoisyLR"),
        Path("C:/Users/krish/semicon2026/Test_NoisyLR/NoisyLR"),
    ]
    for cand in candidates:
        if cand.exists():
            test_dir = cand
            break

    print(f"Loading production model checkpoint from {ckpt_path}...")
    ckpt = load_checkpoint(ckpt_path)
    model = build_model(ckpt["model_config"])

    if "ema" in ckpt and ckpt["ema"] is not None:
        print("Using EMA weights...")
        model.load_state_dict(ckpt["ema"])
    elif "model" in ckpt:
        print("Using model weights...")
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.to(device).eval()
    ensemble_wrapper = SelfEnsemble(model, tta_folds)

    test_files = sorted(list(test_dir.glob("*.npy")))
    print(f"Found {len(test_files)} test images in {test_dir}")
    print(f"Running {tta_folds}-Fold TTA inference on {device}...")

    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    with torch.inference_mode():
        for test_path in tqdm(test_files, desc="Restoring Test Wafers"):
            noisy_np = load_npy(test_path)
            noisy_tensor = torch.from_numpy(noisy_np)[None, None].to(device)

            pred_tensor = ensemble_wrapper(noisy_tensor).float().clamp(0, 1)
            pred_np = pred_tensor.squeeze().cpu().numpy()

            out_path = output_dir / test_path.name
            save_npy(out_path, pred_np)

    elapsed = time.time() - start_time
    fps = len(test_files) / elapsed if test_files else 0.0
    print("\n=======================================================")
    print(f"=== TEST INFERENCE COMPLETED IN {elapsed:.2f}s ({fps:.2f} img/s) ===")
    print(f"Saved all {len(test_files)} restored outputs to {output_dir.resolve()}")
    print("=======================================================\n")

def main():
    parser = argparse.ArgumentParser(description="Production Test Dataset Inference")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"), help="Path to best.pt checkpoint")
    parser.add_argument("--test-dir", type=Path, default=Path("Test_NoisyLR/NoisyLR"), help="Path to test images")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory for restored .npy files")
    parser.add_argument("--tta", type=int, default=8, help="TTA folds (1, 8, 16)")
    args = parser.parse_args()

    run_test_inference(
        ckpt_path=args.checkpoint,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        tta_folds=args.tta,
    )

if __name__ == "__main__":
    main()
