from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from semicon_restore.checkpoint import load_checkpoint
from semicon_restore.data import PairedNpyDataset, pair_paths, read_manifest
from semicon_restore.inference import SelfEnsemble
from semicon_restore.metrics import psnr, ssim
from semicon_restore.models import build_model

def evaluate_checkpoint(
    ckpt_path: Path,
    data_root: Path,
    manifest_path: Path,
    split_group: str = "val",
    tta_folds: int = 8,
    device: torch.device | None = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found!")

    print(f"Loading model checkpoint from {ckpt_path}...")
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

    candidates = [
        (data_root / "train/train/NoisyLR", data_root / "train/train/GT"),
        (data_root / "NoisyLR", data_root / "GT"),
        (Path("C:/Users/krish/semicon2026/train/train/NoisyLR"), Path("C:/Users/krish/semicon2026/train/train/GT")),
    ]

    lr_dir, gt_dir = None, None
    for lr_cand, gt_cand in candidates:
        if lr_cand.exists() and gt_cand.exists():
            lr_dir, gt_dir = lr_cand, gt_cand
            break

    if lr_dir is None or gt_dir is None:
        raise FileNotFoundError(f"Dataset directory not found under {data_root}")

    all_pairs = pair_paths(lr_dir, gt_dir)
    manifest = read_manifest(manifest_path) if manifest_path.exists() else {}
    val_files = set(manifest.get(split_group, []))

    val_pairs = [(lr, gt) for lr, gt in all_pairs if lr.name in val_files] if val_files else all_pairs
    dataset = PairedNpyDataset(val_pairs, crop_size=None, training=False)

    print(f"Evaluating {len(dataset)} validation samples with {tta_folds}-Fold TTA on {device}...")

    psnr_scores = []
    ssim_scores = []

    with torch.inference_mode():
        for batch in tqdm(dataset, desc="Evaluating"):
            noisy_tensor = batch["lr"].unsqueeze(0).to(device)
            gt_tensor = batch["gt"].unsqueeze(0).to(device)

            pred_tensor = ensemble_wrapper(noisy_tensor).float().clamp(0, 1)

            psnr_val = float(psnr(pred_tensor, gt_tensor).item())
            ssim_val = float(ssim(pred_tensor, gt_tensor).item())

            psnr_scores.append(psnr_val)
            ssim_scores.append(ssim_val)

    mean_psnr = float(np.mean(psnr_scores))
    mean_ssim = float(np.mean(ssim_scores))

    print("\n=======================================================")
    print(f"=== PRODUCTION MODEL EVALUATION RESULTS ({tta_folds}-Fold TTA) ===")
    print(f"Checkpoint File : {ckpt_path}")
    print(f"Dataset Split   : {split_group} ({len(dataset)} images)")
    print(f"Average PSNR    : {mean_psnr:.4f} dB")
    print(f"Average SSIM    : {mean_ssim:.5f}")
    print("=======================================================\n")

    return {"psnr": mean_psnr, "ssim": mean_ssim}

def main():
    parser = argparse.ArgumentParser(description="Evaluate Production Model Checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"), help="Path to checkpoint best.pt")
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Data root path")
    parser.add_argument("--manifest", type=Path, default=Path("splits/grouped-v1.json"), help="Validation manifest path")
    parser.add_argument("--tta", type=int, default=8, help="TTA folds (1, 8, 16)")
    args = parser.parse_args()

    evaluate_checkpoint(
        ckpt_path=args.checkpoint,
        data_root=args.data_root,
        manifest_path=args.manifest,
        tta_folds=args.tta,
    )

if __name__ == "__main__":
    main()
