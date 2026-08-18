from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from semicon_restore.checkpoint import load_checkpoint
from semicon_restore.inference import SelfEnsemble
from semicon_restore.metrics import psnr, ssim
from semicon_restore.models import build_model

def load_npy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False).astype(np.float32)

def compute_snr(signal: np.ndarray, target: np.ndarray) -> float:
    signal_power = np.mean(target ** 2)
    noise_power = np.mean((signal - target) ** 2)
    if noise_power < 1e-10:
        return 100.0
    return float(10.0 * np.log10(signal_power / noise_power))

def compare_single_image(
    filename: str,
    ckpt_path: Path,
    data_root: Path,
    output_png: Path,
    tta_folds: int = 8,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidates = [
        (data_root / "train/train/NoisyLR" / filename, data_root / "train/train/GT" / filename),
        (data_root / "NoisyLR" / filename, data_root / "GT" / filename),
        (Path("C:/Users/krish/semicon2026/train/train/NoisyLR") / filename, Path("C:/Users/krish/semicon2026/train/train/GT") / filename),
    ]

    lr_path, gt_path = None, None
    for lr_cand, gt_cand in candidates:
        if lr_cand.exists() and gt_cand.exists():
            lr_path, gt_path = lr_cand, gt_cand
            break

    if lr_path is None or gt_path is None:
        print(f"Error: Sample {filename} not found under {data_root}")
        return

    print(f"Loading model checkpoint from {ckpt_path}...")
    ckpt = load_checkpoint(ckpt_path)
    model = build_model(ckpt["model_config"])

    if "ema" in ckpt and ckpt["ema"] is not None:
        model.load_state_dict(ckpt["ema"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.to(device).eval()
    ensemble_wrapper = SelfEnsemble(model, tta_folds)

    noisy_np = load_npy(lr_path)
    gt_np = load_npy(gt_path)

    noisy_tensor = torch.from_numpy(noisy_np)[None, None].to(device)
    gt_tensor = torch.from_numpy(gt_np)[None, None].to(device)

    noisy_up_tensor = F.interpolate(noisy_tensor, size=gt_np.shape, mode="bicubic", align_corners=False).clamp(0, 1)
    noisy_up_np = noisy_up_tensor.squeeze().cpu().numpy()

    with torch.inference_mode():
        pred_tensor = ensemble_wrapper(noisy_tensor).float().clamp(0, 1)
        pred_np = pred_tensor.squeeze().cpu().numpy()

    in_psnr = float(psnr(noisy_up_tensor, gt_tensor).item())
    out_psnr = float(psnr(pred_tensor, gt_tensor).item())

    in_ssim = float(ssim(noisy_up_tensor, gt_tensor).item())
    out_ssim = float(ssim(pred_tensor, gt_tensor).item())

    in_snr = compute_snr(noisy_up_np, gt_np)
    out_snr = compute_snr(pred_np, gt_np)
    snr_gain = out_snr - in_snr

    residual_err = np.abs(pred_np - gt_np)
    mae = float(np.mean(residual_err))

    print("\n=======================================================")
    print(f"=== SAMPLE RESTORATION & SNR ANALYSIS FOR {filename} ===")
    print(f"Original Noisy Input : PSNR {in_psnr:.2f} dB | SNR {in_snr:.2f} dB | SSIM {in_ssim:.4f}")
    print(f"Restored Output      : PSNR {out_psnr:.2f} dB | SNR {out_snr:.2f} dB | SSIM {out_ssim:.4f}")
    print(f"Net Gain / Error     : +{snr_gain:.2f} dB SNR Gain | MAE Loss: {mae:.5f}")
    print("=======================================================\n")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    plt.subplots_adjust(wspace=0.18)

    axes[0].imshow(noisy_up_np, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Noisy Input ({filename})\nPSNR: {in_psnr:.2f} dB | SNR: {in_snr:.2f} dB", fontsize=9.5, fontweight="bold", color="#d9534f")
    axes[0].axis("off")

    axes[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth\nClean Target Reference", fontsize=9.5, fontweight="bold", color="#5cb85c")
    axes[1].axis("off")

    axes[2].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Restored Output\nPSNR: {out_psnr:.2f} dB | SNR: {out_snr:.2f} dB\nSSIM: {out_ssim:.4f}", fontsize=9.5, fontweight="bold", color="#0275d8")
    axes[2].axis("off")

    im = axes[3].imshow(residual_err, cmap="inferno", vmin=0, vmax=0.15)
    axes[3].set_title(f"Residual Loss Heatmap\nMAE: {mae:.4f} | Noise Power Reduced", fontsize=9.5, fontweight="bold", color="#f0ad4e")
    axes[3].axis("off")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.suptitle(f"Semiconductor Wafer Restoration: {filename} Analysis", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison panel figure to {output_png}")

def main():
    parser = argparse.ArgumentParser(description="Compare Single Wafer Sample & Compute SNR")
    parser.add_argument("--file", type=str, default="002060.npy", help="Sample filename (e.g. 000048.npy or 002060.npy)")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"), help="Path to best.pt checkpoint")
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Data root path")
    parser.add_argument("--output-png", type=Path, default=Path("reports/sample_comparison.png"), help="Output PNG path")
    parser.add_argument("--tta", type=int, default=8, help="TTA folds (1, 8, 16)")
    args = parser.parse_args()

    compare_single_image(
        filename=args.file,
        ckpt_path=args.checkpoint,
        data_root=args.data_root,
        output_png=args.output_png,
        tta_folds=args.tta,
    )

if __name__ == "__main__":
    main()
