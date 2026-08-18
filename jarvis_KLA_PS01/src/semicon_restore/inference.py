from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .checkpoint import load_checkpoint
from .degradation import (
    DegradationParams,
    area_downsample,
    block_repeat,
    resolve_params,
)
from .engine import autocast_context
from .io import discover_npy, load_array, save_array_atomic
from .models import build_model
from .runtime import DeviceInfo

TRANSFORM_COUNTS = frozenset({1, 4, 8, 16})


def dihedral(x: torch.Tensor, index: int) -> torch.Tensor:
    # The eight symmetries of the square, generated as an optional transpose followed by a rotation.
    # Indices 0-3 are the rotations alone, so a four-way ensemble is exactly the first half of the
    # eight-way one and the two modes need no separate tables.
    transposed = x.transpose(-2, -1) if index & 4 else x
    return torch.rot90(transposed, index & 3, dims=(-2, -1))


def dihedral_inverse(x: torch.Tensor, index: int) -> torch.Tensor:
    # (R . T)^-1 = T^-1 . R^-1, and a transpose is its own inverse, so undoing the pair reverses the
    # order and negates the rotation.
    rotated = torch.rot90(x, -(index & 3), dims=(-2, -1))
    return rotated.transpose(-2, -1) if index & 4 else rotated


class SelfEnsemble(nn.Module):
    # Averaging a prediction over the symmetries of the square cancels the part of the error that
    # depends on orientation. Predictions are accumulated rather than stacked so the memory cost stays
    # that of a single output no matter how many transforms are averaged.
    # When transforms=16, 8 geometric transforms at 1.0x scale and 8 at 1.05x scale are averaged.
    def __init__(self, model: nn.Module, transforms: int = 8):
        super().__init__()
        if transforms not in TRANSFORM_COUNTS:
            raise ValueError(f"transforms must be one of {sorted(TRANSFORM_COUNTS)}, got {transforms}")
        self.model = model
        self.transforms = transforms

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        total = None
        count = 0
        geo_count = 8 if self.transforms == 16 else self.transforms

        # Scale 1.0x geometric transforms
        for index in range(geo_count):
            prediction = dihedral_inverse(self.model(dihedral(raw, index)).float(), index)
            total = prediction if total is None else total + prediction
            count += 1

        # Scale 1.05x micro-scale transforms if transforms == 16
        if self.transforms == 16:
            h, w = raw.shape[-2], raw.shape[-1]
            scaled_h, scaled_w = round(h * 1.05) & ~1, round(w * 1.05) & ~1
            raw_scaled = torch.nn.functional.interpolate(raw, size=(scaled_h, scaled_w), mode="bicubic", align_corners=False)
            for index in range(8):
                pred_scaled = dihedral_inverse(self.model(dihedral(raw_scaled, index)).float(), index)
                target_h = pred_scaled.shape[-2] // 2 if pred_scaled.shape[-2] > scaled_h else pred_scaled.shape[-2]
                target_w = pred_scaled.shape[-1] // 2 if pred_scaled.shape[-1] > scaled_w else pred_scaled.shape[-1]
                pred_rescaled = torch.nn.functional.interpolate(pred_scaled, size=(2 * h, 2 * w), mode="bicubic", align_corners=False)
                total = total + pred_rescaled
                count += 1

        return total / count


class Ensemble(nn.Module):
    def __init__(self, models: list[nn.Module]):
        super().__init__()
        if not models:
            raise ValueError("Ensemble needs at least one model")
        self.models = nn.ModuleList(models)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        total = None
        for model in self.models:
            prediction = model(raw).float()
            total = prediction if total is None else total + prediction
        return total / len(self.models)


def back_project(prediction: torch.Tensor, raw: torch.Tensor, params: DegradationParams,
                 variance: float) -> torch.Tensor:
    mean = area_downsample(prediction)
    detail = (area_downsample(prediction * prediction) - mean * mean).clamp_min(0.0)
    gain = variance / (variance + params.noise_variance(mean.clamp_min(0.0), detail))
    # block_repeat spreads a single correction over the four pixels of its block, which is the only
    # honest choice: one observed value constrains the block mean and says nothing about how the
    # detail inside the block is arranged.
    return prediction + block_repeat(gain * (raw - mean))


class BackProjection(nn.Module):
    # The calibrated degradation satisfies E[LR | GT] = A(GT) exactly: the block-mix draw has the block
    # mean as its mean and the multiplicative gamma draw has mean one. Measured on the validation split,
    # mean(LR - A(GT)) = +8.1e-05, so the residual between the observation and the 2x2 average of a
    # prediction really is zero mean and the correction below is well founded rather than heuristic.
    #
    # It still does not help, and the measurement says why. The Wiener gain assumes the model's error and
    # the observation noise are independent, but the model saw that observation: on the validation split
    # the two errors correlate at +0.53, and cov / var(model error) = 1.27, so the model has already
    # absorbed more of the observation than an independent estimator would carry. That drives the
    # optimal gain negative (-0.066), and even the oracle per-noise-level gain recovers only 2.4% of the
    # block-mean error, which is 43.5% of the total, so the ceiling is under 0.04 dB. The best measured
    # constant gain scored +0.012 dB PSNR for -0.0005 SSIM. Keep this disabled unless a model trained on
    # fresh synthetic noise stops echoing the observation; then the premise is worth re-testing.
    def __init__(self, model: nn.Module, params: DegradationParams, variance: float):
        super().__init__()
        if variance <= 0:
            raise ValueError(f"Back-projection variance must be positive, got {variance}")
        self.model = model
        self.params = params
        self.variance = float(variance)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return back_project(self.model(raw).float(), raw.float(), self.params, self.variance)


def load_inference_model(checkpoint_path: str | Path, device: DeviceInfo, use_ema: bool = True) -> torch.nn.Module:
    checkpoint = load_checkpoint(checkpoint_path)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["ema"] if use_ema else checkpoint["model"], strict=True)
    return model.to(device.device).eval()


def load_inference_pipeline(checkpoints: list[str | Path], device: DeviceInfo, use_ema: bool = True,
                            transforms: int = 1, back_projection_variance: float = 0.0,
                            degradation: str | Path | dict | DegradationParams | None = None) -> torch.nn.Module:
    # Refinements compose outwards: the checkpoint ensemble is averaged first, that average is what the
    # geometric ensemble is applied to, and back-projection is outermost so the consistency step acts on
    # the final prediction rather than on one member of an average that is about to move.
    models = [load_inference_model(path, device, use_ema) for path in checkpoints]
    model = models[0] if len(models) == 1 else Ensemble(models)
    if transforms > 1:
        model = SelfEnsemble(model, transforms)
    if back_projection_variance > 0:
        model = BackProjection(model, resolve_params(degradation), back_projection_variance)
    return model.to(device.device).eval()


def restore_directory(model: torch.nn.Module, input_dir: str | Path, output_dir: str | Path, device: DeviceInfo,
                      batch_size: int = 8, overwrite: bool = False) -> dict:
    paths = discover_npy(input_dir)
    if not paths:
        raise ValueError(f"No .npy inputs found in {input_dir}")
    groups: dict[tuple[int, int], list[tuple[Path, np.ndarray]]] = defaultdict(list)
    read_started = time.perf_counter()
    for path in paths:
        array = load_array(path)
        if array.shape not in {(128, 128), (256, 256)}:
            raise ValueError(f"Unsupported official input shape {array.shape}: {path}")
        groups[array.shape].append((path, array))
    read_seconds = time.perf_counter() - read_started
    output_root = Path(output_dir)
    model_seconds, write_seconds = 0.0, 0.0
    amp = device.device.type == "cuda"
    with torch.inference_mode():
        for items in groups.values():
            for start in range(0, len(items), batch_size):
                batch_items = items[start:start + batch_size]
                batch = np.stack([array for _, array in batch_items])[:, None]
                tensor = torch.from_numpy(batch).to(device.device, non_blocking=True)
                if device.device.type == "cuda":
                    torch.cuda.synchronize(device.device)
                model_started = time.perf_counter()
                with autocast_context(device.device, amp, device.precision):
                    prediction = model(tensor)
                if device.device.type == "cuda":
                    torch.cuda.synchronize(device.device)
                model_seconds += time.perf_counter() - model_started
                output = prediction.float().clamp(0, 1).cpu().numpy()[:, 0].astype(np.float32, copy=False)
                write_started = time.perf_counter()
                for (source, source_array), restored in zip(batch_items, output):
                    if restored.shape != (source_array.shape[0] * 2, source_array.shape[1] * 2):
                        raise RuntimeError(f"Incorrect model output shape for {source.name}: {restored.shape}")
                    save_array_atomic(output_root / source.name, restored, overwrite)
                write_seconds += time.perf_counter() - write_started
    return {"images": len(paths), "read_seconds": read_seconds, "model_seconds": model_seconds,
            "write_seconds": write_seconds, "images_per_second_model": len(paths) / max(model_seconds, 1e-9)}
