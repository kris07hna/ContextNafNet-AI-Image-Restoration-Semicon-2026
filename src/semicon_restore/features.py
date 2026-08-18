from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .degradation import DegradationParams, variance_stabilize

INPUT_MODES = {"raw4": 4, "noise_aware": 8}


def input_channels(mode: str) -> int:
    if mode not in INPUT_MODES:
        raise ValueError(f"Unknown input mode {mode}, expected one of {sorted(INPUT_MODES)}")
    return INPUT_MODES[mode]


def gaussian_kernel(sigma: float, radius: int, device=None, dtype=torch.float32) -> torch.Tensor:
    if sigma <= 0 or radius < 1:
        raise ValueError("Gaussian blur needs sigma > 0 and radius >= 1")
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    weights = torch.exp(-0.5 * (offsets / sigma) ** 2)
    return weights / weights.sum()


def gaussian_blur(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    radius = kernel.numel() // 2
    padded = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(F.conv2d(padded, kernel.view(1, 1, 1, -1)), kernel.view(1, 1, -1, 1))


def raw_features(raw: torch.Tensor) -> torch.Tensor:
    bounded = raw.clamp(0, 1)
    overflow = raw - bounded
    return torch.cat((raw, bounded, overflow, overflow.abs()), dim=1)


class NoiseFeatures(nn.Module):
    # The noise standard deviation spans 14.6x across the intensity range, so the useful denoising
    # strength is strongly signal dependent. These channels hand the calibrated relation to the
    # network directly instead of making it infer the sensor model from pixels: a predicted noise
    # level, the residual whitened by it, a variance-stabilised copy of the input, and the ratio of
    # observed local spread to predicted noise, which separates real structure from noise.
    def __init__(self, mode: str = "raw4", quadratic: float = 0.026627, linear: float = 0.0,
                 constant: float = 3.929e-05, vst_margin: float = 0.05, blur_sigma: float = 1.0,
                 blur_radius: int = 2, limit: float = 6.0):
        super().__init__()
        self.mode = mode
        self.channels = input_channels(mode)
        self.limit = float(limit)
        self.quadratic, self.linear, self.constant = float(quadratic), float(linear), float(constant)
        if mode == "raw4":
            return
        noise = DegradationParams(quadratic=quadratic, linear=linear, constant=constant,
                                  detail_mode="none", vst_margin=vst_margin)
        self.stabilizer = noise.stabilizer()
        self.floor = noise.additive_sigma
        self.reference = float(noise.noise_std(1.0))
        self.register_buffer("kernel", gaussian_kernel(blur_sigma, blur_radius), persistent=False)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        if self.mode == "raw4":
            return raw_features(raw)
        dtype = raw.dtype
        value = raw.float()
        bounded = value.clamp(0, 1)
        overflow = value - bounded
        kernel = self.kernel.to(value.dtype)
        # Estimate the noise-free local mean from the in-range signal: values outside [0, 1] are
        # clipping artefacts and would drag the estimate away from the true block intensity.
        mean = gaussian_blur(bounded, kernel)
        local_variance = (gaussian_blur(bounded * bounded, kernel) - mean * mean).clamp_min(0.0)
        sigma = torch.sqrt(self.quadratic * mean * mean + self.linear * mean + self.constant).clamp_min(self.floor)
        features = torch.cat((
            value, bounded, overflow, overflow.abs(),
            sigma / self.reference,
            ((value - mean) / sigma).clamp(-self.limit, self.limit),
            variance_stabilize(value, self.stabilizer),
            (torch.sqrt(local_variance) / sigma).clamp(0.0, self.limit),
        ), dim=1)
        return features.to(dtype)
