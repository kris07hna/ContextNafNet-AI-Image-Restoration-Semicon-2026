from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_PARAMS_PATH = Path("configs/degradation-v1.json")


def _is_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor)


def _sqrt(value):
    return torch.sqrt(value) if _is_tensor(value) else np.sqrt(value)


def _asinh(value):
    return torch.asinh(value) if _is_tensor(value) else np.arcsinh(value)


@dataclass(frozen=True)
class VarianceStabilizer:
    quadratic: float
    linear: float
    constant: float
    scale: float
    denominator: float
    offset: float
    normalizer: float


@dataclass(frozen=True)
class DegradationParams:
    # var(LR - A(GT)) = quadratic * d^2 + linear * d + detail * s^2 + constant, where
    # d = A(GT) is the 2x2 block mean and s^2 = A(GT^2) - d^2 the within-block variance.
    # The detail term is the dominant noise source on edges and is white; no linear
    # downsampling operator removes it, so it is modelled as detail-proportional noise.
    quadratic: float = 0.026627
    linear: float = 0.0
    detail: float = 0.452940
    constant: float = 3.929e-05
    detail_mode: str = "blockmix"
    kernel: tuple[tuple[float, ...], ...] | None = None
    vst_margin: float = 0.05
    clip_low: float = -0.5
    clip_high: float = 3.0
    source: str = "default"

    def __post_init__(self) -> None:
        if self.quadratic <= 0 or self.linear < 0 or self.detail < 0 or self.constant < 0:
            raise ValueError("Degradation noise parameters must satisfy quadratic > 0 and the rest >= 0")
        if self.detail_mode not in {"blockmix", "gaussian", "none"}:
            raise ValueError(f"Unknown detail_mode: {self.detail_mode}")
        if self.detail_mode == "blockmix" and not 0.0 < self.detail < 1.0:
            raise ValueError("blockmix detail must lie in (0, 1)")
        if self.vst_margin <= 0:
            raise ValueError("vst_margin must be positive")
        if self.kernel is not None:
            size = len(self.kernel)
            if size % 2 or any(len(row) != size for row in self.kernel):
                raise ValueError("Degradation kernel must be square with an even side length")

    # Poisson(rate * d) / rate  times  Gamma(shape, 1/shape) has mean d and
    # variance d^2 / shape + (d / rate) * (1 + 1 / shape), which matches the fit
    # when shape = 1 / quadratic and rate = (1 + quadratic) / linear.
    @property
    def gamma_shape(self) -> float:
        return 1.0 / self.quadratic

    @property
    def poisson_rate(self) -> float:
        return (1.0 + self.quadratic) / self.linear if self.linear > 0 else float("inf")

    @property
    def additive_sigma(self) -> float:
        return math.sqrt(self.constant)

    # A convex combination of the four block pixels with Dirichlet(alpha) weights has mean
    # A(GT) and variance s^2 / (4 alpha + 1). Solving for the fitted detail coefficient, after
    # discounting the (1 + quadratic) inflation the later multiplicative draw applies, gives:
    @property
    def dirichlet_alpha(self) -> float:
        effective = self.detail / (1.0 + self.quadratic)
        return 0.25 * (1.0 / effective - 1.0)

    def noise_variance(self, signal, detail_variance=None):
        variance = self.quadratic * signal * signal + self.linear * signal + self.constant
        return variance if detail_variance is None else variance + self.detail * detail_variance

    def noise_std(self, signal, detail_variance=None):
        return _sqrt(self.noise_variance(signal, detail_variance))

    def kernel_array(self) -> np.ndarray | None:
        return None if self.kernel is None else np.asarray(self.kernel, dtype=np.float64)

    def stabilizer(self) -> VarianceStabilizer:
        a, b = self.quadratic, self.linear
        # T(x) = integral of dx / sqrt(a x^2 + b x + c) is defined on all of R only when
        # 4 a c > b^2. With a fitted linear term small enough to violate that, the transform
        # uses a slightly inflated constant: it under-amplifies the darkest pixels a little
        # but keeps T smooth and monotone everywhere instead of clipping the negative tail.
        constant = max(self.constant, b * b / (4.0 * a) * (1.0 + self.vst_margin))
        denominator = math.sqrt(4.0 * a * constant - b * b)
        scale = 1.0 / math.sqrt(a)
        offset = math.asinh(b / denominator)
        normalizer = scale * (math.asinh((2.0 * a + b) / denominator) - offset)
        return VarianceStabilizer(a, b, constant, scale, denominator, offset, normalizer)

    @classmethod
    def from_dict(cls, values: dict) -> DegradationParams:
        values = dict(values)
        kernel = values.get("kernel")
        values["kernel"] = None if kernel is None else tuple(tuple(float(v) for v in row) for row in kernel)
        for key in ("quadratic", "linear", "detail", "constant", "vst_margin", "clip_low", "clip_high"):
            if key in values:
                values[key] = float(values[key])
        for key in ("source", "detail_mode"):
            if key in values:
                values[key] = str(values[key])
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in values.items() if key in known})

    def to_dict(self) -> dict:
        return {
            "quadratic": self.quadratic, "linear": self.linear, "detail": self.detail,
            "constant": self.constant, "detail_mode": self.detail_mode,
            "kernel": None if self.kernel is None else [list(row) for row in self.kernel],
            "vst_margin": self.vst_margin, "clip_low": self.clip_low, "clip_high": self.clip_high,
            "source": self.source,
        }

    @classmethod
    def load(cls, path: str | Path) -> DegradationParams:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload.get("degradation", payload))

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"degradation": self.to_dict()}
        if extra:
            payload.update(extra)
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temp.replace(path)


DEFAULT_PARAMS = DegradationParams()


def resolve_params(spec: DegradationParams | dict | str | Path | None) -> DegradationParams:
    if spec is None:
        return DEFAULT_PARAMS
    if isinstance(spec, DegradationParams):
        return spec
    if isinstance(spec, dict):
        return DegradationParams.from_dict(spec)
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"Degradation parameters not found: {path}")
    return DegradationParams.load(path)


def variance_stabilize(value, stabilizer: VarianceStabilizer, normalize: bool = True):
    inner = (2.0 * stabilizer.quadratic * value + stabilizer.linear) / stabilizer.denominator
    transformed = stabilizer.scale * (_asinh(inner) - stabilizer.offset)
    return transformed / stabilizer.normalizer if normalize else transformed


def area_downsample(image):
    height, width = image.shape[-2], image.shape[-1]
    if height % 2 or width % 2:
        raise ValueError(f"Area downsample needs even spatial dimensions, got {(height, width)}")
    if _is_tensor(image):
        return F.avg_pool2d(image, 2)
    array = np.ascontiguousarray(image)
    return array.reshape(*array.shape[:-2], height // 2, 2, width // 2, 2).mean(axis=(-3, -1))


def block_repeat(image):
    if _is_tensor(image):
        return image.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    return np.repeat(np.repeat(image, 2, axis=-2), 2, axis=-1)


def kernel_downsample(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    size = kernel.shape[0]
    height, width = image.shape[-2], image.shape[-1]
    if height % 2 or width % 2:
        raise ValueError(f"Kernel downsample needs even spatial dimensions, got {(height, width)}")
    before, after = size // 2 - 1, size // 2
    padded = np.pad(np.asarray(image, dtype=np.float64), ((before, after), (before, after)), mode="reflect")
    output = np.zeros((height // 2, width // 2), dtype=np.float64)
    for dy in range(size):
        for dx in range(size):
            weight = kernel[dy, dx]
            if weight:
                output += weight * padded[dy:dy + height:2, dx:dx + width:2]
    return output


def downsample(image: np.ndarray, params: DegradationParams) -> np.ndarray:
    kernel = params.kernel_array()
    return area_downsample(image) if kernel is None else kernel_downsample(image, kernel)


def block_statistics(image: np.ndarray, params: DegradationParams) -> tuple[np.ndarray, np.ndarray]:
    mean = downsample(image, params)
    detail = np.maximum(area_downsample(np.asarray(image, dtype=np.float64) ** 2) - area_downsample(image) ** 2, 0.0)
    return mean, detail


def block_values(image: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(image, dtype=np.float64)
    height, width = array.shape[-2], array.shape[-1]
    if height % 2 or width % 2:
        raise ValueError(f"Block view needs even spatial dimensions, got {(height, width)}")
    return array.reshape(height // 2, 2, width // 2, 2).transpose(1, 3, 0, 2).reshape(4, height // 2, width // 2)


def block_mix(image: np.ndarray, generator: np.random.Generator, alpha: float) -> np.ndarray:
    values = block_values(image)
    weights = generator.gamma(alpha, 1.0, size=values.shape)
    total = weights.sum(axis=0, keepdims=True)
    weights = np.where(total > 0.0, weights / np.maximum(total, 1e-300), 0.25)
    return (weights * values).sum(axis=0)


def apply_noise(mean: np.ndarray, generator: np.random.Generator, params: DegradationParams,
                detail_variance: np.ndarray | float = 0.0) -> np.ndarray:
    signal = np.maximum(np.asarray(mean, dtype=np.float64), 0.0)
    rate, shape = params.poisson_rate, params.gamma_shape
    noisy = generator.poisson(rate * signal) / rate if math.isfinite(rate) else signal
    noisy = noisy * generator.gamma(shape, 1.0 / shape, size=signal.shape)
    variance = params.constant + (params.detail * np.asarray(detail_variance, dtype=np.float64)
                                  if params.detail_mode == "gaussian" else 0.0)
    if np.any(variance > 0):
        noisy = noisy + generator.normal(0.0, 1.0, size=signal.shape) * np.sqrt(variance)
    return np.clip(noisy, params.clip_low, params.clip_high).astype(np.float32, copy=False)


def degrade(gt: np.ndarray, generator: np.random.Generator, params: DegradationParams) -> np.ndarray:
    if params.detail_mode == "blockmix":
        mean = block_mix(gt, generator, params.dirichlet_alpha)
        if params.kernel is not None:
            mean = mean + (downsample(gt, params) - area_downsample(gt))
    else:
        mean = downsample(gt, params)
    _, detail = block_statistics(gt, params)
    return apply_noise(mean, generator, params, detail)


def resample_hr(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape[-2] == size and image.shape[-1] == size:
        return np.ascontiguousarray(image, dtype=np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None]
    shrinking = image.shape[-1] > size
    resized = (F.interpolate(tensor, size=(size, size), mode="area") if shrinking else
               F.interpolate(tensor, size=(size, size), mode="bicubic", align_corners=False))
    return resized[0, 0].clamp(0.0, 1.0).numpy()


def synthesize_pair(gt_source: np.ndarray, generator: np.random.Generator, params: DegradationParams,
                    target_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    size = int(target_size or gt_source.shape[-1])
    if size % 2:
        raise ValueError(f"Synthetic ground-truth size must be even, got {size}")
    gt = resample_hr(gt_source, size)
    return degrade(gt, generator, params), np.ascontiguousarray(gt, dtype=np.float32)
