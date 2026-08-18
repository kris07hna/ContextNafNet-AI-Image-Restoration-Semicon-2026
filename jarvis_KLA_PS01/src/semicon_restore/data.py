from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler, get_worker_info

from .degradation import DegradationParams, resolve_params, synthesize_pair
from .io import load_array

# Scale augmentation is off by default because it measurably smooths the target: downscaling a
# ground-truth region halves the mean within-block variance, so the synthetic pairs would carry
# less high-frequency content than the real ones. Widen the range only to ablate that trade-off.
DEFAULT_SCALE_RANGE = (1.0, 1.0)


def pair_paths(lr_dir: str | Path, gt_dir: str | Path) -> list[tuple[Path, Path]]:
    lr_root, gt_root = Path(lr_dir), Path(gt_dir)
    lr = {p.name: p for p in lr_root.glob("*.npy") if not p.name.startswith("._")}
    gt = {p.name: p for p in gt_root.glob("*.npy") if not p.name.startswith("._")}
    names = sorted(lr.keys() & gt.keys())
    if set(lr) != set(gt):
        raise ValueError(f"Unpaired files: LR-only={sorted(set(lr)-set(gt))[:5]}, GT-only={sorted(set(gt)-set(lr))[:5]}")
    return [(lr[name], gt[name]) for name in names]


def _gradient_energy(patch: np.ndarray) -> float:
    return float(np.abs(np.diff(patch, axis=0)).mean() + np.abs(np.diff(patch, axis=1)).mean())


def _choose_position(image: np.ndarray, size: int, rng: random.Random,
                     probability: float = 0.0, candidates: int = 8) -> tuple[int, int]:
    if image.shape[0] < size or image.shape[1] < size:
        raise ValueError(f"Crop {size} does not fit {image.shape}")
    count = max(1, candidates if rng.random() < probability else 1)
    positions = [(rng.randint(0, image.shape[0] - size), rng.randint(0, image.shape[1] - size))
                 for _ in range(count)]
    if count == 1:
        return positions[0]
    bounded = np.clip(image, 0.0, 1.0)
    return max(positions, key=lambda p: _gradient_energy(bounded[p[0]:p[0] + size, p[1]:p[1] + size]))


def _crop_pair(lr: np.ndarray, gt: np.ndarray, size: int, rng: random.Random,
               detail_probability: float = 0.0, detail_candidates: int = 8) -> tuple[np.ndarray, np.ndarray]:
    y, x = _choose_position(lr, size, rng, detail_probability, detail_candidates)
    return lr[y:y + size, x:x + size], gt[2*y:2*y + 2*size, 2*x:2*x + 2*size]


def _synthetic_pair(gt: np.ndarray, size: int, rng: random.Random, generator: np.random.Generator,
                    params: DegradationParams, scale_range: tuple[float, float],
                    detail_probability: float = 0.0, detail_candidates: int = 8) -> tuple[np.ndarray, np.ndarray]:
    # Scale augmentation: take a source region larger than the wanted ground-truth patch and area
    # downscale it, so the patch carries finer structure than the fixed 2x relation ever presents.
    # Only scales at or above 1 are allowed; upscaling the source would bake interpolation blur
    # into the target and teach the model to predict it.
    target = 2 * size
    low, high = scale_range
    high = min(high, min(gt.shape[-2], gt.shape[-1]) / target)
    scale = low if high <= low else rng.uniform(low, high)
    source = min(min(gt.shape[-2], gt.shape[-1]), max(target, round(target * scale) & ~1))
    y, x = _choose_position(gt, source, rng, detail_probability, detail_candidates)
    return synthesize_pair(gt[y:y + source, x:x + source], generator, params, target_size=target)



def _augment(lr: np.ndarray, gt: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    k = rng.randrange(4)
    lr, gt = np.rot90(lr, k), np.rot90(gt, k)
    if rng.random() < 0.5:
        lr, gt = np.flip(lr, 1), np.flip(gt, 1)
    if rng.random() < 0.5:
        lr, gt = np.flip(lr, 0), np.flip(gt, 0)
    return np.ascontiguousarray(lr), np.ascontiguousarray(gt)


class PairedNpyDataset(Dataset):
    def __init__(self, pairs: list[tuple[Path, Path]], crop_size: int | None = 64, training: bool = True,
                 seed: int = 2026, detail_crop_probability: float = 0.0, detail_crop_candidates: int = 8,
                 synthetic_probability: float = 0.0,
                 degradation: DegradationParams | dict | str | Path | None = None,
                 scale_range: tuple[float, float] = DEFAULT_SCALE_RANGE):
        self.pairs, self.crop_size, self.training = pairs, crop_size, training
        if not 0.0 <= detail_crop_probability <= 1.0:
            raise ValueError("detail_crop_probability must be between 0 and 1")
        if not 0.0 <= synthetic_probability <= 1.0:
            raise ValueError("synthetic_probability must be between 0 and 1")
        if scale_range[0] < 1.0 or scale_range[1] < scale_range[0]:
            raise ValueError("scale_range must be ordered and start at 1.0 or above")
        self.seed = seed
        self.detail_crop_probability = detail_crop_probability
        self.detail_crop_candidates = detail_crop_candidates
        # Synthetic pairs need a crop to define the target size, and would leak the forward model
        # into the metric if they ever reached validation, so both are hard requirements.
        self.synthetic_probability = synthetic_probability if training and crop_size else 0.0
        self.degradation = resolve_params(degradation) if self.synthetic_probability > 0 else None
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.rng = random.Random(seed)
        self.generator = np.random.default_rng(seed)
        self.worker_id: int | None = None

    def set_crop_size(self, crop_size: int | None) -> None:
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        worker = get_worker_info()
        worker_id = worker.id if worker else -1
        if self.worker_id != worker_id:
            worker_seed = worker.seed if worker else self.seed
            self.rng = random.Random(worker_seed)
            self.generator = np.random.default_rng(worker_seed)
            self.worker_id = worker_id
        lr_path, gt_path = self.pairs[index]
        gt = load_array(gt_path)
        synthetic = self.crop_size is not None and self.rng.random() < self.synthetic_probability
        if synthetic:
            lr, gt = _synthetic_pair(gt, self.crop_size, self.rng, self.generator, self.degradation,
                                     self.scale_range, self.detail_crop_probability, self.detail_crop_candidates)
        else:
            lr = load_array(lr_path)
            if gt.shape != (lr.shape[0] * 2, lr.shape[1] * 2):
                raise ValueError(f"Shape mismatch for {lr_path.name}")
            if self.crop_size is not None:
                lr, gt = _crop_pair(lr, gt, self.crop_size, self.rng,
                                    self.detail_crop_probability, self.detail_crop_candidates)
        if self.training:
            lr, gt = _augment(lr, gt, self.rng)
        return {"lr": torch.from_numpy(lr[None]), "gt": torch.from_numpy(gt[None]), "name": lr_path.name,
                "synthetic": torch.tensor(float(synthetic))}


def read_manifest(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def morphology_scores(pairs: list[tuple[Path, Path]]) -> np.ndarray:
    scores = []
    for _, gt_path in pairs:
        gt = load_array(gt_path)
        gradient = np.abs(np.diff(gt, axis=0)).mean() + np.abs(np.diff(gt, axis=1)).mean()
        centered = gt - float(gt.mean())
        power = np.abs(np.fft.rfft2(centered)) ** 2
        fy = np.fft.fftfreq(gt.shape[0])[:, None]
        fx = np.fft.rfftfreq(gt.shape[1])[None, :]
        high_frequency = float(power[np.sqrt(fy * fy + fx * fx) >= 0.25].sum() / (power.sum() + 1e-12))
        scores.append(float(gradient) + high_frequency)
    return np.asarray(scores, dtype=np.float64)


def morphology_balanced_sampler(pairs: list[tuple[Path, Path]], seed: int = 2026,
                                samples_per_epoch: int | None = None) -> WeightedRandomSampler:
    scores = morphology_scores(pairs)
    boundaries = np.quantile(scores, [0.25, 0.5, 0.75])
    groups = np.digitize(scores, boundaries, right=False)
    counts = np.bincount(groups, minlength=4)
    weights = np.asarray([1.0 / max(int(counts[group]), 1) for group in groups], dtype=np.float64)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.from_numpy(weights),
        num_samples=samples_per_epoch or len(pairs),
        replacement=True,
        generator=generator,
    )
