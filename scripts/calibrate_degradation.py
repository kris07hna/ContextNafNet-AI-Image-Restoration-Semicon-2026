from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semicon_restore.data import pair_paths, read_manifest
from semicon_restore.degradation import (
    DegradationParams,
    area_downsample,
    block_statistics,
    degrade,
    kernel_downsample,
    variance_stabilize,
)
from semicon_restore.io import load_array

SIGNAL_BINS = 16
DETAIL_BINS = 6
MINIMUM_COUNT = 256


def select_pairs(lr_dir: Path, gt_dir: Path, manifest: str | None, limit: int) -> list[tuple[Path, Path]]:
    pairs = pair_paths(lr_dir, gt_dir)
    if manifest:
        names = set(read_manifest(manifest)["train"])
        pairs = [pair for pair in pairs if pair[0].name in names]
    if not pairs:
        raise ValueError("No pairs selected for calibration")
    stride = max(1, len(pairs) // limit)
    return pairs[::stride][:limit]


def bounded_lstsq(matrix: np.ndarray, target: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Exact non-negative least squares by enumerating the faces of the positive orthant.
    # Only viable because the design has four columns, which is all this model needs.
    root = np.sqrt(weight)[:, None]
    design, values = matrix * root, target * root[:, 0]
    columns = design.shape[1]
    best = (float(np.sum(values * values)), np.zeros(columns))
    for size in range(1, columns + 1):
        for support in itertools.combinations(range(columns), size):
            solution, *_ = np.linalg.lstsq(design[:, list(support)], values, rcond=None)
            if np.any(solution < 0):
                continue
            full = np.zeros(columns)
            full[list(support)] = solution
            residual = float(np.sum((design @ full - values) ** 2))
            if residual < best[0]:
                best = (residual, full)
    return best[1]


def cell_index(signal: np.ndarray, detail: np.ndarray, signal_edges: np.ndarray,
               detail_edges: np.ndarray) -> np.ndarray:
    signal_bin = np.clip(np.digitize(signal, signal_edges[1:-1], right=False), 0, SIGNAL_BINS - 1)
    detail_bin = np.zeros_like(signal_bin)
    for index in range(SIGNAL_BINS):
        mask = signal_bin == index
        if mask.any():
            detail_bin[mask] = np.clip(np.digitize(detail[mask], detail_edges[index][1:-1], right=False),
                                       0, DETAIL_BINS - 1)
    return signal_bin * DETAIL_BINS + detail_bin


def detail_edges_for(signal: np.ndarray, detail: np.ndarray, signal_edges: np.ndarray) -> list[np.ndarray]:
    signal_bin = np.clip(np.digitize(signal, signal_edges[1:-1], right=False), 0, SIGNAL_BINS - 1)
    edges = []
    for index in range(SIGNAL_BINS):
        values = detail[signal_bin == index]
        if values.size < DETAIL_BINS * MINIMUM_COUNT:
            edges.append(np.linspace(0.0, 1.0, DETAIL_BINS + 1))
            continue
        edges.append(np.quantile(values, np.linspace(0.0, 1.0, DETAIL_BINS + 1)))
    return edges


def cell_statistics(signal: np.ndarray, detail: np.ndarray, residual: np.ndarray,
                    cells: np.ndarray) -> list[dict]:
    rows = []
    for cell in range(SIGNAL_BINS * DETAIL_BINS):
        mask = cells == cell
        count = int(mask.sum())
        if count < MINIMUM_COUNT:
            continue
        values = residual[mask]
        centered = values - values.mean()
        std = float(centered.std())
        rows.append({
            "cell": cell, "signal": float(signal[mask].mean()), "detail": float(detail[mask].mean()),
            "count": count, "std": std,
            "skew": float((centered**3).mean() / max(std**3, 1e-12)),
            "kurtosis": float((centered**4).mean() / max(std**4, 1e-12)),
        })
    return rows


def signal_statistics(signal: np.ndarray, residual: np.ndarray, edges: np.ndarray) -> list[dict]:
    index = np.clip(np.digitize(signal, edges[1:-1], right=False), 0, SIGNAL_BINS - 1)
    rows = []
    for bin_index in range(SIGNAL_BINS):
        mask = index == bin_index
        if int(mask.sum()) < MINIMUM_COUNT:
            continue
        values = residual[mask]
        rows.append({"signal": float(signal[mask].mean()), "count": int(mask.sum()),
                     "std": float(values.std())})
    return rows


def fit_noise_model(rows: list[dict]) -> tuple[np.ndarray, dict]:
    signal = np.asarray([row["signal"] for row in rows])
    detail = np.asarray([row["detail"] for row in rows])
    variance = np.asarray([row["std"] ** 2 for row in rows])
    weight = np.asarray([row["count"] for row in rows], dtype=np.float64)
    design = np.stack([signal * signal, detail, signal, np.ones_like(signal)], axis=1)
    coefficients = bounded_lstsq(design, variance, weight)
    predicted = design @ coefficients
    relative = np.sqrt(np.maximum(predicted, 0.0)) / np.sqrt(variance) - 1.0
    diagnostics = {
        "cells": len(rows),
        "weighted_variance_r2": float(1.0 - np.sum(weight * (variance - predicted) ** 2)
                                     / max(np.sum(weight * (variance - np.average(variance, weights=weight)) ** 2), 1e-30)),
        "worst_cell_std_error": float(np.abs(relative).max()),
        "median_cell_std_error": float(np.median(np.abs(relative))),
    }
    return coefficients, diagnostics


def fit_kernel(pairs: list[tuple[Path, Path]], size: int) -> tuple[np.ndarray, float]:
    before, after = size // 2 - 1, size // 2
    gram = np.zeros((size * size, size * size), dtype=np.float64)
    cross = np.zeros(size * size, dtype=np.float64)
    for lr_path, gt_path in pairs:
        lr, gt = load_array(lr_path).astype(np.float64), load_array(gt_path).astype(np.float64)
        height, width = gt.shape
        padded = np.pad(gt, ((before, after), (before, after)), mode="reflect")
        features = np.stack([padded[dy:dy + height:2, dx:dx + width:2].ravel()
                             for dy in range(size) for dx in range(size)], axis=1)
        gram += features.T @ features
        cross += features.T @ lr.ravel()
    kernel = np.linalg.lstsq(gram, cross, rcond=None)[0].reshape(size, size)
    residuals = [(load_array(lr).astype(np.float64) - kernel_downsample(load_array(gt).astype(np.float64), kernel)).ravel()
                 for lr, gt in pairs]
    return kernel, float(np.concatenate(residuals).std())


def autocorrelation(residual: np.ndarray, lag: int) -> float:
    if lag >= residual.shape[-1]:
        return 0.0
    a, b = residual[..., :, :-lag].ravel(), residual[..., :, lag:].ravel()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).mean() / max(a.std() * b.std(), 1e-12))


def summarize(signal: np.ndarray, detail: np.ndarray, residual: np.ndarray, observed: np.ndarray,
              signal_edges: np.ndarray, cells: np.ndarray, per_image: list[np.ndarray]) -> dict:
    mid = (signal >= 0.4) & (signal <= 0.6)
    centered = residual[mid] - residual[mid].mean() if mid.sum() > MINIMUM_COUNT else residual - residual.mean()
    std = float(centered.std())
    return {
        "signal_bins": signal_statistics(signal, residual, signal_edges),
        "cells": cell_statistics(signal, detail, residual, cells),
        "residual_std": float(residual.std()),
        "mid_skew": float((centered**3).mean() / max(std**3, 1e-12)),
        "mid_kurtosis": float((centered**4).mean() / max(std**4, 1e-12)),
        "lag1": float(np.mean([autocorrelation(image, 1) for image in per_image])),
        "lag2": float(np.mean([autocorrelation(image, 2) for image in per_image])),
        "fraction_below_zero": float((observed < 0.0).mean()),
        "fraction_above_one": float((observed > 1.0).mean()),
        "minimum": float(observed.min()), "maximum": float(observed.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the forward degradation model from real LR/GT pairs.")
    parser.add_argument("--lr-dir", default="train/train/NoisyLR")
    parser.add_argument("--gt-dir", default="train/train/GT")
    parser.add_argument("--split-manifest", default=None, help="Restrict calibration to the manifest train split")
    parser.add_argument("--images", type=int, default=200)
    parser.add_argument("--kernel-size", type=int, default=8, help="Even side length of the fitted stride-2 kernel")
    parser.add_argument("--kernel-gain", type=float, default=0.02,
                        help="Minimum relative residual-std reduction required to prefer the fitted kernel")
    parser.add_argument("--kernel-mode", choices=("auto", "area", "fit"), default="auto")
    parser.add_argument("--detail-mode", choices=("blockmix", "gaussian", "none"), default="blockmix",
                        help="How the within-block detail term is sampled in the forward model")
    parser.add_argument("--constant-floor", type=float, default=0.0, help="Lower bound on the additive variance")
    parser.add_argument("--validation-images", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="configs/degradation-v1.json")
    args = parser.parse_args()

    if args.kernel_size % 2:
        raise ValueError("--kernel-size must be even")
    pairs = select_pairs(Path(args.lr_dir), Path(args.gt_dir), args.split_manifest, args.images)
    print(f"calibration_pairs count={len(pairs)}", flush=True)

    signals, details, residuals, observations, per_image = [], [], [], [], []
    for lr_path, gt_path in pairs:
        lr, gt = load_array(lr_path).astype(np.float64), load_array(gt_path).astype(np.float64)
        mean = area_downsample(gt)
        detail = np.maximum(area_downsample(gt * gt) - mean * mean, 0.0)
        residual = lr - mean
        signals.append(mean.ravel())
        details.append(detail.ravel())
        residuals.append(residual.ravel())
        observations.append(lr.ravel())
        per_image.append(residual)
    signal = np.concatenate(signals)
    detail = np.concatenate(details)
    residual = np.concatenate(residuals)
    observed = np.concatenate(observations)
    signal_edges = np.quantile(signal, np.linspace(0.0, 1.0, SIGNAL_BINS + 1))
    signal_edges[0], signal_edges[-1] = signal.min(), signal.max()
    detail_edges = detail_edges_for(signal, detail, signal_edges)
    cells = cell_index(signal, detail, signal_edges, detail_edges)
    real_cells = cell_statistics(signal, detail, residual, cells)
    print(f"area_residual_std={residual.std():.5f} cells={len(real_cells)} "
          f"mean_detail={detail.mean():.6f}", flush=True)

    coefficients, noise_diagnostics = fit_noise_model(real_cells)
    quadratic, detail_coefficient, linear, constant = (float(value) for value in coefficients)
    constant = max(constant, args.constant_floor)
    print(f"noise_fit quadratic={quadratic:.6f} detail={detail_coefficient:.6f} linear={linear:.6f} "
          f"constant={constant:.3e} sigma={math.sqrt(constant):.5f}", flush=True)
    print(f"fit_quality weighted_r2={noise_diagnostics['weighted_variance_r2']:.5f} "
          f"median_cell_std_error={noise_diagnostics['median_cell_std_error'] * 100:.2f}% "
          f"worst_cell_std_error={noise_diagnostics['worst_cell_std_error'] * 100:.2f}%", flush=True)

    kernel_pairs = pairs[: min(len(pairs), max(20, args.images // 4))]
    kernel, kernel_residual_std = fit_kernel(kernel_pairs, args.kernel_size)
    area_std_on_kernel_pairs = float(np.concatenate([
        (load_array(lr).astype(np.float64) - area_downsample(load_array(gt).astype(np.float64))).ravel()
        for lr, gt in kernel_pairs]).std())
    gain = (area_std_on_kernel_pairs - kernel_residual_std) / max(area_std_on_kernel_pairs, 1e-12)
    use_kernel = args.kernel_mode == "fit" or (args.kernel_mode == "auto" and gain >= args.kernel_gain)
    print(f"kernel_fit size={args.kernel_size} area_std={area_std_on_kernel_pairs:.5f} "
          f"fit_std={kernel_residual_std:.5f} gain={gain * 100:.2f}% use_fitted={use_kernel}", flush=True)

    detail_mode = args.detail_mode
    if detail_mode == "blockmix" and not 0.0 < detail_coefficient < 1.0:
        print(f"detail_mode blockmix needs detail in (0, 1), got {detail_coefficient:.6f}; falling back to gaussian",
              flush=True)
        detail_mode = "gaussian"
    params = DegradationParams(
        quadratic=quadratic, linear=linear, detail=detail_coefficient, constant=constant, detail_mode=detail_mode,
        kernel=tuple(tuple(float(value) for value in row) for row in kernel) if use_kernel else None,
        source=f"calibrate_degradation.py images={len(pairs)} manifest={args.split_manifest or 'none'}",
    )
    alpha = params.dirichlet_alpha if detail_mode == "blockmix" else float("nan")
    print(f"forward_model gamma_shape={params.gamma_shape:.2f} poisson_rate={params.poisson_rate:.2f} "
          f"detail_mode={detail_mode} dirichlet_alpha={alpha:.4f} "
          f"additive_sigma={params.additive_sigma:.5f}", flush=True)

    generator = np.random.default_rng(args.seed)
    validation_pairs = pairs[: min(len(pairs), args.validation_images)]
    sim_signals, sim_details, sim_residuals, sim_observations, sim_per_image = [], [], [], [], []
    for _, gt_path in validation_pairs:
        gt = load_array(gt_path).astype(np.float64)
        reference, cell_detail = area_downsample(gt), block_statistics(gt, params)[1]
        simulated = degrade(gt, generator, params).astype(np.float64)
        sim_signals.append(reference.ravel())
        sim_details.append(cell_detail.ravel())
        sim_residuals.append((simulated - reference).ravel())
        sim_observations.append(simulated.ravel())
        sim_per_image.append(simulated - reference)
    sim_signal = np.concatenate(sim_signals)
    sim_detail = np.concatenate(sim_details)
    sim_cells = cell_index(sim_signal, sim_detail, signal_edges, detail_edges)
    # Summarize the real data over exactly the pairs that were simulated: global statistics such as
    # residual_std depend on the signal distribution, so comparing 200 real against 120 simulated
    # images would report a mismatch that is only a difference in which images were included.
    count = len(validation_pairs)
    real_signal = np.concatenate(signals[:count])
    real_detail = np.concatenate(details[:count])
    real_summary = summarize(real_signal, real_detail, np.concatenate(residuals[:count]),
                             np.concatenate(observations[:count]), signal_edges,
                             cell_index(real_signal, real_detail, signal_edges, detail_edges),
                             per_image[:count])
    simulated_summary = summarize(sim_signal, sim_detail, np.concatenate(sim_residuals),
                                  np.concatenate(sim_observations), signal_edges, sim_cells, sim_per_image)

    print("\nsignal      real_std   sim_std    ratio", flush=True)
    worst_signal = 0.0
    for real_row, sim_row in zip(real_summary["signal_bins"], simulated_summary["signal_bins"]):
        ratio = sim_row["std"] / max(real_row["std"], 1e-12)
        worst_signal = max(worst_signal, abs(ratio - 1.0))
        print(f"{real_row['signal']:8.4f}   {real_row['std']:.5f}    {sim_row['std']:.5f}    {ratio:.3f}", flush=True)
    real_by_cell = {row["cell"]: row for row in real_summary["cells"]}
    ratios = [row["std"] / max(real_by_cell[row["cell"]]["std"], 1e-12)
              for row in simulated_summary["cells"] if row["cell"] in real_by_cell]
    worst_cell = max(abs(ratio - 1.0) for ratio in ratios) if ratios else float("nan")
    print(f"\nworst_signal_bin_std_error={worst_signal * 100:.2f}%  "
          f"median_cell_std_error={np.median(np.abs(np.asarray(ratios) - 1.0)) * 100:.2f}%  "
          f"worst_cell_std_error={worst_cell * 100:.2f}%  cells={len(ratios)}", flush=True)
    for key in ("residual_std", "mid_skew", "mid_kurtosis", "lag1", "lag2",
                "fraction_below_zero", "fraction_above_one", "minimum", "maximum"):
        print(f"{key:22s} real={real_summary[key]:+.5f}  simulated={simulated_summary[key]:+.5f}", flush=True)

    stabilizer = params.stabilizer()
    stabilized_residual = variance_stabilize(observed, stabilizer) - variance_stabilize(signal, stabilizer)
    raw_rows = signal_statistics(signal, residual, signal_edges)
    vst_rows = signal_statistics(signal, stabilized_residual, signal_edges)
    raw_spread = max(row["std"] for row in raw_rows) / max(min(row["std"] for row in raw_rows), 1e-12)
    vst_spread = max(row["std"] for row in vst_rows) / max(min(row["std"] for row in vst_rows), 1e-12)
    print(f"\nheteroscedasticity raw_std_ratio={raw_spread:.2f}x vst_std_ratio={vst_spread:.2f}x "
          f"vst_constant={stabilizer.constant:.3e} vst_normalizer={stabilizer.normalizer:.4f}", flush=True)

    params.save(args.output, extra={"diagnostics": {
        "images": len(pairs), "validation_images": len(validation_pairs), "seed": args.seed,
        "detail_mode": detail_mode,
        "noise_fit": noise_diagnostics,
        "kernel": {"size": args.kernel_size, "area_residual_std": area_std_on_kernel_pairs,
                   "fitted_residual_std": kernel_residual_std, "relative_gain": gain, "used": use_kernel},
        "simulation_check": {"worst_signal_bin_std_error": worst_signal, "worst_cell_std_error": worst_cell,
                             "median_cell_std_error": float(np.median(np.abs(np.asarray(ratios) - 1.0)))},
        "heteroscedasticity": {"raw_std_ratio": raw_spread, "vst_std_ratio": vst_spread},
        "real": real_summary, "simulated": simulated_summary,
    }})
    print(f"\nwrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
