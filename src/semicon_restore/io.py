from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


class ArrayContractError(ValueError):
    pass


def discover_npy(directory: str | Path) -> list[Path]:
    root = Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    return sorted(
        p for p in root.glob("*.npy")
        if p.is_file() and not p.name.startswith("._")
    )


def load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    try:
        array = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ArrayContractError(f"Could not load {path}: {exc}") from exc
    if array.ndim != 2:
        raise ArrayContractError(f"{path}: expected 2D grayscale array, got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ArrayContractError(f"{path}: expected real numeric array, got {array.dtype}")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ArrayContractError(f"{path}: contains NaN or Inf")
    return np.ascontiguousarray(array)


def save_array_atomic(path: str | Path, array: np.ndarray, overwrite: bool = True) -> None:
    path = Path(path)
    if array.ndim != 2 or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ArrayContractError(f"Invalid output array for {path}")
    if np.min(array) < 0 or np.max(array) > 1:
        raise ArrayContractError(f"Output outside [0,1] for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npy", dir=path.parent)
    os.close(fd)
    try:
        np.save(temp_name, array, allow_pickle=False)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def validate_pair(lr_path: Path, gt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    lr, gt = load_array(lr_path), load_array(gt_path)
    if gt.shape != (lr.shape[0] * 2, lr.shape[1] * 2):
        raise ArrayContractError(f"{lr_path.name}: LR {lr.shape} does not match GT {gt.shape}")
    if float(gt.min()) < 0 or float(gt.max()) > 1:
        raise ArrayContractError(f"{gt_path}: GT values must be in [0,1]")
    return lr, gt
