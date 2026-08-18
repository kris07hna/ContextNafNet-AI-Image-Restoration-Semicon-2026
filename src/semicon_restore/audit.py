from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .data import pair_paths
from .io import validate_pair


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_dataset(lr_dir: str | Path, gt_dir: str | Path) -> dict:
    pairs = pair_paths(lr_dir, gt_dir)
    lr_min, lr_max, gt_min, gt_max = float("inf"), float("-inf"), float("inf"), float("-inf")
    lr_outside, lr_values = 0, 0
    shapes, records = {}, []
    for lr_path, gt_path in pairs:
        lr, gt = validate_pair(lr_path, gt_path)
        lr_min, lr_max = min(lr_min, float(lr.min())), max(lr_max, float(lr.max()))
        gt_min, gt_max = min(gt_min, float(gt.min())), max(gt_max, float(gt.max()))
        lr_outside += int(((lr < 0) | (lr > 1)).sum())
        lr_values += lr.size
        key = f"{lr.shape}->{gt.shape}"
        shapes[key] = shapes.get(key, 0) + 1
        records.append({"name": lr_path.name, "lr_sha256": _digest(lr_path), "gt_sha256": _digest(gt_path)})
    return {
        "pair_count": len(pairs), "shapes": shapes, "lr_min": lr_min, "lr_max": lr_max,
        "gt_min": gt_min, "gt_max": gt_max, "lr_outside_fraction": lr_outside / max(lr_values, 1),
        "records": records,
    }


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
