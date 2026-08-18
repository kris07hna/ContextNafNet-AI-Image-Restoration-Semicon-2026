from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .data import pair_paths
from .io import load_array


def descriptor(array: np.ndarray, size: int = 24) -> np.ndarray:
    h, w = array.shape
    ys = np.linspace(0, h - 1, size).astype(np.int64)
    xs = np.linspace(0, w - 1, size).astype(np.int64)
    small = array[np.ix_(ys, xs)].astype(np.float32)
    small -= small.mean()
    norm = np.linalg.norm(small)
    return (small / max(float(norm), 1e-8)).ravel()


def canonical_descriptor(array: np.ndarray, size: int = 24) -> np.ndarray:
    variants = [descriptor(np.rot90(array, k), size) for k in range(4)]
    variants += [descriptor(np.fliplr(np.rot90(array, k)), size) for k in range(4)]
    # A fixed projection chooses the same orientation for rotated/reflected copies.
    projection = np.sin(np.arange(size * size, dtype=np.float32) * 0.017)
    scores = [float(value @ projection) for value in variants]
    return variants[int(np.argmax(scores))]


class UnionFind:
    def __init__(self, count: int):
        self.parent = list(range(count))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def create_grouped_split(lr_dir: str | Path, gt_dir: str | Path, validation_fraction: float = 0.15,
                         threshold: float = 0.985, seed: int = 2026) -> dict:
    pairs = pair_paths(lr_dir, gt_dir)
    descriptors = np.stack([canonical_descriptor(load_array(gt_path)) for _, gt_path in pairs])
    uf = UnionFind(len(pairs))
    block = 256
    for start in range(0, len(pairs), block):
        similarities = descriptors[start:start + block] @ descriptors.T
        rows, cols = np.where(similarities >= threshold)
        for row, col in zip(rows.tolist(), cols.tolist()):
            absolute_row = start + row
            if col > absolute_row:
                uf.union(absolute_row, col)
    groups: dict[int, list[str]] = {}
    for index, (lr_path, _) in enumerate(pairs):
        groups.setdefault(uf.find(index), []).append(lr_path.name)
    rng = np.random.default_rng(seed)
    components = list(groups.values())
    rng.shuffle(components)
    components.sort(key=len, reverse=True)
    target = round(len(pairs) * validation_fraction)
    validation, train = [], []
    for component in components:
        if len(validation) < target and abs(target - (len(validation) + len(component))) <= abs(target - len(validation)):
            validation.extend(component)
        else:
            train.extend(component)
    if len(validation) < target:
        for component in sorted((c for c in components if c[0] in train), key=len):
            if len(validation) >= target:
                break
            for name in component:
                train.remove(name)
            validation.extend(component)
    payload = {
        "version": 1, "algorithm": "d4-multiscale-correlation-components", "threshold": threshold,
        "seed": seed, "validation_fraction": validation_fraction, "train": sorted(train),
        "validation": sorted(validation), "groups": [sorted(group) for group in components],
    }
    canonical = json.dumps({k: payload[k] for k in payload if k != "manifest_sha256"}, sort_keys=True).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload
