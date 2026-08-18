from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


def _merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if key in result and isinstance(result[key], dict) and isinstance(value, dict) else value
    return result


def _read(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: str | Path, base_path: str | Path | None = None) -> dict:
    # A config may name its own base with `extends`, resolved relative to the file that names it so the
    # chain does not depend on the working directory. That allows an overlay to be layered on another
    # overlay: the Kaggle configuration selects paths and hardware, the recipe underneath it selects the
    # loss and data, and the full baseline sits under both. With only the single explicit base the
    # middle layer would have to restate every field of the bottom one to stay loadable.
    layers, seen = [_read(path)], {Path(path).resolve()}
    while "extends" in layers[-1]:
        parent = Path(layers[-1]["extends"])
        parent = parent if parent.is_absolute() else Path(path).parent / parent
        if parent.resolve() in seen:
            raise ValueError(f"Circular config extends chain at {parent}")
        seen.add(parent.resolve())
        path = parent
        layers.append(_read(parent))
    if base_path:
        layers.append(_read(base_path))
    config: dict = {}
    for layer in reversed(layers):
        config = _merge(config, layer)
    config.pop("extends", None)
    return config
