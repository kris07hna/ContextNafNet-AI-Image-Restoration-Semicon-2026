from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

FORMAT_VERSION = 1


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.module = deepcopy(model).eval()
        self.decay = decay
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        current = model.state_dict()
        for name, value in self.module.state_dict().items():
            source = current[name].detach()
            value.copy_(value * self.decay + source * (1 - self.decay) if value.is_floating_point() else source)


def random_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: str | Path, model: torch.nn.Module, ema: ModelEMA, optimizer, scheduler, scaler,
                    model_config: dict, train_config: dict, step: int, epoch: int, best: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION, "model_config": model_config, "train_config": train_config,
        "model": model.state_dict(), "ema": ema.module.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None, "scaler": scaler.state_dict() if scaler else None,
        "step": step, "epoch": epoch, "best": best, "random_state": random_state(),
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format_version')}")
    required = {"model_config", "model", "ema"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Checkpoint missing keys: {sorted(missing)}")
    return checkpoint


def widen_input_weights(state: dict[str, torch.Tensor], model: torch.nn.Module) -> tuple[dict, list[str]]:
    # Widening the stem lets a checkpoint trained on the raw inputs seed a noise-aware model: the
    # added input channels start at zero, so the adapted model initially computes exactly the same
    # function as the checkpoint and fine-tunes from its accuracy instead of from scratch.
    adapted, notes = dict(state), []
    for name, parameter in model.state_dict().items():
        saved = adapted.get(name)
        if saved is None or saved.shape == parameter.shape:
            continue
        compatible = (saved.ndim == 4 and saved.shape[1] < parameter.shape[1]
                      and saved.shape[0] == parameter.shape[0] and saved.shape[2:] == parameter.shape[2:])
        if not compatible:
            raise ValueError(f"Cannot adapt {name}: {tuple(saved.shape)} to {tuple(parameter.shape)}")
        widened = torch.zeros_like(parameter)
        widened[:, : saved.shape[1]] = saved.to(parameter.dtype)
        adapted[name] = widened
        notes.append(f"{name} {tuple(saved.shape)} -> {tuple(parameter.shape)}")
    return adapted, notes
