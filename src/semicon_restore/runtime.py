from __future__ import annotations

import os
import platform
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    name: str
    precision: str
    bf16: bool
    memory_gb: float


def select_device(requested: str = "auto") -> DeviceInfo:
    if requested == "cpu" or not torch.cuda.is_available():
        return DeviceInfo(torch.device("cpu"), "CPU", "fp32", False, 0.0)
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    name = props.name
    precision = "bf16" if bf16 and ("H100" in name or "A100" in name or "L4" in name) else "fp16"
    return DeviceInfo(torch.device(f"cuda:{index}"), name, precision, bf16, props.total_memory / 2**30)


def set_seed(seed: int, deterministic: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def environment_info() -> dict[str, str]:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": str(torch.version.cuda),
        "cuda_available": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cudnn"] = str(torch.backends.cudnn.version())
    return info
