from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.clamp(0, 1)
    mse = (prediction - target).square().flatten(1).mean(1).clamp_min(1e-12)
    return -10 * torch.log10(mse)


def ssim(prediction: torch.Tensor, target: torch.Tensor, window: int = 11) -> torch.Tensor:
    prediction = prediction.clamp(0, 1)
    padding = window // 2
    mu_x = F.avg_pool2d(prediction, window, 1, padding)
    mu_y = F.avg_pool2d(target, window, 1, padding)
    var_x = F.avg_pool2d(prediction.square(), window, 1, padding) - mu_x.square()
    var_y = F.avg_pool2d(target.square(), window, 1, padding) - mu_y.square()
    cov = F.avg_pool2d(prediction * target, window, 1, padding) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + 0.01**2) * (2 * cov + 0.03**2)) / ((mu_x.square() + mu_y.square() + 0.01**2) * (var_x + var_y + 0.03**2))
    return score.flatten(1).mean(1)


def summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    if tensor.numel() == 0:
        return {"count": 0}
    return {
        "count": int(tensor.numel()), "mean": float(tensor.mean()), "median": float(tensor.median()),
        "std": float(tensor.std(correction=0)), "min": float(tensor.min()), "max": float(tensor.max()),
    }
