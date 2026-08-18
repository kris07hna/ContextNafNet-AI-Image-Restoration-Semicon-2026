from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

PIXEL_MODES = frozenset({"charbonnier", "mse"})


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((prediction - target).square() + self.eps**2).mean()


def gradient_charbonnier(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    px, tx = prediction[..., :, 1:] - prediction[..., :, :-1], target[..., :, 1:] - target[..., :, :-1]
    py, ty = prediction[..., 1:, :] - prediction[..., :-1, :], target[..., 1:, :] - target[..., :-1, :]
    return 0.5 * (torch.sqrt((px - tx).square() + eps**2).mean() + torch.sqrt((py - ty).square() + eps**2).mean())


def edge_weighted_gradient_charbonnier(prediction: torch.Tensor, target: torch.Tensor,
                                       edge_weight: float = 1.0, eps: float = 1e-3) -> torch.Tensor:
    prediction, target = prediction.float(), target.float()
    px, tx = prediction[..., :, 1:] - prediction[..., :, :-1], target[..., :, 1:] - target[..., :, :-1]
    py, ty = prediction[..., 1:, :] - prediction[..., :-1, :], target[..., 1:, :] - target[..., :-1, :]
    wx = 1 + edge_weight * tx.abs() / tx.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
    wy = 1 + edge_weight * ty.abs() / ty.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
    return 0.5 * ((wx * torch.sqrt((px - tx).square() + eps**2)).mean()
                  + (wy * torch.sqrt((py - ty).square() + eps**2)).mean())


def haar_highband_charbonnier(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    def highbands(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x00 = tensor[..., 0::2, 0::2]
        x01 = tensor[..., 0::2, 1::2]
        x10 = tensor[..., 1::2, 0::2]
        x11 = tensor[..., 1::2, 1::2]
        return (
            (x00 - x01 + x10 - x11) * 0.5,
            (x00 + x01 - x10 - x11) * 0.5,
            (x00 - x01 - x10 + x11) * 0.5,
        )

    if prediction.shape[-2] % 2 or prediction.shape[-1] % 2:
        raise ValueError("Haar high-band loss requires even spatial dimensions")
    prediction_bands, target_bands = highbands(prediction), highbands(target)
    return sum(
        torch.sqrt((prediction_band - target_band).square() + eps**2).mean()
        for prediction_band, target_band in zip(prediction_bands, target_bands)
    ) / 3


def _ssim(prediction: torch.Tensor, target: torch.Tensor, window: int = 11) -> torch.Tensor:
    padding = window // 2
    mu_x = F.avg_pool2d(prediction, window, 1, padding)
    mu_y = F.avg_pool2d(target, window, 1, padding)
    var_x = F.avg_pool2d(prediction.square(), window, 1, padding) - mu_x.square()
    var_y = F.avg_pool2d(target.square(), window, 1, padding) - mu_y.square()
    cov = F.avg_pool2d(prediction * target, window, 1, padding) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2))).mean()


def _gaussian_ssim(prediction: torch.Tensor, target: torch.Tensor, window: int = 11,
                   sigma: float = 1.5) -> tuple[torch.Tensor, torch.Tensor]:
    prediction, target = prediction.float(), target.float()
    coordinates = torch.arange(window, device=prediction.device, dtype=prediction.dtype) - window // 2
    kernel = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    kernel = (kernel / kernel.sum()).outer(kernel / kernel.sum())[None, None]
    channels = prediction.shape[1]
    kernel = kernel.expand(channels, 1, window, window)
    padding = window // 2
    mu_x = F.conv2d(prediction, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=padding, groups=channels)
    var_x = (F.conv2d(prediction.square(), kernel, padding=padding, groups=channels) - mu_x.square()).clamp_min(0)
    var_y = (F.conv2d(target.square(), kernel, padding=padding, groups=channels) - mu_y.square()).clamp_min(0)
    cov = F.conv2d(prediction * target, kernel, padding=padding, groups=channels) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    luminance = (2 * mu_x * mu_y + c1) / (mu_x.square() + mu_y.square() + c1).clamp_min(1e-12)
    contrast_structure = (2 * cov + c2) / (var_x + var_y + c2).clamp_min(1e-12)
    return (luminance * contrast_structure).flatten(1).mean(1), contrast_structure.flatten(1).mean(1)


def multiscale_ssim(prediction: torch.Tensor, target: torch.Tensor, levels: int = 3) -> torch.Tensor:
    weights = prediction.new_tensor([0.3, 0.3, 0.4][:levels])
    values = []
    current_prediction, current_target = prediction, target
    for level in range(levels):
        score, contrast = _gaussian_ssim(current_prediction, current_target)
        values.append(score if level == levels - 1 else contrast)
        if level < levels - 1:
            current_prediction = F.avg_pool2d(current_prediction, 2, 2)
            current_target = F.avg_pool2d(current_target, 2, 2)
    stacked = torch.stack([value.clamp_min(1e-6) for value in values], dim=1)
    return torch.prod(stacked ** weights[None], dim=1).mean()


def focal_frequency_loss(prediction: torch.Tensor, target: torch.Tensor, alpha: float = 1.0,
                         max_weight: float = 10.0) -> torch.Tensor:
    prediction32, target32 = prediction.float(), target.float()
    height, width = prediction.shape[-2:]
    window_y = torch.hann_window(height, periodic=False, device=prediction.device, dtype=torch.float32)
    window_x = torch.hann_window(width, periodic=False, device=prediction.device, dtype=torch.float32)
    window = window_y[:, None] * window_x[None, :]
    prediction_fft = torch.fft.rfft2((prediction32 - prediction32.mean((-2, -1), keepdim=True)) * window,
                                    norm="ortho")
    target_fft = torch.fft.rfft2((target32 - target32.mean((-2, -1), keepdim=True)) * window, norm="ortho")
    distance = (prediction_fft - target_fft).abs().square()
    scale = distance.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    weights = (distance.detach() / scale).pow(alpha / 2).clamp(max=max_weight)
    return (weights * distance).mean()


class RestorationLoss(nn.Module):
    def __init__(self, pixel_weight: float = 0.70, ssim_weight: float = 0.20, gradient_weight: float = 0.10,
                 frequency_weight: float = 0.0, structural_mode: str = "ssim", edge_weight: float = 0.0,
                 frequency_mode: str = "haar", pixel_mode: str = "charbonnier",
                 lpips_weight: float = 0.0, lpips_net: str = "vgg"):
        super().__init__()
        if pixel_mode not in PIXEL_MODES:
            raise ValueError(f"Unknown pixel_mode: {pixel_mode}")
        self.pixel = CharbonnierLoss()
        self.weights = pixel_weight, ssim_weight, gradient_weight, frequency_weight
        self.structural_mode = structural_mode
        self.edge_weight = edge_weight
        self.frequency_mode = frequency_mode
        # Charbonnier reports roughly the mean absolute error and squared error is its square, so the
        # two modes differ by more than an order of magnitude; a schedule that switches mode has to
        # rescale the pixel weight with it rather than reuse the perceptual-phase value.
        self.pixel_mode = pixel_mode
        self.lpips_weight = float(lpips_weight)
        # The LPIPS network is loaded only when a positive weight is configured. Its parameters are
        # frozen and it stays in eval mode: it is a fixed perceptual distance metric, not a learnable
        # discriminator. VGG is lighter than AlexNet for this use and benchmarks well on restoration.
        self._lpips_net: nn.Module | None = None
        if self.lpips_weight > 0:
            import lpips
            self._lpips_net = lpips.LPIPS(net=lpips_net, verbose=False)
            self._lpips_net.eval()
            for param in self._lpips_net.parameters():
                param.requires_grad_(False)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction32, target32 = prediction.float(), target.float()
        pixel = (F.mse_loss(prediction32, target32) if self.pixel_mode == "mse"
                 else self.pixel(prediction32, target32))
        bounded = prediction32.clamp(0, 1)
        structural = 1 - (multiscale_ssim(bounded, target32) if self.structural_mode == "ms_ssim"
                          else _ssim(bounded, target32))
        gradient = (edge_weighted_gradient_charbonnier(prediction32, target32, self.edge_weight)
                    if self.edge_weight else gradient_charbonnier(prediction32, target32))
        if not self.weights[3]:
            frequency = prediction32.new_zeros(())
        elif self.frequency_mode == "focal":
            frequency = focal_frequency_loss(prediction32, target32)
        else:
            frequency = haar_highband_charbonnier(prediction32, target32)
        # LPIPS expects 3-channel inputs in [-1, 1]. Grayscale is replicated to three channels;
        # the bounded prediction is used so that out-of-range values do not distort the VGG features.
        if self._lpips_net is not None:
            if next(self._lpips_net.parameters()).device != prediction32.device:
                self._lpips_net = self._lpips_net.to(prediction32.device)
            pred_rgb = bounded.expand(-1, 3, -1, -1) * 2 - 1
            targ_rgb = target32.clamp(0, 1).expand(-1, 3, -1, -1) * 2 - 1
            perceptual = self._lpips_net(pred_rgb, targ_rgb).mean()
        else:
            perceptual = prediction32.new_zeros(())
        total = (self.weights[0] * pixel + self.weights[1] * structural + self.weights[2] * gradient
                 + self.weights[3] * frequency + self.lpips_weight * perceptual)
        return total, {"pixel": pixel.detach(), "ssim_loss": structural.detach(), "gradient": gradient.detach(),
                       "frequency": frequency.detach(), "lpips": perceptual.detach()}
