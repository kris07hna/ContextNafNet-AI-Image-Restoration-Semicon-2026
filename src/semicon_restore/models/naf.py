from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..features import NoiseFeatures, input_channels


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance, mean = torch.var_mean(x, dim=1, keepdim=True, correction=0)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2, ffn_expansion: int = 2, kernel_size: int = 3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("NAFBlock kernel_size must be odd")
        hidden = channels * expansion
        ffn = channels * ffn_expansion
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.Conv2d(hidden, hidden, kernel_size, padding=kernel_size // 2, groups=hidden)
        self.gate1 = SimpleGate()
        self.attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden // 2, hidden // 2, 1))
        self.conv2 = nn.Conv2d(hidden // 2, channels, 1)
        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn, 1)
        self.gate2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn // 2, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(self.conv1(self.norm1(x)))
        y = self.gate1(y)
        y = self.conv2(y * self.attention(y))
        x = x + y * self.beta
        y = self.conv4(self.gate2(self.conv3(self.norm2(x))))
        return x + y * self.gamma


class BottleneckAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 8, ffn_expansion: int = 2):
        super().__init__()
        if channels % heads:
            raise ValueError("Attention channels must be divisible by heads")
        self.heads = heads
        self.norm1 = LayerNorm2d(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.qkv_depthwise = nn.Conv2d(channels * 3, channels * 3, 3, padding=1, groups=channels * 3)
        self.project = nn.Conv2d(channels, channels, 1)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        hidden = channels * ffn_expansion
        self.norm2 = LayerNorm2d(channels)
        self.ffn_in = nn.Conv2d(channels, hidden * 2, 1)
        self.ffn_depthwise = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.ffn_out = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q, k, v = self.qkv_depthwise(self.qkv(self.norm1(x))).chunk(3, dim=1)
        head_channels = channels // self.heads
        q = q.reshape(batch, self.heads, head_channels, height * width)
        k = k.reshape(batch, self.heads, head_channels, height * width)
        v = v.reshape(batch, self.heads, head_channels, height * width)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = (q @ k.transpose(-2, -1) * self.temperature).softmax(dim=-1)
        attended = (attention @ v).reshape(batch, channels, height, width)
        x = x + self.project(attended)
        y = self.ffn_depthwise(self.ffn_in(self.norm2(x)))
        first, second = y.chunk(2, dim=1)
        return x + self.ffn_out(F.gelu(first) * second)


class Conditioning(nn.Module):
    def __init__(self, widths: tuple[int, int]):
        super().__init__()
        total = sum(widths) * 2
        self.widths = widths
        self.mlp = nn.Sequential(nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, total))

    @staticmethod
    def statistics(raw: torch.Tensor) -> torch.Tensor:
        bounded = raw.clamp(0, 1)
        overflow = raw - bounded
        gx = raw[..., :, 1:] - raw[..., :, :-1]
        gy = raw[..., 1:, :] - raw[..., :-1, :]
        return torch.stack(
            [raw.mean((1, 2, 3)), raw.std((1, 2, 3)), (raw < 0).float().mean((1, 2, 3)),
             (raw > 1).float().mean((1, 2, 3)), overflow.abs().mean((1, 2, 3)),
             0.5 * (gx.abs().mean((1, 2, 3)) + gy.abs().mean((1, 2, 3)))], dim=1
        )

    def forward(self, raw: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        params = self.mlp(self.statistics(raw))
        result, offset = [], 0
        for width in self.widths:
            scale = params[:, offset:offset + width]
            offset += width
            shift = params[:, offset:offset + width]
            offset += width
            result.append((scale[:, :, None, None], shift[:, :, None, None]))
        return result


@dataclass(frozen=True)
class ModelConfig:
    name: str = "multiscale_naf"
    in_channels: int = 4
    width: int = 48
    blocks: tuple[int, int, int] = (4, 6, 8)
    conditioning: bool = True
    scale: int = 2
    kernel_size: int = 3
    intro_kernel_size: int = 3
    bottleneck_attention_blocks: int = 0
    attention_heads: int = 8
    # Every field below has a default, so checkpoints written before the noise-aware inputs existed
    # still load and still compare equal to a freshly built raw4 configuration.
    input_mode: str = "raw4"
    noise_quadratic: float = 0.026627
    noise_linear: float = 0.0
    noise_constant: float = 3.929e-05
    vst_margin: float = 0.05
    noise_blur_sigma: float = 1.0

    def __post_init__(self) -> None:
        expected = input_channels(self.input_mode)
        if self.in_channels != expected:
            raise ValueError(f"input_mode {self.input_mode} needs in_channels={expected}, got {self.in_channels}")

    @classmethod
    def from_dict(cls, values: dict) -> ModelConfig:
        values = dict(values)
        values["blocks"] = tuple(values.get("blocks", cls.blocks))
        if "input_mode" in values and "in_channels" not in values:
            values["in_channels"] = input_channels(values["input_mode"])
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)


class MultiscaleNAF(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.scale != 2 or len(config.blocks) != 3:
            raise ValueError("MultiscaleNAF requires scale=2 and three block depths")
        w, depths = config.width, config.blocks
        self.config = config
        if config.intro_kernel_size % 2 == 0:
            raise ValueError("intro_kernel_size must be odd")
        self.intro = nn.Conv2d(config.in_channels, w, config.intro_kernel_size,
                               padding=config.intro_kernel_size // 2)
        self.enc1 = nn.Sequential(*(NAFBlock(w, kernel_size=config.kernel_size) for _ in range(depths[0])))
        self.down1 = nn.Conv2d(w, 2 * w, 2, stride=2)
        self.enc2 = nn.Sequential(*(NAFBlock(2 * w, kernel_size=config.kernel_size) for _ in range(depths[1])))
        self.down2 = nn.Conv2d(2 * w, 4 * w, 2, stride=2)
        middle = [NAFBlock(4 * w, kernel_size=config.kernel_size) for _ in range(depths[2])]
        middle.extend(BottleneckAttention(4 * w, config.attention_heads)
                      for _ in range(config.bottleneck_attention_blocks))
        self.middle = nn.Sequential(*middle)
        self.up2 = nn.Sequential(nn.Conv2d(4 * w, 8 * w, 1), nn.PixelShuffle(2))
        self.reduce2 = nn.Conv2d(4 * w, 2 * w, 1)
        self.dec2 = nn.Sequential(*(NAFBlock(2 * w, kernel_size=config.kernel_size) for _ in range(depths[1])))
        self.up1 = nn.Sequential(nn.Conv2d(2 * w, 4 * w, 1), nn.PixelShuffle(2))
        self.reduce1 = nn.Conv2d(2 * w, w, 1)
        self.dec1 = nn.Sequential(*(NAFBlock(w, kernel_size=config.kernel_size) for _ in range(depths[0])))
        self.head = nn.Conv2d(w, config.scale * config.scale, 3, padding=1)
        self.conditioner = Conditioning((4 * w, 2 * w)) if config.conditioning else None
        self.features = NoiseFeatures(config.input_mode, config.noise_quadratic, config.noise_linear,
                                      config.noise_constant, config.vst_margin, config.noise_blur_sigma)

    def input_features(self, raw: torch.Tensor) -> torch.Tensor:
        return self.features(raw)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.ndim != 4 or raw.shape[1] != 1:
            raise ValueError(f"Expected [B,1,H,W], got {tuple(raw.shape)}")
        original_hw = raw.shape[-2:]
        pad_h, pad_w = (-original_hw[0]) % 4, (-original_hw[1]) % 4
        padded = F.pad(raw, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else raw
        conditions = self.conditioner(padded) if self.conditioner else None
        x1 = self.enc1(self.intro(self.input_features(padded)))
        x2 = self.enc2(self.down1(x1))
        middle = self.middle(self.down2(x2))
        if conditions:
            scale, shift = conditions[0]
            middle = middle * (1 + 0.1 * torch.tanh(scale)) + 0.1 * shift
        y2 = self.reduce2(torch.cat((self.up2(middle), x2), dim=1))
        if conditions:
            scale, shift = conditions[1]
            y2 = y2 * (1 + 0.1 * torch.tanh(scale)) + 0.1 * shift
        y2 = self.dec2(y2)
        y1 = self.dec1(self.reduce1(torch.cat((self.up1(y2), x1), dim=1)))
        correction = F.pixel_shuffle(self.head(y1), self.config.scale)
        base = F.interpolate(padded, scale_factor=self.config.scale, mode="bilinear", align_corners=False)
        output = base + correction
        return output[..., : original_hw[0] * 2, : original_hw[1] * 2]


def build_model(config: ModelConfig | dict) -> MultiscaleNAF:
    config = ModelConfig.from_dict(config) if isinstance(config, dict) else config
    if config.name not in {"multiscale_naf", "context_naf"}:
        raise ValueError(f"Unknown model: {config.name}")
    return MultiscaleNAF(config)
