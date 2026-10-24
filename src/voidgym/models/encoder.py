"""IMPALA-style visual encoder for the universal RGB frame stack."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from voidgym.core.observation import FRAME_STACK


class ResidualBlock(nn.Module):
    """Two-convolution residual unit without batch-dependent normalization."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, value: Tensor) -> Tensor:
        residual = F.relu(value)
        residual = self.conv1(residual)
        residual = F.relu(residual)
        return value + cast(Tensor, self.conv2(residual))


class ImpalaBlock(nn.Module):
    """Convolution, downsampling and two residual units."""

    def __init__(self, in_channels: int, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.residual1 = ResidualBlock(channels)
        self.residual2 = ResidualBlock(channels)

    def forward(self, value: Tensor) -> Tensor:
        value = self.conv(value)
        value = F.max_pool2d(value, kernel_size=3, stride=2, padding=1)
        return cast(Tensor, self.residual2(self.residual1(value)))


class ImpalaEncoder(nn.Module):
    """Encode (B,12,108,192) uint8 stacks; the final feature map is (B,32,14,24)."""

    def __init__(self, output_dim: int = 256) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            ImpalaBlock(FRAME_STACK * 3, 16),
            ImpalaBlock(16, 32),
            ImpalaBlock(32, 32),
        )
        self.projection = nn.Linear(32 * 14 * 24, output_dim)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        if image.ndim != 4 or image.shape[1:] != (FRAME_STACK * 3, 108, 192):
            raise ValueError("image must have shape (B,12,108,192)")
        value = image.to(dtype=torch.float32).div(255.0)
        features = cast(Tensor, self.blocks(value))
        vector = cast(Tensor, self.projection(F.relu(features).flatten(start_dim=1)))
        return features, F.relu(vector)
