"""Encoder IMPALA para quadro empilhado 108×192; saída espacial 14×24."""

from typing import cast

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1),
                                    nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, x + self.layers(x))


class ImpalaBlock(nn.Module):
    def __init__(self, incoming: int, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(incoming, channels, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.residual = nn.Sequential(ResidualBlock(channels), ResidualBlock(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.residual(self.pool(self.conv(x))))


class ImpalaEncoder(nn.Module):
    """Pixels uint8 (B,12,108,192) → mapa (B,32,14,24) e vetor 256."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(ImpalaBlock(12, 16), ImpalaBlock(16, 32), ImpalaBlock(32, 32))
        self.projection = nn.Sequential(nn.ReLU(), nn.Flatten(), nn.Linear(32 * 14 * 24, 256), nn.ReLU())

    def forward(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.blocks(pixels.float() / 255.0)
        return features, self.projection(features)
