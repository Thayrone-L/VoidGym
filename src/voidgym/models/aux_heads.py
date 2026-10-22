"""Predições auxiliares que compartilham o encoder, mas não ator."""

from typing import cast

import torch
from torch import nn


class InverseDynamicsHead(nn.Module):
    def __init__(self, actions: int = 43) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, actions))

    def forward(self, current: torch.Tensor, following: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.layers(torch.cat((current, following), dim=-1)))


class HeroMotionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Linear(256, 3)

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.layers(vector))
