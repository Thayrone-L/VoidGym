"""Cabeça de classificação semântica privilegiada para o treino."""

import torch
from torch import nn
from torch.nn import functional


class SemanticHead(nn.Module):
    def __init__(self, classes: int = 10) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, classes, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return functional.interpolate(self.layers(features), (27, 48), mode="bilinear", align_corners=False)


def semantic_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if not labels.ne(255).any():
        return logits.sum() * 0.0
    return functional.cross_entropy(logits, labels.long(), weight=weights, ignore_index=255)
