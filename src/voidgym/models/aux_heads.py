"""Auxiliary inverse-dynamics and hero-motion prediction heads."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812


class InverseDynamicsHead(nn.Module):
    def __init__(self, feature_dim: int = 256, action_count: int = 43) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim * 2, 256), nn.ReLU(),
                                 nn.Linear(256, action_count))

    def forward(self, current: Tensor, following: Tensor) -> Tensor:
        return cast(Tensor, self.net(torch.cat((current, following), dim=-1)))

    @staticmethod
    def loss(logits: Tensor, actions: Tensor, discontinuity: Tensor) -> Tensor:
        valid = ~discontinuity.bool()
        if not valid.any():
            return logits.sum() * 0.0
        return F.cross_entropy(logits[valid], actions[valid].long())


class HeroMotionHead(nn.Module):
    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(), nn.Linear(128, 3))

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        prediction = self.net(features)
        return prediction[..., :2], prediction[..., 2]

    @staticmethod
    def loss(predicted_motion: Tensor, predicted_ground: Tensor,
             target: Tensor, valid: Tensor | None = None) -> Tensor:
        mask = torch.isfinite(target).all(dim=-1)
        if valid is not None:
            mask &= valid.bool()
        if not mask.any():
            return predicted_motion.sum() * 0.0 + predicted_ground.sum() * 0.0
        motion = F.mse_loss(predicted_motion[mask], target[mask, :2])
        ground = F.binary_cross_entropy_with_logits(predicted_ground[mask], target[mask, 2])
        return motion + ground
