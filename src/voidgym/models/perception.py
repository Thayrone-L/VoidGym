"""Semantic-grid prediction head and class-weighted loss."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from voidgym.core.semantics import IGNORE, SEM_GRID, Sem


class SemanticHead(nn.Module):
    """Semantic decoder with an optional first-block 54×96 spatial skip."""

    def __init__(self, in_channels: int = 32) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.out = nn.Conv2d(32, len(Sem), kernel_size=1)

    def forward(self, features: Tensor) -> Tensor:
        logits = self.out(F.relu(self.conv(features)))
        return F.interpolate(logits, size=SEM_GRID, mode="bilinear", align_corners=False)


def inverse_frequency_weights(counts: list[int] | tuple[int, ...], *, power: float = .5,
                              minimum: float = .25, maximum: float = 20.0) -> Tensor:
    """Compute normalized inverse-frequency weights from per-class pixel counts."""
    if len(counts) != len(Sem) or any(count < 0 for count in counts):
        raise ValueError("counts must contain one nonnegative value per semantic class")
    values = torch.tensor(counts, dtype=torch.float32).clamp_min(1)
    weights = (values.sum() / (len(Sem) * values)).pow(power)
    weights = weights.clamp(minimum, maximum)
    return weights / weights.mean()


def save_class_weights(path: str | Path, counts: list[int] | tuple[int, ...]) -> None:
    """Persist count-derived semantic weights and their source pixel counts."""
    weights = inverse_frequency_weights(counts).tolist()
    payload = {"classes": [item.name for item in Sem], "counts": list(counts), "weights": weights}
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_class_weights(path: str | Path) -> Tensor:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("classes") != [item.name for item in Sem]:
        raise ValueError("semantic weight classes do not match Sem ordering")
    weights = torch.tensor(payload["weights"], dtype=torch.float32)
    if weights.shape != (len(Sem),):
        raise ValueError("semantic weights have an invalid shape")
    return weights


def semantic_loss(logits: Tensor, targets: Tensor, weights: Tensor | None = None, *,
                  loss_type: str = "cross_entropy", focal_gamma: float = 2.0) -> Tensor:
    """Class-weighted cross-entropy or focal loss over valid semantic cells."""
    if logits.ndim != 4 or targets.ndim != 3 or logits.shape[0] != targets.shape[0]:
        raise ValueError("expected logits (B,C,H,W) and targets (B,H,W)")
    if loss_type not in {"cross_entropy", "focal"}:
        raise ValueError("loss_type must be 'cross_entropy' or 'focal'")
    if focal_gamma < 0:
        raise ValueError("focal_gamma must be nonnegative")
    if not targets.ne(IGNORE).any():
        return logits.sum() * 0.0
    if loss_type == "cross_entropy":
        return F.cross_entropy(logits, targets.long(), weight=weights, ignore_index=IGNORE)
    valid = targets.ne(IGNORE)
    raw = F.cross_entropy(logits, targets.long(), ignore_index=IGNORE, reduction="none")
    focal = raw * (1.0 - torch.exp(-raw)).pow(focal_gamma)
    if weights is None:
        return focal[valid].mean()
    safe_targets = targets.long().clamp(min=0, max=len(Sem) - 1)
    class_weights = weights.to(device=targets.device, dtype=logits.dtype)[safe_targets]
    weighted = focal * class_weights
    return weighted[valid].sum() / class_weights[valid].sum().clamp_min(torch.finfo(logits.dtype).eps)
