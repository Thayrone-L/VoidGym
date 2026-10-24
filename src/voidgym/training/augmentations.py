"""Sequence-consistent image and semantic-label augmentation used only during updates."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812


def random_shift(images: Tensor, sem: Tensor | None = None,
                 max_pixels: int = 4) -> tuple[Tensor, Tensor | None]:
    """Shift each recurrent sequence by one shared pixel offset with replicated borders."""
    if images.ndim != 5:
        raise ValueError("images must have shape (B,T,C,H,W)")
    batch, steps, channels, height, width = images.shape
    shift_y = torch.randint(-max_pixels, max_pixels + 1, (batch,), device=images.device)
    shift_x = torch.randint(-max_pixels, max_pixels + 1, (batch,), device=images.device)
    padded = F.pad(images.reshape(batch * steps, channels, height, width).float(),
                   (max_pixels,) * 4, mode="replicate").reshape(
                       batch, steps, channels, height + 2 * max_pixels, width + 2 * max_pixels)
    shifted = torch.stack([
        padded[index, :, :, max_pixels + int(shift_y[index]):max_pixels + int(shift_y[index]) + height,
               max_pixels + int(shift_x[index]):max_pixels + int(shift_x[index]) + width]
        for index in range(batch)
    ]).to(images.dtype)
    if sem is None:
        return shifted, None
    if sem.shape[:2] != (batch, steps):
        raise ValueError("semantic labels must align with image sequences")
    sem_h, sem_w = sem.shape[-2:]
    sem_pad = F.pad(sem.float().reshape(batch * steps, 1, sem_h, sem_w),
                    (1, 1, 1, 1), mode="replicate").reshape(batch, steps, sem_h + 2, sem_w + 2)
    sem_y = torch.round(shift_y.float() / 4).to(torch.int64)
    sem_x = torch.round(shift_x.float() / 4).to(torch.int64)
    moved = torch.stack([
        sem_pad[index, :, 1 + int(sem_y[index]):1 + int(sem_y[index]) + sem_h,
                1 + int(sem_x[index]):1 + int(sem_x[index]) + sem_w]
        for index in range(batch)
    ]).to(sem.dtype)
    return shifted, moved


def color_jitter(images: Tensor, amount: float = .1) -> Tensor:
    """Apply sequence-consistent brightness/contrast multipliers to uint8 stacks."""
    if images.ndim != 5:
        raise ValueError("images must have shape (B,T,C,H,W)")
    batch = images.shape[0]
    factor_shape = (batch,) + (1,) * (images.ndim - 1)
    brightness = torch.empty(factor_shape, device=images.device).uniform_(1 - amount, 1 + amount)
    contrast = torch.empty(factor_shape, device=images.device).uniform_(1 - amount, 1 + amount)
    value = images.float()
    mean = value.mean(dim=(-3, -2, -1), keepdim=True)
    # One factor per sequence; use the first timestep's mean to keep color stable over time.
    sequence_mean = mean[:, :1]
    value = (value - sequence_mean) * contrast + sequence_mean
    return (value * brightness).clamp(0, 255).to(images.dtype)


def augment_sequence(images: Tensor, sem: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
    images, sem = random_shift(images, sem)
    return color_jitter(images), sem
