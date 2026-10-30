"""Seeded palette and render style helpers for platform gym."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from functools import lru_cache
from typing import cast

import numpy as np

from voidgym.gyms.platform_gym.params import Split, StyleParams


@dataclass(frozen=True, slots=True)
class Palette:
    background: tuple[int, int, int]
    ground: tuple[int, int, int]
    platform: tuple[int, int, int]
    hazard: tuple[int, int, int]
    enemy: tuple[int, int, int]
    player: tuple[int, int, int]
    pickup: tuple[int, int, int]
    foreground: tuple[int, int, int]


def _hsv(rng: np.random.Generator, low: float = .25, high: float = .9) -> tuple[int, int, int]:
    h, s, v = float(rng.random()), float(rng.uniform(.35, .95)), float(rng.uniform(low, high))
    return cast(tuple[int, int, int], tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v)))


@lru_cache(maxsize=256)
def sample_style(seed: int, *, split: Split = "train") -> tuple[StyleParams, Palette]:
    rng = np.random.default_rng(seed)
    style = StyleParams.sample(rng, split=split)
    background = _hsv(rng, .12, .52)
    foreground = _hsv(rng, .55, .98)
    if style.contrast >= .15:
        # Force a robust luma difference while preserving random hue.
        bg_luma = sum(c * w for c, w in zip(background, (.2126, .7152, .0722), strict=False))
        if abs(bg_luma - sum(c * w for c, w in zip(foreground, (.2126, .7152, .0722), strict=False))) < 38:
            foreground = (245, 220, 70) if bg_luma < 120 else (30, 35, 65)
    palette = Palette(background, _hsv(rng), _hsv(rng), (220, 65, 90), _hsv(rng), _hsv(rng), (245, 210, 55), foreground)
    return style, palette
