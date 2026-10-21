"""Paletas, famílias de textura e personagens sintéticos por seed."""

import colorsys
import random
from dataclasses import dataclass
from typing import cast

import numpy as np

from voidgym.gyms.platform_gym.params import StyleParams


@dataclass(frozen=True)
class RenderStyle:
    background: tuple[int, int, int]
    solid: tuple[int, int, int]
    platform: tuple[int, int, int]
    hazard: tuple[int, int, int]
    enemy: tuple[int, int, int]
    player: tuple[int, int, int]
    pickup: tuple[int, int, int]
    texture: np.ndarray
    style_params: StyleParams
    background_layer: np.ndarray
    texture_delta: np.ndarray
    hud: tuple[tuple[int, int, int, int, tuple[int, int, int]], ...]


def _hsv(rng: random.Random) -> tuple[int, int, int]:
    rgb = colorsys.hsv_to_rgb(rng.random(), rng.uniform(.35, .85), rng.uniform(.4, 1))
    return cast(tuple[int, int, int], tuple(round(channel * 255) for channel in rgb))


def _texture(seed: int, family: str, color: tuple[int, int, int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:16, :16]
    noise = rng.integers(-22, 23, (16, 16))
    patterns = {
        "smooth": np.zeros((16, 16), dtype=int),
        "noise": noise,
        "bricks": np.where((yy % 5 == 0) | (((xx + (yy // 5 % 2) * 4) % 8) == 0), 28, -8),
        "stones": np.where(((xx - 7) ** 2 + (yy - 8) ** 2) % 47 < 12, 25, -10),
        "planks": np.where((yy % 4 == 0), 24, -5) + noise // 3,
        "mesh": np.where((xx % 4 == 0) | (yy % 4 == 0), 30, -12),
        "moss": np.where((noise > 12), 35, noise // 2),
        "stripes": np.where((xx + yy) % 6 < 2, 26, -8),
        "crystals": np.where((xx - 8) ** 2 + (yy - 8) ** 2 < 35, 32, -18),
        "hexagons": np.where(((xx % 8) - 4) ** 2 + ((yy % 8) - 4) ** 2 < 12, 27, -14),
    }
    modulation = patterns[family]
    return np.clip(np.asarray(color, dtype=np.int16)[None, None, :] + modulation[:, :, None], 0, 255).astype(np.uint8)


def _luminance(color: tuple[int, int, int]) -> float:
    return .2126 * color[0] + .7152 * color[1] + .0722 * color[2]


def make_style(seed: int, split: str = "train") -> RenderStyle:
    params = StyleParams.sample(seed, split)
    rng = random.Random(seed)
    background, solid = _hsv(rng), _hsv(rng)
    if not params.low_contrast and abs(_luminance(solid) - _luminance(background)) < 38:
        solid = cast(tuple[int, int, int], tuple(255 - channel for channel in background))
    texture = _texture(seed, params.texture, solid)
    delta = texture.astype(np.int16) - np.asarray(solid, dtype=np.int16)[None, None, :]
    gradient = np.linspace(.65, 1.0, 108, dtype=np.float32)[:, None, None]
    background_layer = np.broadcast_to(np.asarray(background, dtype=np.float32)[None, None, :] * gradient, (108, 192, 3)).astype(np.uint8).copy()
    hud = tuple((int(rng.randrange(2, 175)), int(rng.randrange(2, 96)), int(rng.randrange(5, 15)), int(rng.randrange(2, 5)), _hsv(rng)) for _ in range(rng.randrange(6)))
    return RenderStyle(background, solid, _hsv(rng), _hsv(rng), _hsv(rng), _hsv(rng), _hsv(rng), texture, params, background_layer, delta, hud)
