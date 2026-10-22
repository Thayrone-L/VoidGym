"""Parâmetros de física e estilo aleatórios por sala do ginásio 2D."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import numpy as np

from voidgym.core.observation import STEP_GAME_MS

PHYSICS_DT_SECONDS = 0.02
# A política avança 80 ms por chamada, portanto a física roda quatro passos de 20 ms.
PHYSICS_STEPS_PER_POLICY_STEP = round(STEP_GAME_MS / (PHYSICS_DT_SECONDS * 1000))

TEXTURE_FAMILIES = (
    "smooth", "noise", "bricks", "rounded_stones", "wood_planks",
    "metal_grid", "moss", "diagonal_stripes", "crystals", "hexagons",
)
TEST_TEXTURE_FAMILIES = ("crystals", "hexagons")
CHARACTER_FAMILIES = ("capsule", "horned_ball", "cape_rectangle", "triangle", "eyed_square")
TEST_CHARACTER_FAMILIES = ("triangle",)
TRAIN_TEXTURE_FAMILIES = tuple(name for name in TEXTURE_FAMILIES if name not in TEST_TEXTURE_FAMILIES)
TRAIN_CHARACTER_FAMILIES = tuple(name for name in CHARACTER_FAMILIES if name not in TEST_CHARACTER_FAMILIES)
Split = Literal["train", "test"]


def split_for_seed(seed: int) -> Split:
    """Every tenth seed is held out entirely from training."""
    return "test" if seed % 10 == 0 else "train"


@dataclass(frozen=True, slots=True)
class PhysicsParams:
    """Física sorteada por sala; dimensões são unidades de mundo (u)."""

    run_speed: float = 8.0
    gravity: float = 55.0
    jump_short_height: float = 1.4
    jump_max_height: float = 4.2
    max_fall_speed: float = 20.0
    dash_distance: float = 4.0
    invulnerable_after_hit: float = 1.0
    player_width: float = 0.6
    player_height: float = 1.2

    def __post_init__(self) -> None:
        ranges = (
            (self.run_speed, 6.0, 10.0, "run_speed"),
            (self.gravity, 40.0, 70.0, "gravity"),
            (self.jump_short_height, 1.0, 1.8, "jump_short_height"),
            (self.jump_max_height, 3.0, 5.5, "jump_max_height"),
            (self.max_fall_speed, 15.0, 25.0, "max_fall_speed"),
            (self.dash_distance, 3.0, 5.0, "dash_distance"),
            (self.invulnerable_after_hit, 0.8, 1.2, "invulnerable_after_hit"),
        )
        for value, minimum, maximum, name in ranges:
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
        if self.jump_max_height <= self.jump_short_height:
            raise ValueError("jump_max_height must exceed jump_short_height")
        if not 0.48 <= self.player_width <= 0.72:
            raise ValueError("player_width must be within ±20% of 0.6 u")
        if not 0.96 <= self.player_height <= 1.44:
            raise ValueError("player_height must be within ±20% of 1.2 u")

    @classmethod
    def sample(cls, rng: np.random.Generator) -> PhysicsParams:
        """Sample a valid room physics profile from the declared ranges."""
        return cls(
            run_speed=float(rng.uniform(6.0, 10.0)),
            gravity=float(rng.uniform(40.0, 70.0)),
            jump_short_height=float(rng.uniform(1.0, 1.8)),
            jump_max_height=float(rng.uniform(3.0, 5.5)),
            max_fall_speed=float(rng.uniform(15.0, 25.0)),
            dash_distance=float(rng.uniform(3.0, 5.0)),
            invulnerable_after_hit=float(rng.uniform(0.8, 1.2)),
            player_width=float(rng.uniform(0.48, 0.72)),
            player_height=float(rng.uniform(0.96, 1.44)),
        )

    @property
    def jump_v0(self) -> float:
        """Initial vertical speed for the requested short jump height."""
        return sqrt(2.0 * self.gravity * self.jump_short_height)

    @property
    def jump_hold_max_seconds(self) -> float:
        """Hold time that extends the jump from short to maximum height."""
        return (self.jump_max_height - self.jump_short_height) / self.jump_v0


@dataclass(frozen=True, slots=True)
class StyleParams:
    """Visual randomization parameters; style families never affect physics."""

    texture_family: str
    character_family: str
    palette_seed: int
    contrast: float
    edge_style: str
    background_layers: int
    fog_strength: float
    light_strength: float
    camera_width: float
    hud_icons: int

    def __post_init__(self) -> None:
        if self.texture_family not in TEXTURE_FAMILIES:
            raise ValueError("unknown texture_family")
        if self.character_family not in CHARACTER_FAMILIES:
            raise ValueError("unknown character_family")
        if not 0.0 <= self.contrast <= 1.0:
            raise ValueError("contrast must be in [0, 1]")
        if self.edge_style not in {"straight", "rounded", "irregular", "moss", "shadow"}:
            raise ValueError("unknown edge_style")
        if not 0 <= self.background_layers <= 3:
            raise ValueError("background_layers must be in [0, 3]")
        if not 0.0 <= self.fog_strength <= 1.0 or not 0.0 <= self.light_strength <= 1.0:
            raise ValueError("fog_strength and light_strength must be in [0, 1]")
        if not 18.0 <= self.camera_width <= 32.0:
            raise ValueError("camera_width must be in [18, 32] u")
        if not 0 <= self.hud_icons <= 5:
            raise ValueError("hud_icons must be in [0, 5]")

    @classmethod
    def sample(cls, rng: np.random.Generator, *, split: Split) -> StyleParams:
        """Sample style values while keeping test-only families out of training."""
        if split not in {"train", "test"}:
            raise ValueError("split must be train or test")
        texture_families = TEST_TEXTURE_FAMILIES if split == "test" else TRAIN_TEXTURE_FAMILIES
        character_families = TEST_CHARACTER_FAMILIES if split == "test" else TRAIN_CHARACTER_FAMILIES
        return cls(
            texture_family=str(rng.choice(texture_families)),
            character_family=str(rng.choice(character_families)),
            palette_seed=int(rng.integers(0, 2**32, dtype=np.uint32)),
            contrast=float(rng.uniform(0.15, 1.0) if rng.random() < 0.85 else rng.uniform(0.0, 0.149999)),
            edge_style=str(rng.choice(("straight", "rounded", "irregular", "moss", "shadow"))),
            background_layers=int(rng.integers(0, 4)),
            fog_strength=float(rng.uniform(0.0, 0.65)),
            light_strength=float(rng.uniform(0.0, 1.0)),
            camera_width=float(rng.uniform(18.0, 32.0)),
            hud_icons=int(rng.integers(0, 6)),
        )
