"""Parâmetros do ginásio procedural, com as durações do contrato universal."""

import math
import random
from dataclasses import dataclass

from voidgym.core.observation import STEP_GAME_MS

# 80 ms lease-base × action_repeat 3 = 240 ms simulados por decisão.
POLICY_STEP_MS = STEP_GAME_MS
PHYSICS_DT = 0.02
SUBSTEPS = round(POLICY_STEP_MS / (PHYSICS_DT * 1000))
TEST_TEXTURE_FAMILIES = frozenset({"crystals", "hexagons"})
TEST_CHARACTER_FAMILIES = frozenset({"horned_ball"})
TEXTURE_FAMILIES = ("smooth", "noise", "bricks", "stones", "planks", "mesh", "moss", "stripes", "crystals", "hexagons")
CHARACTER_FAMILIES = ("capsule", "horned_ball", "cape", "triangle", "square_eyes")


@dataclass(frozen=True)
class PhysicsParams:
    run_speed: float = 8.0
    gravity: float = 55.0
    jump_short_height: float = 1.4
    jump_max_height: float = 4.0
    max_fall_speed: float = 20.0
    dash_distance: float = 4.0
    invulnerable_after_hit: float = 1.0
    player_width: float = 0.6
    player_height: float = 1.2

    @property
    def jump_v0(self) -> float:
        return math.sqrt(2 * self.gravity * self.jump_short_height)

    @property
    def hold_max(self) -> float:
        return (self.jump_max_height - self.jump_short_height) / self.jump_v0

    @property
    def jump_air_time(self) -> float:
        return self.hold_max + 2 * self.jump_v0 / self.gravity

    @property
    def max_jump_range(self) -> float:
        return self.run_speed * self.jump_air_time

    @classmethod
    def sample(cls, rng: random.Random) -> "PhysicsParams":
        short = rng.uniform(1.0, 1.8)
        maximum = rng.uniform(max(3.0, short + 0.1), 5.5)
        return cls(rng.uniform(6, 10), rng.uniform(40, 70), short, maximum,
                   rng.uniform(15, 25), rng.uniform(3, 5), rng.uniform(.8, 1.2),
                   rng.uniform(.48, .72), rng.uniform(.96, 1.44))


@dataclass(frozen=True)
class StyleParams:
    seed: int
    split: str
    texture: str
    character: str
    camera_width: float
    low_contrast: bool

    @classmethod
    def sample(cls, seed: int, split: str = "train") -> "StyleParams":
        rng = random.Random(seed)
        holdout = seed % 10 == 0
        is_test = split == "test" or holdout
        textures = list(TEST_TEXTURE_FAMILIES) if holdout else [item for item in TEXTURE_FAMILIES if is_test or item not in TEST_TEXTURE_FAMILIES]
        characters = list(TEST_CHARACTER_FAMILIES) if holdout else [item for item in CHARACTER_FAMILIES if is_test or item not in TEST_CHARACTER_FAMILIES]
        return cls(seed, "test" if is_test else "train", rng.choice(textures), rng.choice(characters), rng.uniform(18, 32), rng.random() < .15)
