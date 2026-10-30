import numpy as np
import pytest

from voidgym.core.observation import STEP_GAME_MS
from voidgym.gyms.platform_gym.params import (
    PHYSICS_DT_SECONDS,
    PHYSICS_STEPS_PER_POLICY_STEP,
    TEST_CHARACTER_FAMILIES,
    TEST_TEXTURE_FAMILIES,
    TRAIN_CHARACTER_FAMILIES,
    TRAIN_TEXTURE_FAMILIES,
    PhysicsParams,
    StyleParams,
    split_for_seed,
)
from voidgym.gyms.platform_gym.world import Rect, Solid


def test_step_time_and_physics_profile_ranges() -> None:
    assert STEP_GAME_MS == 80
    assert PHYSICS_DT_SECONDS == 0.02
    assert PHYSICS_STEPS_PER_POLICY_STEP == 4
    physics = PhysicsParams.sample(np.random.default_rng(81))
    assert 6 <= physics.run_speed <= 10
    assert 40 <= physics.gravity <= 70
    assert 1 <= physics.jump_short_height <= 1.8
    assert 3 <= physics.jump_max_height <= 5.5
    assert physics.jump_v0 > 0
    assert physics.jump_hold_max_seconds > 0


def test_seed_split_and_held_out_style_families() -> None:
    assert split_for_seed(10) == "test"
    assert split_for_seed(11) == "train"
    train = StyleParams.sample(np.random.default_rng(1), split=split_for_seed(11))
    test = StyleParams.sample(np.random.default_rng(2), split=split_for_seed(10))
    assert train.texture_family in TRAIN_TEXTURE_FAMILIES
    assert train.character_family in TRAIN_CHARACTER_FAMILIES
    assert test.texture_family in TEST_TEXTURE_FAMILIES
    assert test.character_family in TEST_CHARACTER_FAMILIES
    assert set(TRAIN_TEXTURE_FAMILIES).isdisjoint(TEST_TEXTURE_FAMILIES)
    assert set(TRAIN_CHARACTER_FAMILIES).isdisjoint(TEST_CHARACTER_FAMILIES)


def test_world_rect_and_solid_contract() -> None:
    rect = Rect(1, 2, 3, 4)
    assert rect.contains_point(4, 6)
    assert rect.intersects(Rect(3, 5, 2, 2))
    assert not rect.intersects(Rect(4, 6, 1, 1))
    assert Solid(rect, "platform").kind == "platform"
    with pytest.raises(ValueError):
        Rect(0, 0, 0, 1)
    with pytest.raises(ValueError):
        Solid(rect, "one_way")  # type: ignore[arg-type]
