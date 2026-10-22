"""AdapterEnv, ações idênticas ao HK e promoção do currículo."""

import numpy as np
from gymnasium.utils.env_checker import check_env

from voidgym.core.actions import UNIVERSAL_ACTIONS
from voidgym.envs.adapter_env import AdapterEnv
from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter
from voidgym.gyms.platform_gym.curriculum import Curriculum
from voidgym.gyms.platform_gym.reward import GymReward


def test_gym_adapter_observation_mask_check_env_and_random_steps(tmp_path) -> None:
    adapter = PlatformGymAdapter()
    assert adapter.action_table() == tuple(action.name for action in UNIVERSAL_ACTIONS)
    assert adapter.game_time_scale() == 1
    env = AdapterEnv(adapter, GymReward())
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=21)
    assert "sem" not in obs and "training_labels" in info
    mask = env.action_masks()
    for index, action in enumerate(UNIVERSAL_ACTIONS):
        if action.special or action.menu:
            assert not mask[index]
    for index in range(10_000):
        available = np.flatnonzero(adapter.action_mask())
        adapter.apply(int(available[index % len(available)]))
    env.close()


def test_curriculum_promotes_persists_and_reloads(tmp_path) -> None:
    curriculum = Curriculum(tmp_path / "curriculum.json")
    for _ in range(100 * 8):
        curriculum.record(True)
    assert curriculum.level == 7
    curriculum.save()
    loaded = Curriculum(tmp_path / "curriculum.json")
    loaded.load()
    assert loaded.level == 7


def test_level_seven_adapter_transitions_between_connected_rooms() -> None:
    adapter = PlatformGymAdapter()
    adapter.reset(seed=120, options={"level": 7})
    assert len(adapter.rooms) >= 3
    adapter.player.x = adapter.room.width - 1.5
    adapter.player.y = 2
    adapter.apply(adapter.action_table().index("NOOP"))
    assert adapter.room_index == 1
    assert not adapter.terminated
