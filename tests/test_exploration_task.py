from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from numpy.typing import NDArray

from hk_agent.action_masking import ActionAvailability
from hk_agent.distributed_knowledge import portable_compatibility_fingerprint
from hk_agent.exploration_task import (
    ExplorationTask,
    ExplorationTaskConfig,
    calibrate_novelty,
    load_lifelong_visits,
)
from hk_agent.scene_change_task import scene_descriptor


class _Backend:
    def release_all(self) -> None:
        pass


class _Environment(gym.Env):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Dict({
            "pixels": gym.spaces.Box(0, 255, (4, 90, 160, 3), dtype=np.uint8),
            "action_history": gym.spaces.MultiDiscrete(np.full((8,), 27, dtype=np.int64)),
        })
        self.action_space = gym.spaces.Discrete(30)
        self.lease_ms = 80
        self.latest_action_availability = None
        self.backend = _Backend()
        self.frame = self._frame(0)
        self.terminal = False
        self.truncated = False
        self.extra = {}

    @staticmethod
    def _frame(shift: int) -> NDArray[np.uint8]:
        y, x = np.indices((90, 160))
        pattern = ((x // 8 + y // 8 + shift) % 2 * 120 + 60).astype(np.uint8)
        return np.stack((pattern, np.roll(pattern, shift * 10, axis=1), 180 - pattern), axis=-1)

    def _obs(self) -> dict[str, NDArray[np.generic]]:
        return {"pixels": np.repeat(self.frame[None, ...], 4, axis=0),
                "action_history": np.zeros((8,), dtype=np.int64)}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.frame = self._frame(0)
        self.terminal = False
        self.truncated = False
        return self._obs(), {}

    def step(self, action: int, *, lease_ms: int):
        if action == 1:
            self.frame = self._frame(1)
        elif action == 2:
            self.frame = self._frame(3)
        terminal = self.terminal
        info = {"valid": True, "termination_reason": "player_dead" if terminal else None,
                "enemy_kills_since_last_frame": 1 if terminal else 0,
                "hp_lost_since_last_frame": 0, "geo_gained_since_last_frame": 10,
                "geo_spent_since_last_frame": 0,
                "shade_hits_since_last_frame": 0,
                "hp_restored_since_last_frame": 0,
                "supervisor": {"player_dead": terminal}, **self.extra}
        return self._obs(), 0.0, terminal, self.truncated, info


def test_noop_static_has_no_novelty_and_camera_shift_does() -> None:
    environment = _Environment()
    task = ExplorationTask(environment, config=ExplorationTaskConfig(horizon_steps=20))
    task.reset()
    _, noop_reward, _, _, noop_info = task.step(0)
    assert noop_info["reward_components"]["novelty"] == 0
    _, active_reward, _, _, active_info = task.step(1)
    assert active_info["reward_components"]["novelty"] == 0.05
    assert active_reward > noop_reward


def test_novelty_budget_and_archive_are_bounded() -> None:
    environment = _Environment()
    config = ExplorationTaskConfig(horizon_steps=20, novelty_budget=0.05, archive_max=1)
    task = ExplorationTask(environment, config=config)
    _, _ = task.reset()
    _, _, _, _, info = task.step(1)
    assert info["reward_components"]["novelty"] == 0.05
    assert info["archive_size"] == 1
    _, _, _, _, repeated = task.step(1)
    assert repeated["reward_components"]["novelty"] == 0


def test_fifo_archive_forgets_old_descriptors() -> None:
    environment = _Environment()
    task = ExplorationTask(environment, config=ExplorationTaskConfig(horizon_steps=20, archive_max=1))
    task.reset()
    _, _, _, _, first = task.step(1)
    _, _, _, _, second = task.step(2)
    _, _, _, _, revisit = task.step(1)
    assert first["reward_components"]["novelty"] == 0.05
    assert second["reward_components"]["novelty"] == 0.05
    assert revisit["reward_components"]["novelty"] == 0.05
    assert revisit["archive_size"] == 1


def test_stable_scene_change_pays_once_without_terminating() -> None:
    environment = _Environment()
    task = ExplorationTask(environment, config=ExplorationTaskConfig(horizon_steps=20))
    task.reset()
    environment.frame = np.zeros((90, 160, 3), dtype=np.uint8)
    task.step(0)
    environment.frame = np.zeros((90, 160, 3), dtype=np.uint8)
    task.step(0)
    events = []
    for _ in range(4):
        environment.frame = _Environment._frame(1)
        events.append(task.step(0))
    paid = [item[4]["reward_components"]["scene_change"] for item in events]
    assert 5.0 in paid
    assert sum(value > 0 for value in paid) == 1
    assert all(not item[2] and not item[3] for item in events)


def test_death_suppresses_positive_rewards_and_terminates() -> None:
    environment = _Environment()
    task = ExplorationTask(environment)
    task.reset()
    environment.terminal = True
    _, reward, terminated, truncated, info = task.step(1)
    assert terminated and not truncated
    assert info["reward_components"]["death"] == -5
    assert info["reward_components"]["novelty"] == 0
    assert info["reward_components"]["scene_change"] == 0
    assert info["reward_components"]["confirmed_enemy_kills"] == 0
    assert reward < 0


def test_base_truncation_suppresses_positive_rewards() -> None:
    environment = _Environment()
    task = ExplorationTask(environment)
    task.reset()
    environment.truncated = True
    _, _, terminated, truncated, info = task.step(1)
    assert not terminated and truncated
    assert info["reward_components"]["novelty"] == 0
    assert info["reward_components"]["scene_change"] == 0


def test_black_and_white_frames_do_not_pay_novelty() -> None:
    for value in (0, 255):
        environment = _Environment()
        task = ExplorationTask(environment)
        task.reset()
        environment.frame = np.full((90, 160, 3), value, dtype=np.uint8)
        _, _, _, _, info = task.step(0)
        assert info["reward_components"]["novelty"] == 0


def test_unavailable_action_is_masked_and_audited() -> None:
    environment = _Environment()
    environment.latest_action_availability = ActionAvailability(
        mp_charge=0, spell_cost=33, focus_cost=33, fireball_level=0, quake_level=0,
        scream_level=0, has_nail_art=False, has_cyclone=False, has_dash_slash=False, has_upward_slash=False,
    )
    task = ExplorationTask(environment)
    task.reset()
    index = __import__("hk_agent.scene_change_task", fromlist=["SCENE_CHANGE_ACTION_NAMES"]).SCENE_CHANGE_ACTION_NAMES.index("FOCUS_HOLD")
    _, _, _, _, info = task.step(index)
    assert info["masked_action_substituted"] is True
    assert info["executed_task_action"] == 0
    assert index in info["action_mask"]


def test_novelty_calibration_requires_noop_floor_and_active_motion() -> None:
    base = np.zeros((144,), dtype=np.float32)
    noop = [base.copy() for _ in range(10)]
    active = [np.full((144,), 0.1 + index * 0.001, dtype=np.float32) for index in range(10)]
    threshold = calibrate_novelty(noop, active)
    assert threshold == 0.02
    try:
        calibrate_novelty(noop, [base.copy() for _ in range(10)])
    except ValueError as error:
        assert "active exploration" in str(error)
    else:
        raise AssertionError("uninformative active calibration must fail")


def test_exploration_contract_is_pixel_only_and_fingerprint_stable() -> None:
    from hk_agent.distributed_knowledge import _public_task_contract

    task = ExplorationTask(_Environment())
    contract = task.task_contract()
    assert contract["reward_version"] == "explore_v1"
    assert contract["information_boundary"] == ["pixels", "applied_action_history"]
    assert contract["privileged_state"] == []
    assert _public_task_contract(contract, from_manifest=False)["reward_version"] == "explore_v1"
    assert portable_compatibility_fingerprint(contract, {}) == portable_compatibility_fingerprint(contract, {})
    assert scene_descriptor(_Environment._frame(0), __import__("hk_agent.scene_change_task", fromlist=["SceneChangeTaskConfig"]).SceneChangeTaskConfig()).shape == (144, 3)


def _v2_config(**kwargs):
    return ExplorationTaskConfig(reward_version="explore_v2", geo_gain_reward=0, geo_spend_reward=0,
        shade_recovery_reward=4.0, **kwargs)


def test_v2_masks_hud_and_requires_persistent_scene_change(tmp_path: Path) -> None:
    env = _Environment()
    task = ExplorationTask(env, config=_v2_config(), task_log_path=tmp_path / "phase" / "task.jsonl")
    task.reset()
    base = env.frame.copy()
    hud = base.copy()
    hud[:27, :64] = 255 - hud[:27, :64]
    env.frame = hud
    _, _, _, _, info = task.step(0)
    assert info["reward_components"]["novelty"] == 0
    env.frame = _Environment._frame(1)
    _, _, _, _, flash = task.step(0)
    assert flash["reward_components"]["novelty"] == 0
    env.frame = base
    task.step(0)
    env.frame = _Environment._frame(3)
    task.step(0)
    slightly_changed = _Environment._frame(3).copy()
    slightly_changed[60:62, 100:102] = np.clip(slightly_changed[60:62, 100:102].astype(np.int16) + 4, 0, 255).astype(np.uint8)
    env.frame = slightly_changed
    _, _, _, _, persistent = task.step(0)
    assert persistent["reward_components"]["novelty"] == 0.05


def test_v2_lifelong_novelty_decays_and_persists_across_task_processes(tmp_path: Path) -> None:
    path = tmp_path / "run" / "phase" / "task.jsonl"
    config = _v2_config(novelty_reward=0.05)
    for expected in (0.05, 0.05 / np.sqrt(2)):
        env = _Environment()
        task = ExplorationTask(env, config=config, task_log_path=path)
        task.reset()
        env.frame = _Environment._frame(1)
        task.step(0)
        _, reward, _, _, info = task.step(0)
        assert info["reward_components"]["novelty"] == pytest.approx(expected)
        assert reward > 0
        task.close()
    assert len(load_lifelong_visits(tmp_path / "run" / "lifelong-novelty.json", 4096)) == 1


def test_v2_contract_and_policy_observation_preserve_full_hud(tmp_path: Path) -> None:
    from hk_agent.distributed_knowledge import portable_compatibility_fingerprint

    env1, env2 = _Environment(), _Environment()
    v1 = ExplorationTask(env1)
    v2 = ExplorationTask(env2, config=_v2_config(), task_log_path=tmp_path / "phase" / "task.jsonl")
    obs1, _ = v1.reset(seed=5)
    obs2, _ = v2.reset(seed=5)
    assert np.array_equal(obs1["pixels"], obs2["pixels"])
    c1, c2 = v1.task_contract(), v2.task_contract()
    assert c2["reward_version"] == "explore_v2"
    assert c1["task_contract_schema_version"] == 1 and c2["task_contract_schema_version"] == 2
    assert c2["information_boundary"] == c1["information_boundary"]
    from hk_agent.distributed_knowledge import _public_task_contract
    assert _public_task_contract(c2, from_manifest=False)["reward_version"] == "explore_v2"
    assert portable_compatibility_fingerprint(c1, {}) != portable_compatibility_fingerprint(c2, {})
    assert c2["config"]["geo_gain_reward"] == c2["config"]["geo_spend_reward"] == 0


def test_v2_geo_hp_shade_rewards_are_bounded_and_reward_only(tmp_path: Path) -> None:
    env = _Environment()
    env.extra = {"hp_restored_since_last_frame": 2, "shade_hits_since_last_frame": 9,
        "shade_distance": 9.0, "scene_name": "Room", "shade_pending": True}
    task = ExplorationTask(env, config=_v2_config(), task_log_path=tmp_path / "phase" / "task.jsonl")
    task.reset()
    observation, _, _, _, info = task.step(0)
    components = info["reward_components"]
    assert components["confirmed_geo_gained"] == components["confirmed_geo_spent"] == 0
    assert components["hp_restored"] == 0.8
    assert components["shade_hits"] == 1.2
    assert "scene_name" in task.task_contract()["reward_only_signals"]
    assert set(observation) == {"pixels", "action_history"}


def test_v2_shade_potential_is_capped_and_requires_same_scene(tmp_path: Path) -> None:
    env = _Environment()
    env.extra = {"shade_distance": 10.0, "scene_name": "Room", "shade_pending": True}
    task = ExplorationTask(env, config=_v2_config(), task_log_path=tmp_path / "phase" / "task.jsonl")
    task.reset()
    _, _, _, _, first = task.step(0)
    assert first["reward_components"]["shade_approach"] == 0
    env.extra = {"shade_distance": 1.0, "scene_name": "Room", "shade_pending": True}
    _, _, _, _, closer = task.step(0)
    assert closer["reward_components"]["shade_approach"] == 0.05
    env.extra = {"shade_distance": 0.0, "scene_name": "Other", "shade_pending": True}
    _, _, _, _, switched = task.step(0)
    assert switched["reward_components"]["shade_approach"] == 0
    assert switched["reward_components"]["time_cost"] == pytest.approx(-0.00108)


def test_v2_stationary_attack_cannot_farm_novelty_but_movement_can(tmp_path: Path) -> None:
    from hk_agent.scene_change_task import SCENE_CHANGE_ACTION_NAMES

    env = _Environment()
    task = ExplorationTask(env, config=_v2_config(), task_log_path=tmp_path / "phase" / "task.jsonl")
    task.reset()
    env.frame = _Environment._frame(1)
    attack = SCENE_CHANGE_ACTION_NAMES.index("ATTACK")
    _, _, _, _, first = task.step(attack)
    _, _, _, _, repeat = task.step(attack)
    assert first["reward_components"]["novelty"] == 0
    assert repeat["reward_components"]["novelty"] == 0
    env.frame = _Environment._frame(3)
    move = SCENE_CHANGE_ACTION_NAMES.index("RIGHT")
    task.step(move)
    env.frame = _Environment._frame(3)
    _, _, _, _, moved = task.step(move)
    assert moved["reward_components"]["novelty"] == pytest.approx(0.05)


def test_v2_reward_safety_constraints_are_enforced() -> None:
    with pytest.raises(ValueError, match="healing reward"):
        _v2_config(damage_penalty=-0.4).validate()
    with pytest.raises(ValueError, match="does not reward Geo"):
        ExplorationTaskConfig(reward_version="explore_v2", shade_recovery_reward=4.0).validate()
