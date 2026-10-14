from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pytest

from hk_agent.training_task import (
    TASK_ACTION_NAMES,
    TASK_ACTION_TEMPLATES,
    TASK_ACTIONS,
    ChannelsFirstObservation,
    InvalidTrainingTransition,
    VisualMotionTask,
    VisualMotionTaskConfig,
    confirmed_combat_components,
    resolve_calibrated_config,
)


def test_motor_task_exposes_nail_attack() -> None:
    attack_index = TASK_ACTION_NAMES.index("ATTACK")
    assert TASK_ACTIONS[attack_index] == 1 << 5


def test_confirmed_combat_rewards_use_per_frame_deltas_only() -> None:
    components = confirmed_combat_components(
        {"enemy_kills_since_last_frame": 2, "hp_lost_since_last_frame": 1},
        enemy_kill_reward=3.0,
        damage_penalty=-2.0,
    )
    assert components == {
        "confirmed_enemy_kills": 6.0,
        "confirmed_hp_lost": -2.0,
        "confirmed_geo_gained": 0.0,
        "confirmed_geo_spent": 0.0,
        "player_death": 0.0,
        "gameplay_run_failed": 0.0,
        "shade_recovered": 0.0,
    }
    with pytest.raises(InvalidTrainingTransition, match="combat delta"):
        confirmed_combat_components(
            {"enemy_kills_since_last_frame": True},
            enemy_kill_reward=3.0,
            damage_penalty=-2.0,
        )


def test_confirmed_geo_rewards_use_deltas_and_do_not_reward_death_loss() -> None:
    components = confirmed_combat_components(
        {"geo_gained_since_last_frame": 12, "geo_spent_since_last_frame": 5},
        enemy_kill_reward=0.0,
        damage_penalty=0.0,
        geo_gain_reward=0.1,
        geo_spend_reward=0.2,
    )
    assert components["confirmed_geo_gained"] == pytest.approx(1.2)
    assert components["confirmed_geo_spent"] == pytest.approx(1.0)

    death = confirmed_combat_components(
        {
            "geo_spent_since_last_frame": 500,
            "supervisor": {"player_dead": True},
        },
        enemy_kill_reward=0.0,
        damage_penalty=0.0,
        geo_spend_reward=1.0,
    )
    assert death["confirmed_geo_spent"] == 0.0


def test_death_penalties_replace_lethal_hp_and_propagate_run_failure() -> None:
    components = confirmed_combat_components(
        {
            "hp_lost_since_last_frame": 1,
            "run_failed": True,
            "supervisor": {"player_dead": True, "run_failed": True},
        },
        enemy_kill_reward=0.5,
        damage_penalty=-0.25,
        death_penalty=-2.0,
        run_failed_penalty=-4.0,
    )
    assert components["confirmed_hp_lost"] == 0
    assert components["player_death"] == -2
    assert components["gameplay_run_failed"] == -4


class FakeMotorEnvironment:
    metadata: ClassVar[dict[str, object]] = {}

    def __init__(self, *, invalid: bool = False) -> None:
        self.observation_space = gym.spaces.Dict(
            {
                "pixels": gym.spaces.Box(0, 255, (2, 10, 10, 3), np.uint8),
                "action_history": gym.spaces.Box(0, 1023, (10,), np.int64),
            }
        )
        self.action_space = gym.spaces.Discrete(1024)
        self.backend = SimpleNamespace(release_all=lambda: None)
        self.invalid = invalid
        self.step_index = 0
        self.closed = False
        self.run_failed = False
        self.geo_gained = 0
        self.geo_spent = 0
        self.enemy_kills = 0
        self.shade_recovered = False
        self.lease_ms = 80
        self.last_action: tuple[int, int] | None = None

    def _observation(self, value: int) -> dict[str, np.ndarray[Any, Any]]:
        pixels = np.full((2, 10, 10, 3), value, dtype=np.uint8)
        return {
            "pixels": pixels,
            "action_history": np.zeros(10, dtype=np.int64),
        }

    def reset(self, **_: object) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, object]]:
        self.step_index = 0
        return self._observation(0), {}

    def step(self, action: int, *, lease_ms: int | None = None) -> tuple[dict[str, np.ndarray[Any, Any]], float, bool, bool, dict[str, object]]:
        self.last_action = (action, self.lease_ms if lease_ms is None else lease_ms)
        self.step_index += 1
        value = 0 if action == 0 else (255 if self.step_index % 2 else 0)
        info: dict[str, object] = {
            "valid": not self.invalid,
            "termination_reason": "operational_failure" if self.invalid else None,
        }
        if self.run_failed:
            info.update(
                {
                    "run_failed": True,
                    "supervisor": {"player_dead": True, "run_failed": True},
                }
            )
        info["geo_gained_since_last_frame"] = self.geo_gained
        info["geo_spent_since_last_frame"] = self.geo_spent
        info["enemy_kills_since_last_frame"] = self.enemy_kills
        info["shade_recovered"] = self.shade_recovered
        return self._observation(value), 0.0, self.run_failed, self.invalid, info

    def close(self) -> None:
        self.closed = True


def _config() -> VisualMotionTaskConfig:
    return VisualMotionTaskConfig(
        horizon_steps=4,
        pixel_delta_threshold=10,
        motion_target_fraction=0.1,
        target_window=2,
        required_target_hits=1,
        crop_top_fraction=0,
        crop_bottom_fraction=1,
        crop_left_fraction=0,
        crop_right_fraction=1,
    )


def test_diagnostic_motion_bonus_is_once_and_never_claims_task_success() -> None:
    task = VisualMotionTask(FakeMotorEnvironment(), config=_config())  # type: ignore[arg-type]
    task.reset(seed=3)
    _, first, terminated, truncated, info = task.step(2)
    assert first > 0
    assert not terminated and not truncated
    assert info["diagnostic_bonus_awarded"] is False
    _, _second, terminated, truncated, info = task.step(2)
    assert info["diagnostic_bonus_awarded"] is True
    assert not terminated and not truncated
    _, _, _, _, third_info = task.step(2)
    assert third_info["diagnostic_bonus_awarded"] is False
    _, _, terminated, truncated, final_info = task.step(2)
    assert not terminated and truncated
    assert final_info["termination_reason"] == "time_limit"
    assert "task_success" not in final_info["reward_components"]


def test_invalid_base_transition_is_raised_before_reaching_learner() -> None:
    task = VisualMotionTask(FakeMotorEnvironment(invalid=True), config=_config())  # type: ignore[arg-type]
    task.reset()
    with pytest.raises(InvalidTrainingTransition, match="discarded before learner"):
        task.step(2)


def test_task_propagates_run_failure_and_applies_terminal_penalties() -> None:
    environment = FakeMotorEnvironment()
    environment.run_failed = True
    task = VisualMotionTask(environment, config=_config())  # type: ignore[arg-type]
    task.reset()
    _, reward, terminated, _, info = task.step(0)
    assert terminated
    assert info["run_failed"] is True
    assert info["reward_components"]["player_death"] == -2
    assert info["reward_components"]["gameplay_run_failed"] == -4
    assert reward < 0


def test_motor_task_bounds_geo_shaping_per_episode() -> None:
    environment = FakeMotorEnvironment()
    environment.geo_gained = 1_000
    environment.geo_spent = 1_000
    config = _config()
    task = VisualMotionTask(environment, config=config)  # type: ignore[arg-type]
    task.reset()
    paid = 0.0
    for _ in range(4):
        _, _, _, _, info = task.step(0)
        components = info["reward_components"]
        paid += components["confirmed_geo_gained"] + components["confirmed_geo_spent"]
    assert paid == pytest.approx(config.geo_reward_budget)


def test_calibration_requires_active_motion_above_noop_noise() -> None:
    calibrated = resolve_calibrated_config(
        _config(), noop_motion=[0.001] * 5, active_motion=[0.4] * 5
    )
    assert calibrated.motion_target_fraction > 0.001
    with pytest.raises(RuntimeError, match="not separable"):
        resolve_calibrated_config(
            _config(), noop_motion=[0.2] * 5, active_motion=[0.2] * 5
        )


def test_channels_first_wrapper_preserves_action_history() -> None:
    task = VisualMotionTask(FakeMotorEnvironment(), config=_config())  # type: ignore[arg-type]
    wrapped = ChannelsFirstObservation(task, target_height=8 + 28, target_width=8 + 28)
    observation, _ = wrapped.reset()
    assert observation["pixels"].shape == (6, 36, 36)
    assert observation["action_history"].shape == (10,)


def test_tap_hold_and_combination_actions_are_distinct_policy_choices() -> None:
    environment = FakeMotorEnvironment()
    task = VisualMotionTask(environment, config=_config())  # type: ignore[arg-type]
    observation, _ = task.reset()
    assert np.all(observation["action_history"] == 0)

    tap = TASK_ACTION_NAMES.index("RIGHT+JUMP_TAP")
    hold = TASK_ACTION_NAMES.index("RIGHT+JUMP_HOLD")
    jump_attack = TASK_ACTION_NAMES.index("JUMP+ATTACK")
    assert TASK_ACTIONS[tap] == TASK_ACTIONS[hold]
    assert TASK_ACTION_TEMPLATES[jump_attack].buttons == ("JUMP", "ATTACK")

    observation, *_ = task.step(tap)
    assert environment.last_action == (TASK_ACTIONS[tap], 50)
    assert observation["action_history"][-1] == tap
    observation, *_ = task.step(hold)
    assert environment.last_action == (TASK_ACTIONS[hold], 160)
    assert observation["action_history"][-2:].tolist() == [tap, hold]


def test_spell_directions_and_continuous_focus_are_explicit_actions() -> None:
    environment = FakeMotorEnvironment()
    task = VisualMotionTask(environment, config=_config())  # type: ignore[arg-type]
    task.reset()
    actions = dict(zip(TASK_ACTION_NAMES, TASK_ACTION_TEMPLATES, strict=True))

    assert actions["CAST"].buttons == ("CAST",)
    assert actions["CAST"].resolve_lease_ms(environment.lease_ms) == 50
    assert actions["UP+CAST"].buttons == ("UP", "CAST")
    assert actions["DOWN+CAST"].buttons == ("DOWN", "CAST")
    assert actions["FOCUS_HOLD"].buttons == ("CAST",)

    focus = TASK_ACTION_NAMES.index("FOCUS_HOLD")
    observation, _, _, _, info = task.step(focus)
    assert environment.last_action == (TASK_ACTIONS[focus], 1_500)
    assert observation["action_history"][-1] == focus
    contract = task.task_contract()["actions"][focus]
    assert contract["duration"] == "focus"
    assert contract["lease_ms"] == 1_500
    assert contract["release_after"] is True
    assert info["reward_components"]["time_cost"] == pytest.approx(
        task.config.time_cost * 1_500 / environment.lease_ms
    )


def test_nail_art_charge_actions_hold_then_release_explicitly() -> None:
    environment = FakeMotorEnvironment()
    task = VisualMotionTask(environment, config=_config())  # type: ignore[arg-type]
    task.reset()
    actions = dict(zip(TASK_ACTION_NAMES, TASK_ACTION_TEMPLATES, strict=True))

    assert actions["NAIL_ART_CHARGE"].buttons == ("ATTACK",)
    assert actions["UP+NAIL_ART_CHARGE"].buttons == ("UP", "ATTACK")
    assert actions["DOWN+NAIL_ART_CHARGE"].buttons == ("DOWN", "ATTACK")
    assert actions["LEFT+NAIL_ART_CHARGE"].buttons == ("LEFT", "ATTACK")
    assert actions["RIGHT+NAIL_ART_CHARGE"].buttons == ("RIGHT", "ATTACK")

    charged = TASK_ACTION_NAMES.index("NAIL_ART_CHARGE")
    observation, *_ = task.step(charged)
    assert environment.last_action == (TASK_ACTIONS[charged], 1_500)
    assert observation["action_history"][-1] == charged
    contract = task.task_contract()["actions"][charged]
    assert contract["duration"] == "charge"
    assert contract["lease_ms"] == 1_500
    assert contract["release_after"] is True


def test_kill_budget_and_shade_recovery_are_bounded_per_episode() -> None:
    environment = FakeMotorEnvironment()
    environment.enemy_kills = 2
    environment.shade_recovered = True
    task = VisualMotionTask(
        environment,  # type: ignore[arg-type]
        config=replace(
            _config(),
            enemy_kill_reward=2.0,
            enemy_kill_reward_budget=3.0,
            shade_recovery_reward=2.0,
        ),
    )
    task.reset()
    _, _, _, _, first = task.step(1)
    _, _, _, _, second = task.step(1)
    assert first["reward_components"]["confirmed_enemy_kills"] == 3.0
    assert first["reward_components"]["shade_recovered"] == 2.0
    assert second["reward_components"]["confirmed_enemy_kills"] == 0.0
    assert second["reward_components"]["shade_recovered"] == 0.0
