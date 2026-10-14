from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from hk_agent.gym_env import HollowKnightEnv
from hk_agent.temporal_actions import (
    TEMPORAL_COMBAT_ACTIONS,
    TemporalActionHistory,
    action_contract,
    temporal_observation_space,
)

VISUAL_MOTION_REWARD_VERSION = "visual_motion_v1"


class InvalidTrainingTransition(RuntimeError):
    """Raised before an operationally invalid transition can reach the learner."""


@dataclass(frozen=True, slots=True)
class VisualMotionTaskConfig:
    """Versioned, map-agnostic first motor task.

    Only pixels already present in the policy observation are used.  The task
    deliberately exposes a small raw-action subset; it contains no macro,
    location, route, scene name, or privileged game state.
    """

    horizon_steps: int | None = 128
    pixel_delta_threshold: int = 12
    motion_target_fraction: float = 0.004
    target_window: int = 5
    required_target_hits: int = 3
    visual_change_scale: float = 0.1
    time_cost: float = -0.001
    diagnostic_target_bonus: float = 1.0
    enemy_kill_reward: float = 2.0
    enemy_kill_reward_budget: float = 6.0
    shade_recovery_reward: float = 2.0
    damage_penalty: float = -0.25
    geo_gain_reward: float = 0.01
    geo_spend_reward: float = 0.02
    geo_reward_budget: float = 2.5
    death_penalty: float = -2.0
    run_failed_penalty: float = -4.0
    crop_top_fraction: float = 0.10
    crop_bottom_fraction: float = 0.85
    crop_left_fraction: float = 0.10
    crop_right_fraction: float = 0.90

    def validate(self) -> None:
        if self.horizon_steps is not None and not 1 <= self.horizon_steps <= 100_000:
            raise ValueError("horizon_steps is outside the safe range")
        if not 1 <= self.pixel_delta_threshold <= 255:
            raise ValueError("pixel_delta_threshold must be in [1, 255]")
        if not 0 < self.motion_target_fraction <= 1:
            raise ValueError("motion_target_fraction must be in (0, 1]")
        if not 1 <= self.required_target_hits <= self.target_window:
            raise ValueError("required_target_hits must be in [1, target_window]")
        if self.visual_change_scale <= 0 or self.diagnostic_target_bonus <= 0:
            raise ValueError("positive reward components must be positive")
        if self.time_cost > 0:
            raise ValueError("time_cost cannot be positive")
        if (
            self.enemy_kill_reward < 0
            or self.enemy_kill_reward_budget < 0
            or self.shade_recovery_reward < 0
            or self.geo_gain_reward < 0
            or self.geo_spend_reward < 0
            or self.damage_penalty > 0
            or self.death_penalty > 0
            or self.run_failed_penalty > 0
        ):
            raise ValueError("combat reward signs violate the task contract")
        if self.shade_recovery_reward > abs(self.death_penalty):
            raise ValueError(
                "shade_recovery_reward cannot exceed the magnitude of death_penalty"
            )
        if not (
            0 <= self.crop_top_fraction < self.crop_bottom_fraction <= 1
            and 0 <= self.crop_left_fraction < self.crop_right_fraction <= 1
        ):
            raise ValueError("crop fractions must form a non-empty normalized rectangle")
        if not 0 <= self.geo_reward_budget <= 2.5:
            raise ValueError("geo_reward_budget must be in [0, 2.5]")


TASK_ACTION_TEMPLATES = TEMPORAL_COMBAT_ACTIONS
TASK_ACTIONS: tuple[int, ...] = tuple(
    action.raw_action_index for action in TASK_ACTION_TEMPLATES
)
TASK_ACTION_NAMES: tuple[str, ...] = tuple(
    action.name for action in TASK_ACTION_TEMPLATES
)


class VisualMotionTask(gym.Env[dict[str, NDArray[np.generic]], int]):
    """Reward wrapper for the first bounded motor-learning experiment."""

    def __init__(
        self,
        environment: HollowKnightEnv,
        *,
        config: VisualMotionTaskConfig | None = None,
        task_log_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.environment = environment
        self.config = config or VisualMotionTaskConfig()
        self.config.validate()
        self.task_log_path = task_log_path
        self.action_space = gym.spaces.Discrete(len(TASK_ACTION_TEMPLATES))  # type: ignore[assignment]
        source_space = cast(gym.spaces.Dict, environment.observation_space)
        self.observation_space = temporal_observation_space(  # type: ignore[assignment]
            environment.observation_space,
            number_of_actions=len(TASK_ACTION_TEMPLATES),
        )
        history_space = source_space["action_history"]
        if history_space.shape is None:
            raise TypeError("action_history must declare a shape")
        self._action_history = TemporalActionHistory(history_space.shape[0])
        self.metadata = environment.metadata
        self._previous_pixels: NDArray[np.uint8] | None = None
        self._recent_motion: deque[bool] = deque(maxlen=self.config.target_window)
        self._target_bonus_paid = False
        self._steps = 0
        self._geo_reward_paid = 0.0
        self._enemy_kill_reward_paid = 0.0
        self._shade_recovery_reward_paid = False
        self._done = True

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, NDArray[np.generic]], dict[str, Any]]:
        super().reset(seed=seed)
        observation, _ = self.environment.reset(seed=seed, options=options)
        self._previous_pixels = observation["pixels"][-1].copy()
        self._recent_motion.clear()
        self._target_bonus_paid = False
        self._steps = 0
        self._geo_reward_paid = 0.0
        self._enemy_kill_reward_paid = 0.0
        self._shade_recovery_reward_paid = False
        self._done = False
        self._action_history.reset()
        return self._action_history.apply(observation), {}

    def step(
        self, action: int
    ) -> tuple[dict[str, NDArray[np.generic]], float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("reset() is required before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid task action: {action!r}")
        action_template = TASK_ACTION_TEMPLATES[int(action)]
        action_lease_ms = action_template.resolve_lease_ms(self.environment.lease_ms)
        observation, _, terminated, truncated, base_info = self.environment.step(
            action_template.raw_action_index, lease_ms=action_lease_ms
        )
        if base_info.get("valid") is not True:
            self._done = True
            reason = base_info.get("termination_reason")
            self.environment.backend.release_all()
            self._append_task_log(
                {
                    "schema_version": 1,
                    "reward_version": VISUAL_MOTION_REWARD_VERSION,
                    "step": self._steps + 1,
                    "valid": False,
                    "termination_reason": reason,
                    "discarded_before_learner": True,
                }
            )
            raise InvalidTrainingTransition(
                f"operationally invalid transition discarded before learner: {reason}"
            )
        self._action_history.append(int(action))
        observation = self._action_history.apply(observation)
        if self._previous_pixels is None:
            raise RuntimeError("task has no previous observation")
        current = observation["pixels"][-1]
        motion_fraction = visual_motion_fraction(
            self._previous_pixels, current, self.config
        )
        self._previous_pixels = current.copy()
        self._steps += 1
        self._recent_motion.append(motion_fraction >= self.config.motion_target_fraction)
        target_reached_now = (
            len(self._recent_motion) == self.config.target_window
            and sum(self._recent_motion) >= self.config.required_target_hits
        )
        award_diagnostic_target = target_reached_now and not self._target_bonus_paid
        if award_diagnostic_target:
            self._target_bonus_paid = True
        horizon = (
            self.config.horizon_steps is not None
            and self._steps >= self.config.horizon_steps
        )
        task_terminated = terminated
        task_truncated = truncated or (horizon and not task_terminated)
        confirmed_components = confirmed_combat_components(
            base_info,
            enemy_kill_reward=self.config.enemy_kill_reward,
            shade_recovery_reward=self.config.shade_recovery_reward,
            damage_penalty=self.config.damage_penalty,
            geo_gain_reward=self.config.geo_gain_reward,
            geo_spend_reward=self.config.geo_spend_reward,
            death_penalty=self.config.death_penalty,
            run_failed_penalty=self.config.run_failed_penalty,
        )
        confirmed_components["confirmed_enemy_kills"] = min(
            confirmed_components["confirmed_enemy_kills"],
            max(
                0.0,
                self.config.enemy_kill_reward_budget
                - self._enemy_kill_reward_paid,
            ),
        )
        self._enemy_kill_reward_paid += confirmed_components[
            "confirmed_enemy_kills"
        ]
        if self._shade_recovery_reward_paid:
            confirmed_components["shade_recovered"] = 0.0
        elif confirmed_components["shade_recovered"] > 0:
            self._shade_recovery_reward_paid = True
        for component_name in ("confirmed_geo_gained", "confirmed_geo_spent"):
            confirmed_components[component_name] = min(
                confirmed_components[component_name],
                max(0.0, self.config.geo_reward_budget - self._geo_reward_paid),
            )
            self._geo_reward_paid += confirmed_components[component_name]
        components = {
            "visual_change": self.config.visual_change_scale
            * min(1.0, motion_fraction / self.config.motion_target_fraction),
            # Temporal actions form a semi-Markov decision process. Charge/focus
            # must pay for the wall-clock time they consume instead of receiving
            # the same cost as a 50 ms tap.
            "time_cost": self.config.time_cost
            * (action_lease_ms / self.environment.lease_ms),
            "diagnostic_target": (
                self.config.diagnostic_target_bonus if award_diagnostic_target else 0.0
            ),
            **confirmed_components,
        }
        reward = float(sum(components.values()))
        reason = base_info.get("termination_reason")
        if horizon and reason is None:
            reason = "time_limit"
        self._done = task_terminated or task_truncated
        if self._done:
            self.environment.backend.release_all()
        info: dict[str, Any] = {
            "valid": True,
            "termination_reason": reason,
            "reward_version": VISUAL_MOTION_REWARD_VERSION,
            "reward_components": components,
            "motion_fraction": motion_fraction,
            "diagnostic_motion_detected": target_reached_now,
            "diagnostic_bonus_awarded": award_diagnostic_target,
        }
        _copy_lifecycle_info(base_info, info)
        self._append_task_log(
            {
                "schema_version": 1,
                "reward_version": VISUAL_MOTION_REWARD_VERSION,
                "step": self._steps,
                "task_action": action_template.name,
                "raw_action_index": action_template.raw_action_index,
                "lease_ms": action_lease_ms,
                "motion_fraction": motion_fraction,
                "diagnostic_motion_detected": target_reached_now,
                "diagnostic_bonus_awarded": award_diagnostic_target,
                "reward": {"total": reward, "components": components},
                "terminated": task_terminated,
                "truncated": task_truncated,
                "termination_reason": reason,
                "valid": True,
            }
        )
        return observation, reward, task_terminated, task_truncated, info
    def close(self) -> None:
        self.environment.close()

    def task_contract(self) -> dict[str, Any]:
        return {
            "reward_version": VISUAL_MOTION_REWARD_VERSION,
            "action_names": list(TASK_ACTION_NAMES),
            "raw_action_indices": list(TASK_ACTIONS),
            "actions": action_contract(
                TASK_ACTION_TEMPLATES, base_lease_ms=self.environment.lease_ms
            ),
            "config": asdict(self.config),
            "information_boundary": ["pixels", "applied_action_history"],
            "reward_only_signals": [
                "enemy_kills_since_last_frame",
                "hp_lost_since_last_frame",
                "geo_gained_since_last_frame",
                "geo_spent_since_last_frame",
                "shade_recovered",
            ],
            "diagnostic_training": True,
            "claim": "motor-signal diagnostic; not game progress or task success",
        }

    def _append_task_log(self, record: dict[str, Any]) -> None:
        if self.task_log_path is None:
            return
        self.task_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
        descriptor = os.open(self.task_log_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            if os.write(descriptor, payload) != len(payload):
                raise OSError("partial task JSONL append")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def confirmed_combat_components(
    info: dict[str, Any],
    *,
    enemy_kill_reward: float,
    damage_penalty: float,
    shade_recovery_reward: float = 0.0,
    geo_gain_reward: float = 0.0,
    geo_spend_reward: float = 0.0,
    death_penalty: float = 0.0,
    run_failed_penalty: float = 0.0,
) -> dict[str, float]:
    """Reward explicit per-frame deltas without exposing them to observations."""

    def nonnegative_delta(name: str) -> int:
        value = info.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InvalidTrainingTransition(f"invalid confirmed combat delta: {name}")
        return int(value)

    raw_kills = nonnegative_delta("enemy_kills_since_last_frame")
    raw_hp_lost = nonnegative_delta("hp_lost_since_last_frame")
    raw_geo_gained = nonnegative_delta("geo_gained_since_last_frame")
    raw_geo_spent = nonnegative_delta("geo_spent_since_last_frame")
    supervisor = info.get("supervisor", {})
    if not isinstance(supervisor, dict):
        raise InvalidTrainingTransition("invalid supervisor lifecycle metadata")
    player_dead = supervisor.get("player_dead", False)
    run_failed = info.get("run_failed", supervisor.get("run_failed", False))
    shade_recovered = info.get(
        "shade_recovered", supervisor.get("shade_recovered", False)
    )
    if (
        not isinstance(player_dead, bool)
        or not isinstance(run_failed, bool)
        or not isinstance(shade_recovered, bool)
    ):
        raise InvalidTrainingTransition("invalid confirmed death metadata")
    return {
        "confirmed_enemy_kills": enemy_kill_reward * raw_kills,
        "confirmed_hp_lost": 0.0 if player_dead else damage_penalty * raw_hp_lost,
        "confirmed_geo_gained": geo_gain_reward * raw_geo_gained,
        # A death moves Geo out of the live balance but is not a purchase.  It
        # must never be rewarded as spending.
        "confirmed_geo_spent": 0.0 if player_dead else geo_spend_reward * raw_geo_spent,
        "player_death": death_penalty if player_dead else 0.0,
        "gameplay_run_failed": run_failed_penalty if run_failed else 0.0,
        "shade_recovered": shade_recovery_reward if shade_recovered else 0.0,
    }


def _copy_lifecycle_info(source: dict[str, Any], target: dict[str, Any]) -> None:
    supervisor = source.get("supervisor")
    if isinstance(supervisor, dict):
        target["supervisor"] = dict(supervisor)
    if "run_failed" in source:
        target["run_failed"] = source["run_failed"]
    if "shade_recovered" in source:
        target["shade_recovered"] = source["shade_recovered"]


class ChannelsFirstObservation(gym.ObservationWrapper):
    """Convert K,H,W,C pixels into the 3-D C*K,H,W form expected by SB3 CNNs."""

    def __init__(
        self,
        environment: gym.Env[dict[str, NDArray[np.generic]], int],
        *,
        target_height: int = 90,
        target_width: int = 160,
    ) -> None:
        super().__init__(environment)
        if target_height < 36 or target_width < 36:
            raise ValueError("policy image dimensions must be at least 36 pixels")
        source_space = cast(gym.spaces.Dict, environment.observation_space)
        pixels = cast(gym.spaces.Box, source_space.spaces["pixels"])
        stack, source_height, source_width, channels = pixels.shape
        self._row_indices = np.linspace(0, source_height - 1, target_height).astype(np.intp)
        self._column_indices = np.linspace(0, source_width - 1, target_width).astype(np.intp)
        self.observation_space = gym.spaces.Dict(
            {
                "pixels": gym.spaces.Box(
                    0,
                    255,
                    shape=(stack * channels, target_height, target_width),
                    dtype=np.uint8,
                ),
                "action_history": source_space.spaces["action_history"],
            }
        )

    def observation(
        self, observation: dict[str, NDArray[np.generic]]
    ) -> dict[str, NDArray[np.generic]]:
        pixels = observation["pixels"]
        resized = pixels[:, self._row_indices][:, :, self._column_indices]
        return {
            "pixels": np.transpose(resized, (0, 3, 1, 2)).reshape(
                resized.shape[0] * resized.shape[3],
                resized.shape[1],
                resized.shape[2],
            ),
            "action_history": observation["action_history"].copy(),
        }


def visual_motion_fraction(
    before: NDArray[np.uint8],
    after: NDArray[np.uint8],
    config: VisualMotionTaskConfig,
) -> float:
    if before.shape != after.shape or before.ndim != 3 or before.shape[2] != 3:
        raise ValueError("motion frames must be equally shaped RGB images")
    height, width, _ = before.shape
    top = int(height * config.crop_top_fraction)
    bottom = max(top + 1, int(height * config.crop_bottom_fraction))
    left = int(width * config.crop_left_fraction)
    right = max(left + 1, int(width * config.crop_right_fraction))
    difference = np.abs(
        after[top:bottom, left:right].astype(np.int16)
        - before[top:bottom, left:right].astype(np.int16)
    )
    return float(np.mean(np.max(difference, axis=2) >= config.pixel_delta_threshold))


def resolve_calibrated_config(
    config: VisualMotionTaskConfig,
    *,
    noop_motion: list[float],
    active_motion: list[float],
) -> VisualMotionTaskConfig:
    """Gate training on a measurable action signal above NOOP animation noise."""

    if len(noop_motion) < 5 or len(active_motion) < 5:
        raise ValueError("calibration requires at least five NOOP and five active samples")
    noop_p95 = float(np.percentile(np.asarray(noop_motion), 95))
    active_median = float(np.median(np.asarray(active_motion)))
    threshold = max(config.motion_target_fraction, noop_p95 * 1.5)
    if active_median <= threshold:
        raise RuntimeError(
            "calibration failed: active motion is not separable from NOOP animation"
        )
    return replace(config, motion_target_fraction=threshold)
