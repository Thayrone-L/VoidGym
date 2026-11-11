from __future__ import annotations

import hashlib
import json
import os
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from hk_agent.action_masking import action_mask_audit, build_action_mask, decide_masked_action
from hk_agent.gym_env import HollowKnightEnv
from hk_agent.route_memory import TopologicalRouteMemory
from hk_agent.temporal_actions import (
    PLAYER_INTERACTION_ACTIONS,
    TEMPORAL_COMBAT_ACTIONS,
    TemporalActionHistory,
    action_contract,
    temporal_observation_space,
)
from hk_agent.training_task import (
    InvalidTrainingTransition,
    _copy_lifecycle_info,
    confirmed_combat_components,
)

SCENE_CHANGE_REWARD_VERSION = "reach_scene_change_v1"


SCENE_CHANGE_ACTION_TEMPLATES = TEMPORAL_COMBAT_ACTIONS
SCENE_CHANGE_ACTIONS: tuple[int, ...] = tuple(
    action.raw_action_index for action in SCENE_CHANGE_ACTION_TEMPLATES
)
SCENE_CHANGE_ACTION_NAMES: tuple[str, ...] = tuple(
    action.name for action in SCENE_CHANGE_ACTION_TEMPLATES
)
SCENE_CHANGE_INTERACTION_ACTION_TEMPLATES = TEMPORAL_COMBAT_ACTIONS + PLAYER_INTERACTION_ACTIONS
SCENE_CHANGE_INTERACTION_ACTION_NAMES: tuple[str, ...] = tuple(
    action.name for action in SCENE_CHANGE_INTERACTION_ACTION_TEMPLATES
)


def detect_scene_change(
    *,
    descriptor: NDArray[np.float32],
    frame: NDArray[np.uint8],
    config: SceneChangeTaskConfig,
    anchor: NDArray[np.float32] | None,
    previous_descriptor: NDArray[np.float32] | None,
    transition_seen: bool,
    dark_run: int,
    stable_run: int,
    base_done: bool,
) -> dict[str, Any]:
    """Shared fade-confirmed scene detector used by both training tasks."""
    luma_mean, dark_fraction, white_fraction, entropy_bits = visual_quality_metrics(frame, config)
    anchor_luma = float(np.mean(anchor)) if anchor is not None else 1.0
    dark_limit = min(config.dark_luma_absolute, anchor_luma * config.dark_luma_ratio)
    is_dark = anchor is not None and luma_mean <= dark_limit and dark_fraction >= config.dark_pixel_fraction
    is_white = white_fraction >= config.white_pixel_fraction
    informative = not is_dark and not is_white and entropy_bits >= config.minimum_entropy_bits
    if not transition_seen:
        dark_run = dark_run + 1 if is_dark else 0
        transition_seen = dark_run >= config.transition_dark_frames
        if transition_seen:
            stable_run = 0
    anchor_distance = descriptor_distance(anchor, descriptor) if anchor is not None else 0.0
    frame_distance = descriptor_distance(previous_descriptor, descriptor) if previous_descriptor is not None else 1.0
    candidate = transition_seen and informative and anchor_distance >= config.new_scene_distance and frame_distance <= config.stable_frame_distance
    stable_run = stable_run + 1 if candidate else 0
    return {
        "transition_seen": transition_seen, "dark_run": dark_run, "stable_run": stable_run,
        "anchor_distance": anchor_distance, "frame_distance": frame_distance,
        "informative": informative, "candidate": candidate,
        "scene_changed": stable_run >= config.stable_frames and not base_done,
        "luma_mean": luma_mean, "dark_fraction": dark_fraction,
        "white_fraction": white_fraction, "entropy_bits": entropy_bits,
    }


def route_detector_signature(config: SceneChangeTaskConfig) -> str:
    fields = {
        name: getattr(config, name)
        for name in (
            "descriptor_rows",
            "descriptor_columns",
            "crop_top_fraction",
            "crop_bottom_fraction",
            "crop_left_fraction",
            "crop_right_fraction",
            "baseline_stable_distance",
            "transition_dark_frames",
            "new_scene_distance",
            "stable_frame_distance",
            "stable_frames",
        )
    }
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class SceneChangeTaskConfig:
    """Pixel-only contract for a conservative, fade-confirmed room transition."""

    horizon_steps: int | None = 512
    warmup_frames: int = 4
    descriptor_rows: int = 9
    descriptor_columns: int = 16
    crop_top_fraction: float = 0.08
    crop_bottom_fraction: float = 0.92
    crop_left_fraction: float = 0.06
    crop_right_fraction: float = 0.94
    dark_luma_absolute: float = 0.08
    dark_luma_ratio: float = 0.45
    dark_pixel_fraction: float = 0.60
    white_luma_absolute: float = 0.95
    white_pixel_fraction: float = 0.75
    minimum_entropy_bits: float = 1.0
    baseline_stable_distance: float = 0.03
    transition_dark_frames: int = 2
    new_scene_distance: float = 0.12
    stable_frame_distance: float = 0.025
    stable_frames: int = 3
    time_cost: float = -0.002
    novelty_shaping_budget: float = 0.0
    success_reward: float = 10.0
    enemy_kill_reward: float = 2.0
    enemy_kill_reward_budget: float = 6.0
    shade_recovery_reward: float = 2.0
    damage_penalty: float = -0.25
    geo_gain_reward: float = 0.01
    geo_spend_reward: float = 0.02
    death_penalty: float = -2.0
    run_failed_penalty: float = -4.0
    terminate_on_scene_change: bool = True

    def validate(self) -> None:
        if self.horizon_steps is not None and not 1 <= self.horizon_steps <= 100_000:
            raise ValueError("horizon_steps is outside the safe range")
        if not 1 <= self.warmup_frames or (
            self.horizon_steps is not None and self.warmup_frames >= self.horizon_steps
        ):
            raise ValueError("warmup_frames must be below horizon_steps")
        if self.descriptor_rows < 2 or self.descriptor_columns < 2:
            raise ValueError("descriptor grid must be at least 2x2")
        if not (
            0 <= self.crop_top_fraction < self.crop_bottom_fraction <= 1
            and 0 <= self.crop_left_fraction < self.crop_right_fraction <= 1
        ):
            raise ValueError("crop fractions must form a non-empty rectangle")
        for name, value in (
            ("dark_luma_absolute", self.dark_luma_absolute),
            ("dark_luma_ratio", self.dark_luma_ratio),
            ("dark_pixel_fraction", self.dark_pixel_fraction),
            ("white_luma_absolute", self.white_luma_absolute),
            ("white_pixel_fraction", self.white_pixel_fraction),
            ("new_scene_distance", self.new_scene_distance),
            ("stable_frame_distance", self.stable_frame_distance),
            ("baseline_stable_distance", self.baseline_stable_distance),
        ):
            if not 0 < value < 1:
                raise ValueError(f"{name} must be in (0, 1)")
        if not 1 <= self.transition_dark_frames <= 16:
            raise ValueError("transition_dark_frames must be in [1, 16]")
        if not 2 <= self.stable_frames <= 16:
            raise ValueError("stable_frames must be in [2, 16]")
        if not 0 <= self.minimum_entropy_bits <= 4:
            raise ValueError("minimum_entropy_bits must be in [0, 4]")
        if self.time_cost > 0 or not 0 <= self.novelty_shaping_budget <= 1:
            raise ValueError("reward signs violate the curriculum contract")
        if self.success_reward < 10 * self.novelty_shaping_budget:
            raise ValueError("terminal reward must dominate bounded novelty shaping")
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
        if self.enemy_kill_reward > self.success_reward / 4:
            raise ValueError("scene progress reward must dominate combat shaping")


class ReachSceneChangeTask(gym.Env[dict[str, NDArray[np.generic]], int]):
    """Detect a stable scene change from pixels without privileged game state.

    A success needs three independent visual facts: a stable initial reference,
    consecutive near-black transition frames, and a stable post-transition view
    that differs materially from the reference. This intentionally favours false
    negatives over rewarding ordinary animation or camera scrolling.
    """

    def __init__(
        self,
        environment: HollowKnightEnv,
        *,
        config: SceneChangeTaskConfig | None = None,
        task_log_path: Path | None = None,
        route_memory: TopologicalRouteMemory | None = None,
    ) -> None:
        super().__init__()
        self.environment = environment
        self.config = config or SceneChangeTaskConfig()
        self.config.validate()
        self.task_log_path = task_log_path
        self.route_memory = route_memory
        self.action_space = gym.spaces.Discrete(len(SCENE_CHANGE_ACTION_TEMPLATES))  # type: ignore[assignment]
        source_space = cast(gym.spaces.Dict, environment.observation_space)
        self.observation_space = temporal_observation_space(  # type: ignore[assignment]
            environment.observation_space,
            number_of_actions=len(SCENE_CHANGE_ACTION_TEMPLATES),
        )
        history_space = source_space["action_history"]
        if history_space.shape is None:
            raise TypeError("action_history must declare a shape")
        self._action_history = TemporalActionHistory(history_space.shape[0])
        self.metadata = environment.metadata
        self._warmup: deque[NDArray[np.float32]] = deque(maxlen=self.config.warmup_frames)
        self._anchor: NDArray[np.float32] | None = None
        self._previous_descriptor: NDArray[np.float32] | None = None
        self._dark_run = 0
        self._stable_run = 0
        self._transition_seen = False
        self._best_novelty = 0.0
        self._shaping_paid = 0.0
        self._steps = 0
        self._combat_reward_paid = 0.0
        self._enemy_kill_reward_paid = 0.0
        self._shade_recovery_reward_paid = False
        self._route_action_trace: list[int] = []
        self._done = True
        self.mask_unavailable_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, NDArray[np.generic]], dict[str, Any]]:
        super().reset(seed=seed)
        observation, _ = self.environment.reset(seed=seed, options=options)
        descriptor = scene_descriptor(observation["pixels"][-1], self.config)
        self._warmup.clear()
        self._warmup.append(descriptor)
        self._anchor = None
        self._previous_descriptor = descriptor
        self._dark_run = 0
        self._stable_run = 0
        self._transition_seen = False
        self._best_novelty = 0.0
        self._shaping_paid = 0.0
        self._steps = 0
        self._combat_reward_paid = 0.0
        self._enemy_kill_reward_paid = 0.0
        self._shade_recovery_reward_paid = False
        self._route_action_trace.clear()
        self._done = False
        self.mask_unavailable_count = 0
        self._action_history.reset()
        if self.route_memory is not None:
            self.route_memory.reset_episode()
        return self._action_history.apply(observation), {}

    def step(
        self, action: int
    ) -> tuple[dict[str, NDArray[np.generic]], float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("reset() is required before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid task action: {action!r}")
        decision = decide_masked_action(
            int(action), SCENE_CHANGE_ACTION_NAMES,
            getattr(self.environment, "latest_action_availability", None),
        )
        if decision.availability is None:
            self.mask_unavailable_count += 1
        action = decision.executed_index
        audit = action_mask_audit(decision, self.mask_unavailable_count)
        action_template = SCENE_CHANGE_ACTION_TEMPLATES[int(action)]
        action_lease_ms = action_template.resolve_lease_ms(self.environment.lease_ms)
        if isinstance(self.environment, HollowKnightEnv):
            observation, _, terminated, truncated, base_info = self.environment.step(
                action_template.raw_action_index,
                lease_ms=action_lease_ms,
                action_audit=audit,
            )
        else:
            observation, _, terminated, truncated, base_info = self.environment.step(
                action_template.raw_action_index, lease_ms=action_lease_ms
            )
        if base_info.get("valid") is not True:
            self._done = True
            reason = base_info.get("termination_reason")
            self.environment.backend.release_all()
            self._append_log(
                {
                    "schema_version": 1,
                    "reward_version": SCENE_CHANGE_REWARD_VERSION,
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
        self._route_action_trace.append(int(action))
        observation = self._action_history.apply(observation)

        current_frame = observation["pixels"][-1]
        descriptor = scene_descriptor(current_frame, self.config)
        luma_mean, dark_fraction, white_fraction, entropy_bits = visual_quality_metrics(
            current_frame, self.config
        )
        self._steps += 1
        route_event: dict[str, Any] | None = None
        if self._anchor is None:
            self._warmup.append(descriptor)
            if len(self._warmup) == self.config.warmup_frames:
                candidate_anchor = np.median(np.stack(tuple(self._warmup)), axis=0).astype(
                    np.float32
                )
                baseline_spread = max(
                    descriptor_distance(candidate_anchor, item) for item in self._warmup
                )
                if baseline_spread <= self.config.baseline_stable_distance:
                    self._anchor = candidate_anchor
                    if (
                        self.route_memory is not None
                        and self.route_memory.current_node is None
                    ):
                        route_event = self.route_memory.observe_start(candidate_anchor)
                        self._route_action_trace.clear()

        # A base-environment terminal/truncation (for example death or a
        # capture failure) always wins over a visual candidate observed in the
        # same step. It must never be relabelled or rewarded as curriculum
        # success.
        base_done = terminated or truncated
        detection = detect_scene_change(
            descriptor=descriptor, frame=current_frame, config=self.config,
            anchor=self._anchor, previous_descriptor=self._previous_descriptor,
            transition_seen=self._transition_seen, dark_run=self._dark_run,
            stable_run=self._stable_run, base_done=base_done,
        )
        self._transition_seen = bool(detection["transition_seen"])
        self._dark_run = int(detection["dark_run"])
        self._stable_run = int(detection["stable_run"])
        anchor_distance = float(detection["anchor_distance"])
        frame_distance = float(detection["frame_distance"])
        luma_mean = float(detection["luma_mean"])
        dark_fraction = float(detection["dark_fraction"])
        white_fraction = float(detection["white_fraction"])
        entropy_bits = float(detection["entropy_bits"])
        regime_changed = bool(detection["scene_changed"])
        if regime_changed and self.route_memory is not None:
            route_event = self.route_memory.observe_transition(
                descriptor, action_trace=tuple(self._route_action_trace)
            )
            self._route_action_trace.clear()
        novelty_increment = 0.0
        if detection["candidate"]:
            novelty = min(1.0, anchor_distance / self.config.new_scene_distance)
            self._best_novelty = max(self._best_novelty, novelty)
        # Optional shaping remains zero until the complete stability gate has
        # been confirmed. The default budget is zero until calibration.
        if regime_changed and self.config.novelty_shaping_budget > 0:
            available = self.config.novelty_shaping_budget - self._shaping_paid
            novelty_increment = min(
                available, self._best_novelty * self.config.novelty_shaping_budget
            )
            self._shaping_paid += novelty_increment
        self._previous_descriptor = descriptor

        horizon = (
            self.config.horizon_steps is not None
            and self._steps >= self.config.horizon_steps
        )
        task_terminated = terminated or (
            regime_changed and self.config.terminate_on_scene_change
        )
        task_truncated = truncated or (horizon and not task_terminated)
        combat_components = confirmed_combat_components(
            base_info,
            enemy_kill_reward=self.config.enemy_kill_reward,
            shade_recovery_reward=self.config.shade_recovery_reward,
            damage_penalty=self.config.damage_penalty,
            geo_gain_reward=self.config.geo_gain_reward,
            geo_spend_reward=self.config.geo_spend_reward,
            death_penalty=self.config.death_penalty,
            run_failed_penalty=self.config.run_failed_penalty,
        )
        combat_components["confirmed_enemy_kills"] = min(
            combat_components["confirmed_enemy_kills"],
            max(
                0.0,
                self.config.enemy_kill_reward_budget
                - self._enemy_kill_reward_paid,
            ),
        )
        self._enemy_kill_reward_paid += combat_components[
            "confirmed_enemy_kills"
        ]
        if self._shade_recovery_reward_paid:
            combat_components["shade_recovered"] = 0.0
        elif combat_components["shade_recovered"] > 0:
            self._shade_recovery_reward_paid = True
        combat_budget = self.config.success_reward / 4
        # All positive side-objectives share one bounded budget so repeatedly
        # farming Geo cannot dominate actual scene progress.
        for component_name in (
            "confirmed_enemy_kills",
            "confirmed_geo_gained",
            "confirmed_geo_spent",
        ):
            raw_reward = combat_components[component_name]
            combat_components[component_name] = min(
                raw_reward,
                max(0.0, combat_budget - self._combat_reward_paid),
            )
            self._combat_reward_paid += combat_components[component_name]
        components = {
            "time_cost": self.config.time_cost
            * (action_lease_ms / self.environment.lease_ms),
            "bounded_novelty": novelty_increment,
            "stable_visual_regime_change": (
                self.config.success_reward if regime_changed else 0.0
            ),
            **combat_components,
        }
        reward = float(sum(components.values()))
        reason = base_info.get("termination_reason")
        if regime_changed and self.config.terminate_on_scene_change:
            reason = "visual_regime_change_detected"
        elif horizon and reason is None:
            reason = "time_limit"
        self._done = task_terminated or task_truncated
        if self._done:
            self.environment.backend.release_all()
        info: dict[str, Any] = {
            "valid": True,
            "termination_reason": reason,
            "reward_version": SCENE_CHANGE_REWARD_VERSION,
            "reward_components": components,
            "scene_change_event": regime_changed,
            "visual_regime_change_detected": regime_changed,
            **audit,
            "detector": {
                "phase": "settle" if self._transition_seen else "seek_transition",
                "dark_run": self._dark_run,
                "stable_run": self._stable_run,
                "anchor_distance": anchor_distance,
                "frame_distance": frame_distance,
                "luma_mean": luma_mean,
                "dark_fraction": dark_fraction,
                "white_fraction": white_fraction,
                "entropy_bits": entropy_bits,
                "shaping_paid": self._shaping_paid,
            },
        }
        if self.route_memory is not None:
            info["route_memory"] = {
                **self.route_memory.summary(),
                "event": route_event,
            }
        _copy_lifecycle_info(base_info, info)
        log_record: dict[str, Any] = {
            "schema_version": 1,
            "reward_version": SCENE_CHANGE_REWARD_VERSION,
            "step": self._steps,
            "task_action": action_template.name,
            "raw_action_index": action_template.raw_action_index,
            "lease_ms": action_lease_ms,
            **audit,
            "reward": {"total": reward, "components": components},
            "scene_change_event": regime_changed,
            "visual_regime_change_detected": regime_changed,
            "detector": info["detector"],
            "terminated": task_terminated,
            "truncated": task_truncated,
            "termination_reason": reason,
            "valid": True,
        }
        if self.route_memory is not None:
            log_record["route_memory"] = info["route_memory"]
        self._append_log(log_record)
        if regime_changed and not self.config.terminate_on_scene_change:
            self._warmup.clear()
            self._warmup.append(descriptor)
            self._anchor = None
            self._dark_run = 0
            self._stable_run = 0
            self._transition_seen = False
            self._best_novelty = 0.0
            self._shaping_paid = 0.0
        return observation, reward, task_terminated, task_truncated, info

    def close(self) -> None:
        self.environment.close()

    def action_masks(self) -> NDArray[np.bool_]:
        return build_action_mask(
            SCENE_CHANGE_ACTION_NAMES,
            getattr(self.environment, "latest_action_availability", None),
        )

    def task_contract(self) -> dict[str, Any]:
        return {
            "reward_version": SCENE_CHANGE_REWARD_VERSION,
            "legal_action_mask": {"version": "action_availability_v1", "source": "frame_action_availability"},
            "action_names": list(SCENE_CHANGE_ACTION_NAMES),
            "raw_action_indices": list(SCENE_CHANGE_ACTIONS),
            "actions": action_contract(
                SCENE_CHANGE_ACTION_TEMPLATES,
                base_lease_ms=self.environment.lease_ms,
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
            "detector_inputs": ["cropped_rgb_pixels", "temporal_pixel_descriptors"],
            "privileged_state": [],
            "route_memory": {
                "enabled": self.route_memory is not None,
                "config": (
                    self.route_memory.configuration()
                    if self.route_memory is not None
                    else TopologicalRouteMemory.contract()
                ),
            },
            "diagnostic_training": True,
            "claim": (
                "pixel-only stable visual-regime-change diagnostic; room transition "
                "not yet validated against a holdout corpus"
            ),
        }

    def _append_log(self, record: dict[str, Any]) -> None:
        if self.task_log_path is None:
            return
        self.task_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
        descriptor = os.open(
            self.task_log_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600
        )
        try:
            if os.write(descriptor, payload) != len(payload):
                raise OSError("partial task JSONL append")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _cropped(frame: NDArray[np.uint8], config: SceneChangeTaskConfig) -> NDArray[np.uint8]:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("scene frames must be RGB images")
    height, width, _ = frame.shape
    top = int(height * config.crop_top_fraction)
    bottom = max(top + 1, int(height * config.crop_bottom_fraction))
    left = int(width * config.crop_left_fraction)
    right = max(left + 1, int(width * config.crop_right_fraction))
    return frame[top:bottom, left:right]


def scene_descriptor(
    frame: NDArray[np.uint8], config: SceneChangeTaskConfig
) -> NDArray[np.float32]:
    crop = _cropped(frame, config).astype(np.float32) / 255.0
    rows = np.linspace(0, crop.shape[0], config.descriptor_rows + 1).astype(np.intp)
    columns = np.linspace(0, crop.shape[1], config.descriptor_columns + 1).astype(np.intp)
    blocks = [
        np.mean(crop[rows[row] : rows[row + 1], columns[column] : columns[column + 1]], axis=(0, 1))
        for row in range(config.descriptor_rows)
        for column in range(config.descriptor_columns)
    ]
    return np.asarray(blocks, dtype=np.float32)


def descriptor_distance(
    before: NDArray[np.float32], after: NDArray[np.float32]
) -> float:
    if before.shape != after.shape:
        raise ValueError("descriptors must have matching shapes")
    return float(np.mean(np.abs(before - after)))


def visual_quality_metrics(
    frame: NDArray[np.uint8], config: SceneChangeTaskConfig
) -> tuple[float, float, float, float]:
    rgb = _cropped(frame, config).astype(np.float32) / 255.0
    luma = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    histogram, _ = np.histogram(luma, bins=16, range=(0.0, 1.0))
    probabilities = histogram[histogram > 0].astype(np.float64) / luma.size
    entropy_bits = float(-np.sum(probabilities * np.log2(probabilities)))
    return (
        float(np.mean(luma)),
        float(np.mean(luma <= config.dark_luma_absolute)),
        float(np.mean(luma >= config.white_luma_absolute)),
        entropy_bits,
    )
