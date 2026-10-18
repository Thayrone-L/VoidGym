from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from hk_agent.action_masking import action_mask_audit, build_action_mask, decide_masked_action
from hk_agent.gym_env import HollowKnightEnv
from hk_agent.scene_change_task import (
    SCENE_CHANGE_ACTION_NAMES,
    SCENE_CHANGE_ACTION_TEMPLATES,
    SceneChangeTaskConfig,
    descriptor_distance,
    detect_scene_change,
    scene_descriptor,
    visual_quality_metrics,
)
from hk_agent.temporal_actions import (
    TemporalActionHistory,
    action_contract,
    temporal_observation_space,
)
from hk_agent.training_task import (
    InvalidTrainingTransition,
    _copy_lifecycle_info,
    confirmed_combat_components,
)

EXPLORE_REWARD_VERSION = "explore_v1"
SUPPORTED_EXPLORE_REWARD_VERSIONS = frozenset({"explore_v1", "explore_v2"})


def descriptor_hash(descriptor: NDArray[np.generic]) -> str:
    quantized = np.clip(np.rint(np.asarray(descriptor, dtype=np.float32) * 7), 0, 7).astype(np.uint8)
    return hashlib.sha256(quantized.tobytes()).hexdigest()


def load_lifelong_visits(path: Path, limit: int) -> dict[str, int]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text("utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(payload.get("visits"), dict):
        raise ValueError("lifelong novelty file has an unsupported schema")
    result: dict[str, int] = {}
    for key, count in payload["visits"].items():
        if not isinstance(key, str) or len(key) != 64 or isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("lifelong novelty file contains an invalid record")
        if len(result) < limit:
            result[key] = count
    return result


def save_lifelong_visits(path: Path, visits: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"schema_version": 1, "visits": visits}, sort_keys=True, separators=(",", ":"), allow_nan=False)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True, slots=True)
class ExplorationTaskConfig:
    reward_version: str = EXPLORE_REWARD_VERSION
    horizon_steps: int = 3000
    novelty_reward: float = 0.05
    novelty_threshold: float = 0.02
    novelty_budget: float = 20.0
    archive_max: int = 4096
    scene_change_reward: float = 5.0
    enemy_kill_reward: float = 1.0
    enemy_kill_reward_budget: float = 5.0
    shade_recovery_reward: float = 1.0
    damage_penalty: float = -0.5
    death_penalty: float = -5.0
    run_failed_penalty: float = -5.0
    geo_gain_reward: float = 0.01
    geo_spend_reward: float = 0.02
    geo_reward_budget: float = 1.0
    time_cost: float = -0.001
    lease_base_ms: int = 80
    ent_coef: float = 0.01
    gamma: float = 0.995
    gae_lambda: float = 0.95
    n_epochs: int = 4
    target_kl: float = 0.03
    learning_rate: float = 2.5e-4
    clip_range: float = 0.2
    lifelong_archive_max: int = 4096
    novelty_persistence_steps: int = 2
    shade_hit_reward: float = 0.3
    hp_restored_reward: float = 0.4

    def validate(self) -> None:
        if not 1 <= self.horizon_steps <= 100_000 or self.archive_max < 1 or self.lifelong_archive_max < 1:
            raise ValueError("exploration horizon and archive must be positive and bounded")
        if self.reward_version not in SUPPORTED_EXPLORE_REWARD_VERSIONS:
            raise ValueError("unsupported exploration reward version")
        if self.novelty_persistence_steps != 2:
            raise ValueError("novelty persistence must be two consecutive steps")
        if self.novelty_reward < 0 or self.novelty_threshold <= 0 or self.novelty_budget < 0:
            raise ValueError("novelty parameters must be non-negative and threshold positive")
        if self.scene_change_reward < 0 or self.enemy_kill_reward < 0 or self.enemy_kill_reward_budget < 0:
            raise ValueError("positive rewards and budgets must be non-negative")
        if self.shade_recovery_reward < 0 or self.shade_recovery_reward >= abs(self.death_penalty):
            raise ValueError("death penalty magnitude must exceed shade recovery reward")
        if self.reward_version == "explore_v2":
            if self.geo_gain_reward != 0 or self.geo_spend_reward != 0:
                raise ValueError("explore_v2 does not reward Geo")
            if self.shade_recovery_reward != 4.0 or self.shade_hit_reward != 0.3:
                raise ValueError("explore_v2 Shade rewards are fixed by the reward contract")
            if self.shade_recovery_reward >= abs(self.death_penalty):
                raise ValueError("Shade recovery reward must remain below death penalty")
            if not 0 <= self.hp_restored_reward < abs(self.damage_penalty):
                raise ValueError("healing reward must be smaller than damage penalty magnitude")
        if self.damage_penalty > 0 or self.death_penalty >= 0 or self.run_failed_penalty >= 0 or self.time_cost > 0:
            raise ValueError("penalty signs are invalid")
        if self.geo_gain_reward < 0 or self.geo_spend_reward < 0 or self.geo_reward_budget < 0:
            raise ValueError("geo reward parameters must be non-negative")
        if self.lease_base_ms < 1:
            raise ValueError("lease_base_ms must be positive")
        if not (0 < self.ent_coef <= 10 and 0 < self.gamma <= 1 and 0 < self.gae_lambda <= 1
                and self.n_epochs >= 1 and 0 < self.target_kl <= 10
                and 0 < self.learning_rate <= 1 and 0 < self.clip_range <= 1):
            raise ValueError("PPO hyperparameters are outside their safe ranges")

    def maximum_positive_episode_reward(self) -> float:
        # At least three stable frames are required per distinct scene event.
        scene_events = self.horizon_steps // 3
        return (
            self.novelty_budget + self.scene_change_reward * scene_events
            + self.enemy_kill_reward_budget + self.geo_reward_budget
            + self.shade_recovery_reward
        )


class ExplorationTask(gym.Env[dict[str, NDArray[np.generic]], int]):
    """Pixel-only exploration rewards; privileged bridge signals affect rewards only."""

    def __init__(self, environment: HollowKnightEnv, *, config: ExplorationTaskConfig | None = None,
                 task_log_path: Path | None = None) -> None:
        super().__init__()
        self.environment = environment
        self.config = config or ExplorationTaskConfig()
        self.config.validate()
        self.task_log_path = task_log_path
        self.action_space = gym.spaces.Discrete(len(SCENE_CHANGE_ACTION_TEMPLATES))  # type: ignore[assignment]
        source = cast(gym.spaces.Dict, environment.observation_space)
        self.observation_space = temporal_observation_space(source, number_of_actions=len(SCENE_CHANGE_ACTION_TEMPLATES))  # type: ignore[arg-type, assignment]
        history = source["action_history"]
        if history.shape is None:
            raise TypeError("action_history must declare a shape")
        self._history = TemporalActionHistory(history.shape[0])
        self._archive: deque[NDArray[np.float32]] = deque(maxlen=self.config.archive_max)
        self._paid_scenes: list[NDArray[np.float32]] = []
        self._steps = 0
        self._novelty_paid = 0.0
        self._kills_paid = 0.0
        self._geo_paid = 0.0
        self._shade_paid = False
        self._anchor: NDArray[np.float32] | None = None
        self._scene_candidate: NDArray[np.float32] | None = None
        self._scene_stable = 0
        self._previous_descriptor: NDArray[np.float32] | None = None
        self._transition_seen = False
        self._dark_run = 0
        self._done = True
        self.mask_unavailable_count = 0
        self._lifelong_path: Path | None = None
        self._lifelong_visits: dict[str, int] = {}
        self._novelty_candidate: str | None = None
        self._novelty_candidate_descriptor: NDArray[np.float32] | None = None
        self._novelty_candidate_steps = 0
        self._shade_distance: tuple[str, float] | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, NDArray[np.generic]], dict[str, Any]]:
        super().reset(seed=seed)
        observation, _ = self.environment.reset(seed=seed, options=options)
        descriptor = self._descriptor(observation["pixels"][-1])
        if self.config.reward_version == "explore_v2":
            run_root = self.task_log_path.parent.parent if self.task_log_path is not None else Path.cwd()
            self._lifelong_path = run_root / "lifelong-novelty.json" if self.task_log_path is not None else None
            self._lifelong_visits = load_lifelong_visits(self._lifelong_path, self.config.lifelong_archive_max) if self._lifelong_path is not None else {}
        self._archive.clear()
        self._archive.append(descriptor)
        self._paid_scenes = [descriptor.copy()]
        self._anchor = descriptor.copy()
        self._previous_descriptor = descriptor.copy()
        self._transition_seen = False
        self._dark_run = self._scene_stable = 0
        self._scene_candidate = None
        self._novelty_candidate = None
        self._novelty_candidate_descriptor = None
        self._novelty_candidate_steps = 0
        self._shade_distance = None
        self._scene_stable = 0
        self._steps = 0
        self._novelty_paid = self._kills_paid = self._geo_paid = 0.0
        self._shade_paid = False
        self._done = False
        self.mask_unavailable_count = 0
        self._history.reset()
        return self._history.apply(observation), {}

    def step(self, action: int) -> tuple[dict[str, NDArray[np.generic]], float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("reset() is required before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid task action: {action!r}")
        decision = decide_masked_action(int(action), SCENE_CHANGE_ACTION_NAMES,
                                        getattr(self.environment, "latest_action_availability", None))
        if decision.availability is None:
            self.mask_unavailable_count += 1
        action = decision.executed_index
        audit = action_mask_audit(decision, self.mask_unavailable_count)
        template = SCENE_CHANGE_ACTION_TEMPLATES[action]
        lease = template.resolve_lease_ms(self.environment.lease_ms)
        if isinstance(self.environment, HollowKnightEnv):
            observation, _, terminated, truncated, base = self.environment.step(
                template.raw_action_index, lease_ms=lease, action_audit=audit)
        else:
            observation, _, terminated, truncated, base = self.environment.step(template.raw_action_index, lease_ms=lease)
        if base.get("valid") is not True:
            self._done = True
            self.environment.backend.release_all()
            raise InvalidTrainingTransition(f"operationally invalid transition: {base.get('termination_reason')}")
        self._history.append(action)
        observation = self._history.apply(observation)
        frame = observation["pixels"][-1]
        descriptor = self._descriptor(frame)
        luma, dark_fraction, white_fraction, entropy = visual_quality_metrics(frame, _descriptor_config())
        informative = dark_fraction < 0.60 and white_fraction < 0.75 and entropy >= 1.0
        self._steps += 1
        components: dict[str, float] = {
            "novelty": 0.0, "scene_change": 0.0, "confirmed_enemy_kills": 0.0,
            "confirmed_geo_gained": 0.0, "confirmed_geo_spent": 0.0,
            "shade_recovered": 0.0, "shade_hits": 0.0, "shade_approach": 0.0,
            "hp_restored": 0.0, "damage_taken": 0.0,
            "death": 0.0, "run_failed": 0.0,
            "time_cost": self.config.time_cost * (lease / self.config.lease_base_ms),
        }
        base_done = terminated or truncated
        novelty = min((descriptor_distance(descriptor, prior) for prior in self._archive), default=1.0)
        eligible = informative and novelty >= self.config.novelty_threshold and not base_done
        if self.config.reward_version == "explore_v2":
            tokens = set(template.name.split("+"))
            stationary_attack = "ATTACK" in tokens and not tokens.intersection({"LEFT", "RIGHT", "JUMP"})
            if stationary_attack:
                eligible = False
        novelty_hash = descriptor_hash(descriptor) if self.config.reward_version == "explore_v2" else None
        if self.config.reward_version == "explore_v2":
            stable_candidate = (
                self._novelty_candidate_descriptor is not None
                and descriptor_distance(descriptor, self._novelty_candidate_descriptor)
                <= self.config.novelty_threshold
            )
            if eligible and stable_candidate:
                self._novelty_candidate_steps += 1
                self._novelty_candidate_descriptor = descriptor.copy()
            elif eligible:
                self._novelty_candidate = novelty_hash
                self._novelty_candidate_descriptor = descriptor.copy()
                self._novelty_candidate_steps = 1
            else:
                self._novelty_candidate = None
                self._novelty_candidate_descriptor = None
                self._novelty_candidate_steps = 0
            eligible = eligible and self._novelty_candidate_steps >= 2
        if eligible and self._novelty_paid < self.config.novelty_budget:
            visits = self._lifelong_visits.get(novelty_hash, 0) if novelty_hash else 0
            lifetime_factor = 1.0 / float(np.sqrt(1 + visits)) if novelty_hash else 1.0
            components["novelty"] = min(self.config.novelty_reward * lifetime_factor, self.config.novelty_budget - self._novelty_paid)
            self._novelty_paid += components["novelty"]
            self._archive.append(descriptor.copy())
            if novelty_hash is not None:
                self._lifelong_visits[novelty_hash] = visits + 1
                if len(self._lifelong_visits) > self.config.lifelong_archive_max:
                    del self._lifelong_visits[next(iter(self._lifelong_visits))]
                if self._lifelong_path is not None:
                    save_lifelong_visits(self._lifelong_path, self._lifelong_visits)
        detection = detect_scene_change(
            descriptor=descriptor, frame=frame, config=_descriptor_config(),
            anchor=self._anchor, previous_descriptor=self._previous_descriptor,
            transition_seen=self._transition_seen, dark_run=self._dark_run,
            stable_run=self._scene_stable, base_done=base_done,
        )
        self._transition_seen = bool(detection["transition_seen"])
        self._dark_run = int(detection["dark_run"])
        self._scene_stable = int(detection["stable_run"])
        changed = bool(detection["scene_changed"])
        self._previous_descriptor = descriptor.copy()
        if changed and not base_done and not any(descriptor_distance(descriptor, prior) < 0.025 for prior in self._paid_scenes):
            components["scene_change"] = self.config.scene_change_reward
            self._paid_scenes.append(descriptor.copy())
            self._anchor = descriptor.copy()
            self._transition_seen = False
            self._dark_run = self._scene_stable = 0
        elif changed:
            self._anchor = descriptor.copy()
            self._transition_seen = False
            self._dark_run = self._scene_stable = 0
        combat = confirmed_combat_components(
            base, enemy_kill_reward=self.config.enemy_kill_reward,
            shade_recovery_reward=(4.0 if self.config.reward_version == "explore_v2" else self.config.shade_recovery_reward),
            damage_penalty=self.config.damage_penalty, geo_gain_reward=self.config.geo_gain_reward,
            geo_spend_reward=self.config.geo_spend_reward, death_penalty=self.config.death_penalty,
            run_failed_penalty=self.config.run_failed_penalty)
        components["confirmed_enemy_kills"] = min(combat["confirmed_enemy_kills"], max(0.0, self.config.enemy_kill_reward_budget-self._kills_paid))
        self._kills_paid += components["confirmed_enemy_kills"]
        components["shade_recovered"] = 0.0 if self._shade_paid else combat["shade_recovered"]
        self._shade_paid |= components["shade_recovered"] > 0
        if self.config.reward_version == "explore_v2":
            hits = int(base.get("shade_hits_since_last_frame", 0))
            if hits < 0:
                raise InvalidTrainingTransition("invalid Shade hit delta")
            components["shade_hits"] = min(hits, 4) * self.config.shade_hit_reward
            restored = int(base.get("hp_restored_since_last_frame", 0))
            if restored < 0:
                raise InvalidTrainingTransition("invalid restored health delta")
            components["hp_restored"] = restored * self.config.hp_restored_reward
            distance = base.get("shade_distance")
            scene = base.get("scene_name")
            if isinstance(distance, (int, float)) and not isinstance(distance, bool) and isinstance(scene, str) and base.get("shade_pending") is True:
                current_distance = float(distance)
                if np.isfinite(current_distance) and self._shade_distance is not None and self._shade_distance[0] == scene:
                    components["shade_approach"] = float(np.clip(0.05 * (self._shade_distance[1] - current_distance), -0.05, 0.05))
                self._shade_distance = (scene, current_distance)
            else:
                self._shade_distance = None
            if base.get("shade_pending") is True:
                components["time_cost"] -= 0.001 * (lease / 1000.0)
        geo = combat["confirmed_geo_gained"] + combat["confirmed_geo_spent"]
        geo_award = min(geo, max(0.0, self.config.geo_reward_budget-self._geo_paid))
        if geo > 0:
            scale = geo_award / geo
            components["confirmed_geo_gained"] = combat["confirmed_geo_gained"] * scale
            components["confirmed_geo_spent"] = combat["confirmed_geo_spent"] * scale
            self._geo_paid += geo_award
        components["damage_taken"] = combat["confirmed_hp_lost"]
        components["death"] = combat["player_death"]
        components["run_failed"] = combat["gameplay_run_failed"]
        # Death/run-failure and any base terminal/truncation suppress all positive rewards on this transition.
        terminal_penalty = components["death"] < 0 or components["run_failed"] < 0
        if base_done or terminal_penalty:
            for name in ("novelty", "scene_change", "confirmed_enemy_kills", "confirmed_geo_gained", "confirmed_geo_spent", "shade_recovered", "shade_hits", "shade_approach", "hp_restored"):
                components[name] = 0.0
        reward = float(sum(components.values()))
        horizon = self._steps >= self.config.horizon_steps
        task_terminated = terminated or terminal_penalty
        task_truncated = truncated or (horizon and not task_terminated)
        termination_reason = base.get("termination_reason")
        if termination_reason is None and components["death"] < 0:
            termination_reason = "player_dead"
        elif termination_reason is None and components["run_failed"] < 0:
            termination_reason = "gameplay_run_failed"
        elif termination_reason is None and horizon:
            termination_reason = "time_limit"
        self._done = task_terminated or task_truncated
        if self._done:
            self.environment.backend.release_all()
        info: dict[str, Any] = {
            "valid": True, "termination_reason": termination_reason,
            "reward_version": self.config.reward_version, "reward_components": components,
            "scene_change_event": changed, "novelty_distance": novelty,
            "novelty_paid": self._novelty_paid, "archive_size": len(self._archive),
            "descriptor": descriptor.tolist(), "visual_quality": [luma, dark_fraction, white_fraction, entropy], **audit,
            "novelty_eligible": eligible,
        }
        _copy_lifecycle_info(base, info)
        self._append_log({"schema_version":1,"reward_version":self.config.reward_version,"step":self._steps,
                          "task_action":template.name,"raw_action_index":template.raw_action_index,"lease_ms":lease,
                          "reward":{"total":reward,"components":components},"novelty_distance":novelty,
                          "scene_change_event":changed,"terminated":task_terminated,"truncated":task_truncated,
                          "termination_reason":info["termination_reason"],"valid":True,**audit})
        return observation, reward, task_terminated, task_truncated, info

    def action_masks(self) -> NDArray[np.bool_]:
        return build_action_mask(SCENE_CHANGE_ACTION_NAMES, getattr(self.environment,"latest_action_availability",None))

    def close(self) -> None:
        self.environment.close()

    def task_contract(self) -> dict[str, Any]:
        return {"task_contract_schema_version": 2 if self.config.reward_version == "explore_v2" else 1,
                "reward_version": self.config.reward_version,
                "legal_action_mask":{"version":"action_availability_v1","source":"frame_action_availability"},
                "action_names":list(SCENE_CHANGE_ACTION_NAMES),
                "actions":action_contract(SCENE_CHANGE_ACTION_TEMPLATES,base_lease_ms=self.environment.lease_ms),
                "config":{key:value for key,value in asdict(self.config).items() if key != "reward_version"},"information_boundary":["pixels","applied_action_history"],
                "reward_only_signals":["enemy_kills_since_last_frame","hp_lost_since_last_frame","geo_gained_since_last_frame","geo_spent_since_last_frame","shade_recovered","death","run_failed"] + (["hp_restored_since_last_frame","shade_hits_since_last_frame","shade_distance","scene_name"] if self.config.reward_version == "explore_v2" else []),
                "detector_inputs":["cropped_rgb_pixels","temporal_pixel_descriptors"],"privileged_state":[],
                "diagnostic_training":False,"claim":"pixel-only exploration with bounded reward-only bridge signals",
                "ppo_hyperparameters": {"ent_coef":self.config.ent_coef,"gamma":self.config.gamma,
                    "gae_lambda":self.config.gae_lambda,"n_epochs":self.config.n_epochs,
                    "target_kl":self.config.target_kl,"learning_rate":self.config.learning_rate,
                    "clip_range":self.config.clip_range}}

    def _descriptor(self, frame: NDArray[np.generic]) -> NDArray[np.float32]:
        if self.config.reward_version == "explore_v1":
            return scene_descriptor(np.asarray(frame, dtype=np.uint8), _descriptor_config())
        image = np.asarray(frame, dtype=np.uint8).copy()
        height, width = image.shape[:2]
        image[: int(height * 0.30), : int(width * 0.40)] = 127
        config = _descriptor_config()
        rows, columns = config.descriptor_rows, config.descriptor_columns
        y_edges = np.linspace(0, height, rows + 1, dtype=np.intp)
        x_edges = np.linspace(0, width, columns + 1, dtype=np.intp)
        grid = np.empty((rows * columns, 3), dtype=np.float32)
        index = 0
        for row in range(rows):
            for column in range(columns):
                cell = image[y_edges[row]:y_edges[row + 1], x_edges[column]:x_edges[column + 1]]
                grid[index] = cell.mean(axis=(0, 1)) / 255.0
                index += 1
        return grid

    def _append_log(self, record: dict[str, Any]) -> None:
        if self.task_log_path is None:
            return
        self.task_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload=(json.dumps(record,separators=(",",":"),allow_nan=False)+"\n").encode()
        fd=os.open(self.task_log_path,os.O_APPEND|os.O_CREAT|os.O_WRONLY,0o600)
        try:
            if os.write(fd,payload)!=len(payload): raise OSError("partial task JSONL append")
            os.fsync(fd)
        finally: os.close(fd)


def _descriptor_config() -> SceneChangeTaskConfig:
    return SceneChangeTaskConfig()


def calibrate_novelty(noop_descriptors: list[NDArray[np.float32]], active_descriptors: list[NDArray[np.float32]], *, default_threshold: float = 0.02) -> float:
    if len(noop_descriptors) < 10 or len(active_descriptors) < 10:
        raise ValueError("exploration calibration requires at least ten NOOP and ten active samples")
    distances=[]
    for index, descriptor in enumerate(noop_descriptors):
        archive=noop_descriptors[:index] or [noop_descriptors[0]]
        distances.append(min(descriptor_distance(descriptor,item) for item in archive))
    threshold=max(default_threshold,float(np.percentile(distances,99))*2)
    active=[]
    archive=list(noop_descriptors)
    for descriptor in active_descriptors:
        active.append(min(descriptor_distance(descriptor,item) for item in archive))
    if float(np.median(active)) <= threshold:
        raise ValueError("active exploration calibration did not exceed the NOOP threshold")
    return threshold
