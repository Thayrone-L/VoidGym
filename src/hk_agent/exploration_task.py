from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from hk_agent.action_masking import action_mask_audit, build_action_mask, decide_masked_action
from hk_agent.gym_env import HollowKnightEnv
from hk_agent.player_knowledge import ITEM_FLAGS, PlayerKnowledge
from hk_agent.run_journey import RunDiary, increment_campaign_count, rebuild_progress
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
SUPPORTED_EXPLORE_REWARD_VERSIONS = frozenset(
    {"explore_v1", "explore_v2", "explore_v3", "explore_v4"}
)


def descriptor_hash(descriptor: NDArray[np.generic]) -> str:
    quantized = np.clip(np.rint(np.asarray(descriptor, dtype=np.float32) * 7), 0, 7).astype(
        np.uint8
    )
    return hashlib.sha256(quantized.tobytes()).hexdigest()


def _ability_aware_novelty_hash(
    descriptor: NDArray[np.generic], knowledge: PlayerKnowledge
) -> str:
    acquired = bytes(int(knowledge.known_flags.get(flag, False)) for flag, _ in ITEM_FLAGS)
    return hashlib.sha256(bytes.fromhex(descriptor_hash(descriptor)) + acquired).hexdigest()


def load_lifelong_visits(path: Path, limit: int) -> dict[str, int]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text("utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("visits"), dict)
    ):
        raise ValueError("lifelong novelty file has an unsupported schema")
    result: dict[str, int] = {}
    for key, count in payload["visits"].items():
        if (
            not isinstance(key, str)
            or len(key) != 64
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
        ):
            raise ValueError("lifelong novelty file contains an invalid record")
        if len(result) < limit:
            result[key] = count
    return result


def save_lifelong_visits(path: Path, visits: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"schema_version": 1, "visits": visits},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
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


def load_campaign_counts(path: Path, limit: int = 4096) -> dict[str, int]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text("utf-8"))
    counts = payload.get("counts") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(counts, dict)
    ):
        raise ValueError(f"campaign memory has an unsupported schema: {path.name}")
    result: dict[str, int] = {}
    for key, count in counts.items():
        if (
            not isinstance(key, str)
            or not key
            or len(key) > 512
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
        ):
            raise ValueError(f"campaign memory contains an invalid record: {path.name}")
        if len(result) < limit:
            result[key] = count
    return result


def save_campaign_counts(path: Path, counts: dict[str, int], limit: int = 4096) -> None:
    if len(counts) > limit:
        for key in list(counts)[: len(counts) - limit]:
            del counts[key]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"schema_version": 1, "counts": counts},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
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


def load_scene_first_seen(path: Path, limit: int = 4096) -> dict[str, int]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text("utf-8"))
    records = payload.get("first_seen_unix_ns") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(records, dict)
    ):
        raise ValueError("scene first-seen file has an unsupported schema")
    result: dict[str, int] = {}
    for name, timestamp in records.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or timestamp <= 0
        ):
            raise ValueError("scene first-seen file contains an invalid record")
        if len(result) < limit:
            result[name] = timestamp
    return result


def save_scene_first_seen(path: Path, records: dict[str, int], limit: int = 4096) -> None:
    if len(records) > limit:
        for key in list(records)[: len(records) - limit]:
            del records[key]
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"schema_version": 1, "first_seen_unix_ns": records},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
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
    shade_hit_budget: float = 1.2
    shade_approach_budget: float = 1.0
    hp_restored_reward: float = 0.4
    hp_restored_budget: float = 4.0
    scene_reward_budget: float = 20.0
    time_cost_budget: float = 20.0
    stuck_seconds: float = 120.0
    evaluation_interval: int = 20

    def validate(self) -> None:
        if (
            not 1 <= self.horizon_steps <= 100_000
            or self.archive_max < 1
            or self.lifelong_archive_max < 1
        ):
            raise ValueError("exploration horizon and archive must be positive and bounded")
        if self.reward_version not in SUPPORTED_EXPLORE_REWARD_VERSIONS:
            raise ValueError("unsupported exploration reward version")
        if self.novelty_persistence_steps != 2:
            raise ValueError("novelty persistence must be two consecutive steps")
        if self.novelty_reward < 0 or self.novelty_threshold <= 0 or self.novelty_budget < 0:
            raise ValueError("novelty parameters must be non-negative and threshold positive")
        if (
            self.scene_change_reward < 0
            or self.enemy_kill_reward < 0
            or self.enemy_kill_reward_budget < 0
        ):
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
        if self.reward_version in {"explore_v3", "explore_v4"}:
            if self.horizon_steps > 3000:
                raise ValueError("explore_v3 horizon exceeds the bounded episode-return contract")
            if self.geo_gain_reward != 0 or self.geo_spend_reward != 0:
                raise ValueError("explore_v3 does not reward Geo")
            if (
                self.shade_recovery_reward != 4.0
                or self.shade_hit_reward != 0.3
                or self.shade_recovery_reward >= abs(self.death_penalty)
            ):
                raise ValueError(
                    "explore_v3 Shade recovery reward must be 4 and below death penalty"
                )
            if not 0 <= self.hp_restored_reward < abs(self.damage_penalty):
                raise ValueError("healing reward must be smaller than damage penalty magnitude")
            if not 0 < self.stuck_seconds <= 3600 or not 1 <= self.evaluation_interval <= 10_000:
                raise ValueError("stuck_seconds/evaluation_interval are outside their safe ranges")
            if (
                self.scene_reward_budget < 0
                or self.hp_restored_budget < 0
                or self.time_cost_budget < 0
            ):
                raise ValueError("explore_v3 episode reward budgets must be non-negative")
            if self.shade_hit_budget < 0 or self.shade_approach_budget < 0:
                raise ValueError("explore_v3 Shade reward budgets must be non-negative")
            if (
                self.novelty_budget > 20.0
                or self.scene_reward_budget > 20.0
                or self.enemy_kill_reward_budget > 5.0
                or self.hp_restored_budget > 4.0
                or self.shade_hit_budget > 1.2
                or self.shade_approach_budget > 1.0
                or self.time_cost_budget > 20.0
            ):
                raise ValueError(
                    "explore_v3 reward budgets exceed the stable episode-return contract"
                )
            if (
                self.novelty_reward != 0.05
                or self.shade_hit_reward != 0.3
                or self.shade_recovery_reward != 4.0
                or self.hp_restored_reward != 0.4
                or self.damage_penalty != -0.5
                or self.death_penalty != -5.0
                or self.run_failed_penalty != -5.0
                or self.time_cost != -0.001
            ):
                raise ValueError("explore_v3 component weights are fixed by the reward version")
        if self.reward_version == "explore_v4":
            if self.horizon_steps > 3000 or self.geo_gain_reward != 0 or self.geo_spend_reward != 0:
                raise ValueError("explore_v4 keeps the bounded horizon and does not reward Geo")
            if self.shade_recovery_reward != 4.0 or self.shade_hit_reward != 0.3:
                raise ValueError("explore_v4 inherits the explore_v3 Shade contract")
            if (
                self.novelty_reward != 0.05
                or self.damage_penalty != -0.5
                or self.death_penalty != -5.0
            ):
                raise ValueError("explore_v4 inherits the explore_v3 core reward contract")
        if (
            self.damage_penalty > 0
            or self.death_penalty >= 0
            or self.run_failed_penalty >= 0
            or self.time_cost > 0
        ):
            raise ValueError("penalty signs are invalid")
        if self.geo_gain_reward < 0 or self.geo_spend_reward < 0 or self.geo_reward_budget < 0:
            raise ValueError("geo reward parameters must be non-negative")
        if self.lease_base_ms < 1:
            raise ValueError("lease_base_ms must be positive")
        if not (
            0 < self.ent_coef <= 10
            and 0 < self.gamma <= 1
            and 0 < self.gae_lambda <= 1
            and self.n_epochs >= 1
            and 0 < self.target_kl <= 10
            and 0 < self.learning_rate <= 1
            and 0 < self.clip_range <= 1
        ):
            raise ValueError("PPO hyperparameters are outside their safe ranges")

    def maximum_positive_episode_reward(self) -> float:
        if self.reward_version in {"explore_v3", "explore_v4"}:
            return (
                self.novelty_budget
                + self.scene_reward_budget
                + self.enemy_kill_reward_budget
                + self.shade_recovery_reward
                + self.hp_restored_budget
                + self.shade_hit_budget
                + self.shade_approach_budget
                + (20.0 if self.reward_version == "explore_v4" else 0.0)
            )
        # At least three stable frames are required per distinct scene event.
        scene_events = self.horizon_steps // 3
        return (
            self.novelty_budget
            + self.scene_change_reward * scene_events
            + self.enemy_kill_reward_budget
            + self.geo_reward_budget
            + self.shade_recovery_reward
        )

    def minimum_negative_episode_reward(self) -> float:
        if self.reward_version in {"explore_v3", "explore_v4"}:
            return -(
                abs(self.damage_penalty) * 9
                + abs(self.death_penalty)
                + abs(self.run_failed_penalty)
                + self.time_cost_budget
                + self.shade_approach_budget
            )
        max_lease = max(
            template.resolve_lease_ms(self.lease_base_ms)
            for template in SCENE_CHANGE_ACTION_TEMPLATES
        )
        max_health = 9
        time = abs(self.time_cost) * self.horizon_steps * (max_lease / self.lease_base_ms)
        urgency = 0.001 * self.horizon_steps * (max_lease / 1000.0)
        shade_approach = (
            self.shade_approach_budget
            if self.reward_version in {"explore_v3", "explore_v4"}
            else 0.0
        )
        return -(
            abs(self.damage_penalty) * max_health
            + abs(self.death_penalty)
            + abs(self.run_failed_penalty)
            + time
            + urgency
            + shade_approach
        )


class ExplorationTask(gym.Env[dict[str, NDArray[np.generic]], int]):
    """Pixel-only exploration rewards; privileged bridge signals affect rewards only."""

    def __init__(
        self,
        environment: HollowKnightEnv,
        *,
        config: ExplorationTaskConfig | None = None,
        task_log_path: Path | None = None,
        lifelong_memory_root: Path | None = None,
    ) -> None:
        super().__init__()
        self.environment = environment
        self.config = config or ExplorationTaskConfig()
        self.config.validate()
        self.task_log_path = task_log_path
        self.lifelong_memory_root = lifelong_memory_root
        self.action_space = gym.spaces.Discrete(len(SCENE_CHANGE_ACTION_TEMPLATES))  # type: ignore[assignment]
        source = cast(gym.spaces.Dict, environment.observation_space)
        task_observation_space = temporal_observation_space(
            source,  # type: ignore[arg-type]
            number_of_actions=len(SCENE_CHANGE_ACTION_TEMPLATES),
        )
        if self.config.reward_version == "explore_v4":
            task_observation_space.spaces["player_knowledge"] = PlayerKnowledge.space()
        self.observation_space = task_observation_space  # type: ignore[assignment]
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
        self._scene_visits: dict[str, int] = {}
        self._enemy_visits: dict[str, int] = {}
        self._scene_paid = 0.0
        self._campaign_scene_path: Path | None = None
        self._campaign_enemy_path: Path | None = None
        self._scene_first_seen_path: Path | None = None
        self._scene_first_seen: dict[str, int] = {}
        self._progress_elapsed_seconds = 0.0
        self._previous_scene_name: str | None = None
        self._episode_scene_names: set[str] = set()
        self._episode_new_scenes: list[str] = []
        self._episode_campaign_new_scenes: list[str] = []
        self._episode_kills = 0
        self._episode_deaths = 0
        self._episode_shade_recovered = 0
        self._episode_shade_lost = 0
        self._steps_to_first_scene: int | None = None
        self._hp_restored_paid = 0.0
        self._shade_hits_paid = 0.0
        self._shade_approach_positive_paid = 0.0
        self._shade_approach_negative_paid = 0.0
        self._damage_paid_health = 0.0
        self._time_cost_paid = 0.0
        self._recent_reward_window: deque[dict[str, float]] = deque(maxlen=2000)
        self._recent_no_novelty_pairs: deque[tuple[str, str]] = deque(maxlen=2000)
        self._farm_alert_latched = False
        self._episode_total_reward = 0.0
        self._evaluation_mode = False
        self._episode_number = 0
        self._milestone_counts: dict[str, int] = {}
        self._milestone_path: Path | None = None
        self._milestone_paid = 0.0
        journey_value = os.environ.get("HK_AGENT_JOURNEY_ROOT")
        self._journey_root = Path(journey_value) if journey_value else None
        self._journey_actor = os.environ.get("HK_AGENT_SLOT_ID", "0")
        self._diary: RunDiary | None = None
        self._reload_checkpoint_next_reset = False
        self._player_knowledge = PlayerKnowledge()
        playthrough_root = os.environ.get("HK_AGENT_PLAYTHROUGH_ROOT")
        self._knowledge_path = (
            Path(playthrough_root) / "player-knowledge.json"
            if playthrough_root
            else task_log_path.with_name("player-knowledge.json") if task_log_path else None
        )
        if self._knowledge_path is not None:
            try:
                if self._knowledge_path.stat().st_size <= 1_048_576:
                    self._player_knowledge = PlayerKnowledge.from_payload(
                        json.loads(self._knowledge_path.read_text("utf-8"))
                    )
            except (OSError, json.JSONDecodeError):
                pass

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, NDArray[np.generic]], dict[str, Any]]:
        super().reset(seed=seed)
        reset_options = dict(options or {})
        if self._reload_checkpoint_next_reset:
            reset_options["reload_checkpoint"] = True
            self._reload_checkpoint_next_reset = False
        observation, reset_info = self.environment.reset(seed=seed, options=reset_options)
        self._player_knowledge.update(reset_info)
        self._persist_player_knowledge()
        descriptor = self._descriptor(observation["pixels"][-1])
        if self.config.reward_version in {"explore_v2", "explore_v3", "explore_v4"}:
            run_root = (
                self.task_log_path.parent.parent if self.task_log_path is not None else Path.cwd()
            )
            memory_root = self.lifelong_memory_root or (
                run_root if self.task_log_path is not None else None
            )
            self._lifelong_path = (
                memory_root / "lifelong-novelty.json" if memory_root is not None else None
            )
            self._lifelong_visits = (
                load_lifelong_visits(self._lifelong_path, self.config.lifelong_archive_max)
                if self._lifelong_path is not None
                else {}
            )
            if (
                self.config.reward_version in {"explore_v3", "explore_v4"}
                and memory_root is not None
            ):
                self._campaign_scene_path = memory_root / "lifelong-scenes.json"
                self._campaign_enemy_path = memory_root / "lifelong-enemies.json"
                self._scene_first_seen_path = memory_root / "lifelong-scene-first-seen.json"
                self._scene_visits = load_campaign_counts(
                    self._campaign_scene_path, self.config.lifelong_archive_max
                )
                self._enemy_visits = load_campaign_counts(
                    self._campaign_enemy_path, self.config.lifelong_archive_max
                )
                self._scene_first_seen = load_scene_first_seen(
                    self._scene_first_seen_path, self.config.lifelong_archive_max
                )
            if self.config.reward_version == "explore_v4" and memory_root is not None:
                self._milestone_path = memory_root / "lifelong-milestones.json"
                self._milestone_counts = load_campaign_counts(
                    self._milestone_path, self.config.lifelong_archive_max
                )
        self._milestone_paid = 0.0
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
        self._hp_restored_paid = 0.0
        self._shade_hits_paid = 0.0
        self._shade_approach_positive_paid = 0.0
        self._shade_approach_negative_paid = 0.0
        self._damage_paid_health = 0.0
        self._time_cost_paid = 0.0
        self._progress_elapsed_seconds = 0.0
        self._episode_scene_names = set()
        self._episode_new_scenes = []
        self._episode_campaign_new_scenes = []
        self._scene_paid = 0.0
        self._episode_kills = self._episode_deaths = 0
        self._episode_shade_recovered = self._episode_shade_lost = 0
        self._steps_to_first_scene = None
        self._episode_total_reward = 0.0
        self._episode_number += 1
        initial_scene = reset_info.get("scene_name")
        self._previous_scene_name = initial_scene if isinstance(initial_scene, str) else None
        if self._previous_scene_name:
            self._episode_scene_names.add(self._previous_scene_name)
        self._diary = None
        if self._journey_root is not None:
            self._diary = RunDiary(
                self._journey_root
                / f"actor-{self._journey_actor}"
                / f"episode-{self._episode_number}",
                self._journey_root,
                f"actor-{self._journey_actor}",
                self._episode_number,
                capture_png=getattr(self.environment.backend, "capture_png", None),
            )
            initial_telemetry = reset_info.get("operator_telemetry", {})
            initial_record = dict(initial_telemetry) if isinstance(initial_telemetry, dict) else {}
            initial_record.setdefault("scene_name", self._previous_scene_name)
            initial_events = initial_record.get("milestone_events_since_last_frame", [])
            initial_record["events"] = (
                list(initial_events) if isinstance(initial_events, list) else []
            )
            initial_events = self._diary.observe(initial_record, monotonic_seconds=time.monotonic())
            self._diary.capture_firsts(initial_events)
            self._update_playthrough(scene=self._previous_scene_name, step=0, events=initial_events)
        self._done = False
        self.mask_unavailable_count = 0
        self._history.reset()
        return self._apply_player_knowledge(self._history.apply(observation)), {}

    def step(
        self, action: int
    ) -> tuple[dict[str, NDArray[np.generic]], float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("reset() is required before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid task action: {action!r}")
        decision = decide_masked_action(
            int(action),
            SCENE_CHANGE_ACTION_NAMES,
            getattr(self.environment, "latest_action_availability", None),
        )
        if decision.availability is None:
            self.mask_unavailable_count += 1
        action = decision.executed_index
        audit = action_mask_audit(decision, self.mask_unavailable_count)
        template = SCENE_CHANGE_ACTION_TEMPLATES[action]
        lease = template.resolve_lease_ms(self.environment.lease_ms)
        if self._diary is not None:
            self._diary.capture_before_exit(template.buttons)
        if isinstance(self.environment, HollowKnightEnv):
            observation, _, terminated, truncated, base = self.environment.step(
                template.raw_action_index, lease_ms=lease, action_audit=audit
            )
        else:
            observation, _, terminated, truncated, base = self.environment.step(
                template.raw_action_index, lease_ms=lease
            )
        previous_skill_epoch = self._player_knowledge.skill_epoch
        self._player_knowledge.update(base)
        self._persist_player_knowledge()
        ability_reexploration = self._player_knowledge.skill_epoch > previous_skill_epoch
        if self._diary is not None:
            telemetry = base.get("operator_telemetry", {})
            record = dict(telemetry) if isinstance(telemetry, dict) else {}
            record.setdefault("scene_name", base.get("scene_name"))
            record["geo_gained_since_last_frame"] = base.get("geo_gained_since_last_frame", 0)
            if "milestone_events_since_last_frame" in base:
                raw_events = base["milestone_events_since_last_frame"]
                record["events"] = list(raw_events) if isinstance(raw_events, list) else []
            else:
                record["events"] = []
            if (
                isinstance(base.get("hp_lost_since_last_frame"), int)
                and base["hp_lost_since_last_frame"] > 0
            ):
                record["events"].append(
                    {
                        "category": "damage",
                        "key": f"episode_{self._episode_number}_step_{self._steps + 1}",
                        "confirmed": True,
                    }
                )
            supervisor = base.get("supervisor")
            if isinstance(supervisor, dict) and supervisor.get("player_dead") is True:
                record["events"].append(
                    {
                        "category": "death",
                        "key": f"episode_{self._episode_number}_death",
                        "confirmed": True,
                    }
                )
            journey_events = self._diary.observe(record)
            self._diary.capture_firsts(journey_events)
            self._update_playthrough(scene=str(record.get("scene_name") or "") or None,
                                     step=self._steps + 1, events=journey_events)
        if base.get("valid") is not True:
            self._done = True
            self.environment.backend.release_all()
            raise InvalidTrainingTransition(
                f"operationally invalid transition: {base.get('termination_reason')}"
            )
        self._history.append(action)
        observation = self._history.apply(observation)
        frame = observation["pixels"][-1]
        descriptor = self._descriptor(frame)
        luma, dark_fraction, white_fraction, entropy = visual_quality_metrics(
            frame, _descriptor_config()
        )
        informative = dark_fraction < 0.60 and white_fraction < 0.75 and entropy >= 1.0
        self._steps += 1
        components: dict[str, float] = {
            "novelty": 0.0,
            "scene_change": 0.0,
            "confirmed_enemy_kills": 0.0,
            "confirmed_geo_gained": 0.0,
            "confirmed_geo_spent": 0.0,
            "shade_recovered": 0.0,
            "shade_hits": 0.0,
            "shade_approach": 0.0,
            "hp_restored": 0.0,
            "damage_taken": 0.0,
            "death": 0.0,
            "run_failed": 0.0,
            "milestone_novelty": 0.0,
            "time_cost": self.config.time_cost * (lease / self.config.lease_base_ms),
        }
        base_done = terminated or truncated
        novelty = min(
            (descriptor_distance(descriptor, prior) for prior in self._archive), default=1.0
        )
        eligible = informative and novelty >= self.config.novelty_threshold and not base_done
        if self.config.reward_version in {"explore_v2", "explore_v3", "explore_v4"}:
            tokens = set(template.name.split("+"))
            stationary_attack = "ATTACK" in tokens and not tokens.intersection(
                {"LEFT", "RIGHT", "JUMP"}
            )
            if stationary_attack:
                eligible = False
        novelty_hash = (
            _ability_aware_novelty_hash(descriptor, self._player_knowledge)
            if self.config.reward_version == "explore_v4"
            else descriptor_hash(descriptor)
            if self.config.reward_version in {"explore_v2", "explore_v3", "explore_v4"}
            else None
        )
        if self.config.reward_version in {"explore_v2", "explore_v3", "explore_v4"}:
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
            components["novelty"] = min(
                self.config.novelty_reward * lifetime_factor,
                self.config.novelty_budget - self._novelty_paid,
            )
            self._novelty_paid += components["novelty"]
            self._archive.append(descriptor.copy())
            if novelty_hash is not None:
                self._lifelong_visits[novelty_hash] = visits + 1
                if len(self._lifelong_visits) > self.config.lifelong_archive_max:
                    del self._lifelong_visits[next(iter(self._lifelong_visits))]
                if self._lifelong_path is not None:
                    save_lifelong_visits(self._lifelong_path, self._lifelong_visits)
        detection = detect_scene_change(
            descriptor=descriptor,
            frame=frame,
            config=_descriptor_config(),
            anchor=self._anchor,
            previous_descriptor=self._previous_descriptor,
            transition_seen=self._transition_seen,
            dark_run=self._dark_run,
            stable_run=self._scene_stable,
            base_done=base_done,
        )
        self._transition_seen = bool(detection["transition_seen"])
        self._dark_run = int(detection["dark_run"])
        self._scene_stable = int(detection["stable_run"])
        changed = bool(detection["scene_changed"])
        self._previous_descriptor = descriptor.copy()
        if (
            changed
            and self.config.reward_version not in {"explore_v3", "explore_v4"}
            and not base_done
            and not any(
                descriptor_distance(descriptor, prior) < 0.025 for prior in self._paid_scenes
            )
        ):
            components["scene_change"] = self.config.scene_change_reward
            self._paid_scenes.append(descriptor.copy())
            self._anchor = descriptor.copy()
            self._transition_seen = False
            self._dark_run = self._scene_stable = 0
        elif changed:
            self._anchor = descriptor.copy()
            self._transition_seen = False
            self._dark_run = self._scene_stable = 0
        scene_event_paid = False
        current_scene = base.get("scene_name")
        if isinstance(current_scene, str):
            self._episode_scene_names.add(current_scene)
            if self._previous_scene_name is None:
                self._previous_scene_name = current_scene
            elif current_scene != self._previous_scene_name:
                self._steps_to_first_scene = (
                    self._steps
                    if self._steps_to_first_scene is None
                    else self._steps_to_first_scene
                )
                if (
                    self.config.reward_version in {"explore_v3", "explore_v4"}
                    and not base_done
                    and current_scene not in self._episode_new_scenes
                ):
                    prior_visits = self._scene_visits.get(current_scene, 0)
                    requested_reward = 2.0 + 8.0 / float(np.sqrt(1 + prior_visits))
                    remaining = max(0.0, self.config.scene_reward_budget - self._scene_paid)
                    components["scene_change"] = min(requested_reward, remaining)
                    if components["scene_change"] > 0:
                        self._scene_paid += components["scene_change"]
                        self._scene_visits[current_scene] = prior_visits + 1
                        self._episode_new_scenes.append(current_scene)
                        scene_event_paid = True
                        if prior_visits == 0:
                            self._episode_campaign_new_scenes.append(current_scene)
                            self._scene_first_seen.setdefault(current_scene, time.time_ns())
                            if self._scene_first_seen_path is not None:
                                save_scene_first_seen(
                                    self._scene_first_seen_path,
                                    self._scene_first_seen,
                                    self.config.lifelong_archive_max,
                                )
                        if self._campaign_scene_path is not None:
                            save_campaign_counts(
                                self._campaign_scene_path,
                                self._scene_visits,
                                self.config.lifelong_archive_max,
                            )
                self._previous_scene_name = current_scene
        combat = confirmed_combat_components(
            base,
            enemy_kill_reward=(
                0.0
                if self.config.reward_version in {"explore_v3", "explore_v4"}
                else self.config.enemy_kill_reward
            ),
            shade_recovery_reward=(
                4.0
                if self.config.reward_version in {"explore_v2", "explore_v3", "explore_v4"}
                else self.config.shade_recovery_reward
            ),
            damage_penalty=self.config.damage_penalty,
            geo_gain_reward=self.config.geo_gain_reward,
            geo_spend_reward=self.config.geo_spend_reward,
            death_penalty=self.config.death_penalty,
            run_failed_penalty=self.config.run_failed_penalty,
        )
        components["confirmed_enemy_kills"] = min(
            combat["confirmed_enemy_kills"],
            max(0.0, self.config.enemy_kill_reward_budget - self._kills_paid),
        )
        if self.config.reward_version in {"explore_v3", "explore_v4"}:
            events = base.get("enemy_events_since_last_frame", [])
            if not isinstance(events, list):
                raise InvalidTrainingTransition("invalid enemy event list")
            rewardable_events = (
                not base_done and combat["player_death"] >= 0 and combat["gameplay_run_failed"] >= 0
            )
            if rewardable_events:
                self._episode_kills += len(events)
            remaining_kill_budget = max(
                0.0, self.config.enemy_kill_reward_budget - self._kills_paid
            )
            for event in events if rewardable_events else []:
                if (
                    not isinstance(event, dict)
                    or not isinstance(event.get("enemy_type"), str)
                    or not isinstance(event.get("scene_name"), str)
                ):
                    raise InvalidTrainingTransition("invalid typed enemy reward event")
                key = f"{event['scene_name']}|{event['enemy_type']}"
                visits = self._enemy_visits.get(key, 0)
                requested = max(0.2, 1.0 / float(np.sqrt(1 + visits)))
                award = min(requested, remaining_kill_budget)
                components["confirmed_enemy_kills"] += award
                remaining_kill_budget -= award
                if award > 0:
                    self._enemy_visits[key] = visits + 1
            if self._campaign_enemy_path is not None and events and self._enemy_visits:
                save_campaign_counts(
                    self._campaign_enemy_path, self._enemy_visits, self.config.lifelong_archive_max
                )
        if self.config.reward_version in {"explore_v3", "explore_v4"}:
            components["confirmed_enemy_kills"] = min(
                components["confirmed_enemy_kills"],
                max(0.0, self.config.enemy_kill_reward_budget - self._kills_paid),
            )
        if (
            self.config.reward_version == "explore_v4"
            and not base_done
            and combat["player_death"] >= 0
            and combat["gameplay_run_failed"] >= 0
        ):
            events = base.get("milestone_events_since_last_frame", [])
            if not isinstance(events, list) or len(events) > 128:
                raise InvalidTrainingTransition("invalid milestone event list")
            remaining = max(0.0, 20.0 - self._milestone_paid)
            for event in events:
                if not isinstance(event, dict) or event.get("confirmed") is not True:
                    continue
                category, identity = event.get("category"), event.get("key")
                if (
                    category not in {"grub", "boss", "item"}
                    or not isinstance(identity, str)
                    or not identity
                ):
                    continue
                key = f"{category}:{identity}"[:512]
                visits = self._milestone_counts.get(key, 0)
                if self._milestone_path is not None:
                    visits = increment_campaign_count(
                        self._milestone_path, key, limit=self.config.lifelong_archive_max
                    )
                base_reward, minimum = {
                    "grub": (3.0, 0.5),
                    "boss": (15.0, 2.0),
                    "item": (5.0, 1.0),
                }[category]
                requested = max(minimum, base_reward / float(np.sqrt(1 + visits)))
                award = min(requested, remaining)
                if award <= 0:
                    continue
                components["milestone_novelty"] += award
                self._milestone_paid += award
                remaining -= award
                self._milestone_counts[key] = visits + 1
        self._kills_paid += components["confirmed_enemy_kills"]
        components["shade_recovered"] = 0.0 if self._shade_paid else combat["shade_recovered"]
        self._shade_paid |= components["shade_recovered"] > 0
        if components["shade_recovered"] > 0:
            self._episode_shade_recovered += 1
        if self.config.reward_version in {"explore_v2", "explore_v3", "explore_v4"}:
            hits = int(base.get("shade_hits_since_last_frame", 0))
            if hits < 0:
                raise InvalidTrainingTransition("invalid Shade hit delta")
            hit_budget = (
                self.config.shade_hit_budget
                if self.config.reward_version in {"explore_v3", "explore_v4"}
                else float("inf")
            )
            components["shade_hits"] = min(
                min(hits, 4) * self.config.shade_hit_reward,
                max(0.0, hit_budget - self._shade_hits_paid),
            )
            self._shade_hits_paid += components["shade_hits"]
            restored = int(base.get("hp_restored_since_last_frame", 0))
            if restored < 0:
                raise InvalidTrainingTransition("invalid restored health delta")
            hp_budget = (
                self.config.hp_restored_budget
                if self.config.reward_version in {"explore_v3", "explore_v4"}
                else float("inf")
            )
            components["hp_restored"] = min(
                restored * self.config.hp_restored_reward,
                max(0.0, hp_budget - self._hp_restored_paid),
            )
            self._hp_restored_paid += components["hp_restored"]
            distance = base.get("shade_distance")
            scene = base.get("scene_name")
            if (
                isinstance(distance, (int, float))
                and not isinstance(distance, bool)
                and isinstance(scene, str)
                and base.get("shade_pending") is True
            ):
                current_distance = float(distance)
                if (
                    np.isfinite(current_distance)
                    and self._shade_distance is not None
                    and self._shade_distance[0] == scene
                ):
                    approach = float(
                        np.clip(0.05 * (self._shade_distance[1] - current_distance), -0.05, 0.05)
                    )
                    if self.config.reward_version in {"explore_v3", "explore_v4"}:
                        if approach > 0:
                            approach = min(
                                approach,
                                max(
                                    0.0,
                                    self.config.shade_approach_budget
                                    - self._shade_approach_positive_paid,
                                ),
                            )
                            self._shade_approach_positive_paid += approach
                        elif approach < 0:
                            approach = max(
                                approach,
                                -max(
                                    0.0,
                                    self.config.shade_approach_budget
                                    - self._shade_approach_negative_paid,
                                ),
                            )
                            self._shade_approach_negative_paid += -approach
                    components["shade_approach"] = approach
                self._shade_distance = (scene, current_distance)
            else:
                self._shade_distance = None
            if base.get("shade_pending") is True:
                components["time_cost"] -= 0.001 * (lease / 1000.0)
        if self.config.reward_version in {"explore_v3", "explore_v4"}:
            payable_time_cost = min(
                abs(components["time_cost"]),
                max(0.0, self.config.time_cost_budget - self._time_cost_paid),
            )
            components["time_cost"] = -payable_time_cost
            self._time_cost_paid += payable_time_cost
        geo = combat["confirmed_geo_gained"] + combat["confirmed_geo_spent"]
        geo_award = min(geo, max(0.0, self.config.geo_reward_budget - self._geo_paid))
        if geo > 0:
            scale = geo_award / geo
            components["confirmed_geo_gained"] = combat["confirmed_geo_gained"] * scale
            components["confirmed_geo_spent"] = combat["confirmed_geo_spent"] * scale
            self._geo_paid += geo_award
        components["damage_taken"] = combat["confirmed_hp_lost"]
        if self.config.reward_version in {"explore_v3", "explore_v4"}:
            raw_damage = int(base.get("hp_lost_since_last_frame", 0))
            if raw_damage < 0:
                raise InvalidTrainingTransition("invalid confirmed damage delta")
            payable_damage = min(float(raw_damage), max(0.0, 9.0 - self._damage_paid_health))
            components["damage_taken"] = -payable_damage * abs(self.config.damage_penalty)
            self._damage_paid_health += payable_damage
        components["death"] = combat["player_death"]
        components["run_failed"] = combat["gameplay_run_failed"]
        if components["death"] < 0:
            self._episode_deaths += 1
        if components["run_failed"] < 0:
            self._episode_shade_lost += 1
        # Death/run-failure and any base terminal/truncation suppress all positive rewards on this transition.
        terminal_penalty = components["death"] < 0 or components["run_failed"] < 0
        if base_done or terminal_penalty:
            for name in (
                "novelty",
                "scene_change",
                "confirmed_enemy_kills",
                "confirmed_geo_gained",
                "confirmed_geo_spent",
                "shade_recovered",
                "shade_hits",
                "shade_approach",
                "hp_restored",
                "milestone_novelty",
            ):
                components[name] = 0.0
        reward = float(sum(components.values()))
        self._episode_total_reward += reward
        if components["novelty"] > 0 or scene_event_paid:
            self._progress_elapsed_seconds = 0.0
        else:
            self._progress_elapsed_seconds += lease / 1000.0
        stuck = (
            self.config.reward_version in {"explore_v3", "explore_v4"}
            and self._progress_elapsed_seconds > self.config.stuck_seconds
            and not base_done
            and not terminal_penalty
        )
        if stuck:
            # The next PPO slice reopens through the validated Continue plan,
            # loading the game's last committed checkpoint instead of replaying
            # the campaign seed. The owner closes the current process safely.
            self._reload_checkpoint_next_reset = True
        horizon = self._steps >= self.config.horizon_steps
        task_terminated = terminated or terminal_penalty
        task_truncated = truncated or ((horizon or stuck) and not task_terminated)
        termination_reason = base.get("termination_reason")
        if termination_reason is None and components["death"] < 0:
            termination_reason = "player_dead"
        elif termination_reason is None and components["run_failed"] < 0:
            termination_reason = "gameplay_run_failed"
        elif termination_reason is None and stuck:
            termination_reason = "stuck"
        elif termination_reason is None and horizon:
            termination_reason = "time_limit"
        self._done = task_terminated or task_truncated
        if self._done and self._diary is not None and self._journey_root is not None:
            self._diary.finish(
                termination_reason,
                deaths=[{"scene": current_scene}] * self._episode_deaths,
                shade={
                    "recovered": self._episode_shade_recovered,
                    "lost": self._episode_shade_lost,
                },
            )
            rebuild_progress(self._journey_root)
        if self._done:
            self.environment.backend.release_all()
        farm_alerts = self._record_farm_window(components, current_scene, template.name)
        episode_progress: dict[str, Any] | None = None
        if self._done and self.config.reward_version in {"explore_v3", "explore_v4"}:
            episode_progress = {
                "episode": self._episode_number,
                "steps": self._steps,
                "distinct_scenes": len(self._episode_scene_names),
                "scene_names": sorted(self._episode_scene_names),
                "new_scenes_in_campaign": self._episode_campaign_new_scenes.copy(),
                "new_scenes_paid_this_episode": self._episode_new_scenes.copy(),
                "deaths": self._episode_deaths,
                "shade_recoveries": self._episode_shade_recovered,
                "shade_losses": self._episode_shade_lost,
                "enemy_kills": self._episode_kills,
                "steps_to_first_scene_change": self._steps_to_first_scene,
                "termination_reason": termination_reason,
                "return": self._episode_total_reward,
            }
        info: dict[str, Any] = {
            "valid": True,
            "termination_reason": termination_reason,
            "reward_version": self.config.reward_version,
            "reward_components": components,
            "scene_change_event": changed,
            "novelty_distance": novelty,
            "novelty_paid": self._novelty_paid,
            "archive_size": len(self._archive),
            "descriptor": descriptor.tolist(),
            "visual_quality": [luma, dark_fraction, white_fraction, entropy],
            **audit,
            "novelty_eligible": eligible,
            "scene_detector_audit": changed,
            "farm_alerts": farm_alerts,
        }
        if episode_progress is not None:
            info["episode_progress"] = episode_progress
        _copy_lifecycle_info(base, info)
        self._append_log(
            {
                "schema_version": 1,
                "reward_version": self.config.reward_version,
                "ability_reexploration": ability_reexploration,
                "step": self._steps,
                "task_action": template.name,
                "raw_action_index": template.raw_action_index,
                "lease_ms": lease,
                "reward": {"total": reward, "components": components},
                "novelty_distance": novelty,
                "scene_change_event": changed,
                "terminated": task_terminated,
                "truncated": task_truncated,
                "termination_reason": info["termination_reason"],
                "valid": True,
                "evaluation_only": self._evaluation_mode,
                "scene_name": current_scene if isinstance(current_scene, str) else None,
                "stuck_seconds": self._progress_elapsed_seconds,
                "farm_alerts": farm_alerts,
                "episode_progress": episode_progress,
                **audit,
            }
        )
        return self._apply_player_knowledge(observation), reward, task_terminated, task_truncated, info

    def _apply_player_knowledge(
        self, observation: dict[str, NDArray[np.generic]]
    ) -> dict[str, NDArray[np.generic]]:
        result = dict(observation)
        if self.config.reward_version == "explore_v4":
            result["player_knowledge"] = self._player_knowledge.vector()
        return result

    def _persist_player_knowledge(self) -> None:
        if self._knowledge_path is None:
            return
        self._knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        pending = self._knowledge_path.with_name(f".{self._knowledge_path.name}.{os.getpid()}.tmp")
        pending.write_text(json.dumps(self._player_knowledge.to_payload(), separators=(",", ":"), allow_nan=False), encoding="utf-8", newline="\n")
        os.replace(pending, self._knowledge_path)

    @staticmethod
    def _update_playthrough(*, scene: str | None, step: int, events: list[dict[str, Any]]) -> None:
        root_value = os.environ.get("HK_AGENT_PLAYTHROUGH_ROOT")
        if not root_value:
            return
        checkpoint = next((event.get("key") for event in events
                           if event.get("category") == "checkpoint" and isinstance(event.get("key"), str)), None)
        try:
            from hk_agent.playthrough import update_playthrough_progress

            update_playthrough_progress(Path(root_value), scene=scene,
                                        progress={"step": step, "checkpoint": checkpoint})
        except (OSError, ValueError, RuntimeError):
            # Progress display is best-effort; the save transaction remains authoritative.
            return

    def action_masks(self) -> NDArray[np.bool_]:
        return build_action_mask(
            SCENE_CHANGE_ACTION_NAMES, getattr(self.environment, "latest_action_availability", None)
        )

    def set_evaluation_mode(self, enabled: bool) -> None:
        self._evaluation_mode = bool(enabled)

    def close(self) -> None:
        self.environment.close()

    def task_contract(self) -> dict[str, Any]:
        schema_version = {"explore_v1": 1, "explore_v2": 2, "explore_v3": 3, "explore_v4": 5}[
            self.config.reward_version
        ]
        return {
            "task_contract_schema_version": schema_version,
            "reward_version": self.config.reward_version,
            "legal_action_mask": {
                "version": "action_availability_v2" if schema_version == 5 else "action_availability_v1",
                "source": "frame_action_availability",
            },
            "action_names": list(SCENE_CHANGE_ACTION_NAMES),
            "actions": action_contract(
                SCENE_CHANGE_ACTION_TEMPLATES, base_lease_ms=self.environment.lease_ms
            ),
            "config": {
                key: value for key, value in asdict(self.config).items() if key != "reward_version"
            },
            "observation_version": "pixels_action_history_player_knowledge_v1" if schema_version == 5 else "pixels_action_history_v1",
            "information_boundary": ["pixels", "applied_action_history", "player_knowledge"] if schema_version == 5 else ["pixels", "applied_action_history"],
            "reward_only_signals": [
                "enemy_kills_since_last_frame",
                "hp_lost_since_last_frame",
                "geo_gained_since_last_frame",
                "geo_spent_since_last_frame",
                "shade_recovered",
                "death",
                "run_failed",
            ]
            + (
                [
                    "hp_restored_since_last_frame",
                    "shade_hits_since_last_frame",
                    "shade_distance",
                    "scene_name",
                ]
                if self.config.reward_version in {"explore_v2", "explore_v3", "explore_v4"}
                else []
            )
            + (
                ["enemy_events_since_last_frame"]
                if self.config.reward_version in {"explore_v3", "explore_v4"}
                else []
            )
            + (
                ["milestone_events_since_last_frame"]
                if self.config.reward_version == "explore_v4"
                else []
            ),
            "detector_inputs": ["cropped_rgb_pixels", "temporal_pixel_descriptors"],
            "privileged_state": [],
            "diagnostic_training": False,
            "claim": "pixel-only exploration with bounded reward-only bridge signals",
            "ppo_hyperparameters": {
                "ent_coef": self.config.ent_coef,
                "gamma": self.config.gamma,
                "gae_lambda": self.config.gae_lambda,
                "n_epochs": self.config.n_epochs,
                "target_kl": self.config.target_kl,
                "learning_rate": self.config.learning_rate,
                "clip_range": self.config.clip_range,
            },
        }

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
                cell = image[y_edges[row] : y_edges[row + 1], x_edges[column] : x_edges[column + 1]]
                grid[index] = cell.mean(axis=(0, 1)) / 255.0
                index += 1
        return grid

    def _record_farm_window(
        self, components: dict[str, float], scene: object, action: str
    ) -> list[str]:
        if self.config.reward_version not in {"explore_v3", "explore_v4"}:
            return []
        self._recent_reward_window.append(
            {name: value for name, value in components.items() if value > 0}
        )
        novelty_seen = components.get("novelty", 0.0) > 0 or components.get("scene_change", 0.0) > 0
        if not novelty_seen:
            self._recent_no_novelty_pairs.append((str(scene or "unknown"), action))
        triggers: list[str] = []
        if len(self._recent_reward_window) == 2000:
            totals: dict[str, float] = {}
            for row in self._recent_reward_window:
                for name, value in row.items():
                    totals[name] = totals.get(name, 0.0) + value
            positive_total = sum(totals.values())
            if positive_total > 0 and max(totals.values(), default=0.0) / positive_total > 0.60:
                triggers.append("component_over_60_percent")
        if len(self._recent_no_novelty_pairs) == 2000:
            counts: dict[tuple[str, str], int] = {}
            for pair in self._recent_no_novelty_pairs:
                counts[pair] = counts.get(pair, 0) + 1
            if max(counts.values(), default=0) / 2000 > 0.40:
                triggers.append("same_scene_action_over_40_percent_without_novelty")
        if triggers and not self._farm_alert_latched:
            self._farm_alert_latched = True
            return triggers
        if not triggers:
            self._farm_alert_latched = False
        return []

    def _append_log(self, record: dict[str, Any]) -> None:
        if self.task_log_path is None:
            return
        self.task_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
        fd = os.open(self.task_log_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            if os.write(fd, payload) != len(payload):
                raise OSError("partial task JSONL append")
            os.fsync(fd)
        finally:
            os.close(fd)


def _descriptor_config() -> SceneChangeTaskConfig:
    return SceneChangeTaskConfig()


def calibrate_novelty(
    noop_descriptors: list[NDArray[np.float32]],
    active_descriptors: list[NDArray[np.float32]],
    *,
    default_threshold: float = 0.02,
) -> float:
    if len(noop_descriptors) < 10 or len(active_descriptors) < 10:
        raise ValueError(
            "exploration calibration requires at least ten NOOP and ten active samples"
        )
    distances = []
    for index, descriptor in enumerate(noop_descriptors):
        archive = noop_descriptors[:index] or [noop_descriptors[0]]
        distances.append(min(descriptor_distance(descriptor, item) for item in archive))
    threshold = max(default_threshold, float(np.percentile(distances, 99)) * 2)
    active = []
    archive = list(noop_descriptors)
    for descriptor in active_descriptors:
        active.append(min(descriptor_distance(descriptor, item) for item in archive))
    if float(np.median(active)) <= threshold:
        raise ValueError("active exploration calibration did not exceed the NOOP threshold")
    return threshold
