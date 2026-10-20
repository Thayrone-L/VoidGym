from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter, deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from hk_agent.action_masking import action_mask_audit, build_action_mask, decide_masked_action
from hk_agent.challenge_practice import (
    ChallengeAttempt,
    ChallengePracticeController,
    ChallengeSnapshot,
    challenge_reward,
    create_challenge_snapshot,
)
from hk_agent.demonstrations import HelpRequestLedger, HelpTriggerConfig
from hk_agent.gym_env import HollowKnightEnv
from hk_agent.player_knowledge import ITEM_FLAGS, PlayerKnowledge
from hk_agent.run_journey import RunDiary, atomic_json, increment_campaign_count, rebuild_progress
from hk_agent.scene_change_task import (
    SCENE_CHANGE_ACTION_NAMES,
    SCENE_CHANGE_ACTION_TEMPLATES,
    SCENE_CHANGE_INTERACTION_ACTION_NAMES,
    SCENE_CHANGE_INTERACTION_ACTION_TEMPLATES,
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


def enemy_kill_reward_for_visit(visits: int, *, continuous: bool) -> float:
    if visits < 0:
        raise ValueError("enemy visit count must be non-negative")
    if continuous:
        return 1.0 / float(1 + visits) ** 2
    return max(0.2, 1.0 / float(np.sqrt(1 + visits)))


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
    tutorial_hint_level: int = 0
    continuous_playthrough: bool = False

    def validate(self) -> None:
        if (
            not 1 <= self.horizon_steps <= 100_000
            or self.archive_max < 1
            or self.lifelong_archive_max < 1
        ):
            raise ValueError("exploration horizon and archive must be positive and bounded")
        if self.reward_version not in SUPPORTED_EXPLORE_REWARD_VERSIONS:
            raise ValueError("unsupported exploration reward version")
        if self.tutorial_hint_level not in {0, 1}:
            raise ValueError("tutorial_hint_level must be 0 (official clues) or 1 (gentle guide)")
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
        self._action_templates = (
            SCENE_CHANGE_INTERACTION_ACTION_TEMPLATES
            if self.config.reward_version == "explore_v4"
            else SCENE_CHANGE_ACTION_TEMPLATES
        )
        self._action_names = (
            SCENE_CHANGE_INTERACTION_ACTION_NAMES
            if self.config.reward_version == "explore_v4"
            else SCENE_CHANGE_ACTION_NAMES
        )
        self.action_space = gym.spaces.Discrete(len(self._action_templates))  # type: ignore[assignment]
        source = cast(gym.spaces.Dict, environment.observation_space)
        task_observation_space = temporal_observation_space(
            source,  # type: ignore[arg-type]
            number_of_actions=len(self._action_templates),
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
        self._lifelong_dirty_steps = 0
        self._novelty_candidate: str | None = None
        self._novelty_candidate_descriptor: NDArray[np.float32] | None = None
        self._novelty_candidate_steps = 0
        self._shade_distance: tuple[str, float] | None = None
        self._scene_visits: dict[str, int] = {}
        self._enemy_visits: dict[str, int] = {}
        self._enemy_visits_dirty = False
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
        self._recent_reward_totals: dict[str, float] = {}
        self._recent_positive_total = 0.0
        self._recent_no_novelty_pairs: deque[tuple[str, str]] = deque(maxlen=2000)
        self._recent_no_novelty_counts: Counter[tuple[str, str]] = Counter()
        self._farm_alert_latched = False
        self._continuous_budget_path: Path | None = None
        self._continuous_budget_state: dict[str, Any] = {
            "schema_version": 2, "game_steps": 0, "game_time_ms": 0,
            "segment_paid": {}, "buckets": {}
        }
        self._continuous_budget_dirty_steps = 0
        self._pending_continuous_budget: dict[str, float] = {}
        self._episode_total_reward = 0.0
        self._evaluation_mode = False
        self._episode_number = 0
        self._scene_origins: dict[str, tuple[float, float]] = {}
        self._scene_max_progress: dict[str, float] = {}
        self._pending_log_records: list[bytes] = []
        self._last_log_flush = time.monotonic()
        self._last_knowledge_persist_step = -32
        self._last_knowledge_persist_ms = 0.0
        self._last_playthrough_update = 0.0
        self._milestone_counts: dict[str, int] = {}
        self._milestone_path: Path | None = None
        self._milestone_paid = 0.0
        journey_value = os.environ.get("HK_AGENT_JOURNEY_ROOT")
        self._journey_root = Path(journey_value) if journey_value else None
        self._journey_actor = os.environ.get("HK_AGENT_SLOT_ID", "0")
        if self._journey_root is not None:
            actor_diary = self._journey_root / f"actor-{self._journey_actor}"
            prior_episodes = [
                int(path.stem.removeprefix("episode-"))
                for path in actor_diary.glob("episode-*.json")
                if path.stem.removeprefix("episode-").isdigit()
            ]
            self._episode_number = max(prior_episodes, default=0)
        self._diary: RunDiary | None = None
        self._reload_checkpoint_next_reset = False
        self._player_knowledge = PlayerKnowledge()
        self._pending_tutorial_events: list[dict[str, Any]] = []
        self._challenge_practice_controller: ChallengePracticeController | None = None
        self._challenge_practice_finished = False
        self._challenge_practice_enabled = os.environ.get(
            "HK_AGENT_CHALLENGE_PRACTICE_ENABLED", "0"
        ) == "1"
        self._help_requests_enabled = os.environ.get("HK_AGENT_HELP_REQUESTS_ENABLED", "0") == "1"
        self._help_request_ledger: HelpRequestLedger | None = None
        shared_root = os.environ.get("HK_AGENT_SHARED_ROOT")
        slot_value = os.environ.get("HK_AGENT_SLOT_ID")
        if self._help_requests_enabled and shared_root and slot_value and slot_value.isdigit():
            try:
                self._help_request_ledger = HelpRequestLedger(
                    Path(shared_root) / "help-requests", slot=int(slot_value),
                    config=HelpTriggerConfig(
                        failed_attempts=int(os.environ.get("HK_AGENT_HELP_FAILED_ATTEMPTS", "100")),
                        stalled_hours=float(os.environ.get("HK_AGENT_HELP_STALLED_HOURS", "6")),
                        stuck_minutes=float(os.environ.get("HK_AGENT_HELP_STUCK_MINUTES", "360")),
                        repeated_stuck_count=int(os.environ.get("HK_AGENT_HELP_STUCK_COUNT", "10")),
                        requests_per_hour=int(os.environ.get("HK_AGENT_HELP_REQUESTS_PER_HOUR", "3")),
                    ),
                )
            except (OSError, ValueError):
                self._help_request_ledger = None
        challenge_context = os.environ.get("HK_AGENT_CHALLENGE_CONTEXT")
        challenge_snapshot = os.environ.get("HK_AGENT_CHALLENGE_SNAPSHOT")
        if self._challenge_practice_enabled and challenge_context and challenge_snapshot:
            kind, separator, challenge_id = challenge_context.partition(":")
            if separator and kind in {"boss", "traversal", "item", "area"} and Path(challenge_snapshot).is_dir():
                snapshot = ChallengeSnapshot(
                    challenge_id, cast(Any, kind),
                    "arena_entrance" if kind == "boss" else "challenge_room",
                    challenge_snapshot, Path(challenge_snapshot).stat().st_mtime_ns,
                )
                controller = ChallengePracticeController(
                    main_actor="actor-main", practice_actor=os.environ.get("HK_AGENT_SLOT_ID", "practice"),
                    required_wins=2, recent_attempts=5, time_limit_seconds=3600.0,
                )
                controller.start(snapshot, lambda _rollout: None)
                self._challenge_practice_controller = controller
        self._challenge_snapshot_index: list[dict[str, Any]] = []
        self._challenge_snapshot_root: Path | None = None
        self._challenge_save_root = Path(os.environ["HK_AGENT_INSTANCE_DATA_ROOT"]) if os.environ.get("HK_AGENT_INSTANCE_DATA_ROOT") else None
        self._personal_save_root = Path(os.environ["HK_AGENT_PERSONAL_ROOT"]) if os.environ.get("HK_AGENT_PERSONAL_ROOT") else None
        playthrough_root = os.environ.get("HK_AGENT_PLAYTHROUGH_ROOT")
        if playthrough_root:
            self._challenge_snapshot_root = Path(playthrough_root).parent / "challenge-snapshots" / Path(playthrough_root).name
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
        self._flush_transition_log()
        self._flush_lifelong_visits()
        self._flush_enemy_visits()
        self._save_continuous_budget(force=True)
        self._persist_player_knowledge(force=True)
        reset_options = dict(options or {})
        if self._reload_checkpoint_next_reset:
            reset_options["reload_checkpoint"] = True
            self._reload_checkpoint_next_reset = False
        observation, reset_info = self.environment.reset(seed=seed, options=reset_options)
        self._player_knowledge.update(
            reset_info, challenge_tracking_enabled=(self._challenge_practice_enabled or self._help_requests_enabled)
        )
        self._record_challenge_practice_events(self._player_knowledge.new_events)
        self._capture_challenge_snapshots(self._player_knowledge.new_events, reset_info)
        self._pending_tutorial_events.extend(self._player_knowledge.new_events)
        self._persist_player_knowledge(force=bool(self._player_knowledge.new_events))
        descriptor = self._descriptor(observation["pixels"][-1])
        if self.config.reward_version in {"explore_v2", "explore_v3", "explore_v4"}:
            run_root = (
                self.task_log_path.parent.parent if self.task_log_path is not None else Path.cwd()
            )
            memory_root = self.lifelong_memory_root or (
                run_root if self.task_log_path is not None else None
            )
            if self.config.continuous_playthrough and memory_root is not None:
                actor_slot = os.environ.get("HK_AGENT_SLOT_ID", "0")
                self._continuous_budget_path = memory_root / f"continuous-reward-slot-{actor_slot}.json"
                self._continuous_budget_state = self._load_continuous_budget(self._continuous_budget_path)
                if bool(reset_options.get("reload_checkpoint")):
                    self._continuous_budget_state["segment_paid"] = {}
                    self._save_continuous_budget(force=True)
            actor_slot = os.environ.get("HK_AGENT_SLOT_ID", "0")
            self._lifelong_path = (
                memory_root / (f"lifelong-novelty-slot-{actor_slot}.json"
                               if self.config.continuous_playthrough else "lifelong-novelty.json")
                if memory_root is not None else None
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
                actor_slot = os.environ.get("HK_AGENT_SLOT_ID", "0")
                self._campaign_enemy_path = (
                    memory_root / f"lifelong-enemies-slot-{actor_slot}.json"
                    if self.config.continuous_playthrough
                    else memory_root / "lifelong-enemies.json"
                )
                self._scene_first_seen_path = memory_root / "lifelong-scene-first-seen.json"
                self._scene_visits = load_campaign_counts(
                    self._campaign_scene_path, self.config.lifelong_archive_max
                )
                self._enemy_visits = load_campaign_counts(
                    self._campaign_enemy_path, self.config.lifelong_archive_max
                )
                self._enemy_visits_dirty = False
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
            initial_position = _player_position(initial_record)
            if self._previous_scene_name and initial_position is not None:
                self._scene_origins.setdefault(self._previous_scene_name, initial_position)
                self._scene_max_progress.setdefault(self._previous_scene_name, 0.0)
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
        step_profile: dict[str, float] = {}
        if self._done:
            raise RuntimeError("reset() is required before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid task action: {action!r}")
        decision = decide_masked_action(
            int(action),
            self._action_names,
            getattr(self.environment, "latest_action_availability", None),
        )
        if decision.availability is None:
            self.mask_unavailable_count += 1
        action = decision.executed_index
        audit = action_mask_audit(decision, self.mask_unavailable_count)
        template = self._action_templates[action]
        lease = template.resolve_lease_ms(self.environment.lease_ms)
        if self._diary is not None:
            self._diary.capture_before_exit(template.buttons)
        self._pending_continuous_budget = {}
        self._advance_continuous_budget_step(lease)
        started = time.perf_counter()
        if isinstance(self.environment, HollowKnightEnv):
            observation, _, terminated, truncated, base = self.environment.step(
                template.raw_action_index, lease_ms=lease, action_audit=audit
            )
        else:
            observation, _, terminated, truncated, base = self.environment.step(
                template.raw_action_index, lease_ms=lease
            )
        step_profile["environment_step_ms"] = (time.perf_counter() - started) * 1000
        raw_milestones = base.get("milestone_events_since_last_frame", [])
        if self.config.continuous_playthrough and isinstance(raw_milestones, list) and any(
            isinstance(event, Mapping) and event.get("confirmed") is True
            and (event.get("category") == "checkpoint" or event.get("type") in {"bench_rest", "first_bench_rest"})
            for event in raw_milestones
        ):
            self._continuous_budget_state["segment_paid"] = {}
            self._save_continuous_budget(force=True)
        started = time.perf_counter()
        previous_skill_epoch = self._player_knowledge.skill_epoch
        prior_scene = self._player_knowledge.last_scene
        prior_cell = self._player_knowledge.last_cell
        prior_visited = (
            set(self._player_knowledge.visited.get(prior_scene, set()))
            if isinstance(prior_scene, str) else set()
        )
        self._player_knowledge.update(
            base, challenge_tracking_enabled=(self._challenge_practice_enabled or self._help_requests_enabled)
        )
        self._record_challenge_practice_events(self._player_knowledge.new_events)
        knowledge_events = bool(self._player_knowledge.new_events)
        tutorial_events = self._pending_tutorial_events + self._player_knowledge.new_events
        self._capture_challenge_snapshots(self._player_knowledge.new_events, base)
        self._pending_tutorial_events = []
        self._persist_player_knowledge(force=knowledge_events)
        step_profile["player_knowledge_ms"] = (time.perf_counter() - started) * 1000
        step_profile["player_knowledge_write_ms"] = self._last_knowledge_persist_ms
        ability_reexploration = self._player_knowledge.skill_epoch > previous_skill_epoch
        started = time.perf_counter()
        if self._diary is not None:
            telemetry = base.get("operator_telemetry", {})
            record = dict(telemetry) if isinstance(telemetry, dict) else {}
            record.setdefault("scene_name", base.get("scene_name"))
            record["geo_gained_since_last_frame"] = base.get("geo_gained_since_last_frame", 0)
            record["enemy_kills_since_last_frame"] = base.get("enemy_kills_since_last_frame", 0)
            record["hp_lost_since_last_frame"] = base.get("hp_lost_since_last_frame", 0)
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
            step_profile["diary_write_ms"] = self._diary.last_persist_ms
            self._update_playthrough(scene=str(record.get("scene_name") or "") or None,
                                     step=self._steps + 1, events=journey_events)
        step_profile["journey_and_observation_ms"] = (time.perf_counter() - started) * 1000
        if base.get("valid") is not True:
            self._done = True
            self.environment.backend.release_all()
            raise InvalidTrainingTransition(
                f"operationally invalid transition: {base.get('termination_reason')}"
            )
        started = time.perf_counter()
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
            "tutorial_reward": 0.0,
            "challenge_reward": 0.0,
            "stuck_pressure": 0.0,
            "stuck_escape_reward": 0.0,
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
        telemetry_for_position = base.get("operator_telemetry")
        position = (
            _player_position(telemetry_for_position)
            if isinstance(telemetry_for_position, Mapping) else None
        )
        current_scene = base.get("scene_name")
        if self.config.continuous_playthrough:
            if isinstance(current_scene, str) and position is not None:
                cell_key = hashlib.sha256((
                    f"position:{current_scene}:{math.floor(position[0] / 4.0)}:"
                    f"{math.floor(position[1] / 4.0)}"
                ).encode()).hexdigest()
                novelty_hash = cell_key
                eligible = eligible and self._lifelong_visits.get(cell_key, 0) == 0
            else:
                # Without reliable player coordinates, visual changes alone
                # are insufficient evidence that the agent reached a new place.
                eligible = False
        if isinstance(current_scene, str) and position is not None:
            origin = self._scene_origins.setdefault(current_scene, position)
            room_progress_distance = math.hypot(
                position[0] - origin[0], position[1] - origin[1]
            )
            self._scene_max_progress[current_scene] = max(
                self._scene_max_progress.get(current_scene, 0.0), room_progress_distance
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
        novelty_remaining = self._remaining_budget(
            "novelty", self.config.novelty_budget, self._novelty_paid
        )
        if eligible and novelty_remaining > 0:
            visits = self._lifelong_visits.get(novelty_hash, 0) if novelty_hash else 0
            lifetime_factor = 1.0 / float(np.sqrt(1 + visits)) if novelty_hash else 1.0
            components["novelty"] = min(
                self.config.novelty_reward * lifetime_factor,
                novelty_remaining,
            )
            self._novelty_paid += components["novelty"]
            self._record_continuous_budget("novelty", components["novelty"])
            self._archive.append(descriptor.copy())
            if novelty_hash is not None:
                self._lifelong_visits[novelty_hash] = visits + 1
                if len(self._lifelong_visits) > self.config.lifelong_archive_max:
                    del self._lifelong_visits[next(iter(self._lifelong_visits))]
                self._lifelong_dirty_steps += 1
                if self._lifelong_dirty_steps >= 32:
                    self._flush_lifelong_visits()
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
                    remaining = self._remaining_budget(
                        "scene_change", self.config.scene_reward_budget, self._scene_paid
                    )
                    components["scene_change"] = min(requested_reward, remaining)
                    if components["scene_change"] > 0:
                        self._scene_paid += components["scene_change"]
                        self._record_continuous_budget("scene_change", components["scene_change"])
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
            remaining_kill_budget = self._remaining_budget(
                "confirmed_enemy_kills", self.config.enemy_kill_reward_budget, self._kills_paid
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
                requested = enemy_kill_reward_for_visit(
                    visits, continuous=self.config.continuous_playthrough
                )
                award = min(requested, remaining_kill_budget)
                components["confirmed_enemy_kills"] += award
                remaining_kill_budget -= award
                # Lifetime decay tracks every confirmed kill, including kills
                # that landed while the segment/window budget was exhausted.
                self._enemy_visits[key] = visits + 1
                self._enemy_visits_dirty = True
        if self.config.reward_version in {"explore_v3", "explore_v4"}:
            components["confirmed_enemy_kills"] = min(
                components["confirmed_enemy_kills"],
                self._remaining_budget(
                    "confirmed_enemy_kills", self.config.enemy_kill_reward_budget, self._kills_paid
                ),
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
            remaining = self._remaining_budget("milestone_novelty", 20.0, self._milestone_paid)
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
                self._record_continuous_budget("milestone_novelty", award)
                remaining -= award
                self._milestone_counts[key] = visits + 1
        if self.config.reward_version == "explore_v4":
            tutorial_values = {
                "objective_completed": 2.0,
                "first_target_arrival": 0.5,
                "first_bench_rest": 0.25,
                "station_opened": 0.5,
                "first_fast_travel": 0.75,
            }
            for event in tutorial_events:
                if not isinstance(event, dict):
                    continue
                requested = tutorial_values.get(str(event.get("type")), 0.0)
                award = min(
                    requested,
                    self._remaining_budget("tutorial_reward", 20.0, self._milestone_paid),
                )
                components["tutorial_reward"] += award
                self._milestone_paid += award
                self._record_continuous_budget("tutorial_reward", award)
            challenge_hits = 0
            survival_record = False
            challenge_success = False
            for event in tutorial_events:
                if not isinstance(event, dict):
                    continue
                if event.get("event") == "challenge_hit":
                    delta = event.get("hit_delta", 0)
                    if isinstance(delta, int) and not isinstance(delta, bool) and delta > 0:
                        challenge_hits += min(delta, 3)
                elif event.get("event") == "challenge_attempt":
                    survival_record |= event.get("survival_improvement") is True
                    challenge_success |= event.get("success") is True
            challenge_components = challenge_reward(
                kind="boss", hit_delta=min(challenge_hits, 3),
                survival_improvement=survival_record, success=challenge_success,
            )
            components["challenge_reward"] = sum(challenge_components.values())
        self._kills_paid += components["confirmed_enemy_kills"]
        self._record_continuous_budget(
            "confirmed_enemy_kills", components["confirmed_enemy_kills"]
        )
        shade_recovered_remaining = self._remaining_budget(
            "shade_recovered", self.config.shade_recovery_reward,
            self.config.shade_recovery_reward if self._shade_paid else 0.0,
        )
        components["shade_recovered"] = min(
            0.0 if self._shade_paid else combat["shade_recovered"], shade_recovered_remaining
        )
        self._record_continuous_budget("shade_recovered", components["shade_recovered"])
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
                self._remaining_budget("shade_hits", hit_budget, self._shade_hits_paid),
            )
            self._shade_hits_paid += components["shade_hits"]
            self._record_continuous_budget("shade_hits", components["shade_hits"])
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
                self._remaining_budget("hp_restored", hp_budget, self._hp_restored_paid),
            )
            self._hp_restored_paid += components["hp_restored"]
            self._record_continuous_budget("hp_restored", components["hp_restored"])
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
                                    self._remaining_budget(
                                        "shade_approach_positive", self.config.shade_approach_budget,
                                        self._shade_approach_positive_paid,
                                    ),
                                ),
                            )
                            self._shade_approach_positive_paid += approach
                        elif approach < 0:
                            approach = max(
                                approach,
                                -max(
                                    0.0,
                                    self._remaining_budget(
                                        "shade_approach_negative", self.config.shade_approach_budget,
                                        self._shade_approach_negative_paid,
                                    ),
                                ),
                            )
                            self._shade_approach_negative_paid += -approach
                    self._record_continuous_budget(
                        "shade_approach_positive" if approach > 0 else "shade_approach_negative",
                        abs(approach),
                    )
                    components["shade_approach"] = approach
                self._shade_distance = (scene, current_distance)
            else:
                self._shade_distance = None
            if base.get("shade_pending") is True:
                components["time_cost"] -= 0.001 * (lease / 1000.0)
        if (
            self.config.reward_version in {"explore_v3", "explore_v4"}
            and not self.config.continuous_playthrough
        ):
            payable_time_cost = min(
                abs(components["time_cost"]),
                max(0.0, self.config.time_cost_budget - self._time_cost_paid),
            )
            components["time_cost"] = -payable_time_cost
            self._time_cost_paid += payable_time_cost
        geo = combat["confirmed_geo_gained"] + combat["confirmed_geo_spent"]
        geo_award = min(
            geo, self._remaining_budget("geo", self.config.geo_reward_budget, self._geo_paid)
        )
        if geo > 0:
            scale = geo_award / geo
            components["confirmed_geo_gained"] = combat["confirmed_geo_gained"] * scale
            components["confirmed_geo_spent"] = combat["confirmed_geo_spent"] * scale
            self._geo_paid += geo_award
            self._record_continuous_budget("geo", geo_award)
        components["damage_taken"] = combat["confirmed_hp_lost"]
        if self.config.reward_version in {"explore_v3", "explore_v4"}:
            raw_damage = int(base.get("hp_lost_since_last_frame", 0))
            if raw_damage < 0:
                raise InvalidTrainingTransition("invalid confirmed damage delta")
            payable_damage = min(
                float(raw_damage), self._remaining_budget("damage_health", 9.0, self._damage_paid_health)
            )
            components["damage_taken"] = -payable_damage * abs(self.config.damage_penalty)
            self._damage_paid_health += payable_damage
            self._record_continuous_budget("damage_health", payable_damage)
        components["death"] = combat["player_death"]
        components["run_failed"] = combat["gameplay_run_failed"]
        if components["death"] < 0:
            self._episode_deaths += 1
        if components["run_failed"] < 0:
            self._episode_shade_lost += 1
        current_cell = self._player_knowledge.last_cell
        if self._help_request_ledger is not None and isinstance(current_scene, str) and current_cell is not None:
            place_stuck_count = self._help_request_ledger.stuck_count(
                scene=current_scene, cell=current_cell
            )
            if place_stuck_count >= 3:
                components["stuck_pressure"] = -min(
                    0.005, abs(self.config.time_cost) * (lease / self.config.lease_base_ms) * 2.0
                )
            prior_place_stuck_count = (
                self._help_request_ledger.stuck_count(scene=prior_scene, cell=prior_cell)
                if isinstance(prior_scene, str) and prior_cell is not None else 0
            )
            if (
                prior_place_stuck_count >= 3 and prior_scene == current_scene and prior_cell is not None
                and current_cell != prior_cell
                and abs(current_cell[0] - prior_cell[0]) + abs(current_cell[1] - prior_cell[1]) == 1
                and current_cell not in prior_visited
            ):
                components["stuck_escape_reward"] = self._help_request_ledger.reward_new_neighbor(
                    scene=current_scene, cell=current_cell
                )
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
                "stuck_escape_reward",
            ):
                components[name] = 0.0
        self._commit_continuous_budget(components)
        if self._enemy_visits_dirty and self._steps % 32 == 0:
            self._flush_enemy_visits()
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
        # A stuck PPO slice ends without reloading the last bench. The live
        # game session preserves unsaved player progress.
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
        help_request: dict[str, Any] | None = None
        if self._help_request_ledger is not None:
            current_progress = self._scene_max_progress.get(current_scene, 0.0) if isinstance(current_scene, str) else 0.0
            if isinstance(current_scene, str):
                self._help_request_ledger.observe_progress(scene=current_scene, progress=current_progress)
            if termination_reason == "stuck" and isinstance(current_scene, str):
                cell = self._player_knowledge.last_cell
                if cell is not None:
                    thumbnail_path: str | None = None
                    if self._journey_root is not None:
                        try:
                            from PIL import Image

                            thumbnail_root = self._journey_root / "help-thumbnails"
                            thumbnail_root.mkdir(parents=True, exist_ok=True)
                            thumbnail = thumbnail_root / f"slot-{self._journey_actor}-{time.time_ns()}.png"
                            Image.fromarray(np.asarray(observation["pixels"][-1], dtype=np.uint8)).save(thumbnail)
                            thumbnail_path = str(thumbnail)
                        except (OSError, ValueError):
                            pass
                    help_request = self._help_request_ledger.record_stuck(
                        scene=current_scene, cell=cell, progress=current_progress,
                        thumbnail_path=thumbnail_path,
                    )
        stuck_location: dict[str, Any] | None = None
        if self._help_request_ledger is not None and isinstance(current_scene, str) and current_cell is not None:
            place = f"{current_scene}:c{current_cell[0]}r{current_cell[1]}"
            stuck_location = next(
                (row for row in self._help_request_ledger.stuck_locations() if row.get("place") == place),
                None,
            )
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
                "max_progress_by_scene": dict(self._scene_max_progress),
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
            "max_progress_by_scene": dict(self._scene_max_progress),
            "task_action": template.name,
            "scene_detector_audit": changed,
            "farm_alerts": farm_alerts,
            "player_knowledge_events": tutorial_events,
            "challenge_practice": (
                self._challenge_practice_controller.status()
                if self._challenge_practice_controller is not None else None
            ),
            "challenge_practice_finished": self._challenge_practice_finished,
            "help_request": help_request,
            "stuck_location": stuck_location,
            "pending_help_requests": (
                [row for row in self._help_request_ledger.requests() if row.get("status") == "pending"]
                if self._help_request_ledger is not None else []
            ),
        }
        timing_profile = base.get("timing_ms")
        if isinstance(timing_profile, Mapping):
            info["timing_ms"] = dict(timing_profile)
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
                "actor_step_profile_ms": step_profile,
                "player_knowledge_events": tutorial_events,
                **audit,
            }
        )
        step_profile["reward_and_log_ms"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        observation = self._apply_player_knowledge(observation)
        step_profile["knowledge_observation_ms"] = (time.perf_counter() - started) * 1000
        info["actor_step_profile_ms"] = step_profile
        return observation, reward, task_terminated, task_truncated, info

    def _apply_player_knowledge(
        self, observation: dict[str, NDArray[np.generic]]
    ) -> dict[str, NDArray[np.generic]]:
        result = dict(observation)
        if self.config.reward_version == "explore_v4":
            result["player_knowledge"] = self._player_knowledge.vector(
                hint_level=self.config.tutorial_hint_level
            )
        return result

    def _capture_challenge_snapshots(
        self, events: list[dict[str, Any]], info: dict[str, Any]
    ) -> None:
        if not self._challenge_practice_enabled:
            return
        if self._challenge_save_root is None or self._challenge_snapshot_root is None or self._personal_save_root is None:
            return
        requests: list[tuple[str, str, str, dict[str, Any] | None]] = []
        for event in events:
            if event.get("event") == "challenge_started" and event.get("snapshot_requested") is True:
                key, kind = event.get("challenge_id"), event.get("kind")
                if isinstance(key, str) and kind in {"boss", "traversal", "item", "area"}:
                    requests.append((key, cast(str, kind), "arena_entrance" if kind == "boss" else "challenge_room", event))
        raw_milestones = info.get("milestone_events_since_last_frame", [])
        if isinstance(raw_milestones, list) and any(
            isinstance(item, dict) and item.get("category") == "checkpoint"
            and item.get("confirmed") is True for item in raw_milestones
        ):
            requests.append(("last-bench", "area", "last_bench", None))
        for challenge_id, kind, source, snapshot_event in requests:
            try:
                snapshot = create_challenge_snapshot(
                    challenge_id=challenge_id, kind=cast(Any, kind), source=cast(Any, source),
                    save_root=self._challenge_save_root, snapshot_root=self._challenge_snapshot_root,
                    personal_root=self._personal_save_root, max_snapshots=32,
                )
                entry = {
                    "challenge_id": challenge_id, "kind": kind, "source": source,
                    "save_path": snapshot.save_path, "created_ns": snapshot.created_ns,
                }
                self._challenge_snapshot_index.append(entry)
                del self._challenge_snapshot_index[:-32]
                if snapshot_event is not None:
                    snapshot_event["snapshot_path"] = snapshot.save_path
                index_path = self._challenge_snapshot_root / "index.json"
                index_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = index_path.with_name(f".{index_path.name}.{os.getpid()}.tmp")
                temporary.write_text(json.dumps({"schema_version": 1, "snapshots": self._challenge_snapshot_index}, allow_nan=False), encoding="utf-8", newline="\n")
                os.replace(temporary, index_path)
            except (OSError, ValueError, RuntimeError) as error:
                if snapshot_event is not None:
                    snapshot_event["snapshot_error"] = type(error).__name__

    def _record_challenge_practice_events(self, events: list[dict[str, Any]]) -> None:
        controller = self._challenge_practice_controller
        if controller is None or self._challenge_practice_finished:
            return
        for event in events:
            if event.get("event") != "challenge_attempt":
                continue
            challenge_id = event.get("challenge_id")
            kind = event.get("kind")
            version = event.get("version", "unknown")
            snapshot = controller.snapshot
            if snapshot is None or challenge_id != snapshot.challenge_id or kind != snapshot.kind:
                continue
            if not isinstance(challenge_id, str) or not isinstance(kind, str) or not isinstance(version, str):
                continue
            attempt = ChallengeAttempt(
                challenge_id=challenge_id, kind=cast(Any, kind), version=version,
                duration_seconds=float(event.get("duration_seconds", 0.0) or 0.0),
                hits=int(event.get("hits", 0) or 0),
                progress=float(event.get("progress", 0.0) or 0.0),
                success=event.get("success") is True,
            )
            if controller.record_attempt(attempt):
                self._challenge_practice_finished = True
                break
        if controller.should_exit():
            controller.active = False
            self._challenge_practice_finished = True

    def _persist_player_knowledge(self, *, force: bool = False) -> None:
        if self._knowledge_path is None:
            return
        current_step = int(getattr(self, "_steps", 0))
        last_persist_step = int(getattr(self, "_last_knowledge_persist_step", -32))
        if not force and current_step - last_persist_step < 32:
            return
        self._knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        pending = self._knowledge_path.with_name(
            f".{self._knowledge_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        started = time.perf_counter()
        try:
            with pending.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(
                    json.dumps(
                        self._player_knowledge.to_payload(),
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                stream.flush()
                os.fsync(stream.fileno())
            for attempt in range(20):
                try:
                    os.replace(pending, self._knowledge_path)
                    break
                except PermissionError:
                    if os.name != "nt" or attempt == 19:
                        raise
                    time.sleep(0.025)
            self._last_knowledge_persist_step = current_step
            self._last_knowledge_persist_ms = (time.perf_counter() - started) * 1000
        finally:
            pending.unlink(missing_ok=True)

    def _update_playthrough(self, *, scene: str | None, step: int, events: list[dict[str, Any]]) -> None:
        root_value = os.environ.get("HK_AGENT_PLAYTHROUGH_ROOT")
        if not root_value:
            return
        checkpoint = next((event.get("key") for event in events
                           if event.get("category") == "checkpoint" and isinstance(event.get("key"), str)), None)
        now = time.monotonic()
        if checkpoint is None and now - self._last_playthrough_update < 5.0:
            return
        try:
            from hk_agent.playthrough import update_playthrough_progress

            update_playthrough_progress(Path(root_value), scene=scene,
                                        progress={"step": step, "checkpoint": checkpoint})
            self._last_playthrough_update = now
        except (OSError, ValueError, RuntimeError):
            # Progress display is best-effort; the save transaction remains authoritative.
            return

    def action_masks(self) -> NDArray[np.bool_]:
        return build_action_mask(
            self._action_names, getattr(self.environment, "latest_action_availability", None)
        )

    def _load_continuous_budget(self, path: Path) -> dict[str, Any]:
        empty = {"schema_version": 2, "game_steps": 0, "game_time_ms": 0,
                 "segment_paid": {}, "buckets": {}}
        if not path.is_file():
            return dict(empty)
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            self._backup_and_reset_continuous_budget(path, empty)
            return dict(empty)
        if not isinstance(payload, dict) or payload.get("schema_version") != 2:
            self._backup_and_reset_continuous_budget(path, empty)
            return dict(empty)
        game_steps = payload.get("game_steps", 0)
        game_time_ms = payload.get("game_time_ms", 0)
        segment = payload.get("segment_paid", {})
        buckets = payload.get("buckets", {})
        if not isinstance(game_steps, int) or isinstance(game_steps, bool) or game_steps < 0:
            self._backup_and_reset_continuous_budget(path, empty)
            return dict(empty)
        if not isinstance(game_time_ms, int) or isinstance(game_time_ms, bool) or game_time_ms < 0:
            self._backup_and_reset_continuous_budget(path, empty)
            return dict(empty)
        return {
            "schema_version": 2,
            "game_steps": game_steps,
            "game_time_ms": game_time_ms,
            "segment_paid": {str(k): float(v) for k, v in segment.items()
                              if isinstance(k, str) and isinstance(v, (int, float)) and math.isfinite(v) and v >= 0}
            if isinstance(segment, dict) else {},
            "buckets": {str(k): {str(name): float(value) for name, value in row.items()
                                 if isinstance(name, str) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0}
                        for k, row in buckets.items() if isinstance(k, str) and isinstance(row, dict)}
            if isinstance(buckets, dict) else {},
        }

    @staticmethod
    def _backup_and_reset_continuous_budget(path: Path, empty: Mapping[str, Any]) -> None:
        try:
            backup = path.with_name(f"{path.stem}.backup-{time.time_ns()}{path.suffix}")
            backup.write_bytes(path.read_bytes())
            atomic_json(path, empty)
        except OSError:
            # The caller still starts from clean in-memory budgets; a writable
            # memory directory will persist the migrated state on the first save.
            return

    def _advance_continuous_budget_step(self, lease_ms: int = 0) -> None:
        if not self.config.continuous_playthrough:
            return
        self._continuous_budget_state["game_steps"] = int(self._continuous_budget_state.get("game_steps", 0)) + 1
        self._continuous_budget_state["game_time_ms"] = int(
            self._continuous_budget_state.get("game_time_ms", 0)
        ) + max(0, lease_ms)
        current_bucket = int(self._continuous_budget_state["game_time_ms"]) // 60_000
        buckets = self._continuous_budget_state.setdefault("buckets", {})
        cutoff = current_bucket - 9
        for key in list(buckets):
            if key.isdigit() and int(key) < cutoff:
                del buckets[key]
        self._continuous_budget_dirty_steps += 1

    def _remaining_budget(self, name: str, budget: float, episode_paid: float) -> float:
        if not self.config.continuous_playthrough:
            return max(0.0, budget - episode_paid)
        segment = self._continuous_budget_state.get("segment_paid", {})
        buckets = self._continuous_budget_state.get("buckets", {})
        current = int(self._continuous_budget_state.get("game_time_ms", 0)) // 60_000
        window_paid = sum(
            float(row.get(name, 0.0)) for key, row in buckets.items()
            if key.isdigit() and current - 9 <= int(key) <= current and isinstance(row, dict)
        )
        segment_paid = float(segment.get(name, 0.0)) if isinstance(segment, dict) else 0.0
        pending = self._pending_continuous_budget.get(name, 0.0)
        return max(0.0, min(
            budget - episode_paid - pending,
            budget - segment_paid - pending,
            budget - window_paid - pending,
        ))

    def _record_continuous_budget(self, name: str, amount: float) -> None:
        if not self.config.continuous_playthrough or amount <= 0:
            return
        self._pending_continuous_budget[name] = self._pending_continuous_budget.get(name, 0.0) + amount

    def _commit_continuous_budget(self, components: Mapping[str, float]) -> None:
        if not self.config.continuous_playthrough:
            return
        allowed = {
            "novelty": components.get("novelty", 0.0) > 0,
            "scene_change": components.get("scene_change", 0.0) > 0,
            "confirmed_enemy_kills": components.get("confirmed_enemy_kills", 0.0) > 0,
            "milestone_novelty": components.get("milestone_novelty", 0.0) > 0,
            "tutorial_reward": components.get("tutorial_reward", 0.0) > 0,
            "shade_hits": components.get("shade_hits", 0.0) > 0,
            "shade_recovered": components.get("shade_recovered", 0.0) > 0,
            "hp_restored": components.get("hp_restored", 0.0) > 0,
            "shade_approach_positive": components.get("shade_approach", 0.0) > 0,
            "shade_approach_negative": components.get("shade_approach", 0.0) < 0,
            "time_cost": False,
            "geo": components.get("confirmed_geo_gained", 0.0) + components.get("confirmed_geo_spent", 0.0) > 0,
            "damage_health": components.get("damage_taken", 0.0) < 0,
        }
        for name, amount in self._pending_continuous_budget.items():
            if not allowed.get(name, False):
                continue
            segment = self._continuous_budget_state.setdefault("segment_paid", {})
            segment[name] = float(segment.get(name, 0.0)) + amount
            current = int(self._continuous_budget_state.get("game_time_ms", 0)) // 60_000
            buckets = self._continuous_budget_state.setdefault("buckets", {})
            row = buckets.setdefault(str(current), {})
            row[name] = float(row.get(name, 0.0)) + amount
        self._pending_continuous_budget.clear()
        self._save_continuous_budget()

    def _save_continuous_budget(self, *, force: bool = False) -> None:
        if self._continuous_budget_path is None:
            return
        if not force and self._continuous_budget_dirty_steps < 32:
            return
        atomic_json(self._continuous_budget_path, self._continuous_budget_state)
        self._continuous_budget_dirty_steps = 0

    def set_evaluation_mode(self, enabled: bool) -> None:
        self._evaluation_mode = bool(enabled)

    def _flush_lifelong_visits(self) -> None:
        if self._lifelong_path is None or self._lifelong_dirty_steps == 0:
            return
        save_lifelong_visits(self._lifelong_path, self._lifelong_visits)
        self._lifelong_dirty_steps = 0

    def _flush_enemy_visits(self) -> None:
        if self._campaign_enemy_path is None or not self._enemy_visits_dirty:
            return
        save_campaign_counts(
            self._campaign_enemy_path, self._enemy_visits, self.config.lifelong_archive_max
        )
        self._enemy_visits_dirty = False

    def close(self) -> None:
        try:
            self._persist_player_knowledge(force=True)
            self._flush_transition_log()
            self._flush_lifelong_visits()
            self._flush_enemy_visits()
            self._save_continuous_budget(force=True)
            if self._diary is not None:
                self._diary.flush()
        finally:
            self.environment.close()

    def task_contract(self) -> dict[str, Any]:
        schema_version = {"explore_v1": 1, "explore_v2": 2, "explore_v3": 3, "explore_v4": 5}[
            self.config.reward_version
        ]
        if self.config.reward_version == "explore_v4":
            schema_version = 9
        return {
            "task_contract_schema_version": schema_version,
            "reward_version": self.config.reward_version,
            "legal_action_mask": {
                "version": "action_availability_v4" if schema_version >= 8 else "action_availability_v3" if schema_version >= 6 else "action_availability_v2" if schema_version == 5 else "action_availability_v1",
                "source": "frame_action_availability",
            },
            "action_names": list(self._action_names),
            "actions": action_contract(
                self._action_templates, base_lease_ms=self.environment.lease_ms
            ),
            "config": {
                key: value for key, value in asdict(self.config).items()
                if key != "reward_version"
                and key != "continuous_playthrough"
                and (key != "tutorial_hint_level" or self.config.reward_version == "explore_v4")
            },
            "observation_version": "pixels_action_history_player_knowledge_charms_v1" if schema_version >= 8 else "pixels_action_history_player_knowledge_bestiary_v1" if schema_version == 7 else "pixels_action_history_player_knowledge_tutorials_v2" if schema_version == 6 else "pixels_action_history_player_knowledge_v1" if schema_version == 5 else "pixels_action_history_v1",
            "information_boundary": ["pixels", "applied_action_history", "player_knowledge"] if schema_version >= 5 else ["pixels", "applied_action_history"],
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
            **({"recurrent_policy": {"name": "maskable_recurrent_v1", "hidden_size": 256, "layers": 1, "sequence_length": 128}} if schema_version >= 8 else {}),
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
        reward_row = {name: value for name, value in components.items() if value > 0}
        if len(self._recent_reward_window) == self._recent_reward_window.maxlen:
            expired = self._recent_reward_window[0]
            for name, value in expired.items():
                self._recent_reward_totals[name] -= value
                self._recent_positive_total -= value
                if self._recent_reward_totals[name] <= 1e-12:
                    del self._recent_reward_totals[name]
        self._recent_reward_window.append(reward_row)
        for name, value in reward_row.items():
            self._recent_reward_totals[name] = self._recent_reward_totals.get(name, 0.0) + value
            self._recent_positive_total += value
        novelty_seen = components.get("novelty", 0.0) > 0 or components.get("scene_change", 0.0) > 0
        if not novelty_seen:
            pair = (str(scene or "unknown"), action)
            if len(self._recent_no_novelty_pairs) == self._recent_no_novelty_pairs.maxlen:
                expired_pair = self._recent_no_novelty_pairs.popleft()
                self._recent_no_novelty_counts[expired_pair] -= 1
                if self._recent_no_novelty_counts[expired_pair] <= 0:
                    del self._recent_no_novelty_counts[expired_pair]
            self._recent_no_novelty_pairs.append(pair)
            self._recent_no_novelty_counts[pair] += 1
        triggers: list[str] = []
        if (
            len(self._recent_reward_window) == 2000
            and self._recent_positive_total > 0
            and max(self._recent_reward_totals.values(), default=0.0)
            / self._recent_positive_total > 0.60
        ):
            triggers.append("component_over_60_percent")
            dominant = max(
                self._recent_reward_totals, key=lambda name: self._recent_reward_totals[name]
            )
            triggers.append(f"farm_component:{dominant}")
        if (
            len(self._recent_no_novelty_pairs) == 2000
            and max(self._recent_no_novelty_counts.values(), default=0) / 2000 > 0.40
        ):
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
        payload = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
        self._pending_log_records.append(payload)
        force = bool(record.get("farm_alerts")) or record.get("terminated") is True or record.get("truncated") is True
        if len(self._pending_log_records) >= 32 or force or time.monotonic() - self._last_log_flush >= 2.0:
            self._flush_transition_log()

    def _flush_transition_log(self) -> None:
        if self.task_log_path is None or not self._pending_log_records:
            return
        self.task_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = b"".join(self._pending_log_records)
        fd = os.open(self.task_log_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            if os.write(fd, payload) != len(payload):
                raise OSError("partial task JSONL append")
            os.fsync(fd)
            self._pending_log_records.clear()
            self._last_log_flush = time.monotonic()
        finally:
            os.close(fd)


def _player_position(value: Mapping[str, Any]) -> tuple[float, float] | None:
    x, y = value.get("x"), value.get("y")
    if (isinstance(x, bool) or isinstance(y, bool)
            or not isinstance(x, (int, float)) or not isinstance(y, (int, float))
            or not math.isfinite(float(x)) or not math.isfinite(float(y))):
        return None
    return float(x), float(y)


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
