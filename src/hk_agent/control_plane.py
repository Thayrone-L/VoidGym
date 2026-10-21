from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from hk_agent.bridge_environment import clean_bridge_environment
from hk_agent.playthrough import (
    backup_playthrough,
    initialize_playthrough,
    persist_playthrough,
    playthrough_status,
    save_signature,
)
from hk_agent.playthrough import playthrough_root as resolve_playthrough_root
from hk_agent.runtime_paths import (
    capture_instance_saves,
    ensure_inside_runtime,
    prepare_instance_saves,
    validate_runtime_root,
)
from hk_agent.win_job import WindowsJob, process_identity, resume_process

_REAL_POPEN = subprocess.Popen

OWNER_LOCK_NAME = ".control-plane.owner.json"


class ControlPlaneBusyError(RuntimeError):
    """Another live process already owns this control plane."""


# Must match SceneChangeProbeConfig.validate (hk_agent.scene_change_probe).
PROBE_MAX_HORIZON = 10_000
PROBE_RUNNER_CAMPAIGNS = frozenset({"scene_change_probe", "parallel_smoke"})


def _windows_job_enabled() -> bool:
    return os.name == "nt"

CAMPAIGN_TO_TOOL = {
    "visual_motion_training": "run_first_training.py",
    "exploration_training": "run_first_training.py",
    "scene_change_probe": "run_scene_change_probe.py",
    "parallel_smoke": "run_scene_change_probe.py",
    "scene_change_training": "run_scene_change_training.py",
}
SCENE_CHANGE_GATE_TYPE = "scene_change_gate_review"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


def _bounded_int(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _bounded_float(value: object, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return result


def _map_view_has_valid_bounds(map_view: object) -> bool:
    bounds = map_view.get("scene_bounds") if isinstance(map_view, Mapping) else None
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return False
    values: list[float] = []
    for value in bounds:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        converted = float(value)
        if not math.isfinite(converted):
            return False
        values.append(converted)
    return values[2] > values[0] and values[3] > values[1]


def _preferred_map_capture(rows: Sequence[tuple[Path, object, int]]) -> tuple[Path, object, int]:
    bounded = [row for row in rows if _map_view_has_valid_bounds(row[1])]
    return max(bounded or rows, key=lambda row: row[2])


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text("utf-8").splitlines()
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _project_python_executable(project_root: Path) -> str:
    candidates = (
        project_root / ".venv" / "Scripts" / "python.exe",
        project_root / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


@dataclass(frozen=True, slots=True)
class CampaignSpec:
    campaign_id: str
    name: str
    campaign_type: str
    seed: int
    timesteps: int
    checkpoint_interval: int
    episodes: int
    horizon: int
    action_repeat: int
    lease_ms: int
    window_width: int = 960
    window_height: int = 540
    duration_mode: str = "limited"
    duration_limit_seconds: float = 0.0
    max_checkpoints: int = 5
    save_source: str = "existing_save"
    save_slot: int = 1
    worker_session_id: str | None = None
    weights: Mapping[str, float] = field(default_factory=dict)
    shared_learner: bool = False
    actors: int = 2
    rollout_steps: int = 256
    batch_size: int = 64
    ent_coef: float | None = None
    gamma: float | None = None
    gae_lambda: float | None = None
    n_epochs: int | None = None
    target_kl: float | None = None
    learning_rate: float | None = None
    clip_range: float | None = None
    reward_version: str = "explore_v3"
    warm_start_checkpoint: str | None = None
    warm_start_source_campaign: str | None = None
    stuck_seconds: float = 120.0
    evaluation_interval: int = 20
    tutorial_hint_level: int = 0
    straggler_timeout_seconds: float = 45.0
    challenge_practice_enabled: bool = False
    help_requests_enabled: bool = False
    help_stuck_count: int = 10
    help_stuck_minutes: float = 60.0
    help_requests_per_hour: int = 3
    help_failed_attempts: int = 100
    help_stalled_hours: float = 6.0
    schema_version: int = 11

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> CampaignSpec:
        campaign_type = str(payload.get("campaign_type", ""))
        if campaign_type not in CAMPAIGN_TO_TOOL:
            raise ValueError("campaign_type is not allowlisted")
        shared_learner = payload.get(
            "shared_learner",
            campaign_type == "exploration_training" and payload.get("reward_version", "explore_v3") in {"explore_v3", "explore_v4"},
        )
        if not isinstance(shared_learner, bool):
            raise TypeError("shared_learner must be a boolean")
        challenge_practice_enabled = payload.get("challenge_practice_enabled", False)
        help_requests_enabled = payload.get("help_requests_enabled", False)
        if not isinstance(challenge_practice_enabled, bool) or not isinstance(help_requests_enabled, bool):
            raise TypeError("challenge_practice_enabled and help_requests_enabled must be booleans")
        if (challenge_practice_enabled or help_requests_enabled) and campaign_type != "exploration_training":
            raise ValueError("challenge practice and help requests require exploration_training")
        if shared_learner and campaign_type not in {"visual_motion_training", "exploration_training"}:
            raise ValueError("shared learner requires a supported PPO training campaign")
        reward_version = payload.get("reward_version", "explore_v3")
        if campaign_type == "exploration_training" and reward_version not in {"explore_v1", "explore_v2", "explore_v3", "explore_v4"}:
            raise ValueError("unsupported exploration reward version")
        warm_start_checkpoint = payload.get("warm_start_checkpoint")
        warm_start_source_campaign = payload.get("warm_start_source_campaign")
        if warm_start_checkpoint is not None and (not isinstance(warm_start_checkpoint, str) or len(warm_start_checkpoint) > 2048):
            raise ValueError("warm_start_checkpoint must be a bounded path")
        if warm_start_source_campaign is not None and (not isinstance(warm_start_source_campaign, str) or len(warm_start_source_campaign) > 128):
            raise ValueError("warm_start_source_campaign must be a bounded campaign id")
        if campaign_type != "exploration_training":
            reward_version = "explore_v1"
        name = str(payload.get("name", "")).strip()
        if not 1 <= len(name) <= 80 or any(ord(char) < 32 for char in name):
            raise ValueError("name must contain 1 to 80 printable characters")
        identifier = str(payload.get("campaign_id", "")).strip().lower()
        if not identifier:
            identifier = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48]
        if not _IDENTIFIER.fullmatch(identifier):
            raise ValueError("campaign_id must be a lowercase slug")
        raw_weights = payload.get("weights", {})
        if not isinstance(raw_weights, Mapping):
            raise TypeError("weights must be an object")
        allowed_weights = (
            set()
            if campaign_type == "parallel_smoke"
            else
            {
                "visual_change_scale",
                "time_cost",
                "success_reward",
                "enemy_kill_reward",
                "enemy_kill_reward_budget",
                "shade_recovery_reward",
                "damage_penalty",
                "geo_gain_reward",
                "geo_spend_reward",
                "death_penalty",
                "run_failed_penalty",
            }
            if campaign_type == "visual_motion_training"
            else {
                "time_cost", "novelty_reward", "novelty_budget", "scene_change_reward",
                "enemy_kill_reward", "enemy_kill_reward_budget", "shade_recovery_reward",
                "damage_penalty", "geo_gain_reward", "geo_spend_reward", "death_penalty",
                "run_failed_penalty",
            }
            if campaign_type == "exploration_training"
            else {
                "time_cost",
                "novelty_shaping_budget",
                "novelty_reward",
                "novelty_budget",
                "scene_change_reward",
                "success_reward",
                "enemy_kill_reward",
                "enemy_kill_reward_budget",
                "shade_recovery_reward",
                "damage_penalty",
                "geo_gain_reward",
                "geo_spend_reward",
                "death_penalty",
                "run_failed_penalty",
            }
        )
        unknown = set(raw_weights) - allowed_weights
        if unknown:
            raise ValueError(f"unknown reward weights: {', '.join(sorted(unknown))}")
        weights = {
            str(key): _bounded_float(value, f"weights.{key}", -100.0, 100.0)
            for key, value in raw_weights.items()
        }
        if weights.get("time_cost", 0.0) > 0:
            raise ValueError("weights.time_cost cannot be positive")
        if weights.get("damage_penalty", 0.0) > 0:
            raise ValueError("weights.damage_penalty cannot be positive")
        if weights.get("death_penalty", 0.0) > 0:
            raise ValueError("weights.death_penalty cannot be positive")
        if weights.get("run_failed_penalty", 0.0) > 0:
            raise ValueError("weights.run_failed_penalty cannot be positive")
        if weights.get("enemy_kill_reward", 0.0) < 0:
            raise ValueError("weights.enemy_kill_reward cannot be negative")
        if weights.get("enemy_kill_reward_budget", 0.0) < 0:
            raise ValueError("weights.enemy_kill_reward_budget cannot be negative")
        if weights.get("shade_recovery_reward", 0.0) < 0:
            raise ValueError("weights.shade_recovery_reward cannot be negative")
        if weights.get("shade_recovery_reward", 0.0) > abs(weights.get("death_penalty", -2.0)):
            raise ValueError("weights.shade_recovery_reward cannot exceed death penalty magnitude")
        if weights.get("geo_gain_reward", 0.0) < 0:
            raise ValueError("weights.geo_gain_reward cannot be negative")
        if weights.get("geo_spend_reward", 0.0) < 0:
            raise ValueError("weights.geo_spend_reward cannot be negative")
        if any(weights.get(key, 1.0) <= 0 for key in ("visual_change_scale", "success_reward")):
            raise ValueError("positive reward weights must be greater than zero")
        novelty = weights.get("novelty_shaping_budget", 0.0)
        if not 0 <= novelty <= 1:
            raise ValueError("weights.novelty_shaping_budget must be in [0, 1]")
        success = weights.get("success_reward", 10.0)
        if campaign_type != "visual_motion_training" and success < 10 * novelty:
            raise ValueError("success_reward must dominate novelty shaping")
        if campaign_type != "visual_motion_training" and success < 4 * weights.get(
            "enemy_kill_reward", 0.0
        ):
            raise ValueError("success_reward must dominate combat shaping")
        save_source = payload.get(
            "save_source",
            "continuous_playthrough" if campaign_type == "exploration_training" else "existing_save",
        )
        if not isinstance(save_source, str) or save_source not in {
            "existing_save",
            "new_game",
            "continuous_playthrough",
        }:
            raise ValueError("save_source must be existing_save, new_game or continuous_playthrough")
        duration_mode = payload.get("duration_mode", "limited")
        if not isinstance(duration_mode, str) or duration_mode not in {
            "limited",
            "continuous",
        }:
            raise ValueError("duration_mode must be limited or continuous")
        duration_limit_seconds = _bounded_float(
            payload.get("duration_limit_seconds", 0.0), "duration_limit_seconds", 0.0, 72 * 3600.0
        )
        if campaign_type == "scene_change_probe" and duration_limit_seconds:
            raise ValueError("scene-change probes do not support a time limit")
        if campaign_type == "scene_change_probe" and duration_mode != "limited":
            raise ValueError("scene-change probes do not support continuous duration")
        if campaign_type == "parallel_smoke" and duration_mode != "limited":
            raise ValueError("parallel smoke does not support continuous duration")
        raw_worker_session = payload.get("worker_session_id")
        worker_session_id: str | None = None
        if raw_worker_session is not None and raw_worker_session != "":
            if not isinstance(raw_worker_session, str) or not _IDENTIFIER.fullmatch(
                raw_worker_session
            ):
                raise ValueError("worker_session_id must be a lowercase slug")
            if campaign_type == "scene_change_probe":
                raise ValueError("scene-change probes cannot consume a worker policy")
            worker_session_id = raw_worker_session
        actors = _bounded_int(payload.get("actors", 2), "actors", 1, 3)
        rollout_steps = _bounded_int(payload.get("rollout_steps", 512 if campaign_type == "exploration_training" else 256), "rollout_steps", 1, 100_000)
        batch_size = _bounded_int(payload.get("batch_size", 256 if campaign_type == "exploration_training" else 64), "batch_size", 1, 100_000)
        shared_checkpoint_default = (
            rollout_steps * actors if shared_learner
            else rollout_steps if campaign_type == "exploration_training"
            else 128
        )
        spec = cls(
            campaign_id=identifier,
            name=name,
            campaign_type=campaign_type,
            seed=_bounded_int(payload.get("seed", 7), "seed", 0, 2**31 - 1),
            timesteps=_bounded_int(payload.get("timesteps", 4096 if campaign_type == "exploration_training" else 256), "timesteps", 1, 10_000_000),
            checkpoint_interval=_bounded_int(
                payload.get("checkpoint_interval", shared_checkpoint_default), "checkpoint_interval", 1, 1_000_000
            ),
            episodes=_bounded_int(payload.get("episodes", 3), "episodes", 1, 10_000),
            horizon=_bounded_int(payload.get("horizon", 3000 if campaign_type == "exploration_training" else 512), "horizon", 8, 100_000),
            action_repeat=_bounded_int(payload.get("action_repeat", 3), "action_repeat", 1, 32),
            lease_ms=_bounded_int(payload.get("lease_ms", 80), "lease_ms", 20, 5_000),
            window_width=_bounded_int(payload.get("window_width", 960), "window_width", 320, 7680),
            window_height=_bounded_int(
                payload.get("window_height", 540), "window_height", 320, 7680
            ),
            duration_mode=duration_mode,
            duration_limit_seconds=duration_limit_seconds,
            max_checkpoints=_bounded_int(
                payload.get("max_checkpoints", 5), "max_checkpoints", 2, 100
            ),
            save_source=save_source,
            save_slot=_bounded_int(payload.get("save_slot", 1), "save_slot", 1, 4),
            worker_session_id=worker_session_id,
            weights=weights,
            shared_learner=shared_learner,
            actors=actors,
            rollout_steps=rollout_steps,
            batch_size=batch_size,
            ent_coef=_bounded_float(payload.get("ent_coef", 0.01), "ent_coef", 0.000001, 10.0) if campaign_type == "exploration_training" else None,
            gamma=_bounded_float(payload.get("gamma", 0.995), "gamma", 0.000001, 1.0) if campaign_type == "exploration_training" else None,
            gae_lambda=_bounded_float(payload.get("gae_lambda", 0.95), "gae_lambda", 0.000001, 1.0) if campaign_type == "exploration_training" else None,
            n_epochs=_bounded_int(payload.get("n_epochs", 4), "n_epochs", 1, 100) if campaign_type == "exploration_training" else None,
            target_kl=_bounded_float(payload.get("target_kl", 0.03), "target_kl", 0.000001, 10.0) if campaign_type == "exploration_training" else None,
            learning_rate=_bounded_float(payload.get("learning_rate", 0.00025), "learning_rate", 0.000001, 1.0) if campaign_type == "exploration_training" else None,
            clip_range=_bounded_float(payload.get("clip_range", 0.2), "clip_range", 0.000001, 1.0) if campaign_type == "exploration_training" else None,
            reward_version=str(reward_version),
            warm_start_checkpoint=warm_start_checkpoint,
            warm_start_source_campaign=warm_start_source_campaign,
            stuck_seconds=_bounded_float(payload.get("stuck_seconds", 120.0), "stuck_seconds", 1.0, 3600.0) if campaign_type == "exploration_training" else 120.0,
            evaluation_interval=_bounded_int(payload.get("evaluation_interval", 20), "evaluation_interval", 1, 10_000) if campaign_type == "exploration_training" else 20,
            tutorial_hint_level=_bounded_int(payload.get("tutorial_hint_level", 0), "tutorial_hint_level", 0, 1) if campaign_type == "exploration_training" else 0,
            straggler_timeout_seconds=_bounded_float(
                payload.get("straggler_timeout_seconds", 300.0),
                "straggler_timeout_seconds", 1.0, 300.0,
            ) if shared_learner else 45.0,
            challenge_practice_enabled=challenge_practice_enabled,
            help_requests_enabled=help_requests_enabled,
            help_stuck_count=_bounded_int(payload.get("help_stuck_count", 10), "help_stuck_count", 1, 1000),
            help_stuck_minutes=_bounded_float(payload.get("help_stuck_minutes", 60.0), "help_stuck_minutes", 5.0, 10080.0),
            help_requests_per_hour=_bounded_int(payload.get("help_requests_per_hour", 3), "help_requests_per_hour", 1, 100),
            help_failed_attempts=_bounded_int(payload.get("help_failed_attempts", 100), "help_failed_attempts", 1, 10000),
            help_stalled_hours=_bounded_float(payload.get("help_stalled_hours", 6.0), "help_stalled_hours", 0.1, 720.0),
        )
        if spec.rollout_steps * spec.actors % spec.batch_size:
            raise ValueError("batch_size must divide rollout_steps times actors")
        if spec.shared_learner and (
            spec.checkpoint_interval % (spec.rollout_steps * spec.actors)
        ):
            raise ValueError("shared checkpoint_interval must be a multiple of rollout_steps times actors")
        if spec.shared_learner and spec.rollout_steps % spec.batch_size:
            raise ValueError("batch_size must divide one actor rollout for straggler quorum")
        if spec.campaign_type == "visual_motion_training" and spec.horizon < 16:
            raise ValueError("visual-motion horizon must be at least 16 for calibration")
        if (
            spec.campaign_type in PROBE_RUNNER_CAMPAIGNS
            and spec.horizon > PROBE_MAX_HORIZON
        ):
            # Keep the panel's contract identical to the probe runner's own
            # validation, so a campaign can never be accepted here and then
            # crash the runner at startup.
            raise ValueError(
                f"horizon must be at most {PROBE_MAX_HORIZON} for {spec.campaign_type}"
            )
        return spec


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{secrets.token_hex(8)}.pending")
    try:
        pending.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        for attempt in range(20):
            try:
                os.replace(pending, path)
                break
            except PermissionError:
                if os.name != "nt" or attempt == 19:
                    raise
                time.sleep(0.01)
    finally:
        pending.unlink(missing_ok=True)


class ControlPlane:
    _MAX_ACTIVE_DRY_RUNS = 2

    def __init__(
        self,
        project_root: Path,
        *,
        allow_execute: bool = False,
        max_active_launches: int = 2,
        parallel_enabled: bool = False,
        runtime_root: Path | None = None,
        training_device: str = "cpu",
        torch_threads: int = 3,
        torch_interop_threads: int = 1,
        preparation_timeout_seconds: float = 600.0,
        monotonic_clock: Any = time.monotonic,
        observer: bool = False,
    ) -> None:
        if max_active_launches < 1:
            raise ValueError("max_active_launches must be positive")
        self.project_root = project_root.resolve()
        self.artifacts_root = self.project_root / "artifacts"
        self.control_root = self.artifacts_root / "control-plane"
        self.allow_execute = allow_execute
        self.max_active_launches = max_active_launches
        self.parallel_enabled = parallel_enabled
        self.runtime_root = (runtime_root or self.control_root / "runtime").resolve()
        self.training_device = training_device
        self.torch_threads = torch_threads
        self.torch_interop_threads = torch_interop_threads
        self.preparation_timeout_seconds = preparation_timeout_seconds
        self._monotonic_clock = monotonic_clock
        self.csrf_token = secrets.token_urlsafe(32)
        self._lock = threading.RLock()
        self._journey_cache_lock = threading.RLock()
        self._journey_cache: dict[str, Any] = {}
        self._launch_records_cache_lock = threading.RLock()
        self._launch_records_cache: tuple[float, dict[str, list[dict[str, Any]]]] = (0.0, {})
        self._campaigns_cache: tuple[float, tuple[tuple[str, int, int], ...], list[dict[str, object]]] = (0.0, (), [])
        self._reports_cache: tuple[float, tuple[tuple[str, int, int], ...], list[dict[str, object]]] = (0.0, (), [])
        self._checkpoint_candidate_cache: dict[tuple[bool, str | None], tuple[float, dict[str, object] | None]] = {}
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._jobs: dict[str, WindowsJob] = {}
        self._launches: dict[str, dict[str, object]] = {}
        self._reattached_pids: dict[str, int] = {}
        self._preparing_ids: set[str] = set()
        self._cancel_preparing: set[str] = set()
        self._preparing_started: dict[str, float] = {}
        self._preparing_paths: dict[str, tuple[Path, Path]] = {}
        self._preparing_watchdogs: dict[str, threading.Event] = {}
        self._slot_owners: dict[int, str] = {}
        self._shared_actor_restarts: dict[str, list[float]] = {}
        self._shared_actor_supervision: set[str] = set()
        self._shared_failed_learners: set[str] = set()
        self._slot_fences: dict[int, str] = {}
        self.observer = observer
        self._system_cpu_previous: tuple[int, int] | None = None
        self._map_art_lock = threading.Lock()
        self._map_art_thread: threading.Thread | None = None
        self._owner_lock_path = self.control_root / OWNER_LOCK_NAME
        if observer:
            # Read-only view: never recovers, rewrites, captures or reserves anything.
            return
        self._acquire_owner_lock()
        self._recover_active_launch()

    def _acquire_owner_lock(self) -> None:
        """Make this process the single owner of control_root.

        A second process that constructed a full ControlPlane used to run the
        "panel restarted" recovery against launches another live process still
        owned, marking them FAILED and capturing saves of running games. Now a live
        foreign owner makes construction fail; use observer=True for read-only
        status. Several instances inside one process (tests, restarts) are allowed,
        and a lock left by a dead process is taken over.
        """
        self.control_root.mkdir(parents=True, exist_ok=True)
        try:
            identity: dict[str, object] = dict(process_identity(os.getpid()))
        except OSError:
            identity = {"pid": os.getpid(), "creation_time": None, "executable": sys.executable}
        payload = json.dumps(identity, ensure_ascii=False).encode("utf-8")
        for _attempt in range(3):
            try:
                descriptor = os.open(
                    self._owner_lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
                )
            except FileExistsError:
                owner = self._read_owner_lock()
                if owner is not None and owner.get("pid") != os.getpid() and self._owner_alive(owner):
                    raise ControlPlaneBusyError(
                        f"control plane at {self.control_root} is owned by live process "
                        f"{owner.get('pid')}; open it with observer=True for read-only status"
                    ) from None
                # Same process or stale owner: replace atomically, then verify.
                temporary = self._owner_lock_path.with_name(
                    f"{self._owner_lock_path.name}.{secrets.token_hex(6)}.tmp"
                )
                temporary.write_bytes(payload)
                os.replace(temporary, self._owner_lock_path)
            else:
                try:
                    os.write(descriptor, payload)
                finally:
                    os.close(descriptor)
            current = self._read_owner_lock()
            if current is not None and current.get("pid") == os.getpid():
                return
        raise ControlPlaneBusyError(f"could not acquire control plane ownership at {self.control_root}")

    def _read_owner_lock(self) -> dict[str, object] | None:
        try:
            value = json.loads(self._owner_lock_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _owner_alive(owner: Mapping[str, object]) -> bool:
        pid = owner.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool):
            return False
        try:
            actual = process_identity(pid)
        except OSError:
            return False
        expected = owner.get("creation_time")
        return expected is None or actual.get("creation_time") == expected

    def close(self) -> None:
        """Release ownership if this process holds it."""
        if self.observer:
            return
        owner = self._read_owner_lock()
        if owner is not None and owner.get("pid") == os.getpid():
            try:
                self._owner_lock_path.unlink()
            except OSError:
                pass

    def _require_owner(self) -> None:
        if self.observer:
            raise PermissionError("this ControlPlane is a read-only observer")

    def _observer_snapshot(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        launch_root = self.control_root / "launches"
        if not launch_root.exists():
            return records
        for path in sorted(launch_root.glob("*/launch.json"), reverse=True):
            try:
                record = json.loads(path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(record, dict) and record.get("status") in {
                "PREPARING",
                "RUNNING",
                "STOP_REQUESTED",
            }:
                records.append(record)
        return records

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            process_query_limited_information = 0x1000
            still_active = 259
            handle = ctypes.windll.kernel32.OpenProcess(
                process_query_limited_information, False, pid
            )
            if not handle:
                return False
            exit_code = wintypes.DWORD()
            try:
                ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                return bool(ok) and exit_code.value == still_active
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        try:
            os.kill(pid, 0)
        except (OSError, ValueError):
            return False
        return True

    def _recover_active_launch(self) -> None:
        launch_root = self.control_root / "launches"
        if not launch_root.exists():
            return
        candidates = sorted(launch_root.glob("*/launch.json"), reverse=True)
        for path in candidates:
            try:
                record = json.loads(path.read_text("utf-8"))
                pid = record.get("pid")
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            if record.get("status") == "PREPARING":
                launch_id = str(record.get("launch_id", path.parent.name))
                record.update({"status": "FAILED", "failure": "painel reiniciado durante a preparação", "finished_unix_ns": time.time_ns()})
                atomic_json(path, record)
                runtime_path = ensure_inside_runtime(self.runtime_root / "launches" / launch_id, self.runtime_root)
                if runtime_path.exists():
                    quarantine = ensure_inside_runtime(self.runtime_root / "quarantine" / launch_id, self.runtime_root)
                    quarantine.parent.mkdir(parents=True, exist_ok=True)
                    if not quarantine.exists():
                        shutil.move(str(runtime_path), str(quarantine))
                continue
            if (
                record.get("status") not in {"RUNNING", "STOP_REQUESTED"}
                or isinstance(pid, bool)
                or not isinstance(pid, int)
            ):
                continue
            runner_identity = self._read_runner_identity(path.parent, record, pid)
            if isinstance(runner_identity, dict):
                record["runner_process"] = runner_identity
            if self._launch_process_matches(record, pid):
                resources = record.get("resources")
                valid_slot = isinstance(resources, Mapping) and isinstance(resources.get("slot_id"), int) and not isinstance(resources.get("slot_id"), bool)
                valid_dry_run = record.get("mode") == "dry-run" and isinstance(resources, Mapping) and resources.get("slot_id") is None
                if (record.get("mode") != "execute" and not valid_dry_run) or (
                    record.get("mode") == "execute" and not valid_slot
                ):
                    record["mode"] = "execute"
                    record["legacy_without_slot"] = True
                record["reattached"] = True
                launch_id = str(record.get("launch_id", path.parent.name))
                record["launch_id"] = launch_id
                self._launches[launch_id] = record
                reattached_identity = record.get("runner_process")
                self._reattached_pids[launch_id] = (
                    int(reattached_identity["pid"])
                    if isinstance(reattached_identity, Mapping)
                    and isinstance(reattached_identity.get("pid"), int)
                    else pid
                )
                if record.get("mode") == "execute" and isinstance(resources, Mapping) and valid_slot:
                    reattached_slot_id = int(resources["slot_id"])
                    self._slot_owners[reattached_slot_id] = launch_id
                    self._slot_fences[reattached_slot_id] = secrets.token_hex(16)
                    record["slot_fence_token"] = self._slot_fences[reattached_slot_id]
                atomic_json(path, record)
                deadline = record.get("deadline_unix_ns")
                if isinstance(deadline, int) and deadline <= time.time_ns():
                    self._request_deadline_stop_locked(launch_id, record)
                continue
            record.update(
                {
                    "status": (
                        "CANCELLED" if record.get("status") == "STOP_REQUESTED" else "FAILED"
                    ),
                    "failure": "processo não pertence mais a este launch",
                    "finished_unix_ns": time.time_ns(),
                }
            )
            resources = record.get("resources")
            if record.get("mode") == "execute" and isinstance(resources, Mapping):
                data_root = resources.get("instance_data_root")
                recovery_slot_id = resources.get("slot_id")
                if isinstance(data_root, str) and isinstance(recovery_slot_id, int) and not isinstance(recovery_slot_id, bool):
                    token = secrets.token_hex(16)
                    record["slot_fence_token"] = token
                    recovered_id = str(record.get("launch_id", path.parent.name))
                    self._slot_owners[recovery_slot_id] = recovered_id
                    self._slot_fences[recovery_slot_id] = token
                    try:
                        recovery_runtime_root = self.runtime_root / "launches" / recovered_id
                        recovery_runtime_root.mkdir(parents=True, exist_ok=True)
                        final_save = capture_instance_saves(
                            Path(data_root), recovery_runtime_root,
                            fence_token=token,
                            is_fence_current=self._fence_checker(recovery_slot_id),
                        )
                        record["final_save"] = str(final_save)
                        record["final_save_status"] = "captured"
                    except (OSError, ValueError, InterruptedError) as error:
                        record["final_save_status"] = "failed"
                        record["final_save_error"] = f"{type(error).__name__}: {error}"
                    finally:
                        self._release_slot_locked(recovered_id)
            record["slot_state"] = "released"
            atomic_json(path, record)

    def list_runs(self) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for area in ("training", "probes", "gates"):
            root = self.artifacts_root / area
            if not root.exists():
                continue
            for path in root.rglob("manifest.json"):
                try:
                    manifest = json.loads(path.read_text("utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(manifest, dict):
                    continue
                relative = path.relative_to(self.project_root)
                results.append(self._summarize_manifest(path, relative, manifest))
        return sorted(
            results,
            key=self._started_ns,
            reverse=True,
        )

    @staticmethod
    def _started_ns(item: Mapping[str, object]) -> int:
        value = item.get("started_unix_ns")
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    def _summarize_manifest(
        self, path: Path, relative: Path, manifest: Mapping[str, object]
    ) -> dict[str, object]:
        checkpoints = manifest.get("checkpoints", [])
        if not isinstance(checkpoints, list):
            checkpoints = []
        checkpoint_count = len(checkpoints)
        if "checkpoints" not in manifest:
            checkpoint_count = len(list((path.parent / "checkpoints").glob("*.zip")))
        failure = manifest.get("failure")
        episodes = manifest.get("episodes")
        return {
            "path": relative.as_posix(),
            "kind": manifest.get("artifact_type") or manifest.get("reward_version") or "training",
            "status": str(manifest.get("status", "UNKNOWN")),
            "gate_passed": manifest.get("gate_passed"),
            "started_unix_ns": manifest.get("started_unix_ns"),
            "finished_unix_ns": manifest.get("finished_unix_ns"),
            "timesteps": manifest.get("completed_timesteps") or manifest.get("total_timesteps"),
            "episodes": len(episodes) if isinstance(episodes, list) else None,
            "reward": manifest.get("reward_total") or manifest.get("total_reward"),
            "checkpoints": checkpoint_count,
            "failure": str(failure)[:500] if failure else None,
            "metrics": {
                key: manifest[key]
                for key in (
                    "positive_events",
                    "death_episodes",
                    "operational_failures",
                    "episode_resets_started",
                    "episode_count",
                    "gameplay_run_index",
                    "gameplay_run_failures",
                    "shade_pending",
                )
                if key in manifest
            },
        }

    def scene_change_gate(self) -> dict[str, object]:
        candidates: list[dict[str, object]] = []
        gate_root = self.artifacts_root / "gates" / "reach-scene-change-v1"
        if gate_root.exists():
            for gate_path in gate_root.glob("*.json"):
                try:
                    review = json.loads(gate_path.read_text("utf-8"))
                    evidence_relative = Path(str(review["evidence_probe"]))
                    evidence_path = (self.project_root / evidence_relative).resolve(strict=True)
                    evidence_path.relative_to((self.artifacts_root / "probes").resolve(strict=True))
                    evidence_bytes = evidence_path.read_bytes()
                    evidence = json.loads(evidence_bytes)
                except (OSError, KeyError, ValueError, json.JSONDecodeError):
                    continue
                if (
                    review.get("artifact_type") == SCENE_CHANGE_GATE_TYPE
                    and review.get("status") == "APPROVED"
                    and review.get("gate_passed") is True
                    and review.get("manual_reviewed") is True
                    and review.get("evidence_sha256") == hashlib.sha256(evidence_bytes).hexdigest()
                    and evidence.get("artifact_type") == "scene_change_probe"
                    and evidence.get("status") == "COMPLETED"
                    and isinstance(evidence.get("positive_events"), int)
                    and evidence["positive_events"] >= 1
                ):
                    candidates.append(
                        {
                            "path": gate_path.relative_to(self.project_root).as_posix(),
                            "evidence": evidence_relative.as_posix(),
                        }
                    )
        return {
            "passed": bool(candidates),
            "evidence": candidates[0]["evidence"] if candidates else None,
        }

    def list_campaigns(self) -> list[dict[str, object]]:
        campaigns: list[dict[str, object]] = []
        root = self.control_root / "campaigns"
        if not root.exists():
            return campaigns
        paths = tuple(sorted(root.glob("*/*.json")))
        signature = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size)
                          for path in paths if path.exists())
        cached_at, cached_signature, cached_campaigns = self._campaigns_cache
        if signature == cached_signature and self._monotonic_clock() - cached_at < 10.0:
            return [dict(row) for row in cached_campaigns]
        for path in paths:
            try:
                payload = json.loads(path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                payload["spec_path"] = path.relative_to(self.project_root).as_posix()
                campaign_id = payload.get("campaign_id")
                if payload.get("shared_learner") is True and isinstance(campaign_id, str):
                    checkpoint_root = self.runtime_root / "shared" / campaign_id / "checkpoints"
                    if not any(checkpoint_root.glob("round-*.zip")):
                        failures = [row for row in self._campaign_launch_records(campaign_id)
                                    if row.get("status") == "FAILED"]
                        if failures:
                            latest = max(failures, key=lambda row: int(row.get("finished_unix_ns", 0) or 0))
                            payload["operator_state"] = "not_started"
                            payload["operator_failure"] = str(latest.get("failure") or "Falha antes da primeira rodada")
                campaigns.append(payload)
        campaigns = sorted(campaigns, key=lambda item: str(item.get("created_at", "")), reverse=True)
        self._campaigns_cache = (self._monotonic_clock(), signature, campaigns)
        return [dict(row) for row in campaigns]

    def _operator_checkpoint_candidate(
        self, *, allow_unknown_algorithm: bool = False, campaign_id: str | None = None,
    ) -> dict[str, object] | None:
        cache_key = (allow_unknown_algorithm, campaign_id)
        cached = self._checkpoint_candidate_cache.get(cache_key)
        if cached and self._monotonic_clock() - cached[0] < 5.0:
            return dict(cached[1]) if cached[1] is not None else None
        requested_campaign_id = campaign_id
        candidates: list[tuple[int, int, int, dict[str, object]]] = []
        for campaign in self.list_campaigns():
            candidate_campaign_id = campaign.get("campaign_id")
            if (campaign.get("campaign_type") != "exploration_training"
                    or campaign.get("reward_version") not in {"explore_v3", "explore_v4"}
                    or campaign.get("shared_learner") is not True
                    or not isinstance(candidate_campaign_id, str)
                    or (requested_campaign_id is not None and candidate_campaign_id != requested_campaign_id)):
                continue
            shared_root = self.runtime_root / "shared" / candidate_campaign_id
            algorithm = self._read_json(shared_root / "manifest.json").get("algorithm")
            if (algorithm not in {"MaskableRecurrentPPO", "MaskablePPO"}
                    and (not allow_unknown_algorithm or algorithm is not None)):
                continue
            checkpoints = sorted(
                (item for item in (shared_root / "checkpoints").glob("round-*.zip") if item.is_file()),
                key=lambda item: int(item.stem.removeprefix("round-"))
                if item.stem.removeprefix("round-").isdigit() else -1,
            )
            if not checkpoints:
                continue
            checkpoint = checkpoints[-1]
            match = re.fullmatch(r"round-(\d+)", checkpoint.stem)
            round_number = int(match.group(1)) if match else 0
            rank = 0 if algorithm == "MaskableRecurrentPPO" else 1
            candidates.append((rank, -round_number, len(candidates), {
                "campaign": campaign, "campaign_id": candidate_campaign_id,
                "checkpoint": checkpoint, "round": round_number,
                "algorithm": algorithm,
            }))
        result = min(candidates, key=lambda item: (item[0], item[1], item[2]))[3] if candidates else None
        self._checkpoint_candidate_cache[cache_key] = (self._monotonic_clock(), result)
        return dict(result) if result is not None else None

    def resumable_operator_campaigns(self) -> list[dict[str, object]]:
        """Return compact checkpoint-backed campaign choices for external clients."""
        rows: list[dict[str, object]] = []
        for campaign in self.list_campaigns():
            campaign_id = campaign.get("campaign_id")
            if not isinstance(campaign_id, str):
                continue
            candidate = self._operator_checkpoint_candidate(
                allow_unknown_algorithm=True, campaign_id=campaign_id,
            )
            if candidate is None:
                continue
            rows.append({
                "campaign_id": campaign_id,
                "name": str(campaign.get("name", campaign_id)),
                "round": candidate["round"],
                "checkpoint": Path(str(candidate["checkpoint"])).name,
                "created_at": campaign.get("created_at"),
            })
        def round_number(row: dict[str, object]) -> int:
            value = row.get("round")
            return value if isinstance(value, int) and not isinstance(value, bool) else 0

        rows.sort(key=round_number, reverse=True)
        return rows

    def _operator_system_resources(self) -> dict[str, object]:
        if os.name != "nt":
            return {"cpu_percent": None, "memory_percent": None, "memory_used_bytes": None, "memory_total_bytes": None}
        import ctypes
        from ctypes import wintypes

        class FileTime(ctypes.Structure):
            _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

        idle, kernel, user = FileTime(), FileTime(), FileTime()
        cpu_percent: float | None = None
        if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            values = tuple((item.high << 32) | item.low for item in (idle, kernel, user))
            total = values[1] + values[2]
            if self._system_cpu_previous is not None:
                previous_total, previous_idle = self._system_cpu_previous
                delta_total = total - previous_total
                delta_idle = values[0] - previous_idle
                if delta_total > 0:
                    cpu_percent = round(max(0.0, min(100.0, 100.0 * (delta_total - delta_idle) / delta_total)), 1)
            self._system_cpu_previous = (total, values[0])

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", wintypes.DWORD), ("memory_load", wintypes.DWORD),
                ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong), ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        memory = MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            return {"cpu_percent": cpu_percent, "memory_percent": int(memory.memory_load),
                    "memory_used_bytes": int(memory.total_physical - memory.available_physical),
                    "memory_total_bytes": int(memory.total_physical)}
        return {"cpu_percent": cpu_percent, "memory_percent": None, "memory_used_bytes": None, "memory_total_bytes": None}

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _operator_reason(reason: object) -> str:
        raw = str(reason or "erro não informado").strip()
        lowered = raw.casefold()
        known = (
            (("warm-start campaigns must use a new shared checkpoint directory",),
             "A pasta de checkpoints já está em uso; continue a campanha existente ou use outra pasta."),
            (("bridge disconnected", "bridge desconectou", "connection reset", "broken pipe"),
             "A conexão com a ponte do jogo foi interrompida."),
            (("relaunch limit", "maximum relaunch", "limite de relançamentos"),
             "A instância atingiu o limite de relançamentos automáticos."),
            (("no shared-learner checkpoint", "no checkpoint", "checkpoint is unavailable"),
             "Não há checkpoint disponível para continuar o treino."),
            (("personal game", "jogo pessoal", "another hollow knight process", "game already open"),
             "O Hollow Knight pessoal parece estar aberto; feche-o antes de iniciar o treino."),
        )
        for needles, message in known:
            if any(needle in lowered for needle in needles):
                return message
        if "learner" in lowered and any(word in lowered for word in ("failed", "failure", "caiu", "falhou")):
            return "cancelada porque o learner caiu"
        if "heartbeat expirou" in lowered or "heartbeat expired" in lowered:
            return "O processo parou de responder; consulte os detalhes técnicos para ver a falha original."
        if raw.casefold() in {"stop requested", "parada solicitada", "parada cooperativa solicitada"}:
            return "parada solicitada pelo operador"
        return "O treino foi encerrado por um erro."

    def _launch_error_detail(self, record: Mapping[str, Any]) -> str | None:
        raw_path = record.get("log_path")
        if not isinstance(raw_path, str) or not raw_path:
            return None
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.project_root / path
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(self.project_root)
            lines = resolved.read_text("utf-8", errors="replace").splitlines()
        except (OSError, ValueError):
            return None
        nonempty = [line.strip() for line in lines if line.strip()]
        if not nonempty:
            return None
        # Prefer the actual tail of the failed process log; the summary reason
        # can be a heartbeat timeout that hides the learner/actor exception.
        detail = nonempty[-1]
        if len(detail) > 500:
            detail = detail[-500:]
        return detail

    def _cached_journey_index(self, campaign_id: str, journey_root: Path) -> dict[str, Any]:
        from hk_agent.run_journey import load_json

        path = journey_root / "campaign-progress.json"
        layout_path = journey_root / "map-layout.json"
        try:
            stat = path.stat()
            layout_stat = layout_path.stat() if layout_path.exists() else None
            signature = (stat.st_mtime_ns, stat.st_size,
                         layout_stat.st_mtime_ns if layout_stat else 0,
                         layout_stat.st_size if layout_stat else 0)
        except OSError:
            signature = (0, 0, 0, 0)
        with self._journey_cache_lock:
            cached = self._journey_cache.get(campaign_id)
            if cached and cached.get("signature") == signature:
                return cast(dict[str, Any], cached["index"])
            index = load_json(path)
            index["_layout_path"] = str(layout_path)
            self._journey_cache[campaign_id] = {"signature": signature, "index": index}
            return index

    def journey_run(self, campaign_id: str, run_id: str, *, max_positions: int = 600) -> dict[str, Any]:
        if not campaign_id or Path(campaign_id).name != campaign_id or not run_id:
            raise ValueError("campaign and run are required")
        index = self._cached_journey_index(campaign_id, self.runtime_root / "shared" / campaign_id / "journey")
        runs = index.get("runs", [])
        run = next((row for row in runs if isinstance(row, Mapping) and str(row.get("id")) == run_id), None) if isinstance(runs, list) else None
        if run is None:
            raise ValueError("journey run not found")
        detail = dict(run)
        positions = detail.get("positions", [])
        if max_positions < 2:
            raise ValueError("max_positions must be at least 2")
        if isinstance(positions, list) and len(positions) > max_positions:
            stride = max(1, math.ceil((len(positions) - 1) / (max_positions - 1)))
            sampled = positions[::stride]
            if positions[-1] not in sampled:
                sampled.append(positions[-1])
            detail["positions"] = sampled[:max_positions]
        route_rows = detail.get("route", [])
        route = [str(row["to"]) for row in route_rows if isinstance(row, Mapping) and isinstance(row.get("to"), str)] if isinstance(route_rows, list) else []
        from hk_agent.run_journey import schematic_svg
        return {"run": detail, "svg": schematic_svg(index, route=route)}

    def operator_snapshot(self) -> dict[str, object]:
        status = self.status()
        system_resources = self._operator_system_resources()
        self._ensure_auto_reports(status)
        active = status.get("active_launches", [])
        active_records = [row for row in active if isinstance(row, Mapping)] if isinstance(active, list) else []
        campaigns = self.list_campaigns()
        resume_candidate = self._operator_checkpoint_candidate()
        campaign_id = next((str(row.get("campaign_id")) for row in active_records if row.get("campaign_id")), None)
        if campaign_id is None and resume_candidate is not None:
            campaign_id = str(resume_candidate["campaign_id"])
        if campaign_id is None and campaigns:
            campaign_id = str(campaigns[0].get("campaign_id", "")) or None
        selected_campaign = next(
            (row for row in campaigns if row.get("campaign_id") == campaign_id), {}
        )
        shared_root_value = next((row.get("shared_root") for row in active_records if isinstance(row.get("shared_root"), str)), None)
        shared_root = Path(str(shared_root_value)) if shared_root_value else (self.runtime_root / "shared" / campaign_id if campaign_id else None)
        manifest = self._read_json(shared_root / "manifest.json") if shared_root is not None else {}
        actor_performance: list[dict[str, Any]] = []
        if shared_root is not None:
            for actor_record in active_records:
                resources = actor_record.get("resources")
                slot = resources.get("slot_id") if isinstance(resources, Mapping) else None
                launch_id = actor_record.get("launch_id")
                if actor_record.get("role") != "actor" or not isinstance(slot, int) or not isinstance(launch_id, str):
                    continue
                telemetry_path = shared_root / "telemetry" / f"actor-slot-{slot}.jsonl"
                try:
                    lines = telemetry_path.read_text("utf-8").splitlines()[-64:]
                except OSError:
                    lines = []
                sample: Mapping[str, Any] = {}
                for line in reversed(lines):
                    try:
                        candidate = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(candidate, Mapping) and candidate.get("launch_id") == launch_id:
                        sample = candidate
                        break
                collect_ms = sample.get("collect_ms") if isinstance(sample, Mapping) else None
                profile = sample.get("step_profile_ms", {}) if isinstance(sample, Mapping) else {}
                actor_performance.append({
                    "slot": slot,
                    "launch_id": launch_id,
                    "steps_per_second": (
                        float(manifest.get("rollout_steps", 0)) / (float(collect_ms) / 1000.0)
                        if isinstance(collect_ms, (int, float)) and collect_ms > 0
                        and isinstance(manifest.get("rollout_steps"), (int, float)) else None
                    ),
                    "collect_ms": collect_ms,
                    "step_profile_ms": dict(profile) if isinstance(profile, Mapping) else {},
                })
        journey: dict[str, object] = {"svg": "", "runs": [], "milestones": [], "records": {}}
        if shared_root is not None:
            try:
                journey_root = shared_root / "journey"
                journey_index = self._cached_journey_index(campaign_id or "", journey_root)
                runs = journey_index.get("runs", [])
                best = max((row for row in runs if isinstance(row, Mapping)),
                           key=lambda row: (int(row.get("distinct_scenes", 0)), -int(row.get("deaths", 0))), default={})
                route_records = best.get("route", []) if isinstance(best, Mapping) else []
                route = [str(row["to"]) for row in route_records if isinstance(row, Mapping) and isinstance(row.get("to"), str)]
                from hk_agent.run_journey import schematic_svg
                journey_index["_layout_path"] = str(journey_root / "map-layout.json")
                # The overview only carries compact metadata. Position samples are fetched
                # on demand by /api/journey/run.
                summary_keys = {"id", "distinct_scenes", "deaths", "geo_collected", "grubs", "bosses",
                                "items", "shade_recovered", "actor", "route"}
                summary_runs = [{key: value for key, value in run.items() if key in summary_keys}
                                for run in runs[-50:] if isinstance(run, Mapping)] if isinstance(runs, list) else []
                cache_row = self._journey_cache.get(campaign_id or "", {})
                journey_svg = cache_row.get("svg")
                if not isinstance(journey_svg, str):
                    journey_svg = schematic_svg(journey_index, route=route)
                    cache_row["svg"] = journey_svg
                journey = {"svg": journey_svg, "runs": summary_runs,
                           "milestones": journey_index.get("milestones", [])[-100:],
                           "records": journey_index.get("records", {}), "best_run": best.get("id") if isinstance(best, Mapping) else None,
                           "previous_comparison": journey_index.get("previous_comparison"),
                           "source": "esquemático"}
            except (OSError, ValueError, TypeError):
                pass
        try:
            journey["map_art"] = self.map_art_status()
        except RuntimeError:
            journey["map_art"] = {"cached": False, "attribution": "Mapa: The Embraced One — hallownest.net. Hollow Knight © Team Cherry."}
        if shared_root is not None and not manifest.get("evaluations"):
            eval_file = shared_root / "evaluations.jsonl"
            if eval_file.is_file():
                try:
                    manifest["evaluations"] = [json.loads(line) for line in eval_file.read_text("utf-8").splitlines() if line.strip()]
                except (OSError, json.JSONDecodeError):
                    manifest["evaluations"] = []
        progress = manifest.get("episode_progress", [])
        progress_rows = [row for row in progress if isinstance(row, Mapping)] if isinstance(progress, list) else []
        evaluations = manifest.get("evaluations", [])
        evaluation_rows = [row for row in evaluations if isinstance(row, Mapping)] if isinstance(evaluations, list) else []
        now_ns = time.time_ns()
        started_values: list[int] = [int(row["started_unix_ns"]) for row in active_records if isinstance(row.get("started_unix_ns"), int)]
        if not started_values and campaign_id:
            records = self._campaign_launch_records(campaign_id)
            started_values = [int(row["started_unix_ns"]) for row in records if isinstance(row.get("started_unix_ns"), int)]
        started_ns = min(started_values) if started_values else None
        elapsed_seconds = max(0.0, (now_ns - started_ns) / 1_000_000_000) if isinstance(started_ns, int) else 0.0
        deadlines: list[int] = [int(row["deadline_unix_ns"]) for row in active_records if isinstance(row.get("deadline_unix_ns"), int)]
        remaining_seconds = max(0.0, (min(deadlines) - now_ns) / 1_000_000_000) if deadlines else None
        completed_timesteps = int(manifest.get("completed_timesteps", 0) or 0)
        learner_round = int(manifest.get("round", 0) or 0)
        hours = max(elapsed_seconds / 3600.0, 1 / 3600.0)
        scene_rows = self._campaign_scene_discoveries(campaign_id) if campaign_id else []
        discoveries = {str(row["scene_name"]): int(row["first_seen_unix_ns"]) for row in scene_rows}
        deaths = sum(int(row.get("deaths", 0) or 0) for row in progress_rows)
        recoveries = sum(int(row.get("shade_recoveries", 0) or 0) for row in progress_rows)
        shade_losses = sum(int(row.get("shade_losses", 0) or 0) for row in progress_rows)
        first_scene_steps = [int(row["steps_to_first_scene_change"]) for row in progress_rows if isinstance(row.get("steps_to_first_scene_change"), int)]
        alerts = [str(item) for item in manifest.get("farm_alerts", []) if isinstance(item, str)] if isinstance(manifest.get("farm_alerts", []), list) else []
        warm_start_warning = manifest.get("warm_start_warning")
        if isinstance(warm_start_warning, str) and warm_start_warning:
            alerts.append(warm_start_warning)
        memory_percent = system_resources.get("memory_percent")
        if isinstance(memory_percent, (int, float)) and not isinstance(memory_percent, bool) and memory_percent > 85:
            alerts.append(f"Memória do sistema acima de 85% ({memory_percent:.0f}%)")
        latest_shared = resume_candidate.get("campaign") if resume_candidate is not None else next((row for row in campaigns if row.get("campaign_type") == "exploration_training" and row.get("reward_version") in {"explore_v3", "explore_v4"} and row.get("shared_learner") is True), None)
        latest_checkpoint_exists = resume_candidate is not None
        resume_campaign = resume_candidate.get("campaign") if resume_candidate is not None else {}
        if not isinstance(resume_campaign, Mapping):
            resume_campaign = {}
        last_completion: dict[str, object] | None = None
        if not active_records and campaign_id:
            terminal_records = self._campaign_launch_records(campaign_id)
            finished = sorted((row for row in terminal_records if row.get("status") in {"CANCELLED", "COMPLETED", "FAILED"}), key=lambda row: int(row.get("finished_unix_ns", 0) or 0))
            if finished:
                newest_group = finished[-1].get("shared_group")
                group_rows = [row for row in finished if row.get("shared_group") == newest_group] if newest_group else [finished[-1]]
                save_states = [row.get("final_save_status") for row in group_rows if row.get("mode") == "execute"]
                last_completion = {"safe": bool(save_states) and all(value == "captured" for value in save_states),
                                  "saves_restored": bool(save_states) and all(value == "captured" for value in save_states),
                                  "status": group_rows[-1].get("status"),
                                  "reason": group_rows[-1].get("stop_reason") or group_rows[-1].get("failure")}
        if elapsed_seconds >= 7200 and (not discoveries or max(discoveries.values(), default=0) < now_ns - 7_200_000_000_000):
            alerts.append("Sem novas salas descobertas nas últimas 2 h")
        restart_count = 0
        instance_rows: list[dict[str, object]] = []
        max_room_progress: dict[str, Any] = {"scene": None, "distance": None}
        shared_root_for_slots = shared_root
        if shared_root_for_slots is not None:
            actors_root = shared_root_for_slots / "actors"
            for slot_file in actors_root.glob("slot-*.json"):
                actor = self._read_json(slot_file)
                if actor.get("status") == "WAITING_POLICY":
                    waiting_since = actor.get("waiting_policy_since_unix_ns")
                    if (isinstance(waiting_since, int) and not isinstance(waiting_since, bool)
                            and (now_ns - waiting_since) / 1_000_000_000 > 600):
                        alerts.append(f"Instância {actor.get('slot')} aguarda política há mais de 10 min")
                progress_by_scene = actor.get("max_progress_by_scene")
                if isinstance(progress_by_scene, Mapping):
                    for scene, distance in progress_by_scene.items():
                        if (isinstance(scene, str) and isinstance(distance, (int, float))
                                and not isinstance(distance, bool) and math.isfinite(float(distance))
                                and (max_room_progress["distance"] is None
                                     or float(distance) > float(max_room_progress["distance"]))):
                            max_room_progress = {"scene": scene, "distance": float(distance)}
                restart_times = actor.get("restart_times_unix_ns", [])
                recent_restarts = [stamp for stamp in restart_times if isinstance(stamp, int) and stamp >= now_ns - 3_600_000_000_000] if isinstance(restart_times, list) else []
                restart_count += len(recent_restarts)
                terminal_reason = actor.get("failure") or actor.get("stop_reason")
                if actor.get("status") in {"CANCELLED", "FAILED"} and not terminal_reason:
                    matching = [row for row in self._campaign_launch_records(campaign_id or "")
                                if row.get("role") == "actor"
                                and isinstance(row.get("resources"), Mapping)
                                and row["resources"].get("slot_id") == actor.get("slot")
                                and row.get("status") in {"CANCELLED", "FAILED"}]
                    if matching:
                        latest = max(matching, key=lambda row: int(row.get("finished_unix_ns", 0) or 0))
                        terminal_reason = latest.get("failure") or latest.get("stop_reason")
                practice_status = actor.get("challenge_practice")
                practice_label = (
                    practice_status.get("label")
                    if isinstance(practice_status, Mapping) and practice_status.get("active") is True
                    and isinstance(practice_status.get("label"), str) else None
                )
                evaluation_steps = actor.get("evaluation_steps")
                activity_label = practice_label or (
                    actor.get("evaluation_label")
                    if actor.get("status") == "EVALUATING" and isinstance(actor.get("evaluation_label"), str)
                    else None
                )
                if actor.get("status") == "EVALUATING" and isinstance(evaluation_steps, int) and evaluation_steps > 1000:
                    alerts.append(f"A avaliação da instância {actor.get('slot')} passou do limite de 1000 passos")
                instance_rows.append({
                    "slot": actor.get("slot"), "status": actor.get("status", "unknown"),
                    "activity_label": activity_label,
                    "evaluation_steps": evaluation_steps if isinstance(evaluation_steps, int) else None,
                    "challenge_practice": dict(practice_status) if isinstance(practice_status, Mapping) else None,
                    "restarts_last_hour": len(recent_restarts),
                    "problem": actor.get("status") in {"FAILED", "CANCELLED"},
                    "reason": (self._operator_reason(terminal_reason)
                               if actor.get("status") in {"CANCELLED", "FAILED"} else None),
                })
                if len(recent_restarts) >= 3:
                    alerts.append(f"Instância {actor.get('slot')} reiniciou {len(recent_restarts)} vezes na última hora")
                if recent_restarts:
                    alerts.append(
                        f"Instância {actor.get('slot')} foi relançada automaticamente "
                        f"({len(recent_restarts)}/3 nesta hora)"
                    )
                if actor.get("status") == "FAILED":
                    detail = actor.get("auto_relaunch_error")
                    suffix = f": {detail}" if isinstance(detail, str) and detail else ""
                    alerts.append(f"Instância {actor.get('slot')} falhou e precisa de atenção{suffix}")
            current_group = next(
                (str(row["shared_group"]) for row in active_records
                 if isinstance(row.get("shared_group"), str)),
                None,
            )
            if current_group is not None:
                cutoff_ns = now_ns - 3_600_000_000_000
                restart_records = [
                    row for row in _read_jsonl_records(shared_root_for_slots / "actor-restarts.jsonl")
                    if row.get("shared_group") == current_group
                    and isinstance(row.get("time_unix_ns"), int)
                    and row["time_unix_ns"] >= cutoff_ns
                ]
                launch_by_id = {
                    str(row.get("launch_id")): row
                    for row in self._campaign_launch_records(campaign_id or "")
                    if isinstance(row.get("launch_id"), str)
                }
                slot_counts: dict[int, int] = {}
                for restart in restart_records:
                    old_id = restart.get("old_launch_id")
                    old_launch = launch_by_id.get(old_id) if isinstance(old_id, str) else None
                    resources = old_launch.get("resources") if isinstance(old_launch, Mapping) else None
                    slot = resources.get("slot_id") if isinstance(resources, Mapping) else None
                    if isinstance(slot, int) and not isinstance(slot, bool):
                        slot_counts[slot] = slot_counts.get(slot, 0) + 1
                if slot_counts:
                    restart_count = max(restart_count, sum(slot_counts.values()))
                    for slot, count in slot_counts.items():
                        instance = next((row for row in instance_rows if row.get("slot") == slot), None)
                        if instance is not None:
                            existing_count = instance.get("restarts_last_hour", 0)
                            previous_count = (
                                existing_count if isinstance(existing_count, int)
                                and not isinstance(existing_count, bool) else 0
                            )
                            instance["restarts_last_hour"] = max(previous_count, count)
                        alert = f"Instância {slot} foi relançada automaticamente ({count}/3 nesta hora)"
                        if alert not in alerts:
                            alerts.append(alert)
        if "same_scene_action_over_40_percent_without_novelty" in alerts:
            alerts[alerts.index("same_scene_action_over_40_percent_without_novelty")] = "A IA parece estar repetindo o mesmo lugar"
        active_learner_groups = {
            row.get("shared_group") for row in active_records
            if row.get("role") == "learner" and isinstance(row.get("shared_group"), str)
        }

        active_started_ns = min(
            (int(row.get("started_unix_ns", 0) or 0) for row in active_records
             if isinstance(row.get("started_unix_ns"), int)), default=0
        )
        for failed_learner in self._campaign_launch_records(campaign_id or "") if campaign_id else []:
            finished_ns = failed_learner.get("finished_unix_ns", 0)
            if (failed_learner.get("role") == "learner" and failed_learner.get("status") == "FAILED"
                    and active_learner_groups
                    and failed_learner.get("shared_group") in active_learner_groups
                    and isinstance(finished_ns, int) and finished_ns >= active_started_ns):
                detail = failed_learner.get("failure") or "erro não informado"
                alerts.append(f"learner caiu: {detail}")
        alerts = [
            f"Componente dominante no alerta de farm: {item.removeprefix('farm_component:')}"
            if item.startswith("farm_component:") else item
            for item in alerts
        ]
        alerts = [
            "Mais de 90% dos passos da última rodada tiveram recompensa zero"
            if item == "reward_zero_over_90_percent" else item
            for item in alerts
        ]
        launch_records = self._campaign_launch_records(campaign_id or "") if campaign_id else []
        current_group = next((row.get("shared_group") for row in active_records
                              if isinstance(row.get("shared_group"), str)), None)
        relevant_records = ([row for row in launch_records if row.get("shared_group") == current_group]
                            if current_group else launch_records)
        latest_group = max(
            (row.get("shared_group") for row in relevant_records if isinstance(row.get("shared_group"), str)),
            key=lambda group: max((int(row.get("finished_unix_ns", row.get("started_unix_ns", 0)) or 0)
                                   for row in relevant_records if row.get("shared_group") == group), default=0),
            default=None,
        )
        if latest_group is not None:
            relevant_records = [row for row in relevant_records if row.get("shared_group") == latest_group]
        failed_records = [
            row for row in relevant_records
            if row.get("status") == "FAILED"
            or (row.get("status") == "CANCELLED" and isinstance(row.get("failure"), str)
                and row.get("failure"))
        ]
        failed_record = max(failed_records, key=lambda row: int(row.get("finished_unix_ns", 0) or 0), default=None)
        status_banner: dict[str, object]
        marker_warning = "A instância 1 está com a DLL diferente do registro — rode a implantação da DLL"
        marker_error: Exception | None = None
        marker_failure_text = " ".join(
            str(row.get("failure", "")) + " " + str(row.get("stop_reason", ""))
            for row in relevant_records
        ).casefold()
        if "managed instance marker" in marker_failure_text or "bridge dll hash differs from marker" in marker_failure_text:
            alerts.append(marker_warning)
        if active_records and alerts:
            running_text = (f" · Treinando — rodada {learner_round}, "
                            f"{completed_timesteps / max(elapsed_seconds, 1.0):.1f} passos/s")
            status_banner = {"kind": "warning", "message": f"Aviso: {alerts[0]}{running_text}",
                             "technical_detail": None}
        elif active_records:
            status_banner = {
                "kind": "running",
                "message": f"Treinando — rodada {learner_round}, {completed_timesteps / max(elapsed_seconds, 1.0):.1f} passos/s",
                "technical_detail": None,
            }
        elif failed_record is not None:
            raw_reason = failed_record.get("failure") or failed_record.get("stop_reason")
            detail = self._launch_error_detail(failed_record)
            reason = self._operator_reason(f'{raw_reason or ""} {detail or ""}')
            if reason == "O treino foi encerrado por um erro.":
                evidence = detail or str(raw_reason or "erro sem detalhe")
                reason = f"Falha: {evidence[:180]}"
            status_banner = {
                "kind": "error",
                "message": f"O treino parou: {reason}",
                "technical_detail": detail,
            }
        elif alerts:
            running_text = (f" · Treinando — rodada {learner_round}, "
                            f"{completed_timesteps / max(elapsed_seconds, 1.0):.1f} passos/s") if active_records else ""
            status_banner = {"kind": "warning", "message": f"Aviso: {alerts[0]}{running_text}",
                             "technical_detail": None}
        elif relevant_records and any(row.get("status") == "CANCELLED" for row in relevant_records):
            cancelled = max((row for row in relevant_records if row.get("status") == "CANCELLED"),
                            key=lambda row: int(row.get("finished_unix_ns", 0) or 0), default={})
            raw_reason = cancelled.get("stop_reason") or cancelled.get("failure") or "parada solicitada"
            status_banner = {"kind": "idle", "message": f"Treino parado: {self._operator_reason(raw_reason)}",
                             "technical_detail": None}
        elif alerts:
            status_banner = {"kind": "warning", "message": f"Aviso: {alerts[0]}",
                             "technical_detail": None}
        else:
            status_banner = {"kind": "idle", "message": "Treino parado — aguardando início.",
                             "technical_detail": None}
        if not active_records and status_banner["kind"] == "idle" and alerts:
            status_banner = {"kind": "warning", "message": f"Aviso: {alerts[0]}",
                             "technical_detail": None}
        raw_capacity = status.get("capacity", 1)
        capacity = raw_capacity if isinstance(raw_capacity, int) and not isinstance(raw_capacity, bool) else 1
        help_requests: list[dict[str, Any]] = []
        help_history: list[dict[str, Any]] = []
        stuck_locations: list[dict[str, Any]] = []
        if shared_root is not None:
            help_root = shared_root / "help-requests"
            for ledger_path in sorted(help_root.glob("slot-*.json")) if help_root.is_dir() else ():
                ledger = self._read_json(ledger_path)
                slot_text = ledger_path.stem.removeprefix("slot-")
                try:
                    slot_number = int(slot_text)
                except ValueError:
                    slot_number = 0
                details = ledger.get("stuck_details", {})
                if isinstance(details, Mapping):
                    for place, raw in details.items():
                        if not isinstance(place, str) or not isinstance(raw, Mapping):
                            continue
                        count = raw.get("count", 0)
                        stamp = raw.get("last_stuck_unix", now_ns / 1_000_000_000)
                        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                            continue
                        minutes = max(0.0, now_ns / 1_000_000_000 - float(stamp)) / 60.0 if isinstance(stamp, (int, float)) else 0.0
                        location = {**dict(raw), "slot": slot_number, "place": place, "stuck_minutes": minutes}
                        stuck_locations.append(location)
                        alerts.append(
                            f"Instância {slot_number + 1} travada há {minutes:.0f} min em {place} "
                            f"({count} travadas aqui)"
                        )
                rows = ledger.get("requests", [])
                if isinstance(rows, list):
                    help_history.extend(dict(row) for row in rows if isinstance(row, Mapping))
            help_history.sort(key=lambda row: float(row.get("created_unix", 0.0))
                              if isinstance(row.get("created_unix"), (int, float))
                              and not isinstance(row.get("created_unix"), bool) else 0.0)
            help_requests = [row for row in help_history if row.get("status") in {"pending", "demonstration_requested", "helping", "completion_requested"}]
            for help_row in help_requests:
                slot = help_row.get("slot")
                actor_path = (shared_root / "actors" / f"slot-{slot}.json"
                              if isinstance(slot, int) and not isinstance(slot, bool) and slot >= 0 else None)
                actor = self._read_json(actor_path) if actor_path is not None and not actor_path.is_symlink() else {}
                if actor.get("help_id") == help_row.get("request_id"):
                    help_row["help_elapsed_seconds"] = actor.get("help_elapsed_seconds", 0.0)
                    help_row["help_coverage_percent"] = actor.get("help_coverage_percent", 0.0)
        stuck_alert = next((item for item in reversed(alerts) if "travada há" in item), None)
        if stuck_alert and status_banner["kind"] != "error":
            running_text = (f" · Treinando — rodada {learner_round}, "
                            f"{completed_timesteps / max(elapsed_seconds, 1.0):.1f} passos/s") if active_records else ""
            status_banner = {"kind": "warning", "message": f"Aviso: {stuck_alert}{running_text}",
                             "technical_detail": None}
        if help_requests:
            help_alert = (f"Há {len(help_requests)} pedido(s) de ajuda pendente(s) no painel")
            alerts.append(help_alert)
            running_text = (f" · Treinando — rodada {learner_round}, "
                            f"{completed_timesteps / max(elapsed_seconds, 1.0):.1f} passos/s") if active_records else ""
            status_banner = {"kind": "warning", "message": f"Aviso: {help_alert}{running_text}",
                             "technical_detail": None}
        playthroughs: list[dict[str, Any]] = []
        try:
            playthroughs = self._operator_playthroughs(campaign_id, campaigns, active_records, journey)
        except Exception as error:  # noqa: BLE001 - panel data must survive a broken optional playthrough block
            marker_error = error
            chain: BaseException | None = error
            marker_mismatch = False
            while chain is not None:
                detail = str(chain).casefold()
                marker_mismatch = marker_mismatch or (
                    "bridge dll hash differs from marker" in detail
                    or "managed instance marker is missing or invalid" in detail
                )
                chain = chain.__cause__
            if marker_mismatch:
                if marker_warning not in alerts:
                    alerts.append(marker_warning)
            else:
                alerts.append(f"Jogadas contínuas indisponíveis: {type(error).__name__}: {error}")
        if marker_warning in alerts:
            status_banner = {
                "kind": "error",
                "message": marker_warning,
                "technical_detail": str(marker_error) if marker_error is not None else None,
            }
        recent_events = []
        for row in sorted(launch_records, key=lambda item: int(item.get("finished_unix_ns", 0) or 0), reverse=True):
            if row.get("status") not in {"FAILED", "CANCELLED", "COMPLETED"}:
                continue
            raw_reason = row.get("failure") or row.get("stop_reason") or row.get("status")
            detail = self._launch_error_detail(row) if row.get("status") == "FAILED" else None
            reason = self._operator_reason(raw_reason)
            if reason == "O treino foi encerrado por um erro.":
                evidence = detail or str(raw_reason or "erro sem detalhe")
                reason = f"Falha: {evidence[:180]}"
            recent_events.append({"status": row.get("status"), "role": row.get("role"),
                                  "finished_unix_ns": row.get("finished_unix_ns"),
                                  "message": reason, "technical_detail": detail})
            if len(recent_events) >= 8:
                break
        status_completion = status.get("last_completion")
        status_completion_safe = (
            bool(status_completion.get("safe"))
            if isinstance(status_completion, Mapping)
            else False
        )
        return {
            "status": {
                "active_count": len(active_records),
                "active_launches": [
                    {"role": row.get("role"), "status": row.get("status")}
                    for row in active_records
                ],
                "last_completion": {
                    "safe": status_completion_safe
                } if isinstance(status_completion, Mapping) else None,
            }, "campaign_id": campaign_id, "elapsed_seconds": elapsed_seconds,
            "actor_performance": actor_performance,
            "max_room_progress": max_room_progress,
            "remaining_seconds": remaining_seconds, "instances": instance_rows,
            "training_steps": completed_timesteps, "steps_per_second": completed_timesteps / max(elapsed_seconds, 1.0),
            "learner_updates": learner_round, "updates_per_hour": learner_round / hours,
            "last_zero_reward_fraction": manifest.get("last_zero_reward_fraction"),
            "self_imitation": manifest.get("self_imitation", {}),
            "deaths_per_hour": deaths / hours, "shade_recoveries_per_hour": recoveries / hours,
            "shade_losses": shade_losses, "average_steps_to_first_scene": sum(first_scene_steps) / len(first_scene_steps) if first_scene_steps else None,
            "episode_progress": progress_rows[-40:], "scene_trend": [{"episode": row.get("episode"), "distinct_scenes": row.get("distinct_scenes", 0)} for row in progress_rows[-40:]],
            "scenes_discovered": scene_rows, "evaluations": evaluation_rows[-50:], "evaluation_trend": manifest.get("evaluation_trend", []),
            "journey": journey,
            "challenge_practice_enabled": selected_campaign.get("challenge_practice_enabled") is True,
            "help_requests_enabled": selected_campaign.get("help_requests_enabled") is True,
            "help_stuck_count": selected_campaign.get("help_stuck_count", 10),
            "help_requests": help_requests[-16:],
            "help_request_history": help_history[-64:],
            "stuck_locations": sorted(
                stuck_locations,
                key=lambda row: (int(row.get("count", 0)), float(row.get("last_stuck_unix", 0))),
                reverse=True,
            )[:64],
            "playthroughs": playthroughs,
            "alerts": sorted(set(alerts)), "status_banner": status_banner, "recent_events": recent_events,
            "instance_restarts_last_hour": restart_count,
            "recommended_instances": min(3, capacity, max(1, (os.cpu_count() or 1) // 4)),
            "max_instances": min(3, capacity),
            "can_continue": latest_checkpoint_exists, "has_campaign": bool(latest_shared),
            "resumable_campaign": ({"campaign_id": resume_candidate["campaign_id"],
                                    "name": resume_campaign.get("name", resume_candidate["campaign_id"]),
                                    "round": resume_candidate["round"],
                                    "checkpoint": Path(str(resume_candidate["checkpoint"])).name,
                                    "algorithm": resume_candidate["algorithm"]}
                                   if resume_candidate is not None else None),
            "last_completion": ({key: last_completion.get(key) for key in ("safe", "status", "reason")}
                                if isinstance(last_completion, Mapping) else None),
            "checkpoints": [Path(str(row)).name for row in manifest.get("checkpoints", [])[-5:]] if isinstance(manifest.get("checkpoints", []), list) else [],
            "reward_version": next((row.get("reward_version") for row in campaigns if row.get("campaign_id") == campaign_id), None),
            "advanced": {"launch_ids": [row.get("launch_id") for row in active_records], "shared_root": str(shared_root) if shared_root else None,
                         "artifact_path": str(shared_root) if shared_root else None},
            "system": system_resources, "reports": self.list_reports()[:10],
        }

    def operator_fallback_snapshot(self, error: Exception) -> dict[str, object]:
        """Small renderable response used when a whole snapshot unexpectedly fails."""
        try:
            status: Mapping[str, object] = self.status()
        except Exception:  # noqa: BLE001 - emergency response must survive a broken status source
            status = {"active_count": 0, "active_launches": []}
        message = f"Não foi possível carregar uma parte do painel: {type(error).__name__}: {error}"
        return {
            "status": dict(status), "campaign_id": None, "elapsed_seconds": 0.0,
            "remaining_seconds": None, "actor_performance": [], "max_room_progress": {},
            "instances": [], "playthroughs": [], "training_steps": 0, "steps_per_second": 0.0,
            "learner_updates": 0, "updates_per_hour": 0.0, "last_zero_reward_fraction": None,
            "self_imitation": {}, "deaths_per_hour": 0.0, "shade_recoveries_per_hour": 0.0,
            "shade_losses": 0, "average_steps_to_first_scene": None, "episode_progress": [],
            "scene_trend": [], "scenes_discovered": [], "evaluations": [], "evaluation_trend": [],
            "journey": {"svg": "", "runs": [], "milestones": [], "records": {}},
            "challenge_practice_enabled": False, "help_requests_enabled": False, "help_stuck_count": 10,
            "help_requests": [], "help_request_history": [], "stuck_locations": [], "alerts": [message],
            "status_banner": {"kind": "error", "message": "O painel carregou com dados limitados.",
                              "technical_detail": message}, "recent_events": [],
            "instance_restarts_last_hour": 0, "recommended_instances": 1, "max_instances": 1,
            "can_continue": False, "has_campaign": False, "resumable_campaign": None,
            "last_completion": None, "checkpoints": [], "reward_version": None,
            "advanced": {"launch_ids": [], "shared_root": None, "artifact_path": None},
            "system": {}, "reports": [],
        }

    def _operator_playthroughs(
        self,
        campaign_id: str | None,
        campaigns: Sequence[Mapping[str, Any]],
        active_records: Sequence[Mapping[str, Any]],
        journey: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        campaign = next((row for row in campaigns if row.get("campaign_id") == campaign_id), None)
        if not campaign or campaign.get("save_source") != "continuous_playthrough" or not campaign_id:
            return []
        from hk_agent.config import load_config

        config = load_config(self.project_root / "config.training.local.toml", allow_training=True)
        active_slots = {
            resources.get("slot_id")
            for row in active_records
            if isinstance((resources := row.get("resources")), Mapping)
        }
        last_scenes: dict[int, str] = {}
        runs = journey.get("runs", [])
        if isinstance(runs, list):
            for run in runs:
                if not isinstance(run, Mapping):
                    continue
                try:
                    slot = int(str(run.get("actor", "")).removeprefix("actor-"))
                except ValueError:
                    continue
                route = run.get("route", [])
                for edge in reversed(route if isinstance(route, list) else []):
                    if isinstance(edge, Mapping) and isinstance(edge.get("to"), str):
                        last_scenes.setdefault(slot, str(edge["to"]))
                        break
        rows: list[dict[str, Any]] = []
        for slot_id in range(min(config.instance_count, 3)):
            root = resolve_playthrough_root(self.runtime_root, campaign_id, slot_id + 1)
            state = playthrough_status(root) or {"last_scene": None, "progress": {}, "backup_count": 0}
            knowledge = self._read_json(root / "player-knowledge.json")
            challenge_memory = knowledge.get("challenge_memory", {})
            state["challenge_memory"] = (
                [
                    {"challenge_id": key, **dict(value)}
                    for key, value in list(challenge_memory.items())[:64]
                    if isinstance(key, str) and isinstance(value, Mapping)
                ]
                if isinstance(challenge_memory, Mapping) else []
            )
            snapshot_index_path = root.parent / "challenge-snapshots" / root.name / "index.json"
            snapshot_payload = self._read_json(snapshot_index_path)
            state["challenge_snapshots"] = [
                dict(row) for row in snapshot_payload.get("snapshots", [])[-32:]
                if isinstance(row, Mapping) and isinstance(row.get("challenge_id"), str)
                and isinstance(row.get("save_path"), str)
            ] if isinstance(snapshot_payload.get("snapshots"), list) else []
            raw_bestiary = knowledge.get("bestiary", {})
            bestiary_rows: list[dict[str, Any]] = []
            if isinstance(raw_bestiary, Mapping):
                from hk_agent.bestiary import BESTIARY_DISPLAY_NAMES

                for enemy_id, experience in raw_bestiary.items():
                    if not isinstance(enemy_id, str) or not isinstance(experience, Mapping):
                        continue
                    encounters = experience.get("encounters", 0)
                    defeated = experience.get("defeated", 0)
                    geo_defeats = experience.get("geo_defeats", 0)
                    if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                               for value in (encounters, defeated, geo_defeats)):
                        continue
                    total_geo = experience.get("total_geo", 0)
                    total_hits = experience.get("total_hits_to_defeat", 0)
                    if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                               for value in (total_geo, total_hits)):
                        continue
                    bestiary_rows.append({
                        "enemy_type_id": enemy_id,
                        "name": BESTIARY_DISPLAY_NAMES.get(enemy_id, "Inimigo ainda não identificado"),
                        "encounters": encounters,
                        "wins": defeated,
                        "win_rate": defeated / encounters if encounters else 0.0,
                        "average_hits_to_defeat": total_hits / defeated if defeated else None,
                        "damage_per_encounter": (experience.get("damage_taken", 0) / encounters)
                        if encounters else 0.0,
                        "times_killed": experience.get("times_killed", 0),
                        "average_geo": total_geo / geo_defeats if geo_defeats else None,
                        "gave_item": experience.get("gave_item") is True,
                        "fight_rate": (experience.get("fought_encounters", 0) / encounters)
                        if encounters else 0.0,
                    })
            bestiary_rows.sort(key=lambda item: (-int(item["encounters"]), str(item["enemy_type_id"])))
            state["bestiary"] = bestiary_rows[:64]
            practice_actor = next((row for row in active_records
                                   if isinstance((resources := row.get("resources")), Mapping)
                                   and resources.get("slot_id") == slot_id
                                   and isinstance(row.get("challenge_practice"), Mapping)), None)
            if practice_actor is not None:
                practice = dict(practice_actor["challenge_practice"])
                practice_memory_path = practice_actor.get("resources", {}).get("challenge_memory_root")
                if isinstance(practice_memory_path, str):
                    practice_knowledge = self._read_json(Path(practice_memory_path) / "player-knowledge.json")
                    raw_memory = practice_knowledge.get("challenge_memory", {})
                    practice["memory"] = [
                        {"challenge_id": key, **dict(value)}
                        for key, value in raw_memory.items()
                        if isinstance(key, str) and isinstance(value, Mapping)
                    ] if isinstance(raw_memory, Mapping) else []
                shared_value = practice_actor.get("shared_root")
                if isinstance(shared_value, str):
                    actor_status = self._read_json(
                        Path(shared_value) / "actors" / f"slot-{slot_id}.json"
                    )
                    practice.update(actor_status.get("challenge_practice", {}))
                state["challenge_practice"] = practice
            if not state.get("last_scene"):
                state["last_scene"] = last_scenes.get(slot_id)
            state.update({"slot": slot_id + 1, "active": slot_id in active_slots,
                          "can_manage": slot_id not in active_slots})
            rows.append(state)
        return rows

    def _require_idle_playthrough_slot(self, campaign_id: object, slot: object) -> tuple[str, int, Any, Path]:
        if not isinstance(campaign_id, str) or not _IDENTIFIER.fullmatch(campaign_id):
            raise ValueError("campaign_id is invalid")
        slot_number = _bounded_int(slot, "slot", 1, 4)
        campaign = next((row for row in self.list_campaigns() if row.get("campaign_id") == campaign_id), None)
        if campaign is None or campaign.get("save_source") != "continuous_playthrough":
            raise ValueError("campaign has no continuous playthrough")
        slot_id = slot_number - 1
        with self._lock:
            if slot_id in self._slot_owners:
                raise ControlPlaneBusyError("this playthrough slot is active")
        active_value = self.status().get("active_launches", [])
        active_rows = active_value if isinstance(active_value, list) else []
        for row in active_rows:
            if isinstance(row, Mapping):
                resources = row.get("resources")
                if isinstance(resources, Mapping) and resources.get("slot_id") == slot_id:
                    raise ControlPlaneBusyError("this instance is active")
        from hk_agent.config import load_config

        config = load_config(self.project_root / "config.training.local.toml", allow_training=True)
        if slot_id >= config.instance_count:
            raise ValueError("slot is not configured on this computer")
        return campaign_id, slot_id, config, resolve_playthrough_root(self.runtime_root, campaign_id, slot_number)

    def start_new_playthrough(self, payload: Mapping[str, object]) -> dict[str, object]:
        campaign_id, slot_id, config, root = self._require_idle_playthrough_slot(
            payload.get("campaign_id"), payload.get("slot")
        )
        campaign = next(row for row in self.list_campaigns() if row.get("campaign_id") == campaign_id)
        prior = self._read_json(root / "playthrough.json") if root.exists() else {}
        source = prior.get("seed_source", "existing_save")
        if source not in {"existing_save", "new_game"}:
            source = "existing_save"
        archive = ensure_inside_runtime(
            self.runtime_root / "playthrough-archives" / campaign_id / f"slot-{slot_id + 1}" / str(time.time_ns()),
            self.runtime_root,
        )
        if root.exists():
            archive.parent.mkdir(parents=True, exist_ok=True)
            root.rename(archive)
        try:
            initialize_playthrough(
                root, config.experimental_save_root, runtime_root=self.runtime_root,
                personal_root=config.personal_save_root, backup_root=config.backup_root,
                game_install=config.game_executable.parent, source=str(source),
                save_slot=_bounded_int(campaign.get("save_slot", 1), "save_slot", 1, 4),
            )
        except BaseException:
            if root.exists():
                shutil.rmtree(root)
            if archive.exists():
                archive.rename(root)
            raise
        return {"campaign_id": campaign_id, "slot": slot_id + 1, "playthrough": playthrough_status(root)}

    def restore_playthrough_backup(self, payload: Mapping[str, object]) -> dict[str, object]:
        campaign_id, slot_id, config, root = self._require_idle_playthrough_slot(
            payload.get("campaign_id"), payload.get("slot")
        )
        from hk_agent.playthrough import restore_last_backup

        restored = restore_last_backup(root, instance_root=config.instance_data_roots[slot_id])
        return {"campaign_id": campaign_id, "slot": slot_id + 1,
                "restored_backup": restored.name, "playthrough": playthrough_status(root)}

    def _campaign_launch_records(self, campaign_id: str) -> list[dict[str, Any]]:
        now = self._monotonic_clock()
        with self._launch_records_cache_lock:
            cached_at, cached_records = self._launch_records_cache
            if now - cached_at < 10.0:
                return cached_records.get(campaign_id, [])
            records_by_campaign: dict[str, list[dict[str, Any]]] = {}
            root = self.control_root / "launches"
            if root.exists():
                for path in root.glob("*/launch.json"):
                    item = self._read_json(path)
                    key = item.get("campaign_id")
                    if isinstance(key, str):
                        records_by_campaign.setdefault(key, []).append(item)
            self._launch_records_cache = (now, records_by_campaign)
            return records_by_campaign.get(campaign_id, [])

    def _campaign_scene_discoveries(self, campaign_id: str | None) -> list[dict[str, Any]]:
        if not campaign_id:
            return []
        discoveries: dict[str, int] = {}
        campaign_memory = self.runtime_root / "campaign-memory" / campaign_id
        for path in campaign_memory.glob("slot-*/lifelong-scene-first-seen.json"):
            memory = self._read_json(path).get("first_seen_unix_ns", {})
            if isinstance(memory, Mapping):
                for scene, timestamp in memory.items():
                    if isinstance(scene, str) and isinstance(timestamp, int):
                        discoveries[scene] = min(discoveries.get(scene, timestamp), timestamp)
        from hk_agent.run_journey import friendly_scene
        return [{"scene_name": name, "scene_name_friendly": friendly_scene(name), "first_seen": time.strftime("%Y-%m-%d %H:%M", time.localtime(stamp / 1_000_000_000)), "first_seen_unix_ns": stamp}
                for name, stamp in sorted(discoveries.items(), key=lambda item: item[1])]

    def _ensure_auto_reports(self, status: Mapping[str, object]) -> None:
        active = status.get("active_launches", [])
        active_campaigns = {str(row.get("campaign_id")) for row in active if isinstance(row, Mapping) and row.get("campaign_id")} if isinstance(active, list) else set()
        candidates = {
            str(row.get("campaign_id"))
            for row in self.list_campaigns()
            if row.get("campaign_id") and row.get("shared_learner") is True
            and row.get("reward_version") in {"explore_v3", "explore_v4"}
        }
        existing: dict[str, int] = {}
        for report in self.list_reports():
            campaign = str(report.get("campaign_id", ""))
            finished = report.get("last_finished_unix_ns")
            if isinstance(finished, int):
                existing[campaign] = max(existing.get(campaign, 0), finished)
        for campaign_id in candidates - active_campaigns:
            records = self._campaign_launch_records(campaign_id)
            if records and all(row.get("status") in {"CANCELLED", "COMPLETED", "FAILED"} for row in records):
                latest_finish = max(
                    (int(row["finished_unix_ns"]) for row in records if isinstance(row.get("finished_unix_ns"), int)),
                    default=0,
                )
                if latest_finish <= existing.get(campaign_id, 0):
                    continue
                try:
                    self.generate_run_report(campaign_id)
                except (OSError, ValueError, TypeError):
                    continue

    def list_reports(self) -> list[dict[str, object]]:
        root = self.artifacts_root / "reports"
        reports = []
        if root.exists():
            paths = tuple(sorted(root.glob("*/*.json")))
            signature = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size)
                              for path in paths if path.exists())
            cached_at, cached_signature, cached_reports = self._reports_cache
            if signature == cached_signature and self._monotonic_clock() - cached_at < 10.0:
                return [dict(row) for row in cached_reports]
            for path in paths:
                item = self._read_json(path)
                reports.append({"campaign_id": item.get("campaign_id", path.parent.name),
                                "created_at": item.get("created_at"),
                                "html": path.with_suffix(".html").relative_to(self.project_root).as_posix(),
                                "json": path.relative_to(self.project_root).as_posix(),
                                "training_steps": item.get("training_steps", 0),
                                "learner_updates": item.get("learner_updates", 0),
                                "last_finished_unix_ns": item.get("last_finished_unix_ns", 0)})
        reports = sorted(reports, key=lambda row: str(row.get("created_at", "")), reverse=True)
        if root.exists():
            self._reports_cache = (self._monotonic_clock(), signature, reports)
        return [dict(row) for row in reports]

    def _campaign_report_data(self, campaign_id: str) -> dict[str, Any]:
        if not _IDENTIFIER.fullmatch(campaign_id):
            raise ValueError("invalid campaign_id")
        records = self._campaign_launch_records(campaign_id)
        shared_root = next((Path(str(row["shared_root"])) for row in reversed(records) if isinstance(row.get("shared_root"), str)), self.runtime_root / "shared" / campaign_id)
        manifest = self._read_json(shared_root / "manifest.json")
        status_rows = [self._read_json(self.control_root / "launches" / str(row.get("launch_id")) / "launch.json") for row in records]
        starts: list[int] = [int(row["started_unix_ns"]) for row in status_rows if isinstance(row.get("started_unix_ns"), int)]
        ends: list[int] = [int(row["finished_unix_ns"]) for row in status_rows if isinstance(row.get("finished_unix_ns"), int)]
        duration = max(0.0, ((max(ends) if ends else time.time_ns()) - min(starts)) / 1_000_000_000) if starts else 0.0
        slots = {resources.get("slot_id") for row in status_rows if isinstance((resources := row.get("resources")), Mapping) and isinstance(resources.get("slot_id"), int)}
        episodes = [row for row in manifest.get("episode_progress", []) if isinstance(row, Mapping)] if isinstance(manifest.get("episode_progress", []), list) else []
        evaluations = [row for row in manifest.get("evaluations", []) if isinstance(row, Mapping)] if isinstance(manifest.get("evaluations", []), list) else []
        scenes = self._campaign_scene_discoveries(campaign_id)
        help_request_history: list[dict[str, Any]] = []
        help_root = shared_root / "help-requests"
        for ledger_path in sorted(help_root.glob("slot-*.json")) if help_root.is_dir() else ():
            rows = self._read_json(ledger_path).get("requests", [])
            if isinstance(rows, list):
                help_request_history.extend(dict(row) for row in rows if isinstance(row, Mapping))
        journey_root = shared_root / "journey"
        journey_progress: dict[str, Any] = {}
        try:
            from hk_agent.run_journey import load_json as load_journey
            from hk_agent.run_journey import schematic_svg

            journey_progress = load_journey(journey_root / "campaign-progress.json")
            journey_progress["_layout_path"] = str(journey_root / "map-layout.json")
            journey_svg = schematic_svg(journey_progress)
        except (OSError, ValueError, TypeError):
            journey_svg = schematic_svg({})
        alerts = [str(item) for item in manifest.get("farm_alerts", []) if isinstance(item, str)] if isinstance(manifest.get("farm_alerts", []), list) else []
        warm_start_warning = manifest.get("warm_start_warning")
        if isinstance(warm_start_warning, str) and warm_start_warning:
            alerts.append(warm_start_warning)
        checkpoints = manifest.get("checkpoints", [])
        if not isinstance(checkpoints, list):
            checkpoints = [str(path) for path in sorted((shared_root / "checkpoints").glob("round-*.zip"))]
        save_states = [row.get("final_save_status") for row in status_rows if row.get("mode") == "execute"]
        save_checks = "restauração concluída" if save_states and all(item == "captured" for item in save_states) else "em andamento ou sem comprovação"
        return {
            "campaign_id": campaign_id, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            "duration_seconds": duration, "instances": len(slots),
            "training_steps": int(manifest.get("completed_timesteps", 0) or 0),
            "learner_updates": int(manifest.get("round", 0) or 0),
            "scenes_discovered": scenes,
            "journey_map_svg": journey_svg,
            "journey_milestones": journey_progress.get("milestones", []),
            "journey_records": journey_progress.get("records", {}),
            "map_art_attribution": "Mapa: The Embraced One — hallownest.net. Hollow Knight © Team Cherry.",
            "sharing_courtesy": "Antes de distribuir o projeto, considere avisar o autor e pedir permissão para compartilhar o mapa.",
            "episode_progress": episodes, "evaluations": evaluations,
            "help_requests": sorted(help_request_history, key=lambda row: (
                float(row.get("created_unix", 0.0))
                if isinstance(row.get("created_unix"), (int, float))
                and not isinstance(row.get("created_unix"), bool) else 0.0
            )),
            "deaths": sum(int(row.get("deaths", 0) or 0) for row in episodes),
            "shade_recoveries": sum(int(row.get("shade_recoveries", 0) or 0) for row in episodes),
            "shade_losses": sum(int(row.get("shade_losses", 0) or 0) for row in episodes),
            "alerts": alerts, "instance_restarts": sum(int(row.get("relaunches", 0) or 0) for row in status_rows),
            "checkpoints_saved": checkpoints, "save_safety_checks": save_checks,
            "last_finished_unix_ns": max(ends, default=0),
            "end_reason": next((row.get("stop_reason") for row in status_rows if row.get("stop_reason")),
                               next((row.get("failure") for row in status_rows if row.get("failure")),
                                    "parada solicitada" if any(row.get("status") == "CANCELLED" for row in status_rows) else "fim desconhecido")),
            "task_contract": next((row.get("reward_version") for row in self.list_campaigns() if row.get("campaign_id") == campaign_id), "explore_v3"),
        }

    def generate_run_report(self, campaign_id: str) -> dict[str, object]:
        from hk_agent.operator_reports import build_report

        data, document = build_report(self._campaign_report_data(campaign_id))
        folder = self.artifacts_root / "reports" / campaign_id
        folder.mkdir(parents=True, exist_ok=True)
        stamp = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime())}-{time.time_ns() % 1_000_000_000:09d}"
        json_path = folder / f"relatorio-{stamp}.json"
        html_path = json_path.with_suffix(".html")
        atomic_json(json_path, data)
        html_path.write_text(document, encoding="utf-8", newline="\n")
        return {"campaign_id": campaign_id, "html": html_path.relative_to(self.project_root).as_posix(),
                "json": json_path.relative_to(self.project_root).as_posix(), "created_at": data["created_at"]}

    def read_journey_capture(self, campaign_id: str, relative_path: str) -> bytes:
        if not _IDENTIFIER.fullmatch(campaign_id):
            raise ValueError("invalid campaign_id")
        root = (self.runtime_root / "shared" / campaign_id / "journey").resolve(strict=True)
        target = (root / relative_path).resolve(strict=True)
        try:
            target.relative_to(root)
        except ValueError as error:
            raise ValueError("capture path is outside the milestone directory") from error
        if target.suffix.lower() != ".png" or not target.is_file() or target.stat().st_size > 24 * 1024 * 1024:
            raise ValueError("journey capture is missing or exceeds its limit")
        from hk_agent.run_journey import validate_png_header

        with target.open("rb") as stream:
            validate_png_header(stream)
        return target.read_bytes()

    def read_help_thumbnail(self, campaign_id: str, request_id: str) -> bytes:
        if not _IDENTIFIER.fullmatch(campaign_id) or not re.fullmatch(r"help-\d+-\d+", request_id):
            raise ValueError("invalid help thumbnail reference")
        root = (self.runtime_root / "shared" / campaign_id / "help-requests").resolve(strict=True)
        target_path: str | None = None
        for ledger_path in root.glob("slot-*.json"):
            ledger = self._read_json(ledger_path)
            rows = ledger.get("requests", [])
            if isinstance(rows, list):
                row = next((item for item in rows if isinstance(item, Mapping)
                            and item.get("request_id") == request_id), None)
                if row is not None and isinstance(row.get("thumbnail_path"), str):
                    target_path = row["thumbnail_path"]
                    break
        if target_path is None:
            raise FileNotFoundError("help request thumbnail was not found")
        target = Path(target_path).resolve(strict=True)
        journey = (self.runtime_root / "shared" / campaign_id / "journey").resolve(strict=True)
        try:
            target.relative_to(journey)
        except ValueError as error:
            raise ValueError("help thumbnail is outside the campaign journey") from error
        if target.suffix.lower() != ".png" or not target.is_file() or target.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("help thumbnail is missing or too large")
        from hk_agent.run_journey import validate_png_header

        with target.open("rb") as stream:
            validate_png_header(stream)
        return target.read_bytes()

    def map_art_runtime_root(self) -> Path:
        local = os.environ.get("LOCALAPPDATA")
        if not local:
            raise RuntimeError("cache local do mapa não está disponível nesta máquina")
        return Path(local) / "VoidLab"

    def map_art_status(self) -> dict[str, object]:
        root = self.map_art_runtime_root() / "cache" / "map-art"
        reference = root / "reference.json"
        value = self._read_json(reference)
        digest = value.get("latest_sha256")
        progress = self._read_json(root / "progress.json")
        from hk_agent.map_art import validate_map_hash

        if isinstance(digest, str):
            try:
                digest = validate_map_hash(digest)
            except ValueError:
                digest = None
        metadata = self._read_json(root / digest / "tiles.json") if digest else {}
        calibrations: Mapping[str, object] = {}
        global_calibration: Mapping[str, object] = {}
        room_offsets: Mapping[str, object] = {}
        if digest:
            from hk_agent.run_journey import load_scene_calibration

            calibrations = load_scene_calibration(root / str(digest) / "calibration.json", map_sha256=digest)
            global_value = self._read_json(root / str(digest) / "global-calibration.json")
            if global_value.get("map_sha256") == digest and global_value.get("schema_version") == 1:
                global_calibration = global_value
            offsets_value = self._read_json(root / str(digest) / "room-offsets.json")
            if offsets_value.get("map_sha256") == digest and offsets_value.get("schema_version") == 1:
                raw_offsets = offsets_value.get("offsets")
                room_offsets = raw_offsets if isinstance(raw_offsets, Mapping) else {}
        layout_path = self.runtime_root / "map-layout" / "game-map-layout.json"
        try:
            from hk_agent.map_layout import load_game_map_layout
            game_layout: Mapping[str, object] = load_game_map_layout(layout_path)
            layout_error = None
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            game_layout = {}
            layout_error = str(error)
        return {"cached": bool(digest), "sha256": digest,
                "reference_changed": value.get("sha256") != digest if value else False,
                "tiles_ready": metadata.get("complete") is True,
                "tiles": metadata if metadata.get("complete") is True else None,
                "progress": progress,
                "calibrations": calibrations,
                "game_layout": game_layout, "game_layout_error": layout_error,
                "global_calibration": global_calibration, "room_offsets": room_offsets,
                "attribution": "Mapa: The Embraced One — hallownest.net · Hollow Knight © Team Cherry"}

    def save_global_map_calibration(self, payload: Mapping[str, object]) -> dict[str, object]:
        from hk_agent.map_layout import fit_global_calibration
        from hk_agent.run_journey import atomic_json

        art = self.map_art_status()
        digest = art.get("sha256")
        if not isinstance(digest, str) or art.get("tiles_ready") is not True:
            raise ValueError("map tiles must be ready before global calibration")
        raw = payload.get("anchors")
        if not isinstance(raw, list) or len(raw) < 2 or len(raw) > 64:
            raise ValueError("global calibration needs 2 to 64 anchors")
        anchors: list[list[float]] = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise TypeError("global anchor must be an object")
            numbers: list[float] = []
            for key in ("map_x", "map_y", "art_x", "art_y"):
                value = item.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError("global anchor coordinates must be numbers")
                number = float(value)
                if not math.isfinite(number) or abs(number) > 10_000_000:
                    raise ValueError("global anchor coordinates must be finite and bounded")
                numbers.append(number)
            anchors.append(numbers)
        fit = fit_global_calibration(anchors)
        result = {**fit, "map_sha256": digest, "anchors": anchors,
                  "saved_unix_ns": time.time_ns()}
        atomic_json(self.map_art_runtime_root() / "cache" / "map-art" / digest / "global-calibration.json", result)
        return result

    def save_map_room_offset(self, payload: Mapping[str, object]) -> dict[str, object]:
        from hk_agent.run_journey import atomic_json

        art = self.map_art_status()
        digest = art.get("sha256")
        scene, dx, dy = payload.get("scene"), payload.get("dx"), payload.get("dy")
        if not isinstance(digest, str) or not isinstance(scene, str) or not scene or len(scene) > 256:
            raise ValueError("choose a valid map and scene")
        if isinstance(dx, bool) or not isinstance(dx, (int, float)) or isinstance(dy, bool) or not isinstance(dy, (int, float)):
            raise TypeError("room offsets must be numeric")
        dx_value, dy_value = float(dx), float(dy)
        if not math.isfinite(dx_value) or not math.isfinite(dy_value) or abs(dx_value) > 100_000 or abs(dy_value) > 100_000:
            raise ValueError("room offsets must be finite bounded pixel values")
        path = self.map_art_runtime_root() / "cache" / "map-art" / digest / "room-offsets.json"
        current = self._read_json(path)
        offsets = current.get("offsets") if current.get("map_sha256") == digest else {}
        if not isinstance(offsets, dict):
            offsets = {}
        offsets[scene] = [dx_value, dy_value]
        atomic_json(path, {"schema_version": 1, "map_sha256": digest, "offsets": offsets})
        return {"scene": scene, "offset": offsets[scene]}

    def download_operator_map_art(self) -> dict[str, object]:
        root = self.map_art_runtime_root() / "cache" / "map-art"
        reference = self._read_json(root / "reference.json")
        digest = reference.get("latest_sha256")
        if isinstance(digest, str):
            cached = root / digest / "map.png"
            if cached.is_file():
                return self._start_map_art_job(source=cached)
        return self._start_map_art_job()

    def _start_map_art_job(self, *, source: Path | None = None) -> dict[str, object]:
        root = self.map_art_runtime_root() / "cache" / "map-art"
        root.mkdir(parents=True, exist_ok=True)
        with self._map_art_lock:
            if self._map_art_thread is not None and self._map_art_thread.is_alive():
                return self.map_art_status()
            from hk_agent.map_art import load_pyvips

            pyvips_available = load_pyvips() is not None
            if not pyvips_available:
                return {"cached": False, "status": "schematic_only",
                        "message": "Instale o extra opcional map para gerar tiles; o mapa esquemático segue disponível."}
            from hk_agent.run_journey import atomic_json

            atomic_json(root / "progress.json", {"status": "starting", "stage": "starting", "percent": 0,
                                                   "completed_bytes": 0, "total_bytes": None})
            self._map_art_thread = threading.Thread(
                target=self._prepare_map_art, args=(root, source), name="operator-map-art", daemon=True,
            )
            self._map_art_thread.start()
        return {"status": "preparing", "message": "Preparação do mapa iniciada; o progresso aparece aqui."}

    def _prepare_map_art(self, root: Path, source: Path | None) -> None:
        from hk_agent.map_art import estimate_tile_bytes, free_disk_bytes, generate_map_tiles
        from hk_agent.run_journey import atomic_json, cache_map_file, download_mediafire_map

        def update(**values: object) -> None:
            current = self._read_json(root / "progress.json")
            current.update(values)
            current["updated_unix_ms"] = int(time.time() * 1000)
            atomic_json(root / "progress.json", current)

        try:
            update(status="running", stage="downloading" if source is None else "validating",
                   percent=0, message="Baixando a arte em cache local." if source is None else "Validando o PNG importado.")
            if source is None:
                def download_progress(done: int, total: int | None) -> None:
                    update(stage="downloading", completed_bytes=done, total_bytes=total,
                           percent=round(done * 100 / total, 1) if total else None,
                           message="Baixando em blocos; um download interrompido pode ser retomado.")

                cached, digest, dimensions, changed = download_mediafire_map(
                    self.map_art_runtime_root(), progress=download_progress, disk_free=free_disk_bytes,
                )
            else:
                cached, digest, dimensions = cache_map_file(source, self.map_art_runtime_root())
                reference = root / "reference.json"
                previous = self._read_json(reference)
                changed = bool(previous.get("sha256") and previous.get("sha256") != digest)
                atomic_json(reference, {"schema_version": 1, "sha256": previous.get("sha256", digest),
                                        "latest_sha256": digest, "source_page": "importado pelo operador",
                                        "attribution": "The Embraced One — hallownest.net; Hollow Knight © Team Cherry"})
            tile_bytes = estimate_tile_bytes(*dimensions)
            available = free_disk_bytes(cached.parent)
            if available < tile_bytes:
                raise OSError(f"espaço insuficiente para tiles: ~{tile_bytes // (1024*1024)} MiB estimados; {available // (1024*1024)} MiB livres")
            update(status="running", stage="tiling", percent=0, sha256=digest,
                   dimensions=list(dimensions), estimated_tile_bytes=tile_bytes,
                   available_bytes=available, reference_changed=changed,
                   message=f"Gerando tiles e miniatura; estimativa de tiles: {tile_bytes // (1024*1024)} MiB.")

            def tile_progress(percent: float) -> None:
                update(stage="tiling", percent=round(percent, 1), sha256=digest,
                       dimensions=list(dimensions), estimated_tile_bytes=tile_bytes,
                       message="Gerando pirâmide Deep Zoom em streaming.")

            metadata = generate_map_tiles(cached, cached.parent, progress=tile_progress)
            if metadata is None:
                update(status="schematic_only", stage="complete", message="Libvips não está disponível; o esquema da jornada continua disponível.")
                return
            from hk_agent.map_art import auto_register_captures
            from hk_agent.run_journey import load_json

            candidates_by_scene: dict[str, list[tuple[Path, object, int]]] = {}
            for progress_file in (self.runtime_root / "shared").glob("*/journey/campaign-progress.json"):
                index = load_json(progress_file)
                for milestone in index.get("milestones", []):
                    if not isinstance(milestone, Mapping) or not isinstance(milestone.get("capture"), str):
                        continue
                    scene = milestone.get("scene")
                    if not isinstance(scene, str) or not scene:
                        continue
                    frame = (progress_file.parent / milestone["capture"]).resolve()
                    if frame.is_file():
                        timestamp = milestone.get("unix_ms", 0)
                        candidates_by_scene.setdefault(scene, []).append(
                            (frame, milestone.get("map_view", {}), timestamp if isinstance(timestamp, int) else 0)
                        )
            current_calibrations = self.map_art_status().get("calibrations", {})
            existing = current_calibrations if isinstance(current_calibrations, Mapping) else {}
            pending: list[tuple[list[tuple[Path, Mapping[str, Any]]], str]] = []
            for scene, rows in candidates_by_scene.items():
                prior = existing.get(scene)
                if isinstance(prior, Mapping) and prior.get("positioned") is True:
                    continue

                captures: list[tuple[Path, Mapping[str, Any]]] = []
                for frame, map_view, _timestamp in rows:
                    if not isinstance(map_view, Mapping) or not _map_view_has_valid_bounds(map_view):
                        continue
                    captures.append((frame, map_view))
                if captures:
                    pending.append((captures, scene))
            auto_results: list[dict[str, object]] = []
            for match_index, (captures, scene) in enumerate(pending):
                update(status="running", stage="matching", percent=100,
                       match_index=match_index + 1, match_count=len(pending),
                       message=f"Posicionando capturas de cenas ({match_index + 1}/{len(pending)}).")
                try:
                    auto_results.append(auto_register_captures(cached, captures, cached.parent, scene=scene))
                except (OSError, RuntimeError, ValueError, TypeError):
                    auto_results.append({"scene": scene, "positioned": False, "confidence": 0.0})
            positioned = sum(result.get("positioned") is True for result in auto_results)
            not_positioned = len(auto_results) - positioned
            update(status="complete", stage="complete", percent=100, sha256=digest,
                   dimensions=list(dimensions), estimated_tile_bytes=tile_bytes,
                   match_count=len(auto_results), positioned=positioned, unpositioned=not_positioned,
                   message=f"Arte pronta no cache local. Cenas posicionadas automaticamente: {positioned}; não posicionadas: {not_positioned}.")
        except Exception as error:  # noqa: BLE001 - worker boundary reports libvips/runtime failures to the panel
            update(status="failed", stage="failed", message=str(error)[:1000])

    def read_map_tile(self, digest: str, level: int, column: int, row: int) -> bytes:
        from hk_agent.map_art import map_cache_directory, read_map_tile

        root = self.map_art_runtime_root() / "cache" / "map-art"
        return read_map_tile(map_cache_directory(root, digest), level, column, row)

    def read_map_thumbnail(self, digest: str) -> bytes:
        from hk_agent.map_art import map_cache_directory, validate_map_hash

        root = (self.map_art_runtime_root() / "cache" / "map-art").resolve(strict=True)
        cache = map_cache_directory(root, validate_map_hash(digest)).resolve(strict=True)
        try:
            cache.relative_to(root)
        except ValueError as error:
            raise ValueError("map path escaped its local cache") from error
        if self._read_json(cache / "tiles.json").get("complete") is not True:
            raise ValueError("map thumbnail is not ready")
        path = cache / "thumbnail.jpg"
        if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("map thumbnail is missing or exceeds its limit")
        return path.read_bytes()

    def save_map_scene_calibration(self, payload: Mapping[str, object]) -> dict[str, object]:
        from hk_agent.run_journey import (
            fit_manual_calibration,
            propagate_room_calibrations,
            save_scene_calibration,
        )

        art = self.map_art_status()
        digest = str(art.get("sha256") or "")
        if not digest or art.get("tiles_ready") is not True:
            raise ValueError("map tiles must be ready before calibration")
        scene, mode, raw_anchors = payload.get("scene"), payload.get("mode"), payload.get("anchors")
        if not isinstance(scene, str) or not scene or len(scene) > 256:
            raise ValueError("choose a valid scene")
        if mode not in {"bounds", "captures"}:
            raise ValueError("calibration mode must be bounds or captures")
        if not isinstance(raw_anchors, list) or len(raw_anchors) < 2 or (mode == "bounds" and len(raw_anchors) != 2):
            raise ValueError("calibration needs two or more point pairs")
        anchors: list[tuple[float, float, float, float]] = []
        for value in raw_anchors[:64]:
            if not isinstance(value, Mapping):
                raise TypeError("each calibration point must include world and map coordinates")
            keys = ("world_x", "world_y", "map_x", "map_y")
            numbers = tuple(value.get(key) for key in keys)
            if any(isinstance(number, bool) or not isinstance(number, (int, float))
                   or not math.isfinite(float(number)) or abs(float(number)) > 10_000_000
                   for number in numbers):
                raise ValueError("calibration coordinates must be finite and bounded")
            anchors.append(tuple(float(number) for number in numbers))  # type: ignore[arg-type]
        transform, scale, rmse = fit_manual_calibration(anchors)
        path = self.map_art_runtime_root() / "cache" / "map-art" / digest / "calibration.json"
        campaign_id = payload.get("campaign_id")
        if not isinstance(campaign_id, str) or not _IDENTIFIER.fullmatch(campaign_id):
            raise ValueError("choose a valid campaign for passage propagation")
        campaign_root = (self.runtime_root / "shared" / campaign_id / "journey").resolve()
        transitions: list[Mapping[str, Any]] = []
        if campaign_root.exists():
            for diary in campaign_root.glob("actor-*/episode-*.json"):
                try:
                    raw = json.loads(diary.read_text("utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                route = raw.get("route") if isinstance(raw, Mapping) else None
                if isinstance(route, list):
                    transitions.extend(row for row in route if isinstance(row, Mapping))
        calibrations = propagate_room_calibrations(
            transitions, anchor_scene=scene, anchor_transform=transform
        )
        save_scene_calibration(path, map_sha256=digest, scene=scene, transform=transform, confidence=1.0)
        for room, calibration in calibrations.items():
            if room == scene:
                continue
            save_scene_calibration(path, map_sha256=digest, scene=room,
                                   transform=calibration["affine"],
                                   confidence=float(calibration["confidence"]))
        tiles = art.get("tiles")
        outside = False
        if isinstance(tiles, Mapping):
            width, height = tiles.get("width"), tiles.get("height")
            if isinstance(width, (int, float)) and isinstance(height, (int, float)):
                outside = any(not (0 <= u <= width and 0 <= v <= height)
                              for _, _, u, v in anchors)
        return {"scene": scene, "mode": mode, "map_sha256": digest,
                "transform": list(transform), "scale": scale, "rmse": rmse,
                "outside_map": outside, "positioned": True,
                "propagated_scenes": sorted(calibrations)}

    def remove_map_scene_calibration(self, payload: Mapping[str, object]) -> dict[str, object]:
        from hk_agent.run_journey import atomic_json

        art = self.map_art_status()
        digest = str(art.get("sha256") or "")
        scene = payload.get("scene")
        if not digest or not isinstance(scene, str) or not scene or len(scene) > 256:
            raise ValueError("choose a valid calibrated scene")
        path = self.map_art_runtime_root() / "cache" / "map-art" / digest / "calibration.json"
        current = self._read_json(path)
        transforms = current.get("transforms")
        if not isinstance(transforms, dict) or scene not in transforms:
            return {"scene": scene, "removed": False}
        transforms.pop(scene, None)
        current["map_sha256"] = digest
        current["schema_version"] = 1
        atomic_json(path, current)
        return {"scene": scene, "removed": True}

    def import_operator_map_art(self, body: Any, length: int) -> dict[str, object]:
        if length < 24 or length > 1_073_741_824:
            raise ValueError("PNG de mapa fora do limite de tamanho")
        from hk_agent.run_journey import cache_map_file

        runtime = self.map_art_runtime_root()
        temporary_root = runtime / "cache" / "map-art"
        temporary_root.mkdir(parents=True, exist_ok=True)
        required = length * 6 + 512 * 1024 * 1024
        available = shutil.disk_usage(temporary_root).free
        if available < required:
            raise OSError(f"espaço insuficiente: precisa de ~{required // (1024*1024)} MiB, há {available // (1024*1024)} MiB livres")
        fd, name = tempfile.mkstemp(prefix=".import-", suffix=".tmp", dir=temporary_root)
        target = Path(name)
        received = 0
        try:
            with os.fdopen(fd, "wb") as output:
                while received < length:
                    chunk = body.read(min(1024 * 1024, length - received))
                    if not chunk:
                        raise ValueError("upload PNG terminou antes do esperado")
                    received += len(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            cached, digest, dimensions = cache_map_file(target, runtime)
            reference = temporary_root / "reference.json"
            previous = self._read_json(reference)
            changed = bool(previous.get("sha256") and previous.get("sha256") != digest)
            atomic_json(reference, {"schema_version": 1, "sha256": previous.get("sha256", digest),
                                    "latest_sha256": digest, "source_page": "importado pelo operador",
                                    "attribution": "The Embraced One — hallownest.net; Hollow Knight © Team Cherry"})
            self._start_map_art_job(source=cached)
            return {"cached": cached.exists(), "sha256": digest, "dimensions": list(dimensions),
                    "reference_changed": changed}
        finally:
            target.unlink(missing_ok=True)

    def read_report(self, relative_path: str) -> tuple[bytes, str]:
        requested = Path(relative_path)
        report_root = (self.artifacts_root / "reports").resolve(strict=False)
        target = (self.project_root / requested).resolve(strict=True)
        try:
            target.relative_to(report_root)
        except ValueError as error:
            raise ValueError("report path is outside artifacts/reports") from error
        if target.suffix not in {".html", ".json"} or not target.is_file():
            raise ValueError("report file was not found")
        content_type = "text/html; charset=utf-8" if target.suffix == ".html" else "application/json; charset=utf-8"
        return target.read_bytes(), content_type

    def create_campaign(self, payload: Mapping[str, object]) -> dict[str, object]:
        self._require_owner()
        spec = CampaignSpec.from_payload(payload)
        directory = self.control_root / "campaigns" / spec.campaign_id
        existing = list(directory.glob("v*.json")) if directory.exists() else []
        version = (
            max((int(path.stem[1:]) for path in existing if path.stem[1:].isdigit()), default=0) + 1
        )
        record: dict[str, object] = {
            **asdict(spec),
            "version": version,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        path = directory / f"v{version:04d}.json"
        atomic_json(path, record)
        record["spec_path"] = path.relative_to(self.project_root).as_posix()
        return record

    @staticmethod
    def _operator_duration(kind: object, value: object) -> float:
        if kind == "until_stopped":
            return 0.0
        if kind == "hours":
            hours = _bounded_float(value, "duration_hours", 0.25, 72.0)
            return hours * 3600.0
        if kind == "until_time":
            if not isinstance(value, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
                raise ValueError("end_time must use HH:MM")
            now = time.localtime()
            hour, minute = (int(part) for part in value.split(":"))
            seconds = (hour * 3600 + minute * 60) - (now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec)
            if seconds <= 0:
                seconds += 24 * 3600
            return float(seconds)
        raise ValueError("duration_kind must be until_stopped, hours, or until_time")

    def start_operator_campaign(
        self, *, instances: object, duration_kind: object, duration_value: object,
        campaign_name: object = None, help_stuck_count: object = 10,
        help_stuck_minutes: object = 60.0,
    ) -> dict[str, object]:
        from hk_agent.config import load_config

        config = load_config(self.project_root / "config.training.local.toml", allow_training=True)
        count = _bounded_int(instances, "instances", 1, 3)
        if count > min(3, config.instance_count):
            raise ValueError(f"This computer has {config.instance_count} managed game instance(s) configured")
        limit_seconds = self._operator_duration(duration_kind, duration_value)
        stuck_count = _bounded_int(help_stuck_count, "help_stuck_count", 1, 1000)
        stuck_minutes = _bounded_float(help_stuck_minutes, "help_stuck_minutes", 5.0, 10080.0)
        seed = secrets.randbelow(2**31)
        now_slug = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        if campaign_name is not None and (
            not isinstance(campaign_name, str) or not campaign_name.strip()
            or len(campaign_name.strip()) > 80
        ):
            raise ValueError("campaign_name must contain 1 to 80 characters")
        name = campaign_name.strip() if isinstance(campaign_name, str) else f"Exploration {time.strftime('%d/%m %H:%M', time.localtime())}"
        already_used_default_id = (
            any(row.get("campaign_id") == "jogada-1" for row in self.list_campaigns())
            or any((self.runtime_root / "shared" / "jogada-1" / "checkpoints").glob("round-*.zip"))
        )
        campaign_id = (f"jogada-1-new-{now_slug}-{secrets.token_hex(2)}"
                       if name.casefold() == "jogada-1" and already_used_default_id
                       else "jogada-1" if name.casefold() == "jogada-1"
                       else f"explore-{now_slug}-{secrets.token_hex(2)}")
        candidate = self._operator_checkpoint_candidate()
        warm_start: tuple[str, str] | None = None
        if candidate is not None:
            warm_start = (str(candidate["checkpoint"]), str(candidate["campaign_id"]))
        payload: dict[str, object] = {
            "campaign_id": campaign_id,
            "name": name,
            "campaign_type": "exploration_training",
            "reward_version": "explore_v4",
            "shared_learner": True,
            "actors": count,
            "rollout_steps": 512,
            "batch_size": 256,
            "checkpoint_interval": 512 * count,
            "max_checkpoints": 100,
            "timesteps": 10_000_000,
            "episodes": 1,
            "horizon": 3000,
            "action_repeat": 3,
            "lease_ms": 80,
            "duration_mode": "continuous",
            "duration_limit_seconds": limit_seconds,
            "seed": seed,
            "save_source": "continuous_playthrough",
            "save_slot": 1,
            "weights": {},
            "challenge_practice_enabled": True,
            "help_requests_enabled": True,
            "help_stuck_count": stuck_count,
            "help_stuck_minutes": stuck_minutes,
        }
        if warm_start is not None:
            payload["warm_start_checkpoint"], payload["warm_start_source_campaign"] = warm_start
        campaign = self.create_campaign(payload)
        launched = self.launch(campaign_id, version=_bounded_int(campaign["version"], "version", 1, 1_000_000), execute=True)
        notice = (f"Warm-start a partir de {warm_start[1]} ({Path(warm_start[0]).name})."
                  if warm_start else "Nenhum checkpoint compatível disponível; a campanha começa do zero.")
        return {"campaign": campaign, "launch": launched, "notice": notice,
                "warm_start_source_campaign": warm_start[1] if warm_start else None,
                "warm_start_checkpoint": warm_start[0] if warm_start else None}

    def continue_operator_campaign(
        self, *, help_stuck_count: object = 10, help_stuck_minutes: object = 60.0,
        campaign_id: object = None, instances: object = None,
    ) -> dict[str, object]:
        stuck_count = _bounded_int(help_stuck_count, "help_stuck_count", 1, 1000)
        stuck_minutes = _bounded_float(help_stuck_minutes, "help_stuck_minutes", 5.0, 10080.0)
        requested_campaign_id = campaign_id if isinstance(campaign_id, str) else None
        if campaign_id is not None and requested_campaign_id is None:
            raise ValueError("campaign_id must be a string")
        candidate = self._operator_checkpoint_candidate(
            allow_unknown_algorithm=True, campaign_id=requested_campaign_id,
        )
        if candidate is None:
            raise RuntimeError("No compatible shared-learner checkpoint is available to continue")
        prior = candidate["campaign"]
        if not isinstance(prior, Mapping):
            raise TypeError("Resumable campaign metadata is invalid")
        campaign_id = str(candidate["campaign_id"])
        shared_root = self.runtime_root / "shared" / campaign_id
        checkpoints = sorted(
            (shared_root / "checkpoints").glob("round-*.zip"),
            key=lambda path: int(path.stem.removeprefix("round-"))
            if path.stem.removeprefix("round-").isdigit() else -1,
        )
        if not checkpoints:
            raise RuntimeError("No shared-learner checkpoint is available to continue")
        source_version = prior.get("reward_version")
        data = dict(prior)
        data.update({
            "campaign_id": (f"{campaign_id[:42]}-v4-{secrets.token_hex(2)}"
                            if source_version == "explore_v3" else campaign_id),
            "name": f"{str(prior.get('name', 'Exploration'))[:55]} v4",
            "seed": _bounded_int(prior.get("seed", 0), "seed", 0, 2**31 - 1),
            "reward_version": "explore_v4",
        })
        data["straggler_timeout_seconds"] = 300.0
        data["max_checkpoints"] = 100
        data["help_requests_enabled"] = True
        data["help_stuck_count"] = stuck_count
        data["help_stuck_minutes"] = stuck_minutes
        data["challenge_practice_enabled"] = True
        if instances is not None:
            from hk_agent.config import load_config

            config = load_config(self.project_root / "config.training.local.toml", allow_training=True)
            actor_count = _bounded_int(instances, "instances", 1, 3)
            if actor_count > min(3, config.instance_count):
                raise ValueError(f"This computer has {config.instance_count} managed game instance(s) configured")
            data["actors"] = actor_count
            data["checkpoint_interval"] = 512 * actor_count
        zero_start_warning = source_version == "explore_v3" and candidate.get("algorithm") is None
        if source_version == "explore_v3" and not zero_start_warning:
            data["warm_start_checkpoint"] = str(checkpoints[-1])
            data["warm_start_source_campaign"] = campaign_id
        else:
            data.pop("warm_start_checkpoint", None)
            data.pop("warm_start_source_campaign", None)
        campaign = self.create_campaign(data)
        new_id = str(campaign["campaign_id"])
        launched = self.launch(
            new_id,
            version=_bounded_int(campaign["version"], "version", 1, 1_000_000),
            execute=True,
        )
        return {
            "campaign": campaign,
            "launch": launched,
            "checkpoint": str(checkpoints[-1]),
            "notice": (
                f"{campaign_id} retomada do checkpoint {checkpoints[-1].name} ({candidate['round']})."
                if not zero_start_warning else
                f"Origem {campaign_id} sem algoritmo compatível no manifesto; v4 começa do zero."
            ),
        }

    def repeat_operator_campaign(self) -> dict[str, object]:
        campaigns = [row for row in self.list_campaigns() if row.get("campaign_type") == "exploration_training" and row.get("reward_version") in {"explore_v3", "explore_v4"} and row.get("shared_learner") is True]
        if not campaigns:
            raise RuntimeError("No operator training campaign is available to repeat")
        prior = campaigns[0]
        data = dict(prior)
        data.pop("version", None)
        data.pop("created_at", None)
        data.pop("spec_path", None)
        data.pop("warm_start_checkpoint", None)
        data.pop("warm_start_source_campaign", None)
        data["campaign_id"] = f"{str(prior['campaign_id'])[:48]}-repeat-{secrets.token_hex(2)}"
        data["name"] = f"{str(prior.get('name', 'Exploration'))[:56]} repeat"
        data["seed"] = secrets.randbelow(2**31)
        campaign = self.create_campaign(data)
        launched = self.launch(str(campaign["campaign_id"]), version=_bounded_int(campaign["version"], "version", 1, 1_000_000), execute=True)
        return {"campaign": campaign, "launch": launched}

    def _resolve_campaign(self, campaign_id: str, version: int | None) -> tuple[CampaignSpec, Path]:
        if not _IDENTIFIER.fullmatch(campaign_id):
            raise ValueError("invalid campaign id")
        directory = self.control_root / "campaigns" / campaign_id
        candidates = sorted(directory.glob("v*.json"))
        if version is not None:
            candidates = [directory / f"v{version:04d}.json"]
        if not candidates or not candidates[-1].is_file():
            raise ValueError("campaign version was not found")
        path = candidates[-1]
        payload = json.loads(path.read_text("utf-8"))
        if isinstance(payload, dict) and payload.get("campaign_type") == "exploration_training" and "reward_version" not in payload:
            payload["reward_version"] = "explore_v1"
        return CampaignSpec.from_payload(payload), path

    def build_command(
        self,
        spec: CampaignSpec,
        *,
        execute: bool,
        stop_file: Path | None = None,
        experimental_save_root: Path | None = None,
        route_memory: Path | None = None,
        seed_override: int | None = None,
        game_install: Path | None = None,
        instance_data_root: Path | None = None,
        log_file: Path | None = None,
        slot_id: int | None = None,
    ) -> list[str]:
        tool_name = CAMPAIGN_TO_TOOL[spec.campaign_type]
        tool_path = (self.project_root / "tools" / tool_name).resolve(strict=True)
        if tool_path.parent != (self.project_root / "tools").resolve():
            raise RuntimeError("tool escaped the allowlisted directory")
        command = [
            _project_python_executable(self.project_root),
            str(tool_path),
            "--seed",
            str(spec.seed if seed_override is None else seed_override),
            "--lease-ms",
            str(spec.lease_ms),
        ]
        command += [
            "--window-width",
            str(spec.window_width),
            "--window-height",
            str(spec.window_height),
        ]
        command += ["--save-source", spec.save_source, "--save-slot", str(spec.save_slot)]
        if execute:
            if game_install is None or instance_data_root is None:
                raise ValueError("execute requires a validated managed installation and data root")
            command += ["--game-install", str(game_install), "--instance-data-root", str(instance_data_root)]
            if log_file is not None:
                command += ["--log-file", str(log_file)]
        if experimental_save_root is not None:
            command += ["--experimental-save-root", str(experimental_save_root)]
        if spec.campaign_type in {"scene_change_probe", "parallel_smoke"} and route_memory is not None:
            command += ["--route-memory", str(route_memory)]
        if spec.campaign_type == "scene_change_probe":
            command += [
                "--episodes",
                str(spec.episodes),
                "--horizon",
                str(spec.horizon),
                "--action-repeat",
                str(spec.action_repeat),
            ]
        elif spec.campaign_type == "parallel_smoke":
            command += [
                "--bridge-smoke-seconds", "90",
                "--episodes", "1", "--horizon", str(spec.horizon),
                "--action-repeat", "1",
            ]
        else:
            command += [
                "--timesteps",
                str(spec.timesteps),
                "--checkpoint-interval",
                str(spec.checkpoint_interval),
                "--max-checkpoints",
                str(spec.max_checkpoints),
                "--horizon",
                str(spec.horizon),
            ]
            if spec.campaign_type in {"visual_motion_training", "exploration_training"}:
                command += ["--rollout-steps", str(spec.rollout_steps), "--batch-size", str(spec.batch_size)]
            if spec.campaign_type == "exploration_training":
                command += ["--task", "exploration", "--reward-version", spec.reward_version, "--ent-coef", str(spec.ent_coef), "--gamma", str(spec.gamma),
                            "--gae-lambda", str(spec.gae_lambda), "--n-epochs", str(spec.n_epochs), "--target-kl", str(spec.target_kl),
                            "--learning-rate", str(spec.learning_rate), "--clip-range", str(spec.clip_range),
                            "--novelty-reward", str(spec.weights.get("novelty_reward", 0.05)),
                            "--novelty-budget", str(spec.weights.get("novelty_budget", 20.0)),
                            "--scene-change-reward", str(spec.weights.get("scene_change_reward", 5.0)),
                            "--enemy-kill-reward", str(spec.weights.get("enemy_kill_reward", 1.0)),
                            "--enemy-kill-reward-budget", str(spec.weights.get("enemy_kill_reward_budget", 5.0)),
                            "--shade-recovery-reward", str(4.0 if spec.reward_version in {"explore_v2", "explore_v3", "explore_v4"} else spec.weights.get("shade_recovery_reward", 1.0)),
                            "--damage-penalty", str(spec.weights.get("damage_penalty", -0.5)),
                            "--death-penalty", str(spec.weights.get("death_penalty", -5.0)),
                            "--run-failed-penalty", str(spec.weights.get("run_failed_penalty", -5.0)),
                            "--geo-gain-reward", str(0.0 if spec.reward_version in {"explore_v2", "explore_v3", "explore_v4"} else spec.weights.get("geo_gain_reward", 0.01)),
                            "--geo-spend-reward", str(0.0 if spec.reward_version in {"explore_v2", "explore_v3", "explore_v4"} else spec.weights.get("geo_spend_reward", 0.02)),
                            "--time-cost", str(spec.weights.get("time_cost", -0.001)),
                            "--stuck-seconds", str(spec.stuck_seconds), "--evaluation-interval", str(spec.evaluation_interval),
                            "--tutorial-hint-level", str(spec.tutorial_hint_level),
                            "--lifelong-memory-root", str(
                                self.runtime_root / "campaign-memory" / spec.campaign_id
                                if spec.reward_version == "explore_v4"
                                else self.runtime_root / "campaign-memory" / spec.campaign_id / f"slot-{slot_id if slot_id is not None else 0}"
                            )]
            if spec.duration_mode == "continuous":
                command.append("--continuous")
            command += [
                "--training-device",
                self.training_device,
                "--torch-threads",
                str(self.torch_threads),
                "--torch-interop-threads",
                str(self.torch_interop_threads),
            ]
            if spec.campaign_type == "scene_change_training" and route_memory is not None:
                command += ["--route-memory", str(route_memory)]
            if spec.worker_session_id is not None:
                from hk_agent.worker_cycle import (
                    load_worker_config,
                    warm_start_from_preflight,
                )

                worker_config = load_worker_config(self.project_root / "config.worker.local.json")
                warm_start = warm_start_from_preflight(worker_config, spec.worker_session_id)
                if warm_start is not None:
                    command += [
                        "--warm-start-policy",
                        str(warm_start.policy),
                        "--warm-start-policy-manifest",
                        str(warm_start.manifest),
                        "--warm-start-policy-sha256",
                        warm_start.policy_sha256,
                        "--warm-start-policy-manifest-sha256",
                        warm_start.manifest_sha256,
                        "--warm-start-policy-manifest-id",
                        warm_start.manifest_id,
                    ]
        for key, value in sorted(spec.weights.items()):
            command += ["--" + key.replace("_", "-"), str(value)]
        if stop_file is not None:
            command += ["--stop-file", str(stop_file)]
        if execute:
            command.append("--execute")
        return command

    @staticmethod
    def _effective_seed(spec: CampaignSpec, slot_id: int | None) -> int:
        """Use the stable real-launch slot as the seed offset; dry-runs use the base seed."""
        return spec.seed if slot_id is None else (spec.seed + slot_id) % 2_147_483_647

    def _is_fence_current(self, slot_id: int, token: str) -> bool:
        with self._lock:
            return self._slot_fences.get(slot_id) == token

    def _fence_checker(self, slot_id: int) -> Callable[[str], bool]:
        def is_current(token: str) -> bool:
            return self._is_fence_current(slot_id, token)

        return is_current

    def _invalidate_slot_fence_locked(self, launch_id: str) -> None:
        for slot_id, owner in self._slot_owners.items():
            if owner == launch_id:
                self._slot_fences[slot_id] = secrets.token_hex(16)
                record = self._launches.get(launch_id)
                if record is not None:
                    record["slot_state"] = "draining"

    def _release_slot_locked(self, launch_id: str) -> None:
        for slot_id, owner in list(self._slot_owners.items()):
            if owner == launch_id:
                self._slot_owners.pop(slot_id, None)
                self._slot_fences.pop(slot_id, None)
                record = self._launches.get(launch_id)
                if record is not None:
                    record["slot_state"] = "released"

    def launch(self, campaign_id: str, *, version: int | None, execute: bool) -> dict[str, object]:
        self._require_owner()
        spec, spec_path = self._resolve_campaign(campaign_id, version)
        if not spec.shared_learner:
            return self._launch_single(campaign_id, version=version, execute=execute)
        if not execute:
            raise RuntimeError("shared learner campaigns require Execute to be enabled")
        from hk_agent.config import load_config

        project_config = load_config(self.project_root / "config.training.local.toml", allow_training=True)
        if spec.actors > project_config.instance_count:
            raise RuntimeError(f"shared learner requests {spec.actors} actors but only {project_config.instance_count} managed slots are configured")
        if spec.actors > 1 and not self.parallel_enabled:
            raise RuntimeError("shared learner with multiple actors requires parallel execution to be enabled")
        with self._lock:
            self._refresh_locked()
            active_slots = sum(1 for record in self._active_records_locked() if record.get("mode") == "execute")
            if active_slots + spec.actors > self.max_active_launches:
                raise RuntimeError("not enough free game slots for shared learner actors")
        version_number = int(spec_path.stem[1:])
        group_id = f"{campaign_id}-v{version_number:04d}"
        shared_root = self.runtime_root / "shared" / campaign_id
        actors: list[dict[str, object]] = []
        learner: dict[str, object] | None = None
        try:
            for _ in range(spec.actors):
                actors.append(
                    self._launch_single(
                        campaign_id,
                        version=version_number,
                        execute=True,
                        shared_role="actor",
                        shared_root=shared_root,
                        shared_group=group_id,
                        shared_actor_count=spec.actors,
                    )
                )
            # Actors wait for the shared task contract. Start them first so their
            # fresh launch identities replace terminal slot state before the
            # learner begins polling old rollout/status files on resume.
            learner = self._launch_shared_learner(
                spec,
                version=version_number,
                group_id=group_id,
                shared_root=shared_root,
            )
        except BaseException:
            for record in [*actors, *([learner] if learner is not None else [])]:
                try:
                    self.stop(str(record["launch_id"]), _shared_member=True)
                except (RuntimeError, ValueError):
                    pass
            raise
        return {
            "status": "RUNNING",
            "campaign_id": campaign_id,
            "campaign_version": version_number,
            "shared_group": group_id,
            "shared_root": str(shared_root),
            "learner": learner,
            "actors": actors,
        }

    def _launch_shared_learner(
        self,
        spec: CampaignSpec,
        *,
        version: int,
        group_id: str,
        shared_root: Path,
    ) -> dict[str, object]:
        self._require_owner()
        launch_id = time.strftime("launch-%Y%m%dT%H%M%S", time.gmtime()) + f"-{secrets.token_hex(3)}"
        started_ns = time.time_ns()
        launch_root = self.control_root / "launches" / launch_id
        launch_root.mkdir(parents=True, exist_ok=False)
        stop_file = launch_root / "stop.request"
        log_path = launch_root / "process.log"
        tool_path = self.project_root / "tools" / "run_shared_learner.py"
        command = [
            _project_python_executable(self.project_root), str(tool_path), "--shared-root", str(shared_root),
            "--stop-file", str(stop_file), "--actors", str(spec.actors),
            "--rollout-steps", str(spec.rollout_steps), "--batch-size", str(spec.batch_size),
            "--checkpoint-interval", str(spec.checkpoint_interval),
            "--max-checkpoints", str(spec.max_checkpoints), "--timesteps", str(spec.timesteps),
            "--seed", str(spec.seed), "--torch-threads", str(self.torch_threads),
            "--torch-interop-threads", str(self.torch_interop_threads),
            "--straggler-timeout-seconds", str(spec.straggler_timeout_seconds),
        ]
        if spec.warm_start_checkpoint:
            warm_path = Path(spec.warm_start_checkpoint).resolve(strict=True)
            allowed_warm_root = (self.runtime_root / "shared").resolve(strict=False)
            try:
                warm_path.relative_to(allowed_warm_root)
            except ValueError as error:
                raise ValueError("warm-start checkpoint must be inside the managed shared runtime") from error
            if warm_path.suffix.lower() != ".zip" or not warm_path.is_file():
                raise ValueError("warm-start checkpoint must be an existing PPO ZIP")
            command += ["--warm-start-checkpoint", str(warm_path),
                        "--warm-start-source-campaign", spec.warm_start_source_campaign or "explore_v3"]
        if spec.duration_mode == "continuous":
            command.append("--continuous")
        record: dict[str, object] = {
            "schema_version": 1, "launch_id": launch_id, "campaign_id": spec.campaign_id,
            "campaign_version": version, "mode": "shared-learner", "role": "learner",
            "shared_group": group_id, "shared_root": str(shared_root),
            "resources": {"slot_id": None, "training_device": "cpu"},
            "started_unix_ns": started_ns, "heartbeat_unix_ns": started_ns,
            "deadline_unix_ns": started_ns + int(spec.duration_limit_seconds * 1_000_000_000) if spec.duration_limit_seconds else None,
            "status": "RUNNING", "phase": "starting", "command": command[1:],
            "log_path": log_path.relative_to(self.project_root).as_posix(),
            "stop_file": stop_file.relative_to(self.project_root).as_posix(),
        }
        environment = clean_bridge_environment(os.environ)
        environment.update({
            "HK_AGENT_LAUNCH_ID": launch_id,
            "OMP_NUM_THREADS": str(self.torch_threads),
            "MKL_NUM_THREADS": str(self.torch_threads),
            "OPENBLAS_NUM_THREADS": str(self.torch_threads),
            "HK_AGENT_CHALLENGE_PRACTICE_ENABLED": "1" if spec.challenge_practice_enabled else "0",
            "HK_AGENT_HELP_REQUESTS_ENABLED": "1" if spec.help_requests_enabled else "0",
            "HK_AGENT_HELP_STUCK_COUNT": str(spec.help_stuck_count),
            "HK_AGENT_HELP_STUCK_MINUTES": str(spec.help_stuck_minutes),
            "HK_AGENT_HELP_REQUESTS_PER_HOUR": str(spec.help_requests_per_hour),
            "HK_AGENT_HELP_FAILED_ATTEMPTS": str(spec.help_failed_attempts),
            "HK_AGENT_HELP_STALLED_HOURS": str(spec.help_stalled_hours),
        })
        log_handle = log_path.open("wb")
        creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        windows_job = WindowsJob.create() if _windows_job_enabled() and subprocess.Popen is _REAL_POPEN else None
        if _windows_job_enabled():
            creation_flags |= getattr(subprocess, "CREATE_SUSPENDED", 0x00000004)
        try:
            process = subprocess.Popen(
                command, cwd=self.project_root, stdin=subprocess.DEVNULL,
                stdout=log_handle, stderr=subprocess.STDOUT, shell=False,
                creationflags=creation_flags, env=environment,
            )
            if windows_job is not None:
                try:
                    windows_job.assign_pid(process.pid)
                    resume_process(process.pid)
                    self._jobs[launch_id] = windows_job
                except BaseException:
                    windows_job.close()
                    process.kill()
                    raise
        except BaseException:
            log_handle.close()
            if windows_job is not None and launch_id not in self._jobs:
                windows_job.close()
            raise
        record["pid"] = process.pid
        record["runner_process"] = process_identity(process.pid)
        record["launch_nonce"] = launch_id
        atomic_json(launch_root / "launch.json", record)
        with self._lock:
            self._launches[launch_id] = record
            self._processes[launch_id] = process
            threading.Thread(
                target=self._wait_for_process,
                args=(launch_id, process, log_handle, launch_root),
                daemon=True,
            ).start()
        return dict(record)

    def _launch_single(
        self,
        campaign_id: str,
        *,
        version: int | None,
        execute: bool,
        shared_role: str | None = None,
        shared_root: Path | None = None,
        shared_group: str | None = None,
        shared_actor_count: int | None = None,
        shared_restart_of: str | None = None,
        challenge_snapshot: Path | None = None,
        challenge_context: str | None = None,
    ) -> dict[str, object]:
        self._require_owner()
        project_config = None
        with self._lock:
            self._refresh_locked()
            active_records = self._active_records_locked()
            if execute and any(record.get("legacy_without_slot") for record in active_records):
                raise RuntimeError("instância legada ativa sem slot; pare-a antes de iniciar outra")
            active_real = sum(record.get("mode") == "execute" for record in active_records)
            active_dry_runs = sum(record.get("mode") == "dry-run" for record in active_records)
            if execute and active_real >= self.max_active_launches:
                for owned_slot, owner in self._slot_owners.items():
                    if self._launches.get(owner, {}).get("status") in {"FAILED", "CANCELLED"}:
                        raise RuntimeError(
                            f"slot {owned_slot} is draining for launch {owner}; preparation has not exited"
                        )
                raise RuntimeError(f"active campaign capacity reached ({self.max_active_launches})")
            if not execute and active_dry_runs >= self._MAX_ACTIVE_DRY_RUNS:
                raise RuntimeError(
                    f"active dry-run capacity reached ({self._MAX_ACTIVE_DRY_RUNS})"
                )
            if execute and any(
                record.get("mode") == "execute" and not self._has_dedicated_save_root(record)
                for record in active_records
            ):
                raise RuntimeError(
                    "an active legacy launch has no dedicated save root; stop it "
                    "before starting parallel execution"
                )
            if execute and active_real and not self.parallel_enabled:
                raise RuntimeError("parallel execution is disabled by config.training.local.toml")
            if execute and not self.allow_execute:
                raise PermissionError("real launches are disabled; restart with --allow-execute")
            spec, spec_path = self._resolve_campaign(campaign_id, version)
            if (
                spec.campaign_type == "scene_change_training"
                and not self.scene_change_gate()["passed"]
            ):
                raise PermissionError(
                    "scene-change PPO is blocked until a positive probe receives a hashed manual gate review"
                )
            if execute:
                from hk_agent.config import load_config

                project_config = load_config(
                    self.project_root / "config.training.local.toml", allow_training=True
                )
                if len(project_config.install_roots) != self.max_active_launches or len(project_config.instance_data_roots) != self.max_active_launches:
                    raise RuntimeError("execute requires one validated managed installation per slot")
            launch_id = (
                time.strftime("launch-%Y%m%dT%H%M%S", time.gmtime()) + f"-{secrets.token_hex(3)}"
            )
            launch_root = self.control_root / "launches" / launch_id
            stop_file = launch_root / "stop.request"
            used_slots = {
                int(resources["slot_id"])
                for record in active_records
                if record.get("mode") == "execute"
                if isinstance((resources := record.get("resources")), Mapping)
                and isinstance(resources.get("slot_id"), int)
                and not isinstance(resources.get("slot_id"), bool)
            }
            used_slots.update(self._slot_owners)
            selected_slot_id: int | None = (
                next(
                    (slot for slot in range(self.max_active_launches) if slot not in used_slots),
                    None,
                )
                if execute
                else None
            )
            slot_id = selected_slot_id
            if execute and slot_id is None:
                raise RuntimeError("no parallel execution slot is available")
            effective_seed = self._effective_seed(spec, slot_id)
            install_root = project_config.install_roots[slot_id] if project_config is not None and slot_id is not None else None
            data_root = project_config.instance_data_roots[slot_id] if project_config is not None and slot_id is not None else None
            save_root = (
                self.runtime_root / "launches" / launch_id / "save-root" if execute else None
            )
            playthrough = (
                resolve_playthrough_root(self.runtime_root, spec.campaign_id, (slot_id or 0) + 1)
                if execute and spec.save_source == "continuous_playthrough" and slot_id is not None
                else None
            )
            route_memory = self._route_memory_path(spec, slot_id, launch_id)
            command = self.build_command(
                spec,
                execute=execute,
                stop_file=stop_file,
                experimental_save_root=save_root,
                route_memory=route_memory,
                seed_override=effective_seed,
                game_install=install_root,
                instance_data_root=data_root,
                log_file=(self.runtime_root / "launches" / launch_id / "Player.log") if execute else None,
                slot_id=slot_id,
            )
            player_log_path = (self.runtime_root / "launches" / launch_id / "Player.log") if execute else None
            log_path = launch_root / "process.log"
            record: dict[str, object] = {
                "schema_version": 2,
                "launch_id": launch_id,
                "campaign_id": campaign_id,
                "campaign_version": int(spec_path.stem[1:]),
                "campaign_spec": {
                    "path": spec_path.relative_to(self.project_root).as_posix(),
                    "sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                },
                "campaign_origin": {
                    "schema_version": 2,
                    "save_source": spec.save_source,
                    "save_slot": spec.save_slot,
                    "isolated_save_root": (str(save_root) if save_root is not None else None),
                },
                "resources": {
                    "slot_id": slot_id,
                    "effective_seed": effective_seed,
                    "training_device": self.training_device,
                    "torch_threads": self.torch_threads,
                    "torch_interop_threads": self.torch_interop_threads,
                    "route_memory": str(route_memory) if route_memory is not None else None,
                    "game_install": str(install_root) if install_root is not None else None,
                    "instance_data_root": str(data_root) if data_root is not None else None,
                    "playthrough_root": str(playthrough) if playthrough is not None else None,
                    "product_name": project_config.instance_product_names[slot_id] if project_config is not None and slot_id is not None else None,
                    "log_file": str(player_log_path) if player_log_path is not None else None,
                },
                "mode": "execute" if execute else "dry-run",
                "started_unix_ns": time.time_ns(),
                "log_path": log_path.relative_to(self.project_root).as_posix(),
                "stop_file": stop_file.relative_to(self.project_root).as_posix(),
                "status": "PREPARING",
                "phase": "reserving",
                "heartbeat_unix_ns": time.time_ns(),
                "command": command[1:],
            }
            if spec.duration_limit_seconds:
                record["deadline_unix_ns"] = _bounded_int(record["started_unix_ns"], "started_unix_ns", 0, 2**63 - 1) + int(spec.duration_limit_seconds * 1_000_000_000)
            self._launches[launch_id] = record
            fence_token = secrets.token_hex(16) if execute and slot_id is not None else None
            if slot_id is not None and fence_token is not None:
                self._slot_owners[slot_id] = launch_id
                self._slot_fences[slot_id] = fence_token
                record["slot_fence_token"] = fence_token
            self._preparing_ids.add(launch_id)
            self._preparing_started[launch_id] = self._monotonic_clock()
            self._preparing_paths[launch_id] = (launch_root, self.runtime_root / "launches" / launch_id)
            watchdog_stop = threading.Event()
            self._preparing_watchdogs[launch_id] = watchdog_stop
            launch_root.mkdir(parents=True, exist_ok=False)
            atomic_json(launch_root / "launch.json", record)
            threading.Thread(
                target=self._preparation_watchdog,
                args=(launch_id, record, launch_root, watchdog_stop),
                daemon=True,
            ).start()

        runtime_launch_root = self.runtime_root / "launches" / launch_id
        try:
            if execute:
                assert project_config is not None and install_root is not None and data_root is not None
                validated_runtime = validate_runtime_root(
                    self.runtime_root,
                    personal=project_config.personal_save_root,
                    experimental=project_config.experimental_save_root,
                    backup=project_config.backup_root,
                    game_install=project_config.game_executable.parent,
                )
                if validated_runtime != project_config.runtime_root:
                    raise ValueError("Control-plane runtime root no longer matches configuration")
                with self._lock:
                    self._check_preparation(launch_id, record, launch_root, runtime_launch_root, "reserving")
                assert save_root is not None
                save_root = ensure_inside_runtime(save_root, validated_runtime)
                runtime_launch_root = ensure_inside_runtime(save_root.parent, validated_runtime)
                runtime_launch_root.mkdir(parents=True, exist_ok=False)
                with self._lock:
                    self._check_preparation(launch_id, record, launch_root, runtime_launch_root, "cloning")
                # Saves live in Unity's per-instance persistentDataPath; never redirect them.
                if install_root is None or data_root is None:
                    raise RuntimeError("validated instance paths are required")
                from hk_agent.instance_marker import load_instance_marker
                marker = load_instance_marker(install_root, expected_index=(slot_id or 0) + 1)
                if Path(marker.persistent_data_path_expected).resolve(strict=False) != data_root.resolve(strict=False):
                    raise RuntimeError("instance marker data root changed after configuration load")
                self._reject_baseline_links(project_config.experimental_save_root)
                assert slot_id is not None and fence_token is not None
                seed_root = project_config.experimental_save_root
                if playthrough is not None:
                    playthrough, _ = initialize_playthrough(
                        playthrough,
                        project_config.experimental_save_root,
                        runtime_root=validated_runtime,
                        personal_root=project_config.personal_save_root,
                        backup_root=project_config.backup_root,
                        game_install=project_config.game_executable.parent,
                        source=spec.save_source,
                        save_slot=spec.save_slot,
                    )
                    seed_root = playthrough
                    resources_record = record["resources"]
                    if not isinstance(resources_record, dict):
                        raise RuntimeError("launch resources record is invalid")
                    resources_record["playthrough_root"] = str(playthrough)
                    atomic_json(launch_root / "launch.json", record)
                if challenge_snapshot is not None:
                    if challenge_context is None:
                        raise ValueError("challenge actor is missing its context")
                    challenge_snapshot = challenge_snapshot.resolve(strict=True)
                    try:
                        challenge_snapshot.relative_to(validated_runtime)
                    except ValueError as error:
                        raise ValueError("challenge snapshot is outside the managed runtime") from error
                    if not challenge_snapshot.is_dir() or challenge_snapshot.is_symlink():
                        raise ValueError("challenge snapshot must be a real directory")
                    seed_root = challenge_snapshot
                    record["challenge_practice"] = {
                        "context": challenge_context,
                        "snapshot": str(challenge_snapshot),
                        "return_to_main": True,
                    }
                prepare_instance_saves(
                    data_root,
                    seed_root,
                    runtime_launch_root,
                    fence_token=fence_token,
                    is_fence_current=lambda token: self._is_fence_current(slot_id, token),
                )
                shutil.copytree(seed_root, save_root, symlinks=False)
                with self._lock:
                    self._check_preparation(launch_id, record, launch_root, runtime_launch_root, "starting")
                if route_memory is not None:
                    route_memory = ensure_inside_runtime(route_memory, validated_runtime)
                    route_memory.parent.mkdir(parents=True, exist_ok=True)
                    legacy_route_memory = (
                        self.artifacts_root
                        / "training"
                        / "reach-scene-change-v1"
                        / f"route-memory-{spec.save_source}-slot-{spec.save_slot}.json"
                    )
                    if (
                        spec.campaign_type == "scene_change_training"
                        and not route_memory.exists()
                        and legacy_route_memory.is_file()
                    ):
                        shutil.copy2(legacy_route_memory, route_memory)
            log_handle = log_path.open("wb")
            creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            windows_job = (
                WindowsJob.create()
                if _windows_job_enabled() and subprocess.Popen is _REAL_POPEN
                else None
            )
            if _windows_job_enabled():
                creation_flags |= getattr(subprocess, "CREATE_SUSPENDED", 0x00000004)
            environment = clean_bridge_environment(os.environ)
            environment.update(
                {
                    "HK_AGENT_LAUNCH_ID": launch_id,
                    "HK_AGENT_GAME_IDENTITY_FILE": str(launch_root / "game-process.json"),
                    "OMP_NUM_THREADS": str(self.torch_threads),
                    "MKL_NUM_THREADS": str(self.torch_threads),
                    "OPENBLAS_NUM_THREADS": str(self.torch_threads),
                    "HK_AGENT_CHALLENGE_PRACTICE_ENABLED": "1" if spec.challenge_practice_enabled else "0",
                    "HK_AGENT_HELP_REQUESTS_ENABLED": "1" if spec.help_requests_enabled else "0",
                    "HK_AGENT_HELP_STUCK_COUNT": str(spec.help_stuck_count),
                    "HK_AGENT_HELP_STUCK_MINUTES": str(spec.help_stuck_minutes),
                    "HK_AGENT_HELP_REQUESTS_PER_HOUR": str(spec.help_requests_per_hour),
                    "HK_AGENT_HELP_FAILED_ATTEMPTS": str(spec.help_failed_attempts),
                    "HK_AGENT_HELP_STALLED_HOURS": str(spec.help_stalled_hours),
                }
            )
            if shared_role == "actor":
                if shared_root is None or shared_actor_count is None:
                    raise RuntimeError("shared actor launch is missing its shared learner contract")
                environment.update({
                    "HK_AGENT_SHARED_ROLE": "actor",
                    "HK_AGENT_SHARED_ROOT": str(shared_root),
                    "HK_AGENT_SHARED_ACTORS": str(shared_actor_count),
                    "HK_AGENT_JOURNEY_ROOT": str(shared_root / "journey"),
                })
                if shared_restart_of is not None:
                    environment["HK_AGENT_RESTART_OF"] = shared_restart_of
                if challenge_context is not None:
                    environment["HK_AGENT_CHALLENGE_CONTEXT"] = challenge_context
                if challenge_snapshot is not None:
                    environment["HK_AGENT_CHALLENGE_SNAPSHOT"] = str(challenge_snapshot)
                record.update({
                    "role": "actor", "shared_group": shared_group,
                    "shared_root": str(shared_root), "shared_actor_count": shared_actor_count,
                })
            if shared_role == "actor":
                if shared_root is None or slot_id is None:
                    raise RuntimeError("shared actor launch has no slot status path")
                from hk_agent.shared_learner import SharedRoundStore

                SharedRoundStore(shared_root).publish_actor_status(
                    slot=slot_id, launch_id=launch_id, status="STARTING",
                    started_unix_ns=record["started_unix_ns"],
                )
            if slot_id is not None:
                assert project_config is not None
                environment["HK_AGENT_SLOT_ID"] = str(slot_id)
                environment["HK_AGENT_GAME_INSTALL"] = str(install_root)
                environment["HK_AGENT_INSTANCE_DATA_ROOT"] = str(data_root)
                environment["HK_AGENT_PERSONAL_ROOT"] = str(project_config.personal_save_root)
                environment["HK_AGENT_PRODUCT_NAME"] = project_config.instance_product_names[slot_id]
                if playthrough is not None:
                    if challenge_context is not None and shared_root is not None:
                        context_digest = hashlib.sha256(challenge_context.encode("utf-8")).hexdigest()[:16]
                        challenge_memory_root = shared_root / "challenge-practice" / f"slot-{slot_id}" / context_digest
                        environment["HK_AGENT_PLAYTHROUGH_ROOT"] = str(challenge_memory_root)
                        resources_record = record.get("resources")
                        if isinstance(resources_record, dict):
                            resources_record["challenge_memory_root"] = str(challenge_memory_root)
                    else:
                        environment["HK_AGENT_PLAYTHROUGH_ROOT"] = str(playthrough)
                environment["HK_AGENT_LOG_FILE"] = str(player_log_path)
                environment["HK_AGENT_OTHER_INSTANCE_DATA_ROOTS"] = os.pathsep.join(str(root) for index, root in enumerate(project_config.instance_data_roots) if index != slot_id)
                if spec.campaign_type == "parallel_smoke":
                    environment["HK_AGENT_SMOKE_EVIDENCE_FILE"] = str(
                        runtime_launch_root / "smoke-bridge-evidence.json"
                    )
            with self._lock:
                self._check_preparation(launch_id, record, launch_root, runtime_launch_root, "starting")
                process = subprocess.Popen(
                    command,
                    cwd=self.project_root,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    creationflags=creation_flags,
                    env=environment,
                )
                if windows_job is not None:
                    windows_job.assign_pid(process.pid)
                    resume_process(process.pid)
                    self._jobs[launch_id] = windows_job
                record.update({"pid": process.pid, "status": "RUNNING"})
                try:
                    runner_identity = process_identity(process.pid)
                except OSError:
                    runner_identity = {"pid": process.pid, "creation_time": None, "executable": None}
                record.update({"runner_process": runner_identity, "launch_nonce": launch_id})
                if _windows_job_enabled():
                    discovered_identity = self._discover_runner_process(process.pid, windows_job)
                    if discovered_identity is not None:
                        record["runner_process"] = discovered_identity
                        atomic_json(launch_root / "runner-process.json", {**discovered_identity, "launch_nonce": launch_id})
                self._preparing_ids.discard(launch_id)
                self._preparing_started.pop(launch_id, None)
                self._preparing_paths.pop(launch_id, None)
                self._preparing_watchdogs.pop(launch_id, None)
                self._processes[launch_id] = process
                atomic_json(launch_root / "launch.json", record)
        except BaseException as error:
            if "process" in locals() and process.poll() is None:
                process.kill()
            job = self._jobs.pop(launch_id, None)
            if job is None and "windows_job" in locals():
                job = windows_job
            if job is not None:
                job.close()
            if "log_handle" in locals():
                log_handle.close()
            with self._lock:
                already_terminal = record.get("status") in {"FAILED", "CANCELLED"}
                if not already_terminal:
                    record.update({"status": "CANCELLED" if launch_id in self._cancel_preparing else "FAILED", "failure": str(error), "finished_unix_ns": time.time_ns()})
                self._preparing_ids.discard(launch_id)
                self._preparing_started.pop(launch_id, None)
                self._preparing_paths.pop(launch_id, None)
                watchdog = self._preparing_watchdogs.pop(launch_id, None)
                if watchdog is not None:
                    watchdog.set()
                atomic_json(launch_root / "launch.json", record)
                self._cancel_preparing.discard(launch_id)
            if shared_role == "actor" and shared_root is not None and slot_id is not None:
                try:
                    from hk_agent.shared_learner import SharedRoundStore

                    SharedRoundStore(shared_root).publish_actor_status(
                        slot=slot_id, launch_id=launch_id,
                        status=str(record.get("status", "FAILED")),
                        failure=str(record.get("failure", error)),
                        finished_unix_ns=time.time_ns(),
                    )
                except (OSError, ValueError):
                    pass
            if execute and data_root is not None and runtime_launch_root.is_dir():
                try:
                    if slot_id is not None and fence_token is not None:
                        capture_instance_saves(
                            data_root,
                            runtime_launch_root,
                            fence_token=fence_token,
                            is_fence_current=lambda token: self._is_fence_current(slot_id, token),
                        )
                    record["final_save"] = str(runtime_launch_root / "final-save")
                    record["final_save_status"] = "captured"
                    if playthrough is not None:
                        persist_playthrough(playthrough, data_root, force_backup=True)
                    atomic_json(launch_root / "launch.json", record)
                except (OSError, ValueError, InterruptedError) as capture_error:
                    record["final_save_status"] = "failed"
                    record["final_save_error"] = f"{type(capture_error).__name__}: {capture_error}"
                    atomic_json(launch_root / "launch.json", record)
            if execute and runtime_launch_root.is_dir() and not (runtime_launch_root / "previous-instance-state").exists():
                shutil.rmtree(ensure_inside_runtime(runtime_launch_root, self.runtime_root))
            with self._lock:
                self._release_slot_locked(launch_id)
            raise

        with self._lock:
            threading.Thread(
                target=self._wait_for_process,
                args=(launch_id, process, log_handle, launch_root),
                daemon=True,
            ).start()
            return dict(record)

    def restart_shared_actor(
        self, launch_id: str, *, reason: str = "supervised actor restart", timeout: float = 90.0,
        challenge_snapshot: Path | None = None, challenge_context: str | None = None,
    ) -> dict[str, object]:
        """Gracefully stop one shared actor and relaunch it in the learner's current round."""
        self._require_owner()
        with self._lock:
            self._refresh_locked()
            record = self._launches.get(launch_id)
            if record is None or record.get("role") != "actor":
                raise ValueError("launch_id is not a shared actor")
            group = record.get("shared_group")
            if not isinstance(group, str):
                raise TypeError("shared actor group is missing")
            now = time.monotonic()
            now_ns = time.time_ns()
            restart_log = Path(str(record.get("shared_root", ""))) / "actor-restarts.jsonl"
            persisted: list[int] = []
            try:
                for line in restart_log.read_text("utf-8").splitlines():
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    stamp = item.get("time_unix_ns") if isinstance(item, Mapping) else None
                    if (isinstance(item, Mapping) and item.get("shared_group") == group
                            and isinstance(stamp, int) and stamp >= now_ns - 3_600_000_000_000):
                        persisted.append(stamp)
            except OSError:
                pass
            recent = [now - max(0.0, (now_ns - stamp) / 1_000_000_000) for stamp in persisted]
            if not persisted:
                recent = [value for value in self._shared_actor_restarts.get(group, []) if now - value < 3600]
            if len(recent) >= 3:
                raise RuntimeError("shared actor restart limit reached (3 per hour)")
            recent.append(now)
            self._shared_actor_restarts[group] = recent
            campaign_id = str(record["campaign_id"])
            version_value = record.get("campaign_version")
            if isinstance(version_value, bool) or not isinstance(version_value, int):
                raise TypeError("shared actor campaign version is invalid")
            version = version_value
            shared_root = Path(str(record["shared_root"]))
            actors_value = record.get("shared_actor_count")
            if isinstance(actors_value, bool) or not isinstance(actors_value, int):
                raise TypeError("shared actor count is invalid")
            actors = actors_value
        if record.get("status") not in {"COMPLETED", "CANCELLED", "FAILED"}:
            from hk_agent.shared_learner import SharedRoundStore

            resources = record.get("resources")
            slot = resources.get("slot_id") if isinstance(resources, Mapping) else None
            if isinstance(slot, bool) or not isinstance(slot, int):
                raise TypeError("shared actor slot id is invalid")
            SharedRoundStore(shared_root).publish_actor_status(
                slot=slot,
                launch_id=launch_id, status="RESTARTING",
                restart_requested_unix_ns=time.time_ns(),
            )
            self.stop(launch_id, _shared_member=True)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                self.status()
                current = self._launches.get(launch_id, {})
                if current.get("status") in {"COMPLETED", "CANCELLED", "FAILED"}:
                    break
                time.sleep(0.25)
            else:
                raise TimeoutError(f"shared actor did not stop cooperatively: {launch_id}")
        replacement = self._launch_single(
            campaign_id, version=version, execute=True, shared_role="actor",
            shared_root=shared_root, shared_group=group, shared_actor_count=actors,
            shared_restart_of=launch_id, challenge_snapshot=challenge_snapshot,
            challenge_context=challenge_context,
        )
        restart_record = {
            "time_unix_ns": time.time_ns(), "shared_group": group,
            "old_launch_id": launch_id, "new_launch_id": replacement["launch_id"],
            "reason": reason, "restart_number_in_hour": len(recent),
        }
        restart_log = shared_root / "actor-restarts.jsonl"
        restart_log.parent.mkdir(parents=True, exist_ok=True)
        with restart_log.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(restart_record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        replacement.update({"restart_of": launch_id, "restart_reason": reason,
                            "restart_number_in_hour": len(recent)})
        launch_path = self.control_root / "launches" / str(replacement["launch_id"]) / "launch.json"
        atomic_json(launch_path, replacement)
        return replacement

    def start_challenge_practice(self, payload: Mapping[str, object]) -> dict[str, object]:
        """Restart one spare actor from an isolated challenge snapshot in the shared campaign."""
        self._require_owner()
        campaign_id, challenge_id, snapshot_slot = payload.get("campaign_id"), payload.get("challenge_id"), payload.get("slot")
        if not isinstance(campaign_id, str) or not campaign_id:
            raise ValueError("campaign_id must be a non-empty string")
        if isinstance(snapshot_slot, bool) or not isinstance(snapshot_slot, int) or snapshot_slot < 1:
            raise ValueError("slot must be a positive one-based instance number")
        if not isinstance(challenge_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", challenge_id):
            raise ValueError("challenge_id is invalid")
        campaign = next(
            (row for row in self.list_campaigns() if row.get("campaign_id") == campaign_id),
            None,
        )
        if campaign is None:
            raise ValueError("campaign was not found")
        if campaign.get("challenge_practice_enabled") is not True:
            raise ValueError("challenge practice is disabled for this campaign")
        active = self.status().get("active_launches", [])
        actors = [row for row in active if isinstance(row, Mapping)
                  and row.get("campaign_id") == campaign_id and row.get("role") == "actor"] if isinstance(active, list) else []
        if len(actors) < 2:
            raise RuntimeError("challenge practice needs at least two active actors so the main playthrough continues")
        shared_roots = {row.get("shared_root") for row in actors}
        if len(shared_roots) != 1 or not isinstance(next(iter(shared_roots)), str):
            raise RuntimeError("active actors do not share one learner")
        main_slot_id = snapshot_slot - 1
        practice = next((row for row in actors
                         if isinstance(row.get("resources"), Mapping)
                         and row["resources"].get("slot_id") != main_slot_id), None)
        if practice is None:
            raise RuntimeError("no spare actor slot is available for challenge practice")
        resources = practice.get("resources")
        slot = resources.get("slot_id") if isinstance(resources, Mapping) else None
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise TypeError("practice actor has no managed slot")
        playthrough = resolve_playthrough_root(self.runtime_root, campaign_id, snapshot_slot)
        snapshot_index = playthrough.parent / "challenge-snapshots" / playthrough.name / "index.json"
        index = self._read_json(snapshot_index)
        snapshots = index.get("snapshots", [])
        matching = [item for item in snapshots if isinstance(item, Mapping)
                    and item.get("challenge_id") == challenge_id and isinstance(item.get("save_path"), str)] if isinstance(snapshots, list) else []
        if not matching:
            raise FileNotFoundError(f"no isolated snapshot exists for challenge {challenge_id}")
        snapshot = Path(str(matching[-1]["save_path"])).resolve(strict=True)
        runtime = self.runtime_root.resolve(strict=True)
        from hk_agent.config import load_config
        personal = load_config(self.project_root / "config.training.local.toml", allow_training=True).personal_save_root.resolve(strict=False)
        try:
            snapshot.relative_to(runtime)
        except ValueError as error:
            raise ValueError("challenge snapshot must be inside the managed runtime") from error
        if not snapshot.is_dir() or snapshot.is_symlink():
            raise ValueError("challenge snapshot must be a real directory")
        if personal and (snapshot == personal or personal in snapshot.parents or snapshot in personal.parents):
            raise ValueError("challenge snapshot points into the personal save tree")
        return self.restart_shared_actor(
            str(practice["launch_id"]), reason=f"challenge practice: {challenge_id}",
            challenge_snapshot=snapshot, challenge_context=f"{matching[-1].get('kind', 'area')}:{challenge_id}",
        )

    def resolve_help_request(self, payload: Mapping[str, object]) -> dict[str, Any]:
        self._require_owner()
        campaign_id, request_id, action = payload.get("campaign_id"), payload.get("request_id"), payload.get("action")
        if not isinstance(campaign_id, str) or not isinstance(request_id, str):
            raise TypeError("campaign_id and request_id must be strings")
        if action not in {"show", "help_now", "now_not", "discard", "complete"}:
            raise ValueError("action must be show, help_now, now_not, discard, or complete")
        campaign = next((row for row in self.list_campaigns() if row.get("campaign_id") == campaign_id), None)
        if campaign is None or campaign.get("help_requests_enabled") is not True:
            raise ValueError("help requests are not enabled for this campaign")
        shared_root = (self.runtime_root / "shared" / campaign_id).resolve(strict=True)
        root = (shared_root / "help-requests").resolve(strict=True)
        if root.parent != shared_root or root.is_symlink():
            raise ValueError("help request store escaped the campaign runtime")
        status = {"show": "demonstration_requested", "help_now": "helping", "discard": "discarded",
                  "complete": "resolved_with_help"}.get(str(action))
        for path in root.glob("slot-*.json"):
            ledger_payload = self._read_json(path)
            rows = ledger_payload.get("requests", [])
            if not isinstance(rows, list) or not any(
                isinstance(row, Mapping) and row.get("request_id") == request_id for row in rows
            ):
                continue
            match = re.fullmatch(r"slot-(\d+)\.json", path.name)
            if match is None:
                continue
            from hk_agent.demonstrations import HelpRequestLedger

            ledger = HelpRequestLedger(root, slot=int(match.group(1)))
            if action == "now_not":
                return ledger.snooze(request_id, minutes=30)
            if action == "complete":
                # The actor saves the recording before it is marked resolved.
                return ledger.resolve(request_id, status="completion_requested")
            return ledger.resolve(request_id, status=str(status))
        raise FileNotFoundError("help request was not found")

    def help_request_snapshot(self, campaign_id: str) -> dict[str, Any]:
        """Read help state without constructing the full operator performance snapshot."""
        if not isinstance(campaign_id, str) or not _IDENTIFIER.fullmatch(campaign_id):
            raise ValueError("invalid campaign_id")
        campaign = next((row for row in self.list_campaigns()
                         if row.get("campaign_id") == campaign_id), None)
        if campaign is None or campaign.get("help_requests_enabled") is not True:
            raise ValueError("help requests are not enabled for this campaign")
        shared_root = (self.runtime_root / "shared" / campaign_id).resolve(strict=True)
        root = shared_root / "help-requests"
        if root.is_symlink():
            raise ValueError("help request store escaped the campaign runtime")
        history: list[dict[str, Any]] = []
        for ledger_path in sorted(root.glob("slot-*.json")) if root.is_dir() else ():
            ledger = self._read_json(ledger_path)
            rows = ledger.get("requests", [])
            if not isinstance(rows, list):
                continue
            for raw in rows:
                if not isinstance(raw, Mapping):
                    continue
                row = dict(raw)
                slot = row.get("slot")
                actor_path = (shared_root / "actors" / f"slot-{slot}.json"
                              if isinstance(slot, int) and not isinstance(slot, bool) and slot >= 0 else None)
                actor = self._read_json(actor_path) if actor_path is not None and not actor_path.is_symlink() else {}
                if actor.get("help_id") == row.get("request_id"):
                    row["help_elapsed_seconds"] = actor.get("help_elapsed_seconds", 0.0)
                    row["help_coverage_percent"] = actor.get("help_coverage_percent", 0.0)
                history.append(row)
        history.sort(key=lambda row: float(row.get("created_unix", 0.0))
                     if isinstance(row.get("created_unix"), (int, float))
                     and not isinstance(row.get("created_unix"), bool) else 0.0)
        opened = [row for row in history if row.get("status") in {
            "pending", "demonstration_requested", "helping", "completion_requested",
        }]
        return {"campaign_id": campaign_id, "help_requests": opened, "help_request_history": history[-128:]}

    def _wait_for_process(
        self,
        launch_id: str,
        process: subprocess.Popen[bytes],
        log_handle: Any,
        launch_root: Path,
    ) -> None:
        resources_before_wait = self._launches.get(launch_id, {}).get("resources")
        playthrough_value = resources_before_wait.get("playthrough_root") if isinstance(resources_before_wait, Mapping) else None
        data_value = resources_before_wait.get("instance_data_root") if isinstance(resources_before_wait, Mapping) else None
        playthrough = Path(playthrough_value) if isinstance(playthrough_value, str) else None
        data_root = Path(data_value) if isinstance(data_value, str) else None
        last_signature = save_signature(data_root) if data_root is not None else {}
        last_persisted = save_signature(playthrough) if playthrough is not None and playthrough.is_dir() else {}
        stable_observations = 0
        last_periodic_backup = time.monotonic()
        while process.poll() is None:
            time.sleep(1.0)
            if playthrough is None or data_root is None or not data_root.is_dir():
                continue
            signature = save_signature(data_root)
            if signature and signature == last_signature:
                stable_observations += 1
            else:
                stable_observations = 0
                last_signature = signature
            if stable_observations >= 2 and signature and signature != last_persisted:
                try:
                    persist_playthrough(playthrough, data_root)
                    last_persisted = signature
                except (OSError, ValueError, RuntimeError):
                    # Keep the previous good playthrough; final shutdown retries.
                    pass
            if time.monotonic() - last_periodic_backup >= 300.0:
                try:
                    backup_playthrough(playthrough, reason="periodic")
                except OSError:
                    pass
                last_periodic_backup = time.monotonic()
        return_code = process.wait()
        log_handle.close()
        # Captured before any draining bookkeeping rewrites the status, so a
        # cooperative stop is never reclassified as an operational failure.
        with self._lock:
            initial_record = self._launches.get(launch_id) or {}
            stop_requested_before_exit = initial_record.get("status") == "STOP_REQUESTED"
        if initial_record.get("role") == "learner":
            learner_status: dict[str, Any] = {}
            try:
                shared_root = Path(str(initial_record["shared_root"]))
                learner_status = json.loads(
                    (shared_root / "learner.json").read_text("utf-8")
                )
            except (OSError, json.JSONDecodeError, KeyError):
                pass
            cooperative_status = learner_status.get("status")
            status = (
                str(cooperative_status)
                if return_code == 0 and cooperative_status in {"CANCELLED", "COMPLETED"}
                else "FAILED"
            )
            failure = None
            if status == "FAILED":
                failure = str(learner_status.get("failure") or
                              f"learner process exited with code {return_code} and status {cooperative_status!r}")
            with self._lock:
                record = self._launches.get(launch_id)
                if record is not None and self._processes.get(launch_id) is process:
                    record.update({
                        "status": status, "return_code": return_code,
                        "finished_unix_ns": time.time_ns(),
                        "final_save_status": "not_applicable",
                        "shared_learner_status": cooperative_status,
                    })
                    if failure is not None:
                        record["failure"] = failure
                    atomic_json(launch_root / "launch.json", record)
                    self._processes.pop(launch_id, None)
                    job = self._jobs.pop(launch_id, None)
                    if job is not None:
                        job.close()
            return
        job = self._jobs.get(launch_id)
        remaining_pids: list[int] = []
        job_failure = False
        if job is not None:
            query_failed = False
            try:
                remaining_pids = job.process_ids()
            except OSError:
                remaining_pids = []
                query_failed = True
            if remaining_pids or query_failed:
                job_failure = True
                with self._lock:
                    record = self._launches.get(launch_id)
                    if record is not None:
                        record["draining_pids"] = remaining_pids
                        drain_reason = (
                            "could not query remaining job processes"
                            if query_failed
                            else "launcher exited while job processes remained"
                        )
                        record["drain_reason"] = drain_reason
                        if stop_requested_before_exit or record.get("status") == "ABORT_REQUESTED":
                            record["status"] = "CANCELLED"
                        else:
                            record["status"] = "FAILED"
                            record["failure"] = drain_reason
                        atomic_json(launch_root / "launch.json", record)
                try:
                    job.terminate()
                except OSError:
                    pass
                while True:
                    try:
                        remaining_pids = job.process_ids()
                    except OSError:
                        time.sleep(0.05)
                        continue
                    if not remaining_pids:
                        break
                    time.sleep(0.05)
                with self._lock:
                    record = self._launches.get(launch_id)
                    if record is not None:
                        record["draining_pids"] = []
                        atomic_json(launch_root / "launch.json", record)
        capture_failure: str | None = None
        with self._lock:
            self._merge_game_identity_locked(launch_id)
            record = self._launches.get(launch_id)
        if record is None:
            return
        resources = record.get("resources")
        data_root_value = resources.get("instance_data_root") if isinstance(resources, Mapping) else None
        if isinstance(data_root_value, str):
            try:
                resources = record.get("resources")
                slot_id = resources.get("slot_id") if isinstance(resources, Mapping) else None
                token = record.get("slot_fence_token")
                capture_instance_saves(
                    Path(data_root_value),
                    self.runtime_root / "launches" / launch_id,
                    fence_token=token if isinstance(token, str) else None,
                    is_fence_current=(
                        (lambda value: self._is_fence_current(slot_id, value))
                        if isinstance(slot_id, int) and not isinstance(slot_id, bool)
                        else None
                    ),
                )
                playthrough_value = resources.get("playthrough_root") if isinstance(resources, Mapping) else None
                if isinstance(playthrough_value, str):
                    persist_playthrough(Path(playthrough_value), Path(data_root_value), force_backup=True)
                record["final_save"] = str(self.runtime_root / "launches" / launch_id / "final-save")
                record["final_save_status"] = "captured"
            except (OSError, ValueError, InterruptedError) as error:
                capture_failure = f"final save capture failed: {type(error).__name__}: {error}"
        with self._lock:
            record = self._launches.get(launch_id)
            if record is not None and self._processes.get(launch_id) is process:
                stop_was_requested = (
                    stop_requested_before_exit or record.get("status") in {"STOP_REQUESTED", "ABORT_REQUESTED"}
                )
                # Draining after a requested stop is cleanup, not a failure; the
                # remaining PIDs stay recorded in draining_pids/drain_reason.
                job_failure = job_failure and not stop_was_requested
                record.update(
                    {
                        "status": (
                            "FAILED"
                            if capture_failure is not None or job_failure
                            else
                            "CANCELLED"
                            if stop_was_requested and (return_code == 0 or "drain_reason" in record)
                            else "COMPLETED"
                            if return_code == 0
                            else "FAILED"
                        ),
                        "return_code": return_code,
                        "finished_unix_ns": time.time_ns(),
                        **({"failure": capture_failure or str(record.get("failure", "launcher exited while job processes remained"))} if capture_failure is not None or job_failure else {}),
                        "final_save_status": "failed" if capture_failure is not None else "captured",
                        **({"final_save_error": capture_failure} if capture_failure is not None else {}),
                    }
                )
                atomic_json(launch_root / "launch.json", record)
                resource_map = record.get("resources")
                if record.get("role") == "actor" and isinstance(resource_map, Mapping):
                    slot = resource_map.get("slot_id")
                    shared_root_value = record.get("shared_root")
                    if isinstance(slot, int) and not isinstance(slot, bool) and isinstance(shared_root_value, str):
                        try:
                            from hk_agent.shared_learner import SharedRoundStore

                            SharedRoundStore(Path(shared_root_value)).publish_actor_status(
                                slot=slot, launch_id=launch_id,
                                status=str(record["status"]),
                                finished_unix_ns=record["finished_unix_ns"],
                            )
                        except (OSError, ValueError):
                            pass
                self._processes.pop(launch_id, None)
                self._release_slot_locked(launch_id)
                job = self._jobs.pop(launch_id, None)
                if job is not None:
                    job.close()

    def _route_memory_path(
        self, spec: CampaignSpec, slot_id: int | None, launch_id: str
    ) -> Path | None:
        """Exclusive route-memory file for one slot; never shared by two instances.

        Training keeps a persistent per-slot memory (seeded from the legacy file);
        probes get their own per-slot memory so diagnostics never pollute training;
        the parallel smoke uses a throwaway file inside its own launch directory.
        """
        if slot_id is None:
            return None
        save_scope = f"{spec.save_source}-slot-{spec.save_slot}"
        if spec.campaign_type == "scene_change_training":
            return self.runtime_root / "route-memory" / save_scope / f"instance-{slot_id}.json"
        if spec.campaign_type == "scene_change_probe":
            return (
                self.runtime_root / "route-memory-probe" / save_scope / f"instance-{slot_id}.json"
            )
        if spec.campaign_type == "parallel_smoke":
            return self.runtime_root / "launches" / launch_id / "route-memory.json"
        return None

    def _merge_game_identity_locked(self, launch_id: str) -> None:
        """Adopt the worker's single-writer sidecar into the panel-owned record."""
        record = self._launches.get(launch_id)
        if record is None:
            return
        sidecar = self.control_root / "launches" / launch_id / "game-process.json"
        try:
            identity = json.loads(sidecar.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(identity, dict) and identity.get("launch_nonce") == launch_id:
            record["game_process"] = identity

    def _refresh_locked(self) -> None:
        # The waiter owns the terminal transition and persists it before releasing
        # the active slot.  Clearing here would race with it and could leave the
        # launch record permanently at STOP_REQUESTED.
        self._enforce_deadlines_locked()
        for tracked_id in set(self._processes) | set(self._reattached_pids):
            self._merge_game_identity_locked(tracked_id)
        for launch_id, pid in list(self._reattached_pids.items()):
            if self._launch_process_matches(self._launches[launch_id], pid):
                continue
            record = self._launches[launch_id]
            stop_was_requested = record.get("status") in {"STOP_REQUESTED", "ABORT_REQUESTED"}
            resources = record.get("resources")
            data_root = resources.get("instance_data_root") if isinstance(resources, Mapping) else None
            capture_failure: str | None = None
            try:
                if isinstance(data_root, str):
                    slot_id = resources.get("slot_id") if isinstance(resources, Mapping) else None
                    token = record.get("slot_fence_token")
                    capture_instance_saves(
                        Path(data_root),
                        self.runtime_root / "launches" / launch_id,
                        fence_token=token if isinstance(token, str) else None,
                        is_fence_current=(
                            self._fence_checker(slot_id)
                            if isinstance(slot_id, int) and not isinstance(slot_id, bool)
                            else None
                        ),
                    )
                    record["final_save"] = str(self.runtime_root / "launches" / launch_id / "final-save")
                    record["final_save_status"] = "captured"
            except (OSError, ValueError, InterruptedError) as error:
                capture_failure = f"final save capture failed: {type(error).__name__}: {error}"
                record["final_save_status"] = "failed"
                record["final_save_error"] = capture_failure
            record.update(
                {
                    "status": "FAILED" if capture_failure else "CANCELLED" if stop_was_requested else "FAILED",
                    "failure": capture_failure or "processo não pertence mais a este launch",
                    "finished_unix_ns": time.time_ns(),
                }
            )
            atomic_json(self.control_root / "launches" / launch_id / "launch.json", record)
            self._reattached_pids.pop(launch_id, None)
            self._release_slot_locked(launch_id)

    def _request_deadline_stop_locked(self, launch_id: str, record: dict[str, object]) -> None:
        if record.get("status") == "STOP_REQUESTED":
            return
        stop_value = record.get("stop_file")
        if not isinstance(stop_value, str) or not stop_value:
            return
        stop_path = Path(stop_value)
        if not stop_path.is_absolute():
            stop_path = self.project_root / stop_path
        pending = stop_path.with_suffix(stop_path.suffix + ".pending")
        pending.write_text(json.dumps({"requested_unix_ns": time.time_ns(), "reason": "time_limit"}), encoding="utf-8")
        os.replace(pending, stop_path)
        record.update({"status": "STOP_REQUESTED", "stop_reason": "time_limit"})
        atomic_json(self.control_root / "launches" / launch_id / "launch.json", record)

    def _enforce_deadlines_locked(self) -> None:
        now = time.time_ns()
        active_ids = set(self._processes) | set(self._reattached_pids) | self._preparing_ids
        expired_groups: set[str] = set()
        for launch_id in active_ids:
            record = self._launches.get(launch_id)
            deadline = record.get("deadline_unix_ns") if isinstance(record, dict) else None
            if isinstance(record, dict) and isinstance(deadline, int) and deadline <= now:
                group = record.get("shared_group")
                if isinstance(group, str):
                    expired_groups.add(group)
                else:
                    self._request_deadline_stop_locked(launch_id, record)
        for group in expired_groups:
            for launch_id in active_ids:
                record = self._launches.get(launch_id)
                if isinstance(record, dict) and record.get("shared_group") == group:
                    self._request_deadline_stop_locked(launch_id, record)

    def _active_records_locked(self) -> list[dict[str, object]]:
        live_ids = set(self._processes) | set(self._reattached_pids) | self._preparing_ids | set(self._slot_owners.values())
        records = []
        for launch_id in live_ids:
            record = dict(self._launches[launch_id])
            shared_root = record.get("shared_root")
            if isinstance(shared_root, str):
                try:
                    shared_status = json.loads((Path(shared_root) / "learner.json").read_text("utf-8"))
                except (OSError, json.JSONDecodeError):
                    shared_status = {}
                if isinstance(shared_status, dict):
                    record["shared_status"] = shared_status
            resources = record.get("resources")
            slot_id = resources.get("slot_id") if isinstance(resources, Mapping) else None
            if isinstance(slot_id, int) and self._slot_owners.get(slot_id) == launch_id:
                record["slot_state"] = (
                    "draining"
                    if record.get("status") in {"FAILED", "CANCELLED"}
                    else "owned"
                )
                record["slot_held_by"] = launch_id
            records.append(record)
        return sorted(records, key=self._started_ns, reverse=True)

    @classmethod
    def _launch_process_matches(cls, record: Mapping[str, object], pid: int) -> bool:
        identity = record.get("runner_process")
        if not isinstance(identity, Mapping):
            return False
        identity_pid = identity.get("pid")
        if isinstance(identity_pid, bool) or not isinstance(identity_pid, int):
            return False
        expected_time = identity.get("creation_time")
        expected_executable = identity.get("executable")
        if expected_time is None or not isinstance(expected_executable, str):
            return False
        if record.get("launch_nonce") != record.get("launch_id") or not cls._pid_is_alive(identity_pid):
            return False
        try:
            actual = process_identity(identity_pid)
        except OSError:
            return False
        return (
            actual.get("pid") == identity_pid
            and actual.get("creation_time") == expected_time
            and isinstance(actual.get("executable"), str)
            and os.path.normcase(str(actual["executable"])) == os.path.normcase(expected_executable)
        )

    @staticmethod
    def _read_runner_identity(launch_root: Path, record: Mapping[str, object], launcher_pid: int) -> dict[str, object] | None:
        try:
            payload = json.loads((launch_root / "runner-process.json").read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or payload.get("launch_nonce") != record.get("launch_id"):
            return None
        identity = {key: payload.get(key) for key in ("pid", "creation_time", "executable", "parent_pid")}
        if identity.get("parent_pid") != launcher_pid:
            return None
        if isinstance(identity.get("pid"), bool) or not isinstance(identity.get("pid"), int):
            return None
        if not isinstance(identity.get("creation_time"), (int, float)) or not isinstance(identity.get("executable"), str):
            return None
        return identity

    @staticmethod
    def _discover_runner_process(launcher_pid: int, job: WindowsJob | None) -> dict[str, object] | None:
        if job is None:
            return None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                pids = job.process_ids()
            except OSError:
                return None
            for pid in pids:
                if pid == launcher_pid:
                    continue
                try:
                    identity = process_identity(pid)
                except OSError:
                    continue
                if identity.get("parent_pid") == launcher_pid:
                    return identity
            time.sleep(0.05)
        return None

    @staticmethod
    def _reject_baseline_links(root: Path) -> None:
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        for directory, dirs, files in os.walk(root, followlinks=False):
            for name in [*dirs, *files]:
                path = Path(directory) / name
                info = os.lstat(path)
                if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & reparse_flag:
                    raise ValueError(f"experimental save baseline contains a link/reparse point: {path}")

    def _check_preparation(
        self, launch_id: str, record: dict[str, object], launch_root: Path,
        runtime_launch_root: Path, phase: str,
    ) -> None:
        if record.get("status") in {"FAILED", "CANCELLED"} or launch_id not in self._preparing_ids:
            if record.get("status") == "FAILED" and record.get("failure") == "launch preparation timed out":
                raise TimeoutError("launch preparation timed out")
            raise InterruptedError("preparation already terminated")
        elapsed = self._monotonic_clock() - self._preparing_started[launch_id]
        if launch_id in self._cancel_preparing:
            raise InterruptedError("preparation cancelled by stop request")
        if elapsed > self.preparation_timeout_seconds:
            raise TimeoutError("launch preparation timed out")
        record.update({"phase": phase, "heartbeat_unix_ns": time.time_ns()})
        atomic_json(launch_root / "launch.json", record)

    def _preparation_watchdog(
        self, launch_id: str, record: dict[str, object], launch_root: Path, stopped: threading.Event
    ) -> None:
        with self._lock:
            started = self._preparing_started[launch_id]
        while True:
            remaining = self.preparation_timeout_seconds - (self._monotonic_clock() - started)
            if remaining <= 0 or stopped.wait(remaining):
                break
        with self._lock:
            if launch_id not in self._preparing_ids:
                return
            cancelled = launch_id in self._cancel_preparing
            timed_out = self._monotonic_clock() - self._preparing_started[launch_id] >= self.preparation_timeout_seconds
            if not cancelled and not timed_out:
                return
            record.update({
                "status": "CANCELLED" if cancelled else "FAILED",
                "failure": "preparation cancelled by stop request" if cancelled else "launch preparation timed out",
                "finished_unix_ns": time.time_ns(),
            })
            self._invalidate_slot_fence_locked(launch_id)
            atomic_json(launch_root / "launch.json", record)

    @staticmethod
    def _has_dedicated_save_root(record: Mapping[str, object]) -> bool:
        origin = record.get("campaign_origin")
        return isinstance(origin, Mapping) and bool(origin.get("isolated_save_root"))

    def _resume_finished_challenge_practices(self) -> None:
        resumptions: list[str] = []
        with self._lock:
            self._refresh_locked()
            for launch_id, record in self._launches.items():
                if (record.get("role") != "actor" or not isinstance(record.get("challenge_practice"), Mapping)
                        or record.get("challenge_return_started") is True
                        or record.get("status") not in {"COMPLETED", "CANCELLED", "FAILED"}):
                    continue
                shared_value = record.get("shared_root")
                resources = record.get("resources")
                slot = resources.get("slot_id") if isinstance(resources, Mapping) else None
                if not isinstance(shared_value, str) or isinstance(slot, bool) or not isinstance(slot, int):
                    continue
                actor_status = self._read_json(Path(shared_value) / "actors" / f"slot-{slot}.json")
                if actor_status.get("challenge_practice_completed") is not True:
                    continue
                main_actor_active = any(
                    other_id != launch_id and other.get("role") == "actor"
                    and other.get("shared_root") == shared_value
                    and other.get("status") in {"RUNNING", "STOP_REQUESTED", "PREPARING"}
                    for other_id, other in self._launches.items()
                )
                if not main_actor_active:
                    continue
                record["challenge_return_started"] = True
                launch_path = self.control_root / "launches" / launch_id / "launch.json"
                atomic_json(launch_path, record)
                resumptions.append(launch_id)
        for launch_id in resumptions:
            try:
                self.restart_shared_actor(launch_id, reason="challenge practice reached exit criteria; return to main playthrough")
            except (OSError, RuntimeError, TypeError, ValueError):
                with self._lock:
                    failed_record = self._launches.get(launch_id)
                    if failed_record is not None:
                        failed_record["challenge_return_started"] = False

    def _supervise_failed_shared_actors(self) -> None:
        """Replace failed actors while their shared learner remains live."""
        candidates: list[str] = []
        with self._lock:
            self._refresh_locked()
            learners = {
                str(row.get("shared_group")): row
                for row in self._launches.values()
                if row.get("role") == "learner" and row.get("status") == "RUNNING"
                and isinstance(row.get("shared_group"), str)
            }
            for launch_id, row in self._launches.items():
                group = row.get("shared_group")
                retry_after = row.get("auto_relaunch_after_unix_ns", 0)
                if (row.get("role") != "actor" or row.get("status") != "FAILED"
                        or not isinstance(group, str) or group not in learners
                        or launch_id in self._shared_actor_supervision
                        or row.get("auto_relaunch_attempted") is True
                        or (isinstance(retry_after, int) and not isinstance(retry_after, bool)
                            and retry_after > time.time_ns())):
                    continue
                shared_value = row.get("shared_root")
                resources = row.get("resources")
                slot = resources.get("slot_id") if isinstance(resources, Mapping) else None
                if not isinstance(shared_value, str) or isinstance(slot, bool) or not isinstance(slot, int):
                    continue
                actor_status = self._read_json(Path(shared_value) / "actors" / f"slot-{slot}.json")
                if actor_status.get("launch_id") != launch_id or actor_status.get("status") != "FAILED":
                    continue
                row["auto_relaunch_in_progress"] = True
                launch_path = self.control_root / "launches" / launch_id / "launch.json"
                atomic_json(launch_path, row)
                self._shared_actor_supervision.add(launch_id)
                candidates.append(launch_id)
        for launch_id in candidates:
            try:
                replacement = self.restart_shared_actor(
                    launch_id, reason="actor process failed while shared learner was running"
                )
                with self._lock:
                    old = self._launches.get(launch_id)
                    if old is not None:
                        old["auto_relaunch_in_progress"] = False
                        old["auto_relaunch_attempted"] = True
                        old["auto_replacement_launch_id"] = replacement.get("launch_id")
                        atomic_json(self.control_root / "launches" / launch_id / "launch.json", old)
            except (OSError, RuntimeError, TypeError, ValueError, TimeoutError) as error:
                with self._lock:
                    old = self._launches.get(launch_id)
                    if old is not None:
                        old["auto_relaunch_in_progress"] = False
                        old["auto_relaunch_error"] = str(error)[:500]
                        delay_ns = 3_600_000_000_000 if "3 per hour" in str(error) else 15_000_000_000
                        old["auto_relaunch_after_unix_ns"] = time.time_ns() + delay_ns
                        atomic_json(self.control_root / "launches" / launch_id / "launch.json", old)
            finally:
                with self._lock:
                    self._shared_actor_supervision.discard(launch_id)

    def _supervise_failed_shared_learners(self) -> None:
        """Stop actors cleanly when their shared learner fails or stalls."""
        actors_to_stop: list[str] = []
        with self._lock:
            self._refresh_locked()
            for learner_id, learner in self._launches.items():
                group = learner.get("shared_group")
                shared_root = learner.get("shared_root")
                if (learner.get("role") != "learner" or not isinstance(group, str)
                        or not isinstance(shared_root, str) or group in self._shared_failed_learners):
                    continue
                learner_status = self._read_json(Path(shared_root) / "learner.json")
                beat = learner_status.get("heartbeat_unix_ns")
                wait_ms = learner_status.get("last_wait_rollouts_ms", 0.0)
                train_ms = learner_status.get("last_train_ms", 0.0)
                straggler_ms = learner_status.get("last_straggler_wait_ms", 0.0)
                expected_round = sum(
                    float(value) / 1000.0
                    for value in (wait_ms, train_ms, straggler_ms)
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                )
                # Shared CPU PPO updates can occasionally exceed ten minutes on
                # memory-constrained machines. Keep one long update from being
                # misreported as a dead learner while retaining a finite timeout.
                stale_after = max(1800.0, 2.0 * expected_round) if expected_round else 1800.0
                launch_started = learner.get("started_unix_ns")
                now_ns = time.time_ns()
                beat_belongs_to_launch = (
                    isinstance(beat, int) and not isinstance(beat, bool)
                    and isinstance(launch_started, int) and not isinstance(launch_started, bool)
                    and beat >= launch_started
                )
                failure = learner.get("failure") or (
                    learner_status.get("failure") if beat_belongs_to_launch else None
                )
                failed = learner.get("status") == "FAILED" or (
                    beat_belongs_to_launch and learner_status.get("status") == "FAILED"
                )
                startup_expired = (
                    isinstance(launch_started, int) and not isinstance(launch_started, bool)
                    and now_ns - launch_started > max(300, int(stale_after * 1_000_000_000))
                )
                beat_stale = (
                    isinstance(beat, int) and not isinstance(beat, bool)
                    and beat_belongs_to_launch
                    and (now_ns - beat) / 1_000_000_000 > stale_after
                )
                stale = beat_stale or (startup_expired and not beat_belongs_to_launch)
                if not failed and learner.get("status") in {"RUNNING", "STARTING"} and stale:
                    failure = f"learner heartbeat expirou após {stale_after:.0f}s"
                    learner_status.update({"status": "FAILED", "failure": failure})
                    atomic_json(Path(shared_root) / "learner.json", learner_status)
                    learner.update({"status": "FAILED", "failure": failure,
                                    "finished_unix_ns": time.time_ns()})
                    atomic_json(self.control_root / "launches" / learner_id / "launch.json", learner)
                    failed = True
                if not failed:
                    continue
                self._shared_failed_learners.add(group)
                for actor_id, actor in self._launches.items():
                    if (actor.get("role") == "actor" and actor.get("shared_group") == group
                            and actor.get("status") in {"RUNNING", "STARTING"}):
                        actors_to_stop.append(actor_id)
        for actor_id in actors_to_stop:
            try:
                self.stop(actor_id, _shared_member=True)
            except (RuntimeError, ValueError):
                pass

    def status(self) -> dict[str, object]:
        if self.observer:
            observed = self._observer_snapshot()
            return {
                "observer": True,
                "allow_execute": False,
                "capacity": self.max_active_launches,
                "parallel_enabled": self.parallel_enabled,
                "active_count": len(observed),
                "active_launches": observed,
                "active": observed[0] if observed else None,
            }
        self._supervise_failed_shared_learners()
        self._supervise_failed_shared_actors()
        self._resume_finished_challenge_practices()
        with self._lock:
            self._refresh_locked()
            active_launches = self._active_records_locked()
            return {
                "allow_execute": self.allow_execute,
                "capacity": self.max_active_launches,
                "parallel_enabled": self.parallel_enabled,
                "resource_profile": {
                    "training_device": self.training_device,
                    "torch_threads_per_launch": self.torch_threads,
                    "torch_interop_threads_per_launch": self.torch_interop_threads,
                    "runtime_root": str(self.runtime_root),
                },
                "active_count": len(active_launches),
                "active_launches": active_launches,
                # Compatibility for clients that predate parallel launches.
                "active": active_launches[0] if active_launches else None,
                "scene_change_gate": self.scene_change_gate(),
            }

    def stop(self, launch_id: str | None = None, *, _shared_member: bool = False) -> dict[str, object]:
        self._require_owner()
        with self._lock:
            self._refresh_locked()
            if launch_id is not None and not _shared_member:
                selected = self._launches.get(launch_id)
                group = selected.get("shared_group") if selected is not None else None
                if isinstance(group, str):
                    members = [
                        item for item in self._active_records_locked()
                        if item.get("shared_group") == group
                    ]
                    results = [
                        self.stop(str(item["launch_id"]), _shared_member=True)
                        for item in members
                    ]
                    return {"status": "STOP_REQUESTED", "shared_group": group, "members": results}
            preparing_ids = set(self._preparing_ids)
            active_ids = set(self._processes) | set(self._reattached_pids)
            if not active_ids and not preparing_ids:
                raise RuntimeError("there is no active campaign process")
            if launch_id is None:
                all_ids = active_ids | preparing_ids
                if len(all_ids) != 1:
                    raise ValueError("launch_id is required when multiple campaigns are active")
                launch_id = next(iter(all_ids))
            if launch_id in preparing_ids:
                self._cancel_preparing.add(launch_id)
                record = self._launches[launch_id]
                record.update({"status": "CANCELLED", "cancel_requested": True, "failure": "preparation cancelled by stop request", "finished_unix_ns": time.time_ns()})
                self._invalidate_slot_fence_locked(launch_id)
                watchdog = self._preparing_watchdogs.get(launch_id)
                if watchdog is not None:
                    watchdog.set()
                paths = self._preparing_paths.get(launch_id)
                if paths:
                    atomic_json(paths[0] / "launch.json", record)
                return {"status": "CANCEL_REQUESTED"}
            if launch_id not in active_ids:
                raise ValueError("launch_id is not an active campaign")
            record = self._launches[launch_id]
            process = self._processes.get(launch_id)
            stop_path = self.project_root / str(record["stop_file"])
            pending = stop_path.with_suffix(stop_path.suffix + ".pending")
            pending.write_text(json.dumps({"requested_unix_ns": time.time_ns()}), encoding="utf-8")
            os.replace(pending, stop_path)
            record["status"] = "STOP_REQUESTED"
            launch_root = stop_path.parent
            atomic_json(launch_root / "launch.json", record)
            pid = process.pid if process is not None else self._reattached_pids[launch_id]
            return {"status": "STOP_REQUESTED", "pid": pid}

    def abort(self, launch_id: str | None = None) -> dict[str, object]:
        """Terminate every process in one shared campaign through owned Job Objects."""
        self._require_owner()
        with self._lock:
            self._refresh_locked()
            active = self._active_records_locked()
            actors = [row for row in active if row.get("role") == "actor"]
            selected = (next((row for row in active if row.get("launch_id") == launch_id), None)
                        if launch_id is not None else (actors[0] if actors else (active[0] if active else None)))
            if selected is None:
                raise RuntimeError("there is no active campaign to abort")
            group = selected.get("shared_group")
            members = ([row for row in active if row.get("shared_group") == group]
                       if isinstance(group, str) else [selected])
            launch_ids = [str(row.get("launch_id")) for row in members]
            jobs = {item: self._jobs.get(item) for item in launch_ids}
            missing = [item for item, job in jobs.items() if job is None]
            if missing:
                raise RuntimeError("owned Job Object unavailable for: " + ", ".join(missing))
            for item in launch_ids:
                record = self._launches[item]
                record["status"] = "ABORT_REQUESTED"
                record["abort_requested"] = True
                atomic_json(self.control_root / "launches" / item / "launch.json", record)
            for job in jobs.values():
                assert job is not None
                job.terminate()
            return {"status": "ABORT_REQUESTED", "shared_group": group, "members": launch_ids}

    def shared_group_stopped(self, shared_group: str) -> bool:
        """A group is stopped only when its learner and every actor are terminal."""
        self.status()
        terminal = {"COMPLETED", "CANCELLED", "FAILED"}
        with self._lock:
            members = [
                row for row in self._launches.values()
                if row.get("shared_group") == shared_group
            ]
            learners = [row for row in members if row.get("role") == "learner"]
            actors = [row for row in members if row.get("role") == "actor"]
            return (
                len(learners) == 1 and bool(actors)
                and learners[0].get("status") in terminal
                and all(row.get("status") in terminal for row in actors)
            )

    def read_log(
        self, offset: int = 0, limit: int = 65_536, launch_id: str | None = None
    ) -> dict[str, object]:
        with self._lock:
            self._refresh_locked()
            if launch_id is None:
                active = self._active_records_locked()
                record = active[0] if active else None
            else:
                record = self._launches.get(launch_id)
                if record is None:
                    launch_path = self.control_root / "launches" / launch_id / "launch.json"
                    try:
                        candidate = json.loads(launch_path.read_text("utf-8"))
                    except (OSError, json.JSONDecodeError):
                        candidate = None
                    record = candidate if isinstance(candidate, dict) else None
        if record is None:
            return {"launch_id": launch_id, "offset": 0, "next_offset": 0, "text": ""}
        recorded_path = Path(str(record["log_path"]))
        path = recorded_path if recorded_path.is_absolute() else self.project_root / recorded_path
        if not path.exists():
            return {"launch_id": launch_id, "offset": 0, "next_offset": 0, "text": ""}
        size = path.stat().st_size
        safe_offset = min(max(0, offset), size)
        with path.open("rb") as stream:
            stream.seek(safe_offset)
            content = stream.read(limit)
        return {
            "launch_id": record.get("launch_id"),
            "offset": safe_offset,
            "next_offset": safe_offset + len(content),
            "text": content.decode("utf-8", errors="replace"),
        }


def parse_json_body(body: bytes) -> Mapping[str, object]:
    if len(body) > 256 * 1024:
        raise ValueError("request body is too large")
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("JSON body must be an object")
    return payload


def require_keys(payload: Mapping[str, object], allowed: Sequence[str]) -> None:
    unknown = set(payload) - set(allowed)
    if unknown:
        raise ValueError(f"unknown fields: {', '.join(sorted(unknown))}")
