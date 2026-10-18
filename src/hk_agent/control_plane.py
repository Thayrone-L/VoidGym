from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hk_agent.bridge_environment import clean_bridge_environment
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
    straggler_timeout_seconds: float = 45.0
    schema_version: int = 7

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> CampaignSpec:
        campaign_type = str(payload.get("campaign_type", ""))
        if campaign_type not in CAMPAIGN_TO_TOOL:
            raise ValueError("campaign_type is not allowlisted")
        shared_learner = payload.get("shared_learner", False)
        if not isinstance(shared_learner, bool):
            raise TypeError("shared_learner must be a boolean")
        if shared_learner and campaign_type not in {"visual_motion_training", "exploration_training"}:
            raise ValueError("shared learner requires a supported PPO training campaign")
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
        save_source = payload.get("save_source", "existing_save")
        if not isinstance(save_source, str) or save_source not in {
            "existing_save",
            "new_game",
        }:
            raise ValueError("save_source must be existing_save or new_game")
        duration_mode = payload.get("duration_mode", "limited")
        if not isinstance(duration_mode, str) or duration_mode not in {
            "limited",
            "continuous",
        }:
            raise ValueError("duration_mode must be limited or continuous")
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
            straggler_timeout_seconds=_bounded_float(
                payload.get("straggler_timeout_seconds", 45.0),
                "straggler_timeout_seconds", 1.0, 300.0,
            ) if shared_learner else 45.0,
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
        self._slot_fences: dict[int, str] = {}
        self.observer = observer
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
        for path in root.glob("*/*.json"):
            try:
                payload = json.loads(path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                payload["spec_path"] = path.relative_to(self.project_root).as_posix()
                campaigns.append(payload)
        return sorted(campaigns, key=lambda item: str(item.get("created_at", "")), reverse=True)

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
    ) -> list[str]:
        tool_name = CAMPAIGN_TO_TOOL[spec.campaign_type]
        tool_path = (self.project_root / "tools" / tool_name).resolve(strict=True)
        if tool_path.parent != (self.project_root / "tools").resolve():
            raise RuntimeError("tool escaped the allowlisted directory")
        command = [
            sys.executable,
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
                command += ["--task", "exploration", "--ent-coef", str(spec.ent_coef), "--gamma", str(spec.gamma),
                            "--gae-lambda", str(spec.gae_lambda), "--n-epochs", str(spec.n_epochs), "--target-kl", str(spec.target_kl),
                            "--learning-rate", str(spec.learning_rate), "--clip-range", str(spec.clip_range),
                            "--novelty-reward", str(spec.weights.get("novelty_reward", 0.05)),
                            "--novelty-budget", str(spec.weights.get("novelty_budget", 20.0)),
                            "--scene-change-reward", str(spec.weights.get("scene_change_reward", 5.0)),
                            "--enemy-kill-reward", str(spec.weights.get("enemy_kill_reward", 1.0)),
                            "--enemy-kill-reward-budget", str(spec.weights.get("enemy_kill_reward_budget", 5.0)),
                            "--shade-recovery-reward", str(spec.weights.get("shade_recovery_reward", 1.0)),
                            "--damage-penalty", str(spec.weights.get("damage_penalty", -0.5)),
                            "--death-penalty", str(spec.weights.get("death_penalty", -5.0)),
                            "--run-failed-penalty", str(spec.weights.get("run_failed_penalty", -5.0)),
                            "--geo-gain-reward", str(spec.weights.get("geo_gain_reward", 0.01)),
                            "--geo-spend-reward", str(spec.weights.get("geo_spend_reward", 0.02)),
                            "--time-cost", str(spec.weights.get("time_cost", -0.001))]
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
        learner = self._launch_shared_learner(
            spec,
            version=version_number,
            group_id=group_id,
            shared_root=shared_root,
        )
        actors: list[dict[str, object]] = []
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
        except BaseException:
            for record in [learner, *actors]:
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
        launch_root = self.control_root / "launches" / launch_id
        launch_root.mkdir(parents=True, exist_ok=False)
        stop_file = launch_root / "stop.request"
        log_path = launch_root / "process.log"
        tool_path = self.project_root / "tools" / "run_shared_learner.py"
        command = [
            sys.executable, str(tool_path), "--shared-root", str(shared_root),
            "--stop-file", str(stop_file), "--actors", str(spec.actors),
            "--rollout-steps", str(spec.rollout_steps), "--batch-size", str(spec.batch_size),
            "--checkpoint-interval", str(spec.checkpoint_interval),
            "--max-checkpoints", str(spec.max_checkpoints), "--timesteps", str(spec.timesteps),
            "--seed", str(spec.seed), "--torch-threads", str(self.torch_threads),
            "--torch-interop-threads", str(self.torch_interop_threads),
            "--straggler-timeout-seconds", str(spec.straggler_timeout_seconds),
        ]
        if spec.duration_mode == "continuous":
            command.append("--continuous")
        record: dict[str, object] = {
            "schema_version": 1, "launch_id": launch_id, "campaign_id": spec.campaign_id,
            "campaign_version": version, "mode": "shared-learner", "role": "learner",
            "shared_group": group_id, "shared_root": str(shared_root),
            "resources": {"slot_id": None, "training_device": "cpu"},
            "started_unix_ns": time.time_ns(), "heartbeat_unix_ns": time.time_ns(),
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
                prepare_instance_saves(
                    data_root,
                    project_config.experimental_save_root,
                    runtime_launch_root,
                    fence_token=fence_token,
                    is_fence_current=lambda token: self._is_fence_current(slot_id, token),
                )
                shutil.copytree(project_config.experimental_save_root, save_root, symlinks=False)
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
                }
            )
            if shared_role == "actor":
                if shared_root is None or shared_actor_count is None:
                    raise RuntimeError("shared actor launch is missing its shared learner contract")
                environment.update({
                    "HK_AGENT_SHARED_ROLE": "actor",
                    "HK_AGENT_SHARED_ROOT": str(shared_root),
                    "HK_AGENT_SHARED_ACTORS": str(shared_actor_count),
                })
                if shared_restart_of is not None:
                    environment["HK_AGENT_RESTART_OF"] = shared_restart_of
                record.update({
                    "role": "actor", "shared_group": shared_group,
                    "shared_root": str(shared_root), "shared_actor_count": shared_actor_count,
                })
            if slot_id is not None:
                assert project_config is not None
                environment["HK_AGENT_SLOT_ID"] = str(slot_id)
                environment["HK_AGENT_GAME_INSTALL"] = str(install_root)
                environment["HK_AGENT_INSTANCE_DATA_ROOT"] = str(data_root)
                environment["HK_AGENT_PERSONAL_ROOT"] = str(project_config.personal_save_root)
                environment["HK_AGENT_PRODUCT_NAME"] = project_config.instance_product_names[slot_id]
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
        self, launch_id: str, *, reason: str = "supervised actor restart", timeout: float = 90.0
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
            shared_restart_of=launch_id,
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

    def _wait_for_process(
        self,
        launch_id: str,
        process: subprocess.Popen[bytes],
        log_handle: Any,
        launch_root: Path,
    ) -> None:
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
                        if stop_requested_before_exit:
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
                record["final_save"] = str(self.runtime_root / "launches" / launch_id / "final-save")
                record["final_save_status"] = "captured"
            except (OSError, ValueError, InterruptedError) as error:
                capture_failure = f"final save capture failed: {type(error).__name__}: {error}"
        with self._lock:
            record = self._launches.get(launch_id)
            if record is not None and self._processes.get(launch_id) is process:
                stop_was_requested = (
                    stop_requested_before_exit or record.get("status") == "STOP_REQUESTED"
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
        for tracked_id in set(self._processes) | set(self._reattached_pids):
            self._merge_game_identity_locked(tracked_id)
        for launch_id, pid in list(self._reattached_pids.items()):
            if self._launch_process_matches(self._launches[launch_id], pid):
                continue
            record = self._launches[launch_id]
            stop_was_requested = record.get("status") == "STOP_REQUESTED"
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
