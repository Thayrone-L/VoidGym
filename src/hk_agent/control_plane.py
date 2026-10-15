from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CAMPAIGN_TO_TOOL = {
    "visual_motion_training": "run_first_training.py",
    "scene_change_probe": "run_scene_change_probe.py",
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
    schema_version: int = 5

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> CampaignSpec:
        campaign_type = str(payload.get("campaign_type", ""))
        if campaign_type not in CAMPAIGN_TO_TOOL:
            raise ValueError("campaign_type is not allowlisted")
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
                "time_cost",
                "novelty_shaping_budget",
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
        if weights.get("shade_recovery_reward", 0.0) > abs(
            weights.get("death_penalty", -2.0)
        ):
            raise ValueError(
                "weights.shade_recovery_reward cannot exceed death penalty magnitude"
            )
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
        if (
            campaign_type != "visual_motion_training"
            and success < 4 * weights.get("enemy_kill_reward", 0.0)
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
        spec = cls(
            campaign_id=identifier,
            name=name,
            campaign_type=campaign_type,
            seed=_bounded_int(payload.get("seed", 7), "seed", 0, 2**31 - 1),
            timesteps=_bounded_int(payload.get("timesteps", 256), "timesteps", 1, 10_000_000),
            checkpoint_interval=_bounded_int(
                payload.get("checkpoint_interval", 128), "checkpoint_interval", 1, 1_000_000
            ),
            episodes=_bounded_int(payload.get("episodes", 3), "episodes", 1, 10_000),
            horizon=_bounded_int(payload.get("horizon", 512), "horizon", 8, 100_000),
            action_repeat=_bounded_int(payload.get("action_repeat", 3), "action_repeat", 1, 32),
            lease_ms=_bounded_int(payload.get("lease_ms", 80), "lease_ms", 20, 5_000),
            window_width=_bounded_int(
                payload.get("window_width", 960), "window_width", 320, 7680
            ),
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
        )
        if spec.campaign_type == "visual_motion_training" and spec.horizon < 16:
            raise ValueError("visual-motion horizon must be at least 16 for calibration")
        return spec


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(pending, path)


class ControlPlane:
    def __init__(
        self,
        project_root: Path,
        *,
        allow_execute: bool = False,
        max_active_launches: int = 2,
    ) -> None:
        if max_active_launches < 1:
            raise ValueError("max_active_launches must be positive")
        self.project_root = project_root.resolve()
        self.artifacts_root = self.project_root / "artifacts"
        self.control_root = self.artifacts_root / "control-plane"
        self.allow_execute = allow_execute
        self.max_active_launches = max_active_launches
        self.csrf_token = secrets.token_urlsafe(32)
        self._lock = threading.RLock()
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._launches: dict[str, dict[str, object]] = {}
        self._reattached_pids: dict[str, int] = {}
        self._recover_active_launch()

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
                ok = ctypes.windll.kernel32.GetExitCodeProcess(
                    handle, ctypes.byref(exit_code)
                )
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
            if (
                not isinstance(record, dict)
                or record.get("status") not in {"RUNNING", "STOP_REQUESTED"}
                or isinstance(pid, bool)
                or not isinstance(pid, int)
            ):
                continue
            if self._pid_is_alive(pid):
                record["reattached"] = True
                launch_id = str(record.get("launch_id", path.parent.name))
                record["launch_id"] = launch_id
                self._launches[launch_id] = record
                self._reattached_pids[launch_id] = pid
                atomic_json(path, record)
                continue
            record.update(
                {
                    "status": (
                        "CANCELLED"
                        if record.get("status") == "STOP_REQUESTED"
                        else "FAILED"
                    ),
                    "failure": "panel restarted after runner exit; exit code unavailable",
                    "finished_unix_ns": time.time_ns(),
                }
            )
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
                    evidence_path.relative_to(
                        (self.artifacts_root / "probes").resolve(strict=True)
                    )
                    evidence_bytes = evidence_path.read_bytes()
                    evidence = json.loads(evidence_bytes)
                except (OSError, KeyError, ValueError, json.JSONDecodeError):
                    continue
                if (
                    review.get("artifact_type") == SCENE_CHANGE_GATE_TYPE
                    and review.get("status") == "APPROVED"
                    and review.get("gate_passed") is True
                    and review.get("manual_reviewed") is True
                    and review.get("evidence_sha256")
                    == hashlib.sha256(evidence_bytes).hexdigest()
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
        spec = CampaignSpec.from_payload(payload)
        directory = self.control_root / "campaigns" / spec.campaign_id
        existing = list(directory.glob("v*.json")) if directory.exists() else []
        version = max((int(path.stem[1:]) for path in existing if path.stem[1:].isdigit()), default=0) + 1
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
    ) -> list[str]:
        tool_name = CAMPAIGN_TO_TOOL[spec.campaign_type]
        tool_path = (self.project_root / "tools" / tool_name).resolve(strict=True)
        if tool_path.parent != (self.project_root / "tools").resolve():
            raise RuntimeError("tool escaped the allowlisted directory")
        command = [sys.executable, str(tool_path), "--seed", str(spec.seed), "--lease-ms", str(spec.lease_ms)]
        command += [
            "--window-width", str(spec.window_width),
            "--window-height", str(spec.window_height),
        ]
        command += ["--save-source", spec.save_source, "--save-slot", str(spec.save_slot)]
        if experimental_save_root is not None:
            command += ["--experimental-save-root", str(experimental_save_root)]
        if spec.campaign_type == "scene_change_probe":
            command += [
                "--episodes", str(spec.episodes),
                "--horizon", str(spec.horizon),
                "--action-repeat", str(spec.action_repeat),
            ]
        else:
            command += [
                "--timesteps", str(spec.timesteps),
                "--checkpoint-interval", str(spec.checkpoint_interval),
                "--max-checkpoints", str(spec.max_checkpoints),
                "--horizon", str(spec.horizon),
            ]
            if spec.duration_mode == "continuous":
                command.append("--continuous")
            if spec.worker_session_id is not None:
                from hk_agent.worker_cycle import (
                    load_worker_config,
                    warm_start_from_preflight,
                )

                worker_config = load_worker_config(
                    self.project_root / "config.worker.local.json"
                )
                warm_start = warm_start_from_preflight(
                    worker_config, spec.worker_session_id
                )
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

    def launch(self, campaign_id: str, *, version: int | None, execute: bool) -> dict[str, object]:
        with self._lock:
            self._refresh_locked()
            if len(self._processes) + len(self._reattached_pids) >= self.max_active_launches:
                raise RuntimeError(
                    f"active campaign capacity reached ({self.max_active_launches})"
                )
            if execute and any(
                record.get("mode") == "execute"
                and not self._has_dedicated_save_root(record)
                for record in self._active_records_locked()
            ):
                raise RuntimeError(
                    "an active legacy launch has no dedicated save root; stop it "
                    "before starting parallel execution"
                )
            if execute and not self.allow_execute:
                raise PermissionError("real launches are disabled; restart with --allow-execute")
            spec, spec_path = self._resolve_campaign(campaign_id, version)
            if spec.campaign_type == "scene_change_training" and not self.scene_change_gate()["passed"]:
                raise PermissionError(
                    "scene-change PPO is blocked until a positive probe receives a hashed manual gate review"
                )
            launch_id = time.strftime("launch-%Y%m%dT%H%M%S", time.gmtime()) + f"-{secrets.token_hex(3)}"
            launch_root = self.control_root / "launches" / launch_id
            launch_root.mkdir(parents=True, exist_ok=False)
            stop_file = launch_root / "stop.request"
            save_root: Path | None = None
            if execute:
                from hk_agent.config import load_config

                project_config = load_config(
                    self.project_root / "config.training.local.toml",
                    allow_training=True,
                )
                save_root = launch_root / "save-root"
                shutil.copytree(project_config.experimental_save_root, save_root)
            command = self.build_command(
                spec,
                execute=execute,
                stop_file=stop_file,
                experimental_save_root=save_root,
            )
            log_path = launch_root / "process.log"
            log_handle = log_path.open("wb")
            creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            try:
                process = subprocess.Popen(
                    command,
                    cwd=self.project_root,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    creationflags=creation_flags,
                )
            except BaseException:
                log_handle.close()
                raise
            record: dict[str, object] = {
                "launch_id": launch_id,
                "campaign_id": campaign_id,
                "campaign_version": int(spec_path.stem[1:]),
                "campaign_spec": {
                    "path": spec_path.relative_to(self.project_root).as_posix(),
                    "sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                },
                "campaign_origin": {
                    "schema_version": 1,
                    "save_source": spec.save_source,
                    "save_slot": spec.save_slot,
                    "isolated_save_root": (
                        save_root.relative_to(self.project_root).as_posix()
                        if save_root is not None
                        else None
                    ),
                },
                "mode": "execute" if execute else "dry-run",
                "pid": process.pid,
                "started_unix_ns": time.time_ns(),
                "log_path": log_path.relative_to(self.project_root).as_posix(),
                "stop_file": stop_file.relative_to(self.project_root).as_posix(),
                "status": "RUNNING",
                "command": command[1:],
            }
            self._processes[launch_id] = process
            self._launches[launch_id] = record
            atomic_json(launch_root / "launch.json", record)
            threading.Thread(
                target=self._wait_for_process,
                args=(launch_id, process, log_handle, launch_root),
                daemon=True,
            ).start()
            return dict(record)

    def _wait_for_process(
        self,
        launch_id: str,
        process: subprocess.Popen[bytes],
        log_handle: Any,
        launch_root: Path,
    ) -> None:
        return_code = process.wait()
        log_handle.close()
        with self._lock:
            record = self._launches.get(launch_id)
            if record is not None and self._processes.get(launch_id) is process:
                stop_was_requested = record.get("status") == "STOP_REQUESTED"
                record.update(
                    {
                        "status": (
                            "CANCELLED"
                            if stop_was_requested and return_code == 0
                            else "COMPLETED" if return_code == 0 else "FAILED"
                        ),
                        "return_code": return_code,
                        "finished_unix_ns": time.time_ns(),
                    }
                )
                atomic_json(launch_root / "launch.json", record)
                self._processes.pop(launch_id, None)

    def _refresh_locked(self) -> None:
        # The waiter owns the terminal transition and persists it before releasing
        # the active slot.  Clearing here would race with it and could leave the
        # launch record permanently at STOP_REQUESTED.
        for launch_id, pid in list(self._reattached_pids.items()):
            if self._pid_is_alive(pid):
                continue
            record = self._launches[launch_id]
            stop_was_requested = record.get("status") == "STOP_REQUESTED"
            record.update(
                {
                    "status": "CANCELLED" if stop_was_requested else "FAILED",
                    "failure": "reattached runner exited; exit code unavailable",
                    "finished_unix_ns": time.time_ns(),
                }
            )
            atomic_json(
                self.control_root / "launches" / launch_id / "launch.json", record
            )
            self._reattached_pids.pop(launch_id, None)

    def _active_records_locked(self) -> list[dict[str, object]]:
        live_ids = set(self._processes) | set(self._reattached_pids)
        return sorted(
            (dict(self._launches[launch_id]) for launch_id in live_ids),
            key=self._started_ns,
            reverse=True,
        )

    @staticmethod
    def _has_dedicated_save_root(record: Mapping[str, object]) -> bool:
        origin = record.get("campaign_origin")
        return isinstance(origin, Mapping) and bool(origin.get("isolated_save_root"))

    def status(self) -> dict[str, object]:
        with self._lock:
            self._refresh_locked()
            active_launches = self._active_records_locked()
            return {
                "allow_execute": self.allow_execute,
                "capacity": self.max_active_launches,
                "active_count": len(active_launches),
                "active_launches": active_launches,
                # Compatibility for clients that predate parallel launches.
                "active": active_launches[0] if active_launches else None,
                "scene_change_gate": self.scene_change_gate(),
            }

    def stop(self, launch_id: str | None = None) -> dict[str, object]:
        with self._lock:
            self._refresh_locked()
            active_ids = set(self._processes) | set(self._reattached_pids)
            if not active_ids:
                raise RuntimeError("there is no active campaign process")
            if launch_id is None:
                if len(active_ids) != 1:
                    raise ValueError("launch_id is required when multiple campaigns are active")
                launch_id = next(iter(active_ids))
            if launch_id not in active_ids:
                raise ValueError("launch_id is not an active campaign")
            record = self._launches[launch_id]
            process = self._processes.get(launch_id)
            stop_path = self.project_root / str(record["stop_file"])
            pending = stop_path.with_suffix(stop_path.suffix + ".pending")
            pending.write_text(
                json.dumps({"requested_unix_ns": time.time_ns()}), encoding="utf-8"
            )
            os.replace(pending, stop_path)
            record["status"] = "STOP_REQUESTED"
            launch_root = stop_path.parent
            atomic_json(launch_root / "launch.json", record)
            pid = process.pid if process is not None else self._reattached_pids[launch_id]
            return {"status": "STOP_REQUESTED", "pid": pid}

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
                    launch_path = (
                        self.control_root / "launches" / launch_id / "launch.json"
                    )
                    try:
                        candidate = json.loads(launch_path.read_text("utf-8"))
                    except (OSError, json.JSONDecodeError):
                        candidate = None
                    record = candidate if isinstance(candidate, dict) else None
        if record is None:
            return {"launch_id": launch_id, "offset": 0, "next_offset": 0, "text": ""}
        path = self.project_root / str(record["log_path"])
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
