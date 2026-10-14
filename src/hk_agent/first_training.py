from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from hk_agent.distributed_knowledge import portable_compatibility_fingerprint
from hk_agent.training_task import VISUAL_MOTION_REWARD_VERSION


class TrainingTaskContract(Protocol):
    def task_contract(self) -> dict[str, Any]: ...


class OperationalTrainingFailure(RuntimeError):
    """Raised when the environment explicitly declares the whole run invalid."""


@dataclass(frozen=True, slots=True)
class FirstTrainingConfig:
    seed: int = 7
    total_timesteps: int = 256
    checkpoint_interval: int = 128
    rollout_steps: int = 128
    batch_size: int = 64
    continuous: bool = False
    max_checkpoints: int = 5
    resume_checkpoint: Path | None = None

    def validate(self) -> None:
        if not 1 <= self.total_timesteps <= 10_000_000:
            raise ValueError("total_timesteps is outside the safe range")
        if not 1 <= self.checkpoint_interval <= 1_000_000:
            raise ValueError("checkpoint_interval is outside the safe range")
        if not self.continuous and self.checkpoint_interval > self.total_timesteps:
            raise ValueError("checkpoint_interval cannot exceed total_timesteps")
        if not self.continuous and self.total_timesteps % self.rollout_steps != 0:
            raise ValueError("total_timesteps must be a multiple of rollout_steps")
        if self.checkpoint_interval % self.rollout_steps != 0:
            raise ValueError("checkpoint_interval must be a multiple of rollout_steps")
        if self.rollout_steps % self.batch_size != 0:
            raise ValueError("batch_size must divide rollout_steps")
        if not 2 <= self.max_checkpoints <= 100:
            raise ValueError("max_checkpoints must be in [2, 100]")
        if self.resume_checkpoint is not None and self.resume_checkpoint.suffix != ".zip":
            raise ValueError("resume_checkpoint must be a .zip archive")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_resume_manifest(
    checkpoint: Path,
    *,
    task_contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = checkpoint.resolve(strict=True)
    manifest_path = resolved.parent.parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("resume checkpoint has no readable training manifest") from error
    if not isinstance(manifest, dict):
        raise TypeError("resume manifest must be an object")
    if manifest.get("task_contract") != dict(task_contract):
        raise ValueError("resume checkpoint task contract differs from this campaign")
    expected_fingerprint = portable_compatibility_fingerprint(task_contract, provenance)
    source_fingerprint = manifest.get("compatibility_fingerprint")
    if not isinstance(source_fingerprint, str):
        source_contract = manifest.get("task_contract")
        source_provenance = manifest.get("provenance")
        if not isinstance(source_contract, Mapping) or not isinstance(
            source_provenance, Mapping
        ):
            raise TypeError("resume checkpoint has no compatible provenance")
        source_fingerprint = portable_compatibility_fingerprint(
            source_contract, source_provenance
        )
    if source_fingerprint != expected_fingerprint:
        raise ValueError("resume checkpoint provenance differs from this campaign")
    entries = manifest.get("checkpoints")
    if not isinstance(entries, list):
        raise TypeError("resume manifest has no checkpoint inventory")
    digest = sha256_file(resolved)
    matched = any(
        isinstance(entry, Mapping)
        and entry.get("sha256") == digest
        for entry in entries
    )
    if not matched:
        raise ValueError("resume checkpoint hash is absent from its manifest")
    return manifest


def train_ppo(
    environment: gym.Env[Any, Any],
    *,
    task: TrainingTaskContract,
    config: FirstTrainingConfig,
    output_root: Path,
    provenance: Mapping[str, Any] | None = None,
    stop_file: Path | None = None,
) -> dict[str, Any]:
    """Run PPO in bounded or cooperative-stop continuous mode."""

    config.validate()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    task_contract = task.task_contract()
    reward_version = str(task_contract.get("reward_version", VISUAL_MOTION_REWARD_VERSION))
    diagnostic_training = bool(task_contract.get("diagnostic_training", True))
    claim = str(
        task_contract.get(
            "claim", "motor-signal diagnostic; not game progress or task success"
        )
    )
    manifest: dict[str, Any] = {
        "schema_version": 3,
        "status": "RUNNING",
        "algorithm": "PPO",
        "seed": config.seed,
        "duration_mode": "continuous" if config.continuous else "limited",
        "total_timesteps": None if config.continuous else config.total_timesteps,
        "checkpoint_interval": config.checkpoint_interval,
        "max_checkpoints": config.max_checkpoints,
        "reward_version": reward_version,
        "diagnostic_training": diagnostic_training,
        "resumable_checkpoint": False,
        "claim": claim,
        "task_contract": task_contract,
        "provenance": dict(provenance or {}),
        "compatibility_fingerprint": portable_compatibility_fingerprint(
            task_contract, dict(provenance or {})
        ),
        "checkpoints": [],
        "pruned_checkpoints": [],
        "gameplay_run_index": 1,
        "gameplay_run_failures": 0,
        "episode_count": 0,
        "shade_pending": False,
        "started_unix_ns": time.time_ns(),
    }
    atomic_json(manifest_path, manifest)
    completed = 0
    try:
        try:
            stable_baselines3 = importlib.import_module("stable_baselines3")
            torch = importlib.import_module("torch")
            PPO = stable_baselines3.PPO
        except ImportError as error:
            raise RuntimeError(
                "training dependencies are absent; install the project training extra"
            ) from error
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        manifest["versions"] = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
        }
        atomic_json(manifest_path, manifest)
        if config.resume_checkpoint is not None:
            resume_path = config.resume_checkpoint.resolve(strict=True)
            source_manifest = _validated_resume_manifest(
                resume_path,
                task_contract=task_contract,
                provenance=dict(provenance or {}),
            )
            model = PPO.load(str(resume_path), env=environment, device="auto")
            completed = int(model.num_timesteps)
            manifest["resumed_from"] = {
                "path": str(resume_path),
                "sha256": sha256_file(resume_path),
                "completed_timesteps": source_manifest.get("completed_timesteps"),
            }
        else:
            model = PPO(
                "MultiInputPolicy",
                environment,
                seed=config.seed,
                n_steps=config.rollout_steps,
                batch_size=config.batch_size,
                verbose=1,
                device="auto",
            )

        run_failures = 0
        episode_count = 0

        def continue_training(locals_: dict[str, Any], _: dict[str, Any]) -> bool:
            nonlocal episode_count, run_failures
            changed = False
            dones = locals_.get("dones", ())
            if isinstance(dones, (list, tuple, np.ndarray)):
                newly_done = sum(bool(item) for item in dones)
                if newly_done:
                    episode_count += newly_done
                    manifest["episode_count"] = episode_count
                    changed = True
            infos = locals_.get("infos", ())
            if isinstance(infos, (list, tuple)):
                for info in infos:
                    if isinstance(info, Mapping) and info.get("run_failed") is True:
                        run_failures += 1
                        manifest["gameplay_run_failures"] = run_failures
                        changed = True
                    if isinstance(info, Mapping):
                        supervisor = info.get("supervisor")
                        if isinstance(supervisor, Mapping):
                            for source, target in (
                                ("gameplay_run_index", "gameplay_run_index"),
                                ("recovery_required", "shade_pending"),
                                ("failed_gameplay_runs", "gameplay_run_failures"),
                            ):
                                value = supervisor.get(source)
                                if value is not None and manifest.get(target) != value:
                                    manifest[target] = value
                                    changed = True
            if changed:
                atomic_json(manifest_path, manifest)
            return True

        def save_checkpoint() -> None:
            if completed <= 0:
                return
            checkpoint_root = output_root / "checkpoints"
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            temporary_archive = checkpoint_root / f".step-{completed}.pending.zip"
            final_archive = checkpoint_root / f"step-{completed}.zip"
            model.save(temporary_archive)
            os.replace(temporary_archive, final_archive)
            checkpoints = manifest["checkpoints"]
            if not isinstance(checkpoints, list):
                raise TypeError("checkpoint manifest state is invalid")
            checkpoints[:] = [
                item
                for item in checkpoints
                if not isinstance(item, Mapping) or item.get("step") != completed
            ]
            checkpoints.append(
                {
                    "step": completed,
                    "path": str(final_archive),
                    "sha256": sha256_file(final_archive),
                }
            )
            manifest["resumable_checkpoint"] = True
            manifest["completed_timesteps"] = completed
            manifest["gameplay_run_failures"] = run_failures
            expired_paths: list[Path] = []
            while len(checkpoints) > config.max_checkpoints:
                expired = checkpoints.pop(0)
                if isinstance(expired, Mapping):
                    expired_path = Path(str(expired.get("path", "")))
                    expired_paths.append(expired_path)
                    pruned = manifest["pruned_checkpoints"]
                    if isinstance(pruned, list):
                        pruned.append(
                            {
                                "step": expired.get("step"),
                                "path": expired.get("path"),
                                "sha256": expired.get("sha256"),
                                "pruned_unix_ns": time.time_ns(),
                            }
                        )
                        del pruned[:-1000]
            atomic_json(manifest_path, manifest)
            # Persist the new inventory before deleting expired archives. A
            # crash may leave an unreferenced old zip, never a manifest that
            # points at a file already removed.
            for expired_path in expired_paths:
                if expired_path.is_file():
                    expired_path.unlink()

        while config.continuous or completed < config.total_timesteps:
            if stop_file is not None and stop_file.is_file():
                manifest.update({"status": "CANCELLED", "completed_timesteps": completed})
                break
            chunk = config.rollout_steps
            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                callback=continue_training,
            )
            completed = int(model.num_timesteps)
            reached_checkpoint = completed % config.checkpoint_interval == 0
            reached_limit = not config.continuous and completed >= config.total_timesteps
            stop_requested = stop_file is not None and stop_file.is_file()
            if reached_checkpoint or reached_limit or stop_requested:
                save_checkpoint()
            if stop_file is not None and stop_file.is_file():
                manifest.update({"status": "CANCELLED", "completed_timesteps": completed})
                break
        if manifest["status"] != "CANCELLED":
            manifest.update({"status": "COMPLETED", "completed_timesteps": completed})
    except BaseException as error:
        manifest.update(
            {
                "status": "FAILED",
                "completed_timesteps": completed,
                "failure": f"{type(error).__name__}: {error}",
            }
        )
        raise
    finally:
        manifest["finished_unix_ns"] = time.time_ns()
        atomic_json(manifest_path, manifest)
        environment.close()
    return manifest
