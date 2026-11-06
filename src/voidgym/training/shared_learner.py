"""File-coordinated recurrent PPO learner for universal policies."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from voidgym.models.policy import UniversalPolicy
from voidgym.training.checkpoints import load_universal, prune_checkpoints, save_universal
from voidgym.training.recurrent_ppo import RecurrentPPOTrainer
from voidgym.training.runtime_monitor import resource_snapshot
from voidgym.training.shared_rollouts import load_rollout, publish_json, rollout_path
from voidgym_shared.learner_contract import learner_monitor_state

_MIN_COMMIT_BYTES = 3 * 1024**3
_MIN_ACTUAL_COMMIT_BYTES = 3 * 1024**3 // 2


def _commit_guard_failure(stats: Mapping[str, Any]) -> str | None:
    current = stats.get("commit_available_bytes")
    potential = stats.get("potential_commit_available_bytes")
    if isinstance(potential, int) and potential < _MIN_COMMIT_BYTES:
        return (f"learner commit guard: {current} bytes current and {potential} bytes potential; "
                "requires 1.5 GiB current and 3 GiB potential")
    if not isinstance(current, int):
        return None
    effective = potential if isinstance(potential, int) else current
    if current >= _MIN_ACTUAL_COMMIT_BYTES and effective >= _MIN_COMMIT_BYTES:
        return None
    return (f"learner commit guard: {current} bytes current and {effective} bytes potential; "
            "requires 1.5 GiB current and 3 GiB potential")


@dataclass(frozen=True)
class _RolloutWait:
    sequences: list[Any]
    missing_actors: tuple[str, ...]
    wait_ms: float
    straggler_wait_ms: float


def _wait_for_commit_guard(
    root: Path, round_id: int, completed_steps: int, *,
    progress: dict[str, Any] | None = None, progress_lock: threading.Lock | None = None,
    stop_file: Path | None = None, timeout_seconds: float = 30 * 60,
    poll_seconds: float = 5.0, snapshot_fn: Any = resource_snapshot,
    monotonic_fn: Any = time.monotonic, sleep_fn: Any = time.sleep,
) -> bool:
    """Wait for current commit recovery, publishing RUNNING heartbeats meanwhile."""
    below_since: float | None = None
    while True:
        if stop_file is not None and stop_file.is_file():
            if progress is not None and progress_lock is not None:
                with progress_lock:
                    progress["waiting_memory"] = False
            return False
        stats = snapshot_fn(shared_root=root)
        potential = stats.get("potential_commit_available_bytes")
        current = stats.get("commit_available_bytes")
        failure = _commit_guard_failure(stats)
        if isinstance(potential, int) and potential < _MIN_COMMIT_BYTES:
            raise MemoryError(failure or "learner potential commit below 3 GiB")
        if not isinstance(current, int) or failure is None:
            if progress is not None and progress_lock is not None:
                with progress_lock:
                    progress["waiting_memory"] = False
            if below_since is not None:
                _publish_monitor_state(root, "RUNNING", round_id, completed_steps,
                                       metrics={"waiting_memory": 0.0}, waiting_memory=False)
            return True
        now = monotonic_fn()
        if below_since is None:
            below_since = now
        if now - below_since >= timeout_seconds:
            raise MemoryError(f"{failure}; remained below the current-commit guard for "
                              f"{timeout_seconds:.0f} seconds")
        if progress is not None and progress_lock is not None:
            with progress_lock:
                progress["waiting_memory"] = True
                progress["phase"] = "waiting_memory"
                progress["phase_started_monotonic"] = now
        _publish_monitor_state(root, "RUNNING", round_id, completed_steps,
                               metrics={"waiting_memory": 1.0}, waiting_memory=True)
        sleep_fn(min(poll_seconds, max(0.0, timeout_seconds - (now - below_since))))


def _publish_monitor_state(
    root: Path, status: str, round_id: int, completed_steps: int,
    *, metrics: dict[str, float] | None = None, failure: str | None = None,
    waiting_memory: bool = False,
    last_wait_rollouts_ms: float = 0.0, last_train_ms: float = 0.0,
    last_straggler_wait_ms: float = 0.0,
) -> None:
    state = learner_monitor_state(
        status=status, round_id=round_id, completed_steps=completed_steps,
        resources=resource_snapshot(shared_root=root), metrics=metrics, failure=failure,
        waiting_memory=waiting_memory,
        last_wait_rollouts_ms=last_wait_rollouts_ms,
        last_train_ms=last_train_ms,
        last_straggler_wait_ms=last_straggler_wait_ms,
    )
    publish_json(root / "heartbeat.json", state)
    publish_json(root / "learner.json", state)


def _self_imitation_update(
    root: Path, sequences: list[Any], model: UniversalPolicy,
    trainer: RecurrentPPOTrainer, *, round_id: int, game_id: str | None,
) -> dict[str, Any]:
    from voidgym.training.self_imitation import (
        SelfImitationBank,
        sequence_rollout,
        train_sil_update,
    )

    bank = SelfImitationBank(root / "self-imitation")
    bank.ingest([sequence_rollout(sequence) for sequence in sequences],
                round_number=round_id, gamma=.995)
    result = train_sil_update(
        model, trainer.optimizer, bank.load(limit=8), game_id=game_id,
        max_grad_norm=trainer.max_grad_norm,
    )
    return {**bank.summary(), **result}


def _demonstration_update(
    root: Path, model: UniversalPolicy, trainer: RecurrentPPOTrainer, *,
    actions: tuple[str, ...], game_id: str | None, round_id: int,
) -> dict[str, float]:
    if game_id is None or game_id not in model.game_extras:
        return {"demo_active": 0.0, "demo_samples": 0.0, "demo_loss": 0.0}
    from voidgym.training.demonstrations import apply_verified_demonstrations

    return apply_verified_demonstrations(
        root, model, trainer.optimizer, actions=actions, game_id=game_id,
        round_number=round_id, max_grad_norm=trainer.max_grad_norm,
    )


def _read_latest(root: Path) -> tuple[int, Path] | None:
    try:
        data = json.loads((root / "latest.json").read_text(encoding="utf-8"))
        round_id, relative = data.get("round"), data.get("policy")
        if type(round_id) is not int or round_id < 0 or not isinstance(relative, str):
            raise ValueError("invalid latest v2 policy pointer")
        path = (root / relative).resolve(strict=True)
        path.relative_to(root.resolve(strict=True))
        if not path.is_dir() or not (path / "manifest.json").is_file():
            raise ValueError("latest v2 policy checkpoint is incomplete")
        return round_id, path
    except FileNotFoundError:
        return None


def _wait_for_rollouts(root: Path, round_id: int, actors: tuple[str, ...],
                       policy_round: int, action_table: tuple[str, ...],
                       device: torch.device, timeout: float,
                       straggler_timeout: float, stop_file: Path | None,
                       progress: dict[str, Any], progress_lock: threading.Lock
                       ) -> _RolloutWait | None:
    if timeout <= 0 or straggler_timeout <= 0:
        raise ValueError("rollout and straggler timeouts must be positive")
    straggler_timeout = min(straggler_timeout, timeout)
    started = time.monotonic()
    deadline = started + timeout
    paths = {actor: rollout_path(root, round_id, actor) for actor in actors}
    while True:
        now = time.monotonic()
        if now >= deadline:
            break
        if stop_file is not None and stop_file.is_file():
            return None
        present = [actor for actor, path in paths.items() if path.is_file()]
        elapsed = now - started
        with progress_lock:
            if present and "first_rollout_monotonic" not in progress:
                progress["first_rollout_monotonic"] = now
            first_rollout = progress.get("first_rollout_monotonic")
            straggler_elapsed = (now - first_rollout if isinstance(first_rollout, float)
                                 else elapsed)
            progress["phase"] = "waiting_rollouts"
            progress["phase_started_monotonic"] = started
            progress["last_wait_rollouts_ms"] = elapsed * 1000.0
            progress["last_straggler_wait_ms"] = straggler_elapsed * 1000.0
        if len(present) == len(actors) or elapsed + 1e-9 >= straggler_timeout:
            sequences = []
            for actor in present:
                sequences.extend(load_rollout(
                    paths[actor], expected_round=round_id, expected_actor=actor,
                    expected_policy_round=policy_round, action_table=action_table,
                    device=device,
                ))
            missing = tuple(actor for actor in actors if actor not in present)
            if not sequences:
                raise TimeoutError(
                    f"no actor published round {round_id} within {straggler_timeout:g}s; missing {list(missing)}"
                )
            if missing:
                straggler_path = root / "stragglers.jsonl"
                straggler_path.parent.mkdir(parents=True, exist_ok=True)
                with straggler_path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps({
                        "round": round_id, "missing_actors": list(missing),
                        "present_actors": present, "wait_ms": elapsed * 1000.0,
                        "time_unix_ns": time.time_ns(),
                    }, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            with progress_lock:
                progress["last_wait_rollouts_ms"] = elapsed * 1000.0
                progress["last_straggler_wait_ms"] = straggler_elapsed * 1000.0 if missing else 0.0
                progress["phase"] = "training"
                progress["phase_started_monotonic"] = time.monotonic()
                progress.pop("first_rollout_monotonic", None)
            return _RolloutWait(
                sequences=sequences, missing_actors=missing,
                wait_ms=elapsed * 1000.0,
                straggler_wait_ms=straggler_elapsed * 1000.0 if missing else 0.0,
            )
        time.sleep(.1)
    missing_at_deadline = [actor for actor, path in paths.items() if not path.is_file()]
    raise TimeoutError(f"actors did not publish round {round_id}: {missing_at_deadline}")


def _setup_learner(
    root: Path, checkpoint: Path, actions: tuple[str, ...], game_id: str | None,
    extras_dim: int, device: torch.device, *, learning_rate: float, clip_range: float,
    entropy_coef: float, c_sem: float, c_inv: float, c_mot: float,
    encoder_scale: float,
) -> tuple[UniversalPolicy, RecurrentPPOTrainer, Path, int, int]:
    policy_root = root / "policies"
    policy_root.mkdir(parents=True, exist_ok=True)
    model = UniversalPolicy(action_count=len(actions))
    if extras_dim:
        assert game_id is not None
        model.add_game_extras(game_id, extras_dim)
    model.to(device)
    trainer = RecurrentPPOTrainer(
        model, learning_rate=learning_rate, clip=clip_range,
        entropy_coef=entropy_coef, c_sem=c_sem, c_inv=c_inv, c_mot=c_mot,
        augment=True, lr_encoder_scale=encoder_scale, game_id=game_id,
    )
    latest = _read_latest(root)
    if latest is None:
        base = load_universal(checkpoint, model, actions)
        steps = int(base.get("trained_steps", 0))
        round_id = int((base.get("metrics") or {}).get("shared_round", 0))
        initial_path = policy_root / f"universal-{steps:012d}"
        if not initial_path.is_dir():
            save_universal(initial_path, model, actions, steps=steps,
                           metrics={"shared_round": float(round_id)},
                           optimizer=trainer.optimizer)
        latest_data = {"schema_version": 1, "round": round_id,
                       "policy": initial_path.relative_to(root).as_posix(),
                       "action_table": list(actions), "trained_steps": steps}
        publish_json(root / "latest.json", latest_data)
    else:
        round_id, initial_path = latest
        saved = load_universal(initial_path, model, actions, optimizer=trainer.optimizer)
        steps = int(saved.get("trained_steps", 0))
    return model.to(device).eval(), trainer, policy_root, steps, round_id


def _train_shared_round(
    root: Path, policy_root: Path, model: UniversalPolicy, trainer: RecurrentPPOTrainer,
    *, round_id: int, completed_steps: int, actors: tuple[str, ...], actions: tuple[str, ...],
    device: torch.device, timeout: float, straggler_timeout: float,
    stop_file: Path | None, progress: dict[str, Any],
    progress_lock: threading.Lock,
    freeze_encoder: bool, freeze_rounds: int, rollout_steps: int,
    sequence_length: int, batch_size: int, epochs: int, game_id: str | None,
) -> tuple[int, int, dict[str, float], dict[str, Any]] | None:
    next_round = round_id + 1
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(not (freeze_encoder and next_round <= freeze_rounds))
    waited = _wait_for_rollouts(root, next_round, actors, round_id,
                                actions, device, timeout, straggler_timeout,
                                stop_file, progress, progress_lock)
    if waited is None:
        return None
    sequences = waited.sequences
    train_started = time.monotonic()
    with progress_lock:
        progress["phase"] = "training"
        progress["phase_started_monotonic"] = train_started
    model.train()
    metrics = trainer.update(
        sequences, epochs=epochs,
        sequence_batch_size=max(1, batch_size // sequence_length),
    )
    if not metrics or any(not math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("v2 PPO returned empty or non-finite metrics")
    sil = _self_imitation_update(root, sequences, model, trainer,
                                 round_id=next_round, game_id=game_id)
    metrics.update({f"sil_{key}": float(value) for key, value in sil.items()
                    if isinstance(value, (int, float)) and value is not None})
    demonstrations = _demonstration_update(
        root, model, trainer, actions=actions, game_id=game_id, round_id=next_round,
    )
    metrics.update(demonstrations)
    train_ms = (time.monotonic() - train_started) * 1000.0
    with progress_lock:
        progress["last_train_ms"] = train_ms
        progress["phase"] = "idle"
        progress["phase_started_monotonic"] = time.monotonic()
    actual_steps = sum(int(item.actions.numel()) for item in sequences)
    completed_steps += actual_steps
    metrics.update({"shared_round": float(next_round),
                    "steps_per_update": float(actual_steps),
                    "train_device_cuda": float(device.type == "cuda")})
    target = policy_root / f"universal-{completed_steps:012d}"
    save_universal(target, model, actions, steps=completed_steps,
                   metrics=metrics, optimizer=trainer.optimizer)
    relative = target.relative_to(root).as_posix()
    publish_json(root / "latest.json", {
        "schema_version": 1, "round": next_round, "policy": relative,
        "action_table": list(actions), "trained_steps": completed_steps,
        "metrics": metrics,
    })
    row = {"round": next_round, "completed_timesteps": completed_steps,
           "policy": relative, **metrics}
    # The HK control plane consumes learner.json for shared-group supervision.
    _publish_monitor_state(root, "RUNNING", next_round, completed_steps,
                           metrics=metrics,
                           last_wait_rollouts_ms=waited.wait_ms,
                           last_train_ms=train_ms,
                           last_straggler_wait_ms=waited.straggler_wait_ms)
    model.eval()
    prune_checkpoints(policy_root, keep=5)
    return next_round, completed_steps, metrics, row


def _validate_learner_contract(
    actors: tuple[str, ...], rollout_steps: int, sequence_length: int,
    total_timesteps: int, continuous: bool, batch_size: int,
    freeze_encoder_rounds: int, game_extras_dim: int, game_id: str | None,
    device_name: str,
) -> torch.device:
    if not actors or len(set(actors)) != len(actors):
        raise ValueError("actor ids must be unique and nonempty")
    if any(not actor or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in actor) for actor in actors):
        raise ValueError("actor id contains unsafe characters")
    if rollout_steps < 1 or sequence_length < 1 or rollout_steps % sequence_length:
        raise ValueError("rollout_steps must be a positive multiple of sequence_length")
    if not continuous and total_timesteps < rollout_steps * len(actors):
        raise ValueError("total_timesteps must contain at least one full shared round")
    if batch_size < 1 or rollout_steps % batch_size:
        raise ValueError("batch_size must divide the actor rollout length")
    if freeze_encoder_rounds < 0:
        raise ValueError("freeze_encoder_rounds must be non-negative")
    if game_extras_dim < 0 or (game_extras_dim and not game_id):
        raise ValueError("game_extras_dim requires a game_id and cannot be negative")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the v2 learner but is unavailable")
    return torch.device(device_name)


def _start_learner_heartbeat(
    root: Path, round_id: int, completed_steps: int, device: torch.device,
) -> tuple[threading.Event, threading.Thread, dict[str, Any], threading.Lock]:
    stopped = threading.Event()
    lock = threading.Lock()
    progress: dict[str, Any] = {
        "round": round_id, "steps": completed_steps,
        "phase": "starting", "phase_started_monotonic": time.monotonic(),
        "last_wait_rollouts_ms": 0.0, "last_train_ms": 0.0,
        "last_straggler_wait_ms": 0.0,
        "waiting_memory": False,
    }

    def run() -> None:
        while not stopped.wait(5.0):
            with lock:
                current_round, current_steps = progress["round"], progress["steps"]
                wait_ms = float(progress["last_wait_rollouts_ms"])
                train_ms = float(progress["last_train_ms"])
                straggler_ms = float(progress["last_straggler_wait_ms"])
                phase = progress["phase"]
                phase_started = progress["phase_started_monotonic"]
                waiting_memory = bool(progress.get("waiting_memory", False))
                if phase == "waiting_rollouts":
                    wait_ms = max(wait_ms, (time.monotonic() - phase_started) * 1000.0)
                elif phase == "training":
                    train_ms = max(train_ms, (time.monotonic() - phase_started) * 1000.0)
            payload = {"learner_pid": float(os.getpid()),
                       **resource_snapshot(device.type, shared_root=root)}
            _publish_monitor_state(root, "RUNNING", current_round, current_steps,
                                   metrics=payload,
                                   last_wait_rollouts_ms=wait_ms,
                                   last_train_ms=train_ms,
                                   last_straggler_wait_ms=straggler_ms,
                                   waiting_memory=waiting_memory)

    thread = threading.Thread(target=run, name="hk-learner-heartbeat", daemon=True)
    thread.start()
    return stopped, thread, progress, lock


def _train_rounds(
    root: Path, model: UniversalPolicy, trainer: RecurrentPPOTrainer,
    policy_root: Path, device: torch.device, *, round_id: int, completed_steps: int,
    actors: tuple[str, ...], action_table: tuple[str, ...], continuous: bool,
    total_timesteps: int, rollout_steps: int, sequence_length: int, batch_size: int,
    epochs: int, freeze_encoder_rounds: int, game_id: str | None,
    wait_timeout: float, straggler_timeout: float, stop_file: Path | None,
    progress: dict[str, Any],
    progress_lock: threading.Lock,
) -> tuple[str, int, int, list[dict[str, Any]]]:
    rounds: list[dict[str, Any]] = []
    status = "RUNNING"
    while continuous or completed_steps < total_timesteps:
        if not _wait_for_commit_guard(
            root, round_id, completed_steps, progress=progress,
            progress_lock=progress_lock, stop_file=stop_file,
        ):
            status = "CANCELLED"
            break
        update = _train_shared_round(
            root, policy_root, model, trainer, round_id=round_id,
            completed_steps=completed_steps, actors=actors, actions=action_table,
            device=device, timeout=wait_timeout,
            straggler_timeout=straggler_timeout, stop_file=stop_file,
            freeze_encoder=True, freeze_rounds=freeze_encoder_rounds,
            rollout_steps=rollout_steps, sequence_length=sequence_length,
            batch_size=batch_size, epochs=epochs, game_id=game_id,
            progress=progress, progress_lock=progress_lock,
        )
        if update is None:
            status = "CANCELLED"
            break
        round_id, completed_steps, _metrics, row = update
        rounds.append(row)
        with progress_lock:
            progress["round"], progress["steps"] = round_id, completed_steps
    if status == "RUNNING":
        status = "COMPLETED"
    return status, round_id, completed_steps, rounds


def _finalize_shared_learner(
    root: Path, status: str, round_id: int, completed_steps: int,
    actors: tuple[str, ...], rounds: list[dict[str, Any]],
    failure_message: str | None,
) -> dict[str, Any]:
    finished_ns = time.time_ns()
    manifest = {
        "schema_version": 1, "algorithm": "RecurrentPPOUniversal",
        "status": status, "round": round_id,
        "completed_timesteps": completed_steps, "actors": list(actors),
        "rounds": rounds, "finished_unix_ns": finished_ns,
    }
    publish_json(root / "manifest.json", manifest)
    _publish_monitor_state(root, status, round_id, completed_steps,
                           failure=failure_message)
    terminal: dict[str, Any] = {
        "status": status, "round": round_id, "completed_timesteps": completed_steps,
        "finished_unix_ns": finished_ns, "metrics": rounds[-1] if rounds else {},
    }
    if failure_message is not None:
        terminal["error"] = failure_message[:2000]
    publish_json(root / "status.json", terminal)
    return manifest


def run_shared_learner(
    *, shared_root: Path, universal_checkpoint: Path,
    action_table: tuple[str, ...], actors: tuple[str, ...],
    total_timesteps: int, rollout_steps: int, sequence_length: int,
    game_id: str | None = None, game_extras_dim: int = 0,
    learning_rate: float, lr_encoder_scale: float,
    gamma: float, gae_lambda: float, clip_range: float,
    entropy_coef: float, epochs: int, batch_size: int,
    c_sem: float, c_inv: float, c_mot: float,
    device_name: str = "cpu", seed: int = 7,
    continuous: bool = False, freeze_encoder_rounds: int = 0,
    wait_timeout: float = 3600.0, straggler_timeout: float = 300.0,
    stop_file: Path | None = None,
) -> dict[str, Any]:
    device = _validate_learner_contract(
        actors, rollout_steps, sequence_length, total_timesteps, continuous,
        batch_size, freeze_encoder_rounds, game_extras_dim, game_id, device_name,
    )
    torch.set_num_threads(3)
    torch.manual_seed(seed)
    shared_root.mkdir(parents=True, exist_ok=True)
    fault_handle = (shared_root / "faulthandler-learner.log").open("a", encoding="utf-8")
    import faulthandler

    faulthandler.enable(file=fault_handle, all_threads=True)
    round_id, completed_steps = 0, 0
    rounds: list[dict[str, Any]] = []
    status, failure_message = "FAILED", None
    heartbeat_stop: threading.Event | None = None
    heartbeat_thread: threading.Thread | None = None
    try:
        if not _wait_for_commit_guard(shared_root, round_id, completed_steps,
                                      stop_file=stop_file):
            status = "CANCELLED"
        else:
            model, trainer, policy_root, completed_steps, round_id = _setup_learner(
                shared_root, universal_checkpoint, action_table, game_id, game_extras_dim,
                device, learning_rate=learning_rate, clip_range=clip_range,
                entropy_coef=entropy_coef, c_sem=c_sem, c_inv=c_inv, c_mot=c_mot,
                encoder_scale=lr_encoder_scale,
            )
            _publish_monitor_state(shared_root, "RUNNING", round_id, completed_steps)
            heartbeat_stop, heartbeat_thread, progress, progress_lock = _start_learner_heartbeat(
                shared_root, round_id, completed_steps, device,
            )
            status, round_id, completed_steps, rounds = _train_rounds(
                shared_root, model, trainer, policy_root, device,
                round_id=round_id, completed_steps=completed_steps, actors=actors,
                action_table=action_table, continuous=continuous,
                total_timesteps=total_timesteps, rollout_steps=rollout_steps,
                sequence_length=sequence_length, batch_size=batch_size, epochs=epochs,
                freeze_encoder_rounds=freeze_encoder_rounds, game_id=game_id,
                wait_timeout=wait_timeout, straggler_timeout=straggler_timeout,
                stop_file=stop_file, progress=progress,
                progress_lock=progress_lock,
            )
    except BaseException as error:
        status = "FAILED"
        failure_message = f"{type(error).__name__}: {error}"
        _publish_monitor_state(shared_root, status, round_id, completed_steps,
                               failure=failure_message)
        raise
    finally:
        if heartbeat_stop is not None:
            heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=6.0)
        manifest = _finalize_shared_learner(
            shared_root, status, round_id, completed_steps, actors, rounds,
            failure_message,
        )
        faulthandler.disable()
        fault_handle.close()
    return manifest
