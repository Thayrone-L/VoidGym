"""File-coordinated recurrent PPO learner for universal policies."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from voidgym.models.policy import UniversalPolicy
from voidgym.training.checkpoints import load_universal, prune_checkpoints, save_universal
from voidgym.training.recurrent_ppo import RecurrentPPOTrainer
from voidgym.training.shared_rollouts import load_rollout, publish_json, rollout_path


def _publish_monitor_state(
    root: Path, status: str, round_id: int, completed_steps: int,
    *, metrics: dict[str, float] | None = None, failure: str | None = None,
) -> None:
    state: dict[str, Any] = {
        "status": status, "round": round_id,
        "completed_timesteps": completed_steps,
        "updated_unix_ns": time.time_ns(),
    }
    if metrics is not None:
        state["metrics"] = metrics
    if failure is not None:
        state["error"] = failure[:2000]
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
                       stop_file: Path | None) -> list[Any] | None:
    deadline = time.monotonic() + timeout
    paths = {actor: rollout_path(root, round_id, actor) for actor in actors}
    while time.monotonic() < deadline:
        if stop_file is not None and stop_file.is_file():
            return None
        if all(path.is_file() for path in paths.values()):
            sequences = []
            for actor in actors:
                sequences.extend(load_rollout(
                    paths[actor], expected_round=round_id, expected_actor=actor,
                    expected_policy_round=policy_round, action_table=action_table,
                    device=device,
                ))
            return sequences
        time.sleep(.1)
    missing = [actor for actor, path in paths.items() if not path.is_file()]
    raise TimeoutError(f"actors did not publish round {round_id}: {missing}")


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
    device: torch.device, timeout: float, stop_file: Path | None,
    freeze_encoder: bool, freeze_rounds: int, rollout_steps: int,
    sequence_length: int, batch_size: int, epochs: int, game_id: str | None,
) -> tuple[int, int, dict[str, float], dict[str, Any]] | None:
    next_round = round_id + 1
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(not (freeze_encoder and next_round <= freeze_rounds))
    sequences = _wait_for_rollouts(root, next_round, actors, round_id,
                                   actions, device, timeout, stop_file)
    if sequences is None:
        return None
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
    completed_steps += rollout_steps * len(actors)
    metrics.update({"shared_round": float(next_round),
                    "steps_per_update": float(rollout_steps * len(actors)),
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
                           metrics=metrics)
    model.eval()
    prune_checkpoints(policy_root, keep=5)
    return next_round, completed_steps, metrics, row


def run_shared_learner(  # noqa: C901 - validates the shared learner contract before launch
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
                       wait_timeout: float = 3600.0, stop_file: Path | None = None) -> dict[str, Any]:
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
    device = torch.device(device_name)
    torch.set_num_threads(3)
    torch.manual_seed(seed)
    shared_root.mkdir(parents=True, exist_ok=True)
    model, trainer, policy_root, completed_steps, round_id = _setup_learner(
        shared_root, universal_checkpoint, action_table, game_id, game_extras_dim,
        device, learning_rate=learning_rate, clip_range=clip_range,
        entropy_coef=entropy_coef, c_sem=c_sem, c_inv=c_inv, c_mot=c_mot,
        encoder_scale=lr_encoder_scale,
    )
    _publish_monitor_state(shared_root, "RUNNING", round_id, completed_steps)
    rounds: list[dict[str, Any]] = []
    status = "RUNNING"
    try:
        while continuous or completed_steps < total_timesteps:
            update = _train_shared_round(
                shared_root, policy_root, model, trainer, round_id=round_id,
                completed_steps=completed_steps, actors=actors, actions=action_table,
                device=device, timeout=wait_timeout, stop_file=stop_file,
                freeze_encoder=True, freeze_rounds=freeze_encoder_rounds,
                rollout_steps=rollout_steps, sequence_length=sequence_length,
                batch_size=batch_size, epochs=epochs, game_id=game_id,
            )
            if update is None:
                status = "CANCELLED"
                break
            round_id, completed_steps, _metrics, row = update
            rounds.append(row)
        if status == "RUNNING":
            status = "COMPLETED"
    except BaseException as error:
        status = "FAILED"
        _publish_monitor_state(shared_root, status, round_id, completed_steps,
                               failure=str(error))
        raise
    finally:
        manifest = {"schema_version": 1, "algorithm": "RecurrentPPOUniversal",
                    "status": status, "round": round_id,
                    "completed_timesteps": completed_steps, "actors": list(actors),
                    "rounds": rounds, "finished_unix_ns": time.time_ns()}
        publish_json(shared_root / "manifest.json", manifest)
        _publish_monitor_state(shared_root, status, round_id, completed_steps)
    return manifest
