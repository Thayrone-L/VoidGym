"""Safe, atomic serialization for shared universal-policy actor rollouts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from voidgym.training.buffers import SequenceBatch

_SCHEMA = 1
_FIELDS = (
    "actions", "old_log_probs", "old_values", "returns", "advantages",
    "action_masks", "episode_starts", "sem", "hero_motion", "discontinuity",
    "privileged", "has_labels",
)


def rollout_path(root: Path, round_id: int, actor_id: str) -> Path:
    if round_id < 1 or not actor_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in actor_id):
        raise ValueError("invalid round or actor id")
    return root / "rollouts" / f"round-{round_id:08d}-{actor_id}.npz"


def publish_rollout(root: Path, *, round_id: int, actor_id: str,
                    policy_round: int, action_table: tuple[str, ...],
                    sequences: list[SequenceBatch]) -> Path:
    if round_id < 1 or policy_round < 0 or not sequences or not action_table:
        raise ValueError("rollout identity, action table and sequences are required")
    if len(set(action_table)) != len(action_table):
        raise ValueError("action table must contain unique names")
    target = rollout_path(root, round_id, actor_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"schema_version": _SCHEMA, "round": round_id,
                "actor_id": actor_id, "policy_round": policy_round,
                "action_table": list(action_table), "sequence_count": len(sequences)}
    arrays: dict[str, np.ndarray] = {"meta": np.asarray(json.dumps(metadata, sort_keys=True))}
    for index, sequence in enumerate(sequences):
        prefix = f"s{index:04d}__"
        for key, value in sequence.observations.items():
            if not key or "__" in key:
                raise ValueError("observation key is invalid for rollout serialization")
            arrays[prefix + "obs__" + key] = value.detach().cpu().numpy()
        arrays[prefix + "state_h"] = sequence.initial_state[0].detach().cpu().numpy()
        arrays[prefix + "state_c"] = sequence.initial_state[1].detach().cpu().numpy()
        for name in _FIELDS:
            value = getattr(sequence, name)
            arrays[prefix + name] = value.detach().cpu().numpy()
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_rollout(path: Path, *, expected_round: int, expected_actor: str,
                 expected_policy_round: int, action_table: tuple[str, ...],
                 device: torch.device) -> list[SequenceBatch]:
    with np.load(path, allow_pickle=False) as archive:
        if "meta" not in archive.files:
            raise ValueError("rollout metadata is missing")
        metadata = json.loads(str(archive["meta"].item()))
        if (not isinstance(metadata, dict) or metadata.get("schema_version") != _SCHEMA
                or metadata.get("round") != expected_round
                or metadata.get("actor_id") != expected_actor
                or metadata.get("policy_round") != expected_policy_round
                or metadata.get("action_table") != list(action_table)):
            raise ValueError("rollout contract does not match the active learner round")
        count = metadata.get("sequence_count")
        if type(count) is not int or count < 1 or count > 100_000:
            raise ValueError("rollout sequence count is invalid")
        expected_files: set[str] = {"meta"}
        result: list[SequenceBatch] = []
        for index in range(count):
            prefix = f"s{index:04d}__"
            obs: dict[str, Tensor] = {}
            for name in ("pixels", "action_history", "game_extras"):
                key = prefix + "obs__" + name
                if key in archive.files:
                    expected_files.add(key)
                    obs[name] = torch.from_numpy(archive[key].copy()).to(device)
            if not {"pixels", "action_history"}.issubset(obs):
                raise ValueError("rollout is missing universal observations")
            for suffix in ("state_h", "state_c", *_FIELDS):
                key = prefix + suffix
                if key not in archive.files:
                    raise ValueError(f"rollout is missing {suffix}")
                expected_files.add(key)
            def tensor(name: str, current_prefix: str = prefix) -> Tensor:
                return torch.from_numpy(archive[current_prefix + name].copy()).to(device)
            observations = obs
            result.append(SequenceBatch(
                observations=observations,
                actions=tensor("actions").long(),
                old_log_probs=tensor("old_log_probs").float(),
                old_values=tensor("old_values").float(),
                returns=tensor("returns").float(),
                advantages=tensor("advantages").float(),
                action_masks=tensor("action_masks").bool(),
                episode_starts=tensor("episode_starts").bool(),
                initial_state=(tensor("state_h").float(), tensor("state_c").float()),
                sem=tensor("sem").to(torch.uint8),
                hero_motion=tensor("hero_motion").float(),
                discontinuity=tensor("discontinuity").bool(),
                privileged=tensor("privileged").float(),
                has_labels=tensor("has_labels").bool(),
            ))
        if set(archive.files) != expected_files:
            raise ValueError("rollout contains unexpected arrays")
    return result


def publish_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
