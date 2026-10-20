from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import secrets
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np

from hk_agent.policy_artifact import export_policy_artifact, import_policy_artifact

MAX_ROLLOUT_BYTES = 512 * 1024 * 1024
LEARNER_STALE_SECONDS = 300.0
ACTOR_HEARTBEAT_STALE_SECONDS = 90.0
DEFAULT_STRAGGLER_TIMEOUT_SECONDS = 300.0
SHARED_SCHEMA_VERSION = 1


class SharedLearnerError(RuntimeError):
    """Shared actor/learner protocol error."""


class SharedLearnerStopped(SharedLearnerError):
    """Cooperative stop requested while waiting for shared calibration."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_default
    ).encode()


def _json_default(item: Any) -> Any:
    if isinstance(item, np.generic):
        return item.item()
    if isinstance(item, np.ndarray):
        return item.tolist()
    raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
    try:
        with pending.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(20):
            try:
                os.replace(pending, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.025)
    finally:
        pending.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        json.dumps(value, indent=2, allow_nan=False, default=_json_default).encode("utf-8") + b"\n",
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_warm_start_spaces(
    source_observation: gym.Space[Any], source_action: gym.Space[Any],
    target_observation: gym.Space[Any], target_action: gym.Space[Any],
) -> None:
    source_payload = _space_to_payload(source_observation)
    target_payload = _space_to_payload(target_observation)
    if not _compatible_warm_start_observation(source_payload, target_payload):
        raise SharedLearnerError("warm-start observation spaces differ from the new campaign")
    if _space_to_payload(source_action) == _space_to_payload(target_action):
        return
    if (isinstance(source_action, gym.spaces.Discrete)
            and isinstance(target_action, gym.spaces.Discrete)
            and source_action.start == target_action.start and source_action.n < target_action.n):
        return
    raise SharedLearnerError("warm-start action spaces differ from the new campaign")


def _compatible_warm_start_observation(source: Mapping[str, Any], target: Mapping[str, Any]) -> bool:
    if source == target:
        return True
    source_spaces, target_spaces = source.get("spaces"), target.get("spaces")
    if (source.get("type") != "Dict" or target.get("type") != "Dict"
            or not isinstance(source_spaces, Mapping) or not isinstance(target_spaces, Mapping)
            or set(source_spaces) != set(target_spaces)):
        return False
    for name, source_space in source_spaces.items():
        target_space = target_spaces.get(name)
        if not isinstance(source_space, Mapping) or not isinstance(target_space, Mapping):
            return False
        if source_space == target_space:
            continue
        if (name == "action_history" and source_space.get("type") == "MultiDiscrete"
                and target_space.get("type") == "MultiDiscrete"
                and source_space.get("dtype") == target_space.get("dtype")
                and source_space.get("start") == target_space.get("start")):
            old_nvec, new_nvec = source_space.get("nvec"), target_space.get("nvec")
            if (isinstance(old_nvec, list) and isinstance(new_nvec, list)
                    and len(old_nvec) == len(new_nvec)
                    and all(isinstance(old, int) and isinstance(new, int) and old <= new
                            for old, new in zip(old_nvec, new_nvec, strict=True))):
                continue
        return False
    return True


def compatible_warm_start_state(
    source_state: Mapping[str, Any], target_state: Mapping[str, Any],
    source_action_names: Sequence[str] | None = None,
    target_action_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Copy matching tensors and align action heads by name when the action set grows."""
    compatible: dict[str, Any] = {}
    for name, tensor in source_state.items():
        if name not in target_state:
            continue
        target_tensor = target_state[name]
        if getattr(target_tensor, "shape", None) == getattr(tensor, "shape", None):
            compatible[name] = tensor
            continue
        source_shape = cast(tuple[int, ...], getattr(tensor, "shape", ()))
        target_shape = cast(tuple[int, ...], getattr(target_tensor, "shape", ()))
        if ("action_net" not in name or len(source_shape) != len(target_shape) or not source_shape
                or source_shape[1:] != target_shape[1:] or source_shape[0] >= target_shape[0]):
            continue
        clone = getattr(target_tensor, "clone", None)
        expanded = clone() if callable(clone) else target_tensor.copy()
        if source_action_names is not None and target_action_names is not None:
            if len(source_action_names) != source_shape[0] or len(target_action_names) != target_shape[0]:
                raise SharedLearnerError("warm-start action names do not match policy output dimensions")
            target_index = {action: index for index, action in enumerate(target_action_names)}
            for source_index, action in enumerate(source_action_names):
                if action in target_index:
                    expanded[target_index[action]] = tensor[source_index]
        else:
            expanded[:source_shape[0]] = tensor
        compatible[name] = expanded
    if not compatible or not any("features_extractor" in name for name in compatible):
        raise SharedLearnerError("warm-start has no compatible visual feature extractor tensors")
    return compatible

def _space_to_payload(space: gym.Space[Any]) -> dict[str, Any]:
    if isinstance(space, gym.spaces.Dict):
        return {
            "type": "Dict",
            "spaces": {name: _space_to_payload(child) for name, child in space.spaces.items()},
        }
    if isinstance(space, gym.spaces.Box):
        low = np.asarray(space.low)
        high = np.asarray(space.high)
        low_value: int | float | list[Any] = (
            low.reshape(-1)[0].item()
            if low.size and np.all(low == low.reshape(-1)[0])
            else low.tolist()
        )
        high_value: int | float | list[Any] = (
            high.reshape(-1)[0].item()
            if high.size and np.all(high == high.reshape(-1)[0])
            else high.tolist()
        )
        return {
            "type": "Box",
            "shape": list(space.shape),
            "dtype": np.dtype(space.dtype).str,
            "low": low_value,
            "high": high_value,
        }
    if isinstance(space, gym.spaces.Discrete):
        return {"type": "Discrete", "n": space.n, "start": space.start}
    if isinstance(space, gym.spaces.MultiDiscrete):
        return {
            "type": "MultiDiscrete", "nvec": space.nvec.tolist(),
            "start": space.start.tolist(), "dtype": np.dtype(space.dtype).str,
        }
    raise SharedLearnerError(f"unsupported shared learner space: {type(space).__name__}")


def _space_from_payload(payload: Mapping[str, Any]) -> gym.Space[Any]:
    kind = payload.get("type")
    if kind == "Dict":
        children = payload.get("spaces")
        if not isinstance(children, Mapping) or not children:
            raise SharedLearnerError("calibration observation Dict space is invalid")
        if not all(isinstance(name, str) and isinstance(child, Mapping) for name, child in children.items()):
            raise SharedLearnerError("calibration Dict child space is invalid")
        return gym.spaces.Dict({name: _space_from_payload(child) for name, child in children.items()})
    if kind == "Box":
        shape = payload.get("shape")
        dtype = payload.get("dtype")
        if not isinstance(shape, list) or not shape or not all(isinstance(v, int) and v > 0 for v in shape):
            raise SharedLearnerError("calibration Box shape is invalid")
        if not isinstance(dtype, str):
            raise SharedLearnerError("calibration Box dtype is invalid")
        low_payload, high_payload = payload.get("low"), payload.get("high")
        low = low_payload if isinstance(low_payload, (int, float)) else np.asarray(low_payload, dtype=np.dtype(dtype))
        high = high_payload if isinstance(high_payload, (int, float)) else np.asarray(high_payload, dtype=np.dtype(dtype))
        return gym.spaces.Box(
            low=low,
            high=high,
            shape=tuple(shape),
            dtype=np.dtype(dtype),
        )
    if kind == "Discrete":
        n = payload.get("n")
        start = payload.get("start", 0)
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise SharedLearnerError("calibration Discrete space is invalid")
        if isinstance(start, bool) or not isinstance(start, int):
            raise SharedLearnerError("calibration Discrete start is invalid")
        return gym.spaces.Discrete(n, start=start)
    if kind == "MultiDiscrete":
        nvec_value, start_value, dtype_value = payload.get("nvec"), payload.get("start"), payload.get("dtype")
        if not isinstance(dtype_value, str) or not isinstance(nvec_value, list) or not isinstance(start_value, list):
            raise SharedLearnerError("calibration MultiDiscrete space is invalid")
        try:
            dtype = np.dtype(dtype_value)
            nvec = np.asarray(nvec_value)
            start = np.asarray(start_value)
        except (TypeError, ValueError) as error:
            raise SharedLearnerError("calibration MultiDiscrete values are invalid") from error
        if (not np.issubdtype(dtype, np.integer) or not np.issubdtype(nvec.dtype, np.integer)
                or not np.issubdtype(start.dtype, np.integer) or nvec.size == 0
                or nvec.shape != start.shape or np.any(nvec < 1)):
            raise SharedLearnerError("calibration MultiDiscrete dimensions are invalid")
        return gym.spaces.MultiDiscrete(nvec.astype(dtype), dtype=dtype, start=start.astype(dtype))
    raise SharedLearnerError("calibration space type is unsupported")


def _safe_fingerprint(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise SharedLearnerError("task contract fingerprint must be lowercase SHA-256")
    return value


def _resolve_candidate_lock(path: Path, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > 60:
                    path.unlink(missing_ok=True)
            except OSError:
                pass
            time.sleep(0.05)
            continue
        os.close(descriptor)
        return True
    raise TimeoutError("timed out acquiring shared calibration lock")


def share_calibration(root: Path, candidate: Mapping[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    """First actor publishes calibration metrics; every actor reuses that exact result."""
    path = root / "calibration.json"
    lock = root / ".calibration.lock"
    deadline = time.monotonic() + timeout
    while True:
        if path.is_file():
            try:
                current = json.loads(path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                current = None
            if isinstance(current, dict) and current.get("status") == "PASSED":
                return current
        if time.monotonic() >= deadline:
            raise TimeoutError("shared calibration was not published")
        if _resolve_candidate_lock(lock, min(30.0, max(0.1, deadline - time.monotonic()))):
            try:
                if path.is_file():
                    continue
                _atomic_json(path, {"schema_version": 1, "status": "PASSED", **dict(candidate)})
            finally:
                lock.unlink(missing_ok=True)


class SharedRoundStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=False)
        self.policy_root = self.root / "policy"
        self.rollout_root = self.root / "rollouts"

    def actor_status_path(self, slot: int) -> Path:
        return self.root / "actors" / f"slot-{slot}.json"

    def publish_actor_status(
        self, *, slot: int, launch_id: str, status: str, **values: Any
    ) -> bool:
        """Publish a slot heartbeat without letting a retired actor overwrite its replacement."""
        path = self.actor_status_path(slot)
        try:
            current = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
        waiting_since = current.get("waiting_policy_since_unix_ns")
        if status == "WAITING_POLICY":
            if current.get("status") != "WAITING_POLICY" or not isinstance(waiting_since, int):
                waiting_since = time.time_ns()
        else:
            waiting_since = None
        if current.get("status") == "RESTARTING" and current.get("launch_id") == launch_id:
            return False
        if current.get("launch_id") not in (None, launch_id) and current.get("status") not in {
            "RESTARTING", "CANCELLED", "FAILED", "COMPLETED",
        }:
            return False
        _atomic_json(path, {
            **current, **values, "slot": slot, "launch_id": launch_id,
            "status": status, "heartbeat_unix_ns": time.time_ns(),
            "waiting_policy_since_unix_ns": waiting_since,
        })
        return True

    def _read_actor_status(self, slot: int) -> dict[str, Any]:
        try:
            value = json.loads(self.actor_status_path(slot).read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def actor_available(self, slot: int, *, now_unix_ns: int | None = None) -> bool | None:
        """Return None until known, then whether the current slot heartbeat is live."""
        try:
            record = json.loads(self.actor_status_path(slot).read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if record.get("status") in {"RESTARTING", "CANCELLED", "FAILED", "COMPLETED"}:
            return False
        beat = record.get("heartbeat_unix_ns")
        if not isinstance(beat, int):
            return False
        now = time.time_ns() if now_unix_ns is None else now_unix_ns
        return (now - beat) / 1_000_000_000 <= ACTOR_HEARTBEAT_STALE_SECONDS

    def record_stale_rollout(self, *, round_number: int, slot: int, launch_id: str) -> None:
        path = self.root / "telemetry" / "stale-rollouts.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({
            "event": "stale_rollout", "round": round_number, "slot": slot,
            "launch_id": launch_id, "time_unix_ns": time.time_ns(),
        }, allow_nan=False).encode("utf-8") + b"\n"
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, line)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def stale_rollout_count(self) -> int:
        path = self.root / "telemetry" / "stale-rollouts.jsonl"
        try:
            with path.open("rb") as stream:
                return sum(1 for _ in stream)
        except OSError:
            return 0

    @property
    def learner_status_path(self) -> Path:
        return self.root / "learner.json"

    def publish_contract(
        self,
        *,
        task_contract: Mapping[str, Any],
        fingerprint: str,
        observation_space: gym.Space[Any],
        action_space: gym.Space[Any],
        rollout_steps: int,
        actors: int,
    ) -> dict[str, Any]:
        fingerprint = _safe_fingerprint(fingerprint)
        calibration_path = self.root / "calibration.json"
        json.loads(calibration_path.read_text("utf-8"))
        contract = {
            "task_contract": dict(task_contract),
            "task_contract_fingerprint": fingerprint,
            "observation_space": _space_to_payload(observation_space),
            "action_space": _space_to_payload(action_space),
            "rollout_steps": rollout_steps,
            "actors": actors,
        }
        lock = self.root / ".calibration.lock"
        _resolve_candidate_lock(lock)
        try:
            current = json.loads(calibration_path.read_text("utf-8"))
            if not isinstance(current, dict):
                raise SharedLearnerError("shared calibration document is invalid")
            existing = current.get("contract")
            if existing is not None and _canonical_json(existing) != _canonical_json(contract):
                raise SharedLearnerError("actor task contract differs from shared calibration")
            current["contract"] = contract
            _atomic_json(calibration_path, current)
            return cast(dict[str, Any], current)
        finally:
            lock.unlink(missing_ok=True)

    def wait_for_contract(self, timeout: float = 600.0, stop_file: Path | None = None) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        path = self.root / "calibration.json"
        while time.monotonic() < deadline:
            if stop_file is not None and stop_file.is_file():
                raise SharedLearnerStopped("stop requested while waiting for actor calibration")
            try:
                value = json.loads(path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                value = None
            if isinstance(value, dict) and isinstance(value.get("contract"), dict):
                return value
            time.sleep(0.25)
        raise TimeoutError("shared actor contract was not published")

    def heartbeat(self, **values: Any) -> None:
        current: dict[str, Any] = {}
        try:
            loaded = json.loads(self.learner_status_path.read_text("utf-8"))
            if isinstance(loaded, dict):
                current = loaded
        except (OSError, json.JSONDecodeError):
            pass
        current.update(values)
        current["heartbeat_unix_ns"] = time.time_ns()
        _atomic_json(self.learner_status_path, current)

    def learner_alive(self) -> bool:
        try:
            status = json.loads(self.learner_status_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        beat = status.get("heartbeat_unix_ns") if isinstance(status, dict) else None
        return (
            isinstance(beat, int)
            and not isinstance(beat, bool)
            and (time.time_ns() - beat) / 1_000_000_000 <= LEARNER_STALE_SECONDS
            and status.get("status") in {"STARTING", "WAITING", "TRAINING", "RUNNING"}
        )

    def publish_policy(
        self, round_number: int, policy: Any, fingerprint: str, *,
        reuse_existing: bool = False, replace_existing: bool = False,
    ) -> dict[str, Any]:
        fingerprint = _safe_fingerprint(fingerprint)
        stem = f"round-{round_number:08d}"
        pointer_path = self.policy_root / f"{stem}.json"
        if pointer_path.exists() and not replace_existing:
            if reuse_existing:
                try:
                    self.import_policy(round_number, policy, fingerprint)
                except (OSError, ValueError, SharedLearnerError) as error:
                    logging.getLogger(__name__).warning(
                        "ignoring invalid policy pointer for round %s: %s", round_number, error
                    )
                    pointer_path.unlink(missing_ok=True)
                else:
                    return cast(dict[str, Any], json.loads(pointer_path.read_text("utf-8")))
            else:
                raise SharedLearnerError(f"policy round {round_number} was already published")
        generation = secrets.token_hex(8)
        artifact = self.policy_root / f"{stem}-{generation}.safetensors"
        sidecar = self.policy_root / f"{stem}-{generation}.policy.manifest.json"
        exported = export_policy_artifact(
            policy, artifact, compatibility_fingerprint=fingerprint, manifest_path=sidecar
        )
        pointer = {
            "schema_version": SHARED_SCHEMA_VERSION,
            "policy_round": round_number,
            "artifact": artifact.name,
            "manifest": sidecar.name,
            "sha256": exported["artifact"]["sha256"],
            "manifest_sha256": _sha256(sidecar.read_bytes()),
            "manifest_id": exported["manifest_id"],
            "task_contract_fingerprint": fingerprint,
        }
        pending_pointer = self.policy_root / f".{stem}.{generation}.pointer.tmp"
        try:
            with pending_pointer.open("xb") as stream:
                stream.write(_canonical_json(pointer) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            if replace_existing and pointer_path.exists():
                os.replace(pending_pointer, pointer_path)
            else:
                try:
                    os.link(pending_pointer, pointer_path)
                except FileExistsError as error:
                    raise SharedLearnerError(
                        f"policy round {round_number} was already published"
                    ) from error
        except BaseException as error:
            if not pointer_path.exists() or replace_existing:
                artifact.unlink(missing_ok=True)
                sidecar.unlink(missing_ok=True)
            raise SharedLearnerError(
                f"policy round {round_number} publication failed before pointer commit; "
                f"new generation {generation} was cleaned: {error}"
            ) from error
        finally:
            pending_pointer.unlink(missing_ok=True)
        self._cleanup_policy_generations(round_number)
        return pointer

    def _latest_checkpoint_round(self) -> int | None:
        shared_root = self.policy_root.parent
        manifest_path = shared_root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        candidate = manifest.get("latest_checkpoint") if isinstance(manifest, dict) else None
        paths = [Path(candidate)] if isinstance(candidate, str) else []
        paths.extend((shared_root / "checkpoints").glob("round-*.zip"))
        valid: list[int] = []
        for path in paths:
            match = re.fullmatch(r"round-(\d{8})\.zip", path.name)
            if match and path.is_file():
                valid.append(int(match.group(1)))
        return max(valid) if valid else None

    def _cleanup_policy_generations(self, current_round: int) -> None:
        checkpoint_round = self._latest_checkpoint_round()
        protected_rounds = {current_round, current_round - 1, current_round - 2}
        if checkpoint_round is not None:
            protected_rounds.add(checkpoint_round)
        pointers: list[tuple[int, Path, dict[str, Any]]] = []
        for path in self.policy_root.glob("round-*.json"):
            match = re.fullmatch(r"round-(\d{8})\.json", path.name)
            if match is None:
                continue
            try:
                value = json.loads(path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                logging.getLogger(__name__).warning("removing unreadable policy pointer %s", path)
                path.unlink(missing_ok=True)
                continue
            if not isinstance(value, dict):
                path.unlink(missing_ok=True)
                continue
            artifact_name, manifest_name = value.get("artifact"), value.get("manifest")
            if not isinstance(artifact_name, str) or Path(artifact_name).name != artifact_name:
                logging.getLogger(__name__).warning("removing unsafe policy pointer %s", path)
                path.unlink(missing_ok=True)
                continue
            artifact = self.policy_root / artifact_name
            sidecar = self.policy_root / manifest_name if isinstance(manifest_name, str) and Path(manifest_name).name == manifest_name else None
            policy_round = int(match.group(1))
            if not artifact.is_file() or (sidecar is not None and not sidecar.is_file()):
                logging.getLogger(__name__).warning("removing policy pointer with missing artifact: %s", path)
                path.unlink(missing_ok=True)
                if policy_round not in protected_rounds:
                    artifact.unlink(missing_ok=True)
                    if sidecar is not None:
                        sidecar.unlink(missing_ok=True)
                continue
            pointers.append((policy_round, path, value))
        retained_references = {
            name for policy_round, _path, value in pointers if policy_round in protected_rounds
            for name in (value.get("artifact"), value.get("manifest")) if isinstance(name, str)
        }
        for policy_round, pointer_path, value in pointers:
            if policy_round in protected_rounds:
                continue
            pointer_path.unlink(missing_ok=True)
            for key in ("artifact", "manifest"):
                name = value.get(key)
                if isinstance(name, str) and name not in retained_references and Path(name).name == name:
                    (self.policy_root / name).unlink(missing_ok=True)
        referenced = {
            name for _round, _path, value in pointers
            for name in (value.get("artifact"), value.get("manifest")) if isinstance(name, str)
        }
        for artifact in self.policy_root.glob("round-*.safetensors"):
            match = re.match(r"round-(\d{8})-", artifact.name)
            if match and int(match.group(1)) not in protected_rounds and artifact.name not in referenced:
                artifact.unlink(missing_ok=True)
                artifact.with_name(artifact.name.removesuffix(".safetensors") + ".policy.manifest.json").unlink(missing_ok=True)

    def import_policy(self, round_number: int, policy: Any, fingerprint: str) -> dict[str, Any]:
        stem = f"round-{round_number:08d}"
        pointer = json.loads((self.policy_root / f"{stem}.json").read_text("utf-8"))
        if pointer.get("policy_round") != round_number:
            raise SharedLearnerError("policy round does not match request")
        if pointer.get("task_contract_fingerprint") != _safe_fingerprint(fingerprint):
            raise SharedLearnerError("policy task contract fingerprint differs")
        artifact_name, manifest_name = pointer.get("artifact"), pointer.get("manifest")
        if (not isinstance(artifact_name, str) or Path(artifact_name).name != artifact_name
                or not isinstance(manifest_name, str) or Path(manifest_name).name != manifest_name):
            raise SharedLearnerError(f"policy pointer for round {round_number} has unsafe artifact paths")
        artifact = self.policy_root / artifact_name
        manifest = self.policy_root / manifest_name
        if not artifact.is_file() or not manifest.is_file():
            logging.getLogger(__name__).warning(
                "policy round %s points to missing files (artifact=%s, manifest=%s)",
                round_number, artifact.is_file(), manifest.is_file(),
            )
            raise SharedLearnerError(
                f"policy pointer for round {round_number} references a missing artifact or manifest"
            )
        return import_policy_artifact(
            policy,
            artifact,
            expected_compatibility_fingerprint=fingerprint,
            manifest_path=manifest,
            expected_artifact_sha256=str(pointer.get("sha256")),
            expected_manifest_sha256=str(pointer.get("manifest_sha256")),
            expected_manifest_id=str(pointer.get("manifest_id")),
        )

    def publish_rollout(
        self,
        *,
        round_number: int,
        slot: int,
        launch_id: str,
        buffer: Any,
        fingerprint: str,
    ) -> dict[str, Any]:
        if not launch_id or slot < 0 or buffer.pos < 1:
            raise SharedLearnerError("actor rollout identity or size is invalid")
        steps = int(buffer.buffer_size if buffer.full else buffer.pos)
        folder = self.rollout_root / f"round-{round_number:08d}"
        artifact = folder / f"slot-{slot}.npz"
        sidecar = folder / f"slot-{slot}.json"
        arrays: dict[str, np.ndarray] = {}
        observations = buffer.observations
        if not isinstance(observations, Mapping):
            raise SharedLearnerError("actor rollout observations are not a Dict")
        for name, values in observations.items():
            arrays[f"obs__{name}"] = np.asarray(values[:steps, 0]).copy()
        for name in ("actions", "rewards", "episode_starts", "values", "log_probs", "advantages", "returns", "action_masks"):
            value = getattr(buffer, name, None)
            if value is None:
                raise SharedLearnerError(f"actor rollout is missing {name}")
            array = np.asarray(value[:steps, 0]).copy()
            if name == "action_masks":
                if not np.isfinite(array).all() or not np.isin(array, (0, 1)).all():
                    raise SharedLearnerError("actor rollout action masks must contain only finite 0/1 values")
                array = array.astype(np.bool_, copy=False)
            if name in {"values", "log_probs", "advantages", "returns", "rewards", "episode_starts"}:
                array = array.reshape(steps)
            arrays[name] = array
        recurrent_state_names = {
            "lstm_hidden_pi": "hidden_states_pi",
            "lstm_cell_pi": "cell_states_pi",
            "lstm_hidden_vf": "hidden_states_vf",
            "lstm_cell_vf": "cell_states_vf",
        }
        present_recurrent = {key: getattr(buffer, attr, None) for key, attr in recurrent_state_names.items()}
        if any(value is not None for value in present_recurrent.values()):
            if any(value is None for value in present_recurrent.values()):
                raise SharedLearnerError("recurrent rollout is missing an LSTM state array")
            for key, value in present_recurrent.items():
                arrays[key] = np.asarray(cast(Any, value)[:steps, 0]).copy()
        pending = artifact.with_name(f".{artifact.name}.{secrets.token_hex(6)}.tmp")
        folder.mkdir(parents=True, exist_ok=True)
        try:
            with pending.open("wb") as stream:
                np.savez_compressed(stream, **cast(Any, arrays))
                stream.flush()
                os.fsync(stream.fileno())
            if pending.stat().st_size > MAX_ROLLOUT_BYTES:
                raise SharedLearnerError("actor rollout exceeds size limit")
            digest = _sha256(pending.read_bytes())
            os.replace(pending, artifact)
            metadata = {
                "schema_version": SHARED_SCHEMA_VERSION,
                "policy_round": round_number,
                "slot": slot,
                "launch_id": launch_id,
                "n_steps": steps,
                "sha256": digest,
                "task_contract_fingerprint": _safe_fingerprint(fingerprint),
            }
            _atomic_json(sidecar, metadata)
            return metadata
        finally:
            pending.unlink(missing_ok=True)

    def read_rollout(
        self,
        *,
        round_number: int,
        slot: int,
        fingerprint: str,
        n_steps: int,
        observation_space: gym.spaces.Dict,
        action_count: int,
    ) -> dict[str, np.ndarray]:
        folder = self.rollout_root / f"round-{round_number:08d}"
        artifact = folder / f"slot-{slot}.npz"
        sidecar = folder / f"slot-{slot}.json"
        try:
            metadata = json.loads(sidecar.read_text("utf-8"))
            payload = artifact.read_bytes()
        except (OSError, json.JSONDecodeError) as error:
            raise SharedLearnerError("rollout files are missing or malformed") from error
        if len(payload) > MAX_ROLLOUT_BYTES or _sha256(payload) != metadata.get("sha256"):
            raise SharedLearnerError("rollout sha256 does not match metadata")
        if metadata.get("policy_round") != round_number:
            raise SharedLearnerError("rollout belongs to a different policy round")
        if metadata.get("slot") != slot or not isinstance(metadata.get("launch_id"), str):
            raise SharedLearnerError("rollout actor identity is invalid")
        if metadata.get("task_contract_fingerprint") != _safe_fingerprint(fingerprint):
            raise SharedLearnerError("rollout task contract fingerprint differs")
        if metadata.get("n_steps") != n_steps:
            raise SharedLearnerError("rollout n_steps differs from contract")
        supported_obs = (gym.spaces.Box, gym.spaces.Discrete, gym.spaces.MultiDiscrete)
        if any(not isinstance(space, supported_obs) for space in observation_space.spaces.values()):
            raise SharedLearnerError("rollout Dict contains an unsupported observation space")
        expected_obs = {
            f"obs__{name}": (n_steps, *tuple(space.shape or ()))
            for name, space in observation_space.spaces.items()
        }
        required = set(expected_obs) | {"actions", "rewards", "episode_starts", "values", "log_probs", "advantages", "returns", "action_masks"}
        recurrent_fields = {"lstm_hidden_pi", "lstm_cell_pi", "lstm_hidden_vf", "lstm_cell_vf"}
        try:
            with np.load(artifact, allow_pickle=False) as packed:
                if set(packed.files) not in (required, required | recurrent_fields):
                    raise SharedLearnerError("rollout NPZ field inventory is invalid")
                arrays = {name: packed[name].copy() for name in packed.files}
        except (ValueError, OSError, zipfile.BadZipFile) as error:
            raise SharedLearnerError("rollout NPZ format is invalid") from error
        for name, shape in expected_obs.items():
            array = arrays[name]
            space = observation_space.spaces[name.removeprefix("obs__")]
            if array.shape != shape or array.dtype != space.dtype:
                raise SharedLearnerError(f"rollout observation shape/dtype differs for {name}")
        for name in ("rewards", "episode_starts", "values", "log_probs", "advantages", "returns"):
            if arrays[name].shape != (n_steps,):
                raise SharedLearnerError(f"rollout {name} shape is invalid")
        if arrays["actions"].shape not in {(n_steps,), (n_steps, 1)}:
            raise SharedLearnerError("rollout actions shape is invalid")
        if arrays["action_masks"].shape != (n_steps, action_count) or arrays["action_masks"].dtype != np.bool_:
            raise SharedLearnerError("rollout action_masks shape or dtype is invalid")
        if recurrent_fields <= set(arrays):
            reference_shape = arrays["lstm_hidden_pi"].shape
            if len(reference_shape) != 3 or reference_shape[0] != n_steps or min(reference_shape[1:]) < 1:
                raise SharedLearnerError("rollout LSTM state shape is invalid")
            for name in recurrent_fields:
                if arrays[name].shape != reference_shape or arrays[name].dtype != np.float32:
                    raise SharedLearnerError(f"rollout {name} shape or dtype is invalid")
        if not np.all(arrays["action_masks"].any(axis=1)):
            raise SharedLearnerError("rollout contains an empty action mask")
        actions = arrays["actions"].reshape(n_steps).astype(np.int64)
        if np.any(actions < 0) or np.any(actions >= action_count) or not np.all(arrays["action_masks"][np.arange(n_steps), actions]):
            raise SharedLearnerError("rollout contains an unavailable executed action")
        for name, array in arrays.items():
            if array.dtype == object or (np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all()):
                raise SharedLearnerError(f"rollout contains invalid numeric data in {name}")
        return arrays

    def reject(self, *, round_number: int, slot: int, reason: str) -> None:
        path = self.root / "rejected.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = json.dumps({"round": round_number, "slot": slot, "reason": reason, "time_unix_ns": time.time_ns()}) + "\n"
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(row)
            stream.flush()
            os.fsync(stream.fileno())

    def append_evaluation(self, record: Mapping[str, Any]) -> None:
        path = self.root / "evaluations.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        measure = record.get("measure") or (
            "deterministic" if record.get("deterministic") is True else "stochastic"
        )
        key = (record.get("round"), record.get("slot"), measure)
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(existing, Mapping):
                    continue
                existing_measure = existing.get("measure") or (
                    "deterministic" if existing.get("deterministic") is True else "stochastic"
                )
                if (existing.get("round"), existing.get("slot"), existing_measure) == key:
                    return
        payload = json.dumps(dict(record), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

    def read_evaluations(self, limit: int = 200) -> list[dict[str, Any]]:
        path = self.root / "evaluations.jsonl"
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                rows.append(record)
        return rows[-limit:]

    def wait_rollouts(
        self,
        *,
        round_number: int,
        actors: int,
        fingerprint: str,
        n_steps: int,
        observation_space: gym.spaces.Dict,
        action_count: int,
        timeout: float = 600.0,
        straggler_timeout: float = DEFAULT_STRAGGLER_TIMEOUT_SECONDS,
        stop_file: Path | None = None,
    ) -> tuple[list[dict[str, np.ndarray]], list[int], dict[str, Any]]:
        if straggler_timeout <= 0:
            raise ValueError("straggler_timeout must be positive")
        deadline = time.monotonic() + timeout
        accepted: dict[int, dict[str, np.ndarray]] = {}
        inspected: dict[int, tuple[int, int]] = {}
        first_rollout_at: float | None = None
        wait_started = time.monotonic()
        stale_count = self.stale_rollout_count()
        while time.monotonic() < deadline and len(accepted) < actors:
            if stop_file is not None and stop_file.is_file():
                break
            for slot in range(actors):
                if slot in accepted:
                    continue
                sidecar = self.rollout_root / f"round-{round_number:08d}" / f"slot-{slot}.json"
                try:
                    marker = (sidecar.stat().st_size, sidecar.stat().st_mtime_ns)
                    if inspected.get(slot) == marker:
                        continue
                    try:
                        rollout_metadata = json.loads(sidecar.read_text("utf-8"))
                    except (OSError, json.JSONDecodeError) as error:
                        raise SharedLearnerError("rollout metadata is missing or malformed") from error
                    actor_status = self._read_actor_status(slot)
                    current_launch_id = actor_status.get("launch_id")
                    rollout_launch_id = rollout_metadata.get("launch_id") if isinstance(rollout_metadata, Mapping) else None
                    if (isinstance(current_launch_id, str) and current_launch_id
                            and rollout_launch_id != current_launch_id):
                        raise SharedLearnerError(
                            "rollout belongs to a retired actor launch, not the current slot owner"
                        )
                    data = self.read_rollout(
                        round_number=round_number,
                        slot=slot,
                        fingerprint=fingerprint,
                        n_steps=n_steps,
                        observation_space=observation_space,
                        action_count=action_count,
                    )
                    accepted[slot] = data
                    inspected[slot] = marker
                    if first_rollout_at is None:
                        first_rollout_at = time.monotonic()
                except FileNotFoundError:
                    continue
                except (OSError, SharedLearnerError, zipfile.BadZipFile, json.JSONDecodeError) as error:
                    try:
                        marker = (sidecar.stat().st_size, sidecar.stat().st_mtime_ns)
                    except OSError:
                        marker = (-1, -1)
                    if inspected.get(slot) != marker:
                        self.reject(round_number=round_number, slot=slot, reason=str(error))
                        inspected[slot] = marker
            self.heartbeat(round=round_number, status="WAITING", accepted_rollouts=len(accepted))
            if len(accepted) >= actors:
                break
            availability = [self.actor_available(slot) for slot in range(actors)]
            if first_rollout_at is not None:
                unavailable = [slot for slot in range(actors)
                               if slot not in accepted and availability[slot] is False]
                if unavailable or time.monotonic() - first_rollout_at >= straggler_timeout:
                    break
            time.sleep(0.2)
        if not accepted:
            if stop_file is not None and stop_file.is_file():
                return [], [], {
                    "wait_rollouts_ms": round((time.monotonic() - wait_started) * 1000, 3),
                    "straggler_wait_ms": 0.0, "received_slots": [], "stale_rollouts": stale_count,
                }
            raise TimeoutError("no valid actor rollouts arrived before quorum timeout")
        waited = time.monotonic() - wait_started
        straggler_wait = 0.0 if first_rollout_at is None else max(0.0, time.monotonic() - first_rollout_at)
        return [accepted[slot] for slot in sorted(accepted)], sorted(accepted), {
            "wait_rollouts_ms": round(waited * 1000, 3),
            "straggler_wait_ms": round(straggler_wait * 1000, 3),
            "received_slots": sorted(accepted),
            "stale_rollouts": self.stale_rollout_count() - stale_count,
        }


def train_shared_round(model: Any, rollouts: Sequence[Mapping[str, np.ndarray]]) -> int:
    """Train one shared round from feed-forward or recurrent, mask-validated actor rollouts."""
    if not rollouts:
        raise SharedLearnerError("at least one actor rollout is required")
    import importlib
    torch = importlib.import_module("torch")
    steps = len(rollouts[0]["rewards"])
    if any(len(data["rewards"]) != steps for data in rollouts):
        raise SharedLearnerError("actor rollout lengths differ")
    state_fields = {"lstm_hidden_pi", "lstm_cell_pi", "lstm_hidden_vf", "lstm_cell_vf"}
    recurrent_flags = [state_fields <= set(data) for data in rollouts]
    if any(recurrent_flags) and not all(recurrent_flags):
        raise SharedLearnerError("recurrent and feed-forward actor rollouts cannot be mixed")
    recurrent = all(recurrent_flags)
    buffer: Any
    if recurrent:
        from sb3_contrib.common.recurrent.type_aliases import RNNStates

        from hk_agent.recurrent_sb3 import MaskableRecurrentDictRolloutBuffer
        lstm = model.policy.lstm_actor
        hidden_shape = (steps, lstm.num_layers, len(rollouts), lstm.hidden_size)
        buffer = MaskableRecurrentDictRolloutBuffer(
            buffer_size=steps, observation_space=model.observation_space,
            action_space=model.action_space, hidden_state_shape=hidden_shape,
            device=model.device, gae_lambda=model.gae_lambda, gamma=model.gamma,
            n_envs=len(rollouts),
        )
        buffer.sequence_length = model.sequence_length
    else:
        from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer
        buffer = MaskableDictRolloutBuffer(
            buffer_size=steps, observation_space=model.observation_space,
            action_space=model.action_space, device=model.device,
            gae_lambda=model.gae_lambda, gamma=model.gamma, n_envs=len(rollouts),
        )
    buffer.reset()
    obs_spaces = model.observation_space.spaces
    for row in range(steps):
        obs = {name: np.stack([data[f"obs__{name}"][row] for data in rollouts]) for name in obs_spaces}
        actions = np.stack([data["actions"][row].reshape(-1) for data in rollouts])
        rewards = np.asarray([data["rewards"][row] for data in rollouts], dtype=np.float32)
        episode_starts = np.asarray([data["episode_starts"][row] for data in rollouts], dtype=np.float32)
        values = torch.as_tensor([data["values"][row] for data in rollouts], device=model.device)
        log_probs = torch.as_tensor([data["log_probs"][row] for data in rollouts], device=model.device)
        action_masks = np.stack([data["action_masks"][row] for data in rollouts])
        if recurrent:
            state_values = {
                name: torch.as_tensor(np.stack([data[name][row] for data in rollouts], axis=1), device=model.device)
                for name in state_fields
            }
            states = RNNStates(
                (state_values["lstm_hidden_pi"], state_values["lstm_cell_pi"]),
                (state_values["lstm_hidden_vf"], state_values["lstm_cell_vf"]),
            )
            buffer.add(
                obs, actions, rewards, episode_starts, values, log_probs,
                lstm_states=states, action_masks=action_masks,
            )
        else:
            buffer.add(obs, actions, rewards, episode_starts, values, log_probs, action_masks=action_masks)
    buffer.advantages[:] = np.stack([data["advantages"] for data in rollouts], axis=1)
    buffer.returns[:] = np.stack([data["returns"] for data in rollouts], axis=1)
    if not buffer.full:
        raise SharedLearnerError("assembled learner rollout buffer is incomplete")
    model.rollout_buffer = buffer
    transitions = steps * len(rollouts)
    model.num_timesteps += transitions
    model.train()
    return transitions


def zero_reward_fraction(rollouts: Sequence[Mapping[str, np.ndarray]]) -> float:
    """Fraction of transitions whose reward is exactly zero in one learner round."""
    arrays = [np.asarray(row.get("rewards", []), dtype=np.float32).reshape(-1) for row in rollouts]
    total = sum(len(values) for values in arrays)
    if total == 0:
        return 0.0
    zero = sum(int(np.count_nonzero(values == 0.0)) for values in arrays)
    return zero / total


def train_behavioral_cloning_batch(model: Any, batch: Mapping[str, np.ndarray], *, weight: float) -> float:
    """Apply one masked supervised update to a context-matched human batch."""
    if not 0 < weight <= 1:
        raise ValueError("behavioral-cloning weight must be in (0, 1]")
    import torch

    device = model.device
    observation_names = ("pixels", "action_history", "player_knowledge")
    observations = {
        name: torch.as_tensor(batch[name], device=device)
        for name in observation_names
    }
    actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=device).flatten()
    masks = torch.as_tensor(batch["action_masks"], dtype=torch.bool, device=device)
    if masks.shape != (len(actions), int(model.action_space.n)) or not masks.any(dim=1).all():
        raise SharedLearnerError("demonstration action mask differs from the policy")
    if not masks.gather(1, actions.view(-1, 1)).all():
        raise SharedLearnerError("demonstration contains an unavailable action")
    if hasattr(model.policy, "lstm_actor"):
        from sb3_contrib.common.recurrent.type_aliases import RNNStates

        lstm = model.policy.lstm_actor
        sequence_count = int(batch.get("sequence_count", np.asarray(1)).item())
        if sequence_count < 1 or len(actions) % sequence_count:
            raise SharedLearnerError("demonstration sequence layout is malformed")
        shape = (lstm.num_layers, sequence_count, lstm.hidden_size)
        zero = torch.zeros(shape, dtype=torch.float32, device=device)
        starts = torch.zeros(len(actions), dtype=torch.float32, device=device)
        sequence_length = len(actions) // sequence_count
        starts[::sequence_length] = 1.0
        _values, log_probability, _entropy = model.policy.evaluate_actions(
            observations, actions, RNNStates((zero, zero), (zero, zero)), starts, masks,
        )
    else:
        _values, log_probability, _entropy = model.policy.evaluate_actions(
            observations, actions, action_masks=masks,
        )
    loss = -log_probability.mean()
    model.policy.optimizer.zero_grad(set_to_none=True)
    (weight * loss).backward()
    torch.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
    model.policy.optimizer.step()
    return float(loss.detach().cpu())


def apply_context_demonstrations(
    model: Any, shared_root: Path, store: SharedRoundStore, *,
    actor_slots: Sequence[int], round_number: int, manifest: dict[str, Any],
    enabled: bool = False, initial_weight: float = 0.25, decay_rounds: int = 20,
) -> list[dict[str, Any]]:
    """Train only on verified demonstrations matching a live actor's challenge and game version."""
    if not enabled:
        return []
    from hk_agent.demonstrations import demonstration_batch, imitation_schedule

    contexts: dict[str, set[str]] = {}
    for slot in actor_slots:
        try:
            status = json.loads(store.actor_status_path(slot).read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        context, version = status.get("challenge_context"), status.get("game_version")
        if isinstance(context, str) and isinstance(version, str) and context and version:
            contexts.setdefault(context, set()).add(version)
    used: list[dict[str, Any]] = []
    history = manifest.setdefault("demonstration_training", [])
    if not isinstance(history, list):
        history = []
        manifest["demonstration_training"] = history
    for context, versions in contexts.items():
        digest = hashlib.sha256(context.encode("utf-8")).hexdigest()[:16]
        folder = shared_root / "demonstrations" / digest
        for artifact in sorted(folder.glob("*.npz")) if folder.is_dir() else ():
            try:
                metadata = json.loads(artifact.with_suffix(".json").read_text("utf-8"))
                version = metadata.get("game_version")
                if not isinstance(version, str) or version not in versions:
                    continue
                first_used = metadata.get("first_used_round")
                if not isinstance(first_used, int) or first_used < 0:
                    first_used = round_number
                    metadata["first_used_round"] = first_used
                    artifact.with_suffix(".json").write_text(
                        json.dumps(metadata, sort_keys=True), encoding="utf-8"
                    )
                sample_count = metadata.get("samples", 0)
                sample_count = sample_count if isinstance(sample_count, int) and not isinstance(sample_count, bool) else 0
                weight = imitation_schedule(
                    initial_weight=initial_weight, decay_rounds=decay_rounds,
                    round_number=round_number - first_used, samples_in_bank=sample_count,
                )
                if weight <= 0:
                    continue
                batch = demonstration_batch(
                    artifact, expected_game_version=version, expected_context=context,
                    observation_space=model.observation_space,
                    action_count=int(model.action_space.n),
                )
                from hk_agent.demonstrations import augment_demonstration_batch
                from hk_agent.scene_change_task import (
                    SCENE_CHANGE_ACTION_TEMPLATES,
                    SCENE_CHANGE_INTERACTION_ACTION_TEMPLATES,
                )

                if int(model.action_space.n) == len(SCENE_CHANGE_ACTION_TEMPLATES):
                    batch = augment_demonstration_batch(batch, SCENE_CHANGE_ACTION_TEMPLATES)
                elif int(model.action_space.n) == len(SCENE_CHANGE_INTERACTION_ACTION_TEMPLATES):
                    batch = augment_demonstration_batch(batch, SCENE_CHANGE_INTERACTION_ACTION_TEMPLATES)
                loss = train_behavioral_cloning_batch(model, batch, weight=weight)
                row = {
                    "round": round_number, "context": context, "artifact": artifact.name,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "weight": weight, "loss": loss, "samples": len(batch["actions"]),
                    "game_version": version,
                }
                history.append(row)
                used.append(row)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                history.append({
                    "round": round_number, "context": context, "artifact": artifact.name,
                    "status": "rejected", "reason": f"{type(error).__name__}: {error}",
                })
    return used


class GhostDictEnv(gym.Env[dict[str, np.ndarray], int]):
    def __init__(self, observation_space: gym.spaces.Dict, action_space: gym.spaces.Discrete) -> None:
        super().__init__()
        self.observation_space = observation_space  # type: ignore[assignment]
        self.action_space = action_space  # type: ignore[assignment]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        observation_space = cast(gym.spaces.Dict, self.observation_space)
        return {name: np.zeros(cast(tuple[int, ...], space.shape), dtype=space.dtype) for name, space in observation_space.spaces.items()}, {}

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        raise RuntimeError("ghost learner environments must never execute steps")

    def action_masks(self) -> np.ndarray:
        return np.ones(cast(gym.spaces.Discrete, self.action_space).n, dtype=np.bool_)


def _shared_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    files = provenance.get("files", {})
    stable_files = {
        str(name): value.get("sha256")
        for name, value in files.items()
        if isinstance(files, Mapping) and isinstance(value, Mapping)
    } if isinstance(files, Mapping) else {}
    return {
        "game_profile": provenance.get("game_profile"),
        "campaign_origin": provenance.get("campaign_origin"),
        "files": stable_files,
        "observation_dimensions": provenance.get("observation_dimensions"),
        "shared_learner_protocol": SHARED_SCHEMA_VERSION,
    }


def _environment_game_version(environment: Any) -> str | None:
    """Read the connected bridge version through the task and Gym wrappers."""
    current = environment
    for _ in range(6):
        backend = getattr(current, "backend", None)
        bridge = getattr(backend, "_bridge", None)
        version = getattr(bridge, "game_version", None)
        if isinstance(version, str) and version:
            return version
        nested = getattr(current, "environment", None)
        if nested is None:
            return None
        current = nested
    return None


def actor_join_round(learner_status: Mapping[str, Any]) -> int:
    """Return the current policy round so a restarted actor skips completed rounds."""
    value = learner_status.get("round", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return cast(int, value)


def run_deterministic_evaluation(
    model: Any, environment: Any, *, round_number: int, slot: int,
    max_steps: int = 1000, stop_file: Path | None = None,
    heartbeat: Callable[[int], None] | None = None,
    deterministic: bool = True, stall_steps: int = 300,
    stall_seconds: float = 30.0,
) -> dict[str, Any]:
    """Run a short, direct-policy probe without submitting training rollouts."""
    from sb3_contrib.common.maskable.utils import get_action_masks

    max_steps = min(1000, max(1, int(max_steps)))
    stall_steps = min(300, max(1, int(stall_steps)))
    try:
        environment.env_method("set_evaluation_mode", True)
    except (AttributeError, NotImplementedError):
        pass
    previous_obs = getattr(model, "_last_obs", None)
    previous_starts = getattr(model, "_last_episode_starts", None)
    observation = environment.reset()
    policy = getattr(model, "policy", None)
    recurrent = hasattr(policy, "predict_masked") and hasattr(model, "sequence_length")
    evaluation_state = None
    evaluation_starts = np.ones((int(getattr(environment, "num_envs", 1)),), dtype=np.bool_)
    episode_return = 0.0
    episode_progress: dict[str, Any] = {}
    completed = False
    steps = 0
    max_progress = 0.0
    progress_scene: str | None = None
    last_progress_step = 0
    last_progress_at = time.monotonic()
    evaluation_started_at = last_progress_at
    last_heartbeat_at = last_progress_at
    stop_reason = "step_limit"
    stopped_early = False
    try:
        for steps in range(1, max_steps + 1):
            if stop_file is not None and stop_file.is_file():
                stop_reason = "stop_requested"
                stopped_early = True
                break
            masks = get_action_masks(environment)
            if recurrent and policy is not None:
                actions, evaluation_state = policy.predict_masked(
                    observation, evaluation_state, evaluation_starts, masks,
                    deterministic=deterministic,
                )
            else:
                actions, _ = model.predict(
                    observation, deterministic=deterministic, action_masks=masks
                )
            observation, rewards, dones, infos = environment.step(actions)
            evaluation_starts = np.asarray(dones, dtype=np.bool_)
            episode_return += float(np.asarray(rewards).reshape(-1)[0])
            info = infos[0] if infos else {}
            current_scene = info.get("scene_name") if isinstance(info, Mapping) else None
            current_max = max_progress
            progress_map = info.get("max_progress_by_scene") if isinstance(info, Mapping) else None
            if not isinstance(progress_map, Mapping) and isinstance(info, Mapping):
                step_progress = info.get("episode_progress")
                progress_map = step_progress.get("max_progress_by_scene") if isinstance(step_progress, Mapping) else None
            if isinstance(progress_map, Mapping):
                valid_progress = {
                    str(scene): float(value) for scene, value in progress_map.items()
                    if isinstance(scene, str) and isinstance(value, (int, float))
                    and not isinstance(value, bool) and math.isfinite(float(value))
                } if isinstance(progress_map, Mapping) else {}
                if valid_progress:
                    current_max = max(current_max, max(valid_progress.values()))
                    episode_progress["max_progress_by_scene"] = valid_progress
            changed_scene = isinstance(current_scene, str) and current_scene != progress_scene
            improved = current_max >= max_progress + 0.25 or changed_scene
            if improved:
                max_progress = max(max_progress, current_max)
                progress_scene = current_scene if isinstance(current_scene, str) else progress_scene
                last_progress_step = steps
                last_progress_at = time.monotonic()
            if isinstance(info, Mapping) and isinstance(info.get("episode_progress"), Mapping):
                episode_progress.update(dict(info["episode_progress"]))
            if bool(np.asarray(dones).reshape(-1)[0]):
                completed = True
                stop_reason = "episode_ended"
                break
            now = time.monotonic()
            stalled_for_steps = steps - last_progress_step
            stalled_for_seconds = now - last_progress_at
            if stalled_for_steps >= stall_steps or stalled_for_seconds >= stall_seconds:
                stop_reason = "stuck"
                stopped_early = True
                break
            if heartbeat is not None and now - last_heartbeat_at >= 5.0:
                heartbeat(steps)
                last_heartbeat_at = now
    finally:
        try:
            environment.env_method("set_evaluation_mode", False)
        except (AttributeError, NotImplementedError):
            pass
    if previous_obs is not None:
        model._last_obs = previous_obs
    if previous_starts is not None:
        model._last_episode_starts = previous_starts
    evaluation_elapsed = time.monotonic() - evaluation_started_at
    episode_progress["evaluation_max_progress"] = max_progress
    return {
        "schema_version": 2,
        "round": round_number,
        "slot": slot,
        "deterministic": deterministic,
        "measure": "deterministic" if deterministic else "stochastic",
        "steps": steps,
        "return": episode_return,
        "completed": completed,
        "rollout_submitted": False,
        "episode_progress": episode_progress,
        "evaluation_max_progress": max_progress,
        "time_until_stuck_steps": steps if stop_reason == "stuck" else None,
        "time_until_stuck_seconds": round(time.monotonic() - last_progress_at, 2) if stop_reason == "stuck" else None,
        "elapsed_seconds": round(evaluation_elapsed, 2),
        "stop_reason": stop_reason,
        "stopped_early": stopped_early,
        "finished_unix_ns": time.time_ns(),
    }


def evaluation_slot_for_round(round_number: int, interval: int, actor_count: int) -> int:
    """Alternate the single actor that probes at each evaluation boundary."""
    if round_number < 1 or interval < 1 or actor_count < 1:
        raise ValueError("evaluation round, interval, and actor count must be positive")
    evaluation_number = max(0, round_number // interval - 1)
    return evaluation_number % actor_count


def _shared_ppo_parameters(task_contract: Mapping[str, Any]) -> dict[str, Any]:
    if task_contract.get("reward_version") not in {"explore_v1", "explore_v2", "explore_v3", "explore_v4"}:
        return {}
    defaults = {
        "ent_coef": 0.01, "gamma": 0.995, "gae_lambda": 0.95,
        "n_epochs": 4, "target_kl": 0.03, "learning_rate": 2.5e-4,
        "clip_range": 0.2,
    }
    supplied = task_contract.get("ppo_hyperparameters")
    if isinstance(supplied, Mapping):
        defaults.update({key: supplied[key] for key in defaults if key in supplied})
    return defaults


def run_shared_actor(
    environment: gym.Env[Any, Any],
    *,
    task_contract: Mapping[str, Any],
    config: Any,
    output_root: Path,
    provenance: Mapping[str, Any],
    stop_file: Path | None,
    shared_root: Path,
    slot: int,
    launch_id: str,
    actors: int,
) -> dict[str, Any]:
    """Collect masked actor rollouts and publish them without running PPO updates."""
    import importlib

    importlib.import_module("stable_baselines3")
    torch = importlib.import_module("torch")
    contrib = importlib.import_module("sb3_contrib")
    from stable_baselines3.common.callbacks import BaseCallback
    # Leave CPU capacity for the learner: actors only execute one policy forward
    # per environment step, and additional BLAS workers contend with the game.
    actor_threads = 1
    torch.set_num_threads(actor_threads)
    torch.set_num_interop_threads(config.torch_interop_threads)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    recurrent_config = task_contract.get("recurrent_policy")
    recurrent_enabled = isinstance(recurrent_config, Mapping) and recurrent_config.get("name") == "maskable_recurrent_v1"
    if recurrent_enabled:
        from hk_agent.recurrent_sb3 import MaskableRecurrentPPO
        recurrent_settings = cast(Mapping[str, Any], recurrent_config)
        model = MaskableRecurrentPPO(
            "MultiInputLstmPolicy", environment, seed=config.seed,
            n_steps=config.rollout_steps, batch_size=config.batch_size,
            sequence_length=int(recurrent_settings["sequence_length"]), verbose=0,
            device=config.training_device,
            policy_kwargs={"lstm_hidden_size": int(recurrent_settings["hidden_size"]), "n_lstm_layers": int(recurrent_settings["layers"])},
            **_shared_ppo_parameters(task_contract),
        )
    else:
        model = contrib.MaskablePPO(
            "MultiInputPolicy", environment, seed=config.seed,
            n_steps=config.rollout_steps, batch_size=config.batch_size, verbose=0,
            device=config.training_device, **_shared_ppo_parameters(task_contract),
        )
    store = SharedRoundStore(shared_root)

    class ActorHeartbeatCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.round_number = 0
            self.last_heartbeat = 0.0
            self.resets = 0
            self.latest_episode_progress: dict[str, Any] = {}
            self.max_progress_by_scene: dict[str, float] = {}
            previous_actor = store._read_actor_status(slot)
            previous_progress = previous_actor.get("max_progress_by_scene")
            if isinstance(previous_progress, Mapping):
                self.max_progress_by_scene = {
                    str(scene): float(value) for scene, value in previous_progress.items()
                    if isinstance(scene, str) and isinstance(value, (int, float))
                    and not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0
                }
            self.action_counts: dict[str, int] = {}
            self.farm_alerts: set[str] = set()
            self.challenge_practice_completed = False
            self.challenge_practice_status: dict[str, Any] = {}
            self.pending_help_requests: list[dict[str, Any]] = []
            self.help_context: str | None = None
            self.step_profile_ms: dict[str, float] = {}
            self.task_profile_totals: dict[str, float] = {}
            self.bridge_profile_totals: dict[str, float] = {}
            self.profile_steps = 0

        def _on_step(self) -> bool:
            if stop_file is not None and stop_file.is_file():
                return False
            dones = self.locals.get("dones")
            if dones is not None:
                self.resets += int(np.count_nonzero(dones))
            infos = self.locals.get("infos")
            if isinstance(infos, (list, tuple)):
                for info in infos:
                    if not isinstance(info, Mapping):
                        continue
                    room_progress = info.get("max_progress_by_scene")
                    if isinstance(room_progress, Mapping):
                        for scene, value in room_progress.items():
                            if isinstance(scene, str) and isinstance(value, (int, float)) and not isinstance(value, bool):
                                self.max_progress_by_scene[scene] = max(
                                    self.max_progress_by_scene.get(scene, 0.0), float(value)
                                )
                    action_name = info.get("task_action")
                    if isinstance(action_name, str):
                        self.action_counts[action_name] = self.action_counts.get(action_name, 0) + 1
                    task_profile = info.get("actor_step_profile_ms")
                    if isinstance(task_profile, Mapping):
                        self.profile_steps += 1
                        for name, value in task_profile.items():
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                self.task_profile_totals[str(name)] = (
                                    self.task_profile_totals.get(str(name), 0.0) + float(value)
                                )
                    timing_profile = info.get("timing_ms")
                    if isinstance(timing_profile, Mapping):
                        for name, value in timing_profile.items():
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                self.bridge_profile_totals[str(name)] = (
                                    self.bridge_profile_totals.get(str(name), 0.0) + float(value)
                                )
                    if info.get("challenge_practice_finished") is True:
                        self.challenge_practice_completed = True
                    practice_status = info.get("challenge_practice")
                    if isinstance(practice_status, Mapping):
                        self.challenge_practice_status = dict(practice_status)
                    help_rows = info.get("pending_help_requests")
                    if isinstance(help_rows, list):
                        self.pending_help_requests = [dict(row) for row in help_rows if isinstance(row, Mapping)][:8]
                        context = next((row.get("context") for row in self.pending_help_requests
                                        if isinstance(row.get("context"), str)), None)
                        self.help_context = context
                    progress = info.get("episode_progress")
                    if isinstance(progress, Mapping) and info.get("evaluation_only") is not True:
                        self.latest_episode_progress = dict(progress)
                    alerts = info.get("farm_alerts")
                    if isinstance(alerts, list):
                        self.farm_alerts.update(item for item in alerts if isinstance(item, str))
            now = time.monotonic()
            if now - self.last_heartbeat >= 1.0:
                store.publish_actor_status(
                    slot=slot, launch_id=launch_id, status="COLLECTING",
                    round=self.round_number, collected_steps=self.n_calls,
                    episode_progress=self.latest_episode_progress,
                    max_progress_by_scene=self.max_progress_by_scene,
                    action_counts=self.action_counts,
                    farm_alerts=sorted(self.farm_alerts),
                    challenge_context=self.help_context or os.environ.get("HK_AGENT_CHALLENGE_CONTEXT"),
                    game_version=_environment_game_version(environment),
                    challenge_practice=self.challenge_practice_status,
                    pending_help_requests=self.pending_help_requests,
                )
                self.last_heartbeat = now
            return True

    callback = ActorHeartbeatCallback()
    model._setup_learn(
        total_timesteps=config.rollout_steps,
        callback=callback,
        reset_num_timesteps=True,
    )
    contract = store.publish_contract(
        task_contract=task_contract,
        fingerprint=__import__("hk_agent.distributed_knowledge", fromlist=["portable_compatibility_fingerprint"])
        .portable_compatibility_fingerprint(task_contract, _shared_provenance(provenance)),
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        rollout_steps=config.rollout_steps,
        actors=actors,
    )
    contract_body = contract["contract"]
    if not isinstance(contract_body, Mapping):
        raise SharedLearnerError("shared calibration has no actor contract")
    fingerprint = _safe_fingerprint(contract_body.get("task_contract_fingerprint"))
    try:
        previous_actor_status = json.loads(store.actor_status_path(slot).read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        previous_actor_status = {}
    restart_times = previous_actor_status.get("restart_times_unix_ns", []) if isinstance(previous_actor_status, Mapping) else []
    restart_times = [stamp for stamp in restart_times if isinstance(stamp, int) and stamp > time.time_ns() - 86_400_000_000_000] if isinstance(restart_times, list) else []
    if os.environ.get("HK_AGENT_RESTART_OF"):
        restart_times.append(time.time_ns())
    store.publish_actor_status(
        slot=slot, launch_id=launch_id, status="STARTING",
        relaunches=len(restart_times), restart_times_unix_ns=restart_times,
        challenge_context=os.environ.get("HK_AGENT_CHALLENGE_CONTEXT"),
        game_version=_environment_game_version(environment),
    )
    status: dict[str, Any] = {
        "schema_version": SHARED_SCHEMA_VERSION,
        "status": "RUNNING",
        "role": "actor",
        "slot": slot,
        "launch_id": launch_id,
        "algorithm": "MaskablePPO",
        "task_contract_fingerprint": fingerprint,
        "rounds_published": 0,
        "policy_import_retries": 0,
        "policy_import_skips": 0,
        "stale_rollouts": 0,
        "relaunches": len(restart_times),
        "restart_times_unix_ns": restart_times,
        "started_unix_ns": time.time_ns(),
    }
    _atomic_json(output_root / "manifest.json", status)
    learner_status: dict[str, Any] = {}
    try:
        try:
            learner_status = json.loads(store.learner_status_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        round_number = actor_join_round(learner_status)
        while True:
            if stop_file is not None and stop_file.is_file():
                status["status"] = "CANCELLED"
                break
            try:
                learner_status = json.loads(store.learner_status_path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                learner_status = {}
            learner_state = learner_status.get("status")
            if learner_state in {"COMPLETED", "CANCELLED", "FAILED"}:
                status["status"] = "CANCELLED" if learner_state != "FAILED" else "FAILED"
                break
            beat = learner_status.get("heartbeat_unix_ns")
            if not isinstance(beat, int) or (time.time_ns() - beat) / 1_000_000_000 > LEARNER_STALE_SECONDS:
                status["status"] = "FAILED"
                status["failure"] = "learner heartbeat expired"
                break
            policy_path = store.policy_root / f"round-{round_number:08d}.json"
            wait_policy_started = time.perf_counter()
            last_wait_heartbeat = 0.0
            while not policy_path.is_file():
                if stop_file is not None and stop_file.is_file():
                    status["status"] = "CANCELLED"
                    break
                now = time.monotonic()
                if now - last_wait_heartbeat >= 1.0:
                    store.publish_actor_status(
                        slot=slot, launch_id=launch_id,
                        status="WAITING_POLICY", round=round_number,
                    )
                    last_wait_heartbeat = now
                time.sleep(0.25)
            if status["status"] == "CANCELLED":
                break
            wait_policy_ms = (time.perf_counter() - wait_policy_started) * 1000
            imported = False
            for attempt in range(3):
                try:
                    store.import_policy(round_number, model.policy, fingerprint)
                    imported = True
                    break
                except (OSError, UnicodeError, json.JSONDecodeError, SharedLearnerError, ValueError) as error:
                    status["policy_import_retries"] = int(status["policy_import_retries"]) + 1
                    if attempt < 2:
                        time.sleep(0.2)
                    else:
                        status["policy_import_skips"] = int(status["policy_import_skips"]) + 1
                        status["last_policy_import_error"] = f"{type(error).__name__}: {error}"
                        _atomic_json(output_root / "manifest.json", status)
            if not imported:
                round_number += 1
                continue
            callback.round_number = round_number
            callback.resets = 0
            contract_config = task_contract.get("config", {})
            evaluation_interval = (
                int(contract_config.get("evaluation_interval", 20))
                if isinstance(contract_config, Mapping) else 20
            )
            evaluation_round = (
                task_contract.get("reward_version") in {"explore_v3", "explore_v4"}
                and round_number > 0 and evaluation_interval > 0
                and round_number % evaluation_interval == 0
            )
            scheduled_slot = (
                evaluation_slot_for_round(round_number, evaluation_interval, actors)
                if evaluation_round else -1
            )
            if evaluation_round and slot == scheduled_slot:
                existing_evaluations = [
                    row for row in store.read_evaluations()
                    if row.get("round") == round_number and row.get("slot") == slot
                ]
                existing_measures = {
                    row.get("measure") or ("deterministic" if row.get("deterministic") is True else "stochastic")
                    for row in existing_evaluations
                }
                missing_measures = [
                    (True, "deterministic"), (False, "stochastic"),
                ]
                missing_measures = [item for item in missing_measures if item[1] not in existing_measures]
                probe_max_steps = min(1000, max(1, int(contract_config.get("horizon_steps", 1000))))
                for deterministic, measure in missing_measures:
                    measure_label = "determinística" if deterministic else "estocástica"
                    label = f"Avaliando (prova rápida {measure_label}, até ~2,5 min)"
                    store.publish_actor_status(
                        slot=slot, launch_id=launch_id, status="EVALUATING", round=round_number,
                        evaluation_steps=0, evaluation_max_steps=probe_max_steps,
                        evaluation_measure=measure, evaluation_label=label,
                    )
                    def publish_evaluation_heartbeat(
                        steps: int, round_number: int = round_number,
                        probe_max_steps: int = probe_max_steps, measure: str = measure,
                        label: str = label,
                    ) -> None:
                        store.publish_actor_status(
                            slot=slot, launch_id=launch_id, status="EVALUATING",
                            round=round_number, evaluation_steps=steps,
                            evaluation_max_steps=probe_max_steps,
                            evaluation_measure=measure, evaluation_label=label,
                        )
                    evaluation = run_deterministic_evaluation(
                        model, model.get_env(), round_number=round_number, slot=slot,
                        max_steps=probe_max_steps, stop_file=stop_file,
                        heartbeat=publish_evaluation_heartbeat, deterministic=deterministic,
                    )
                    store.append_evaluation(evaluation)
                    status["evaluations_completed"] = int(status.get("evaluations_completed", 0)) + 1
                    _atomic_json(output_root / "manifest.json", status)
                    if stop_file is not None and stop_file.is_file():
                        status["status"] = "CANCELLED"
                        break
                if status["status"] == "CANCELLED":
                    break
            store.publish_actor_status(
                slot=slot, launch_id=launch_id, status="COLLECTING",
                round=round_number, collected_steps=0,
            )
            collect_started = time.perf_counter()
            if not model.collect_rollouts(
                model.get_env(), callback, model.rollout_buffer, n_rollout_steps=config.rollout_steps
            ):
                status["status"] = "CANCELLED"
                break
            collect_ms = (time.perf_counter() - collect_started) * 1000
            try:
                learner_status = json.loads(store.learner_status_path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                learner_status = {}
            latest_round = actor_join_round(learner_status)
            if latest_round > round_number:
                store.record_stale_rollout(
                    round_number=round_number, slot=slot, launch_id=launch_id
                )
                status["stale_rollouts"] = int(status["stale_rollouts"]) + 1
                store.publish_actor_status(
                    slot=slot, launch_id=launch_id, status="WAITING_POLICY",
                    round=latest_round, stale_rollouts=status["stale_rollouts"],
                )
                _append_actor_telemetry(shared_root, slot, {
                    "round": round_number, "launch_id": launch_id,
                    "wait_policy_ms": wait_policy_ms, "collect_ms": collect_ms,
                "publish_ms": 0.0, "stale_rollouts": 1,
                "step_profile_ms": callback.step_profile_ms,
                    "relaunches": int(
                        bool(os.environ.get("HK_AGENT_RESTART_OF"))
                        and not status.get("relaunch_recorded")
                    ), "resets": callback.resets,
                })
                status["resets"] = int(status.get("resets", 0)) + callback.resets
                status["relaunch_recorded"] = True
                round_number = latest_round
                _atomic_json(output_root / "manifest.json", status)
                continue
            store.publish_actor_status(
                slot=slot, launch_id=launch_id, status="PUBLISHING", round=round_number
            )
            publish_started = time.perf_counter()
            store.publish_rollout(
                round_number=round_number,
                slot=slot,
                launch_id=launch_id,
                buffer=model.rollout_buffer,
                fingerprint=fingerprint,
            )
            publish_ms = (time.perf_counter() - publish_started) * 1000
            callback.step_profile_ms.update({
                f"task_{name}": value / max(1, callback.profile_steps)
                for name, value in callback.task_profile_totals.items()
            })
            callback.step_profile_ms.update({
                f"bridge_{name}_ms": value / max(1, callback.profile_steps)
                for name, value in callback.bridge_profile_totals.items()
            })
            _append_actor_telemetry(shared_root, slot, {
                "round": round_number, "launch_id": launch_id,
                "wait_policy_ms": wait_policy_ms, "collect_ms": collect_ms,
                "publish_ms": publish_ms, "stale_rollouts": 0,
                "step_profile_ms": callback.step_profile_ms,
                "max_progress_by_scene": callback.max_progress_by_scene,
                "action_counts": callback.action_counts,
                "action_entropy": -sum(
                    (count / max(1, sum(callback.action_counts.values())))
                    * math.log(count / max(1, sum(callback.action_counts.values())))
                    for count in callback.action_counts.values() if count > 0
                ),
                "relaunches": int(
                    bool(os.environ.get("HK_AGENT_RESTART_OF"))
                    and not status.get("relaunch_recorded")
                ), "resets": callback.resets,
            })
            status["resets"] = int(status.get("resets", 0)) + callback.resets
            status["relaunch_recorded"] = True
            round_number += 1
            status["rounds_published"] = int(status["rounds_published"]) + 1
            status["last_policy_round"] = round_number - 1
            _atomic_json(output_root / "manifest.json", status)
            store.publish_actor_status(
                slot=slot, launch_id=launch_id, status="WAITING_POLICY",
                round=round_number, rounds_published=status["rounds_published"],
                stale_rollouts=status["stale_rollouts"],
            )
            if callback.challenge_practice_completed:
                status["status"] = "CANCELLED"
                status["challenge_practice_completed"] = True
                status["challenge_context"] = os.environ.get("HK_AGENT_CHALLENGE_CONTEXT")
                status["challenge_practice"] = callback.challenge_practice_status
                break
    except BaseException as error:
        status.update({"status": "FAILED", "failure": f"{type(error).__name__}: {error}"})
        raise
    finally:
        status["finished_unix_ns"] = time.time_ns()
        _atomic_json(output_root / "manifest.json", status)
        store.publish_actor_status(
            slot=slot, launch_id=launch_id,
            status=str(status.get("status", "FAILED")), round=round_number,
            rounds_published=status["rounds_published"], stale_rollouts=status["stale_rollouts"],
            challenge_practice_completed=status.get("challenge_practice_completed", False),
            challenge_context=status.get("challenge_context"),
            challenge_practice=status.get("challenge_practice", {}),
        )
        environment.close()
    return status


def _append_actor_telemetry(shared_root: Path, slot: int, row: Mapping[str, Any]) -> None:
    path = shared_root / "telemetry" / f"actor-slot-{slot}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(dict(row), allow_nan=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _restore_latest_checkpoint(
    model_class: Any, checkpoints: Path, environment: gym.Env[Any, Any]
) -> tuple[Any, int, int] | None:
    candidates = [
        path for path in checkpoints.glob("round-*.zip")
        if path.is_file() and re.fullmatch(r"round-\d{8}\.zip", path.name)
    ]
    for path in sorted(candidates, key=lambda item: item.name, reverse=True):
        round_text = path.stem.removeprefix("round-")
        try:
            model = model_class.load(str(path), env=environment, device="cpu")
        except Exception as error:  # noqa: BLE001 - corrupted checkpoint archives may raise library-specific errors
            logging.getLogger(__name__).warning("ignoring invalid checkpoint %s: %s", path, error)
            continue
        completed = getattr(model, "num_timesteps", None)
        if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
            logging.getLogger(__name__).warning("ignoring checkpoint with invalid timestep count: %s", path)
            continue
        return model, int(round_text), completed
    return None


def _warm_start_model_class(algorithm: object, feedforward_class: Any) -> Any:
    if algorithm == "MaskableRecurrentPPO":
        from hk_agent.recurrent_sb3 import MaskableRecurrentPPO
        return MaskableRecurrentPPO
    if algorithm in {None, "MaskablePPO"}:
        return feedforward_class
    raise SharedLearnerError(f"warm-start source algorithm {algorithm!r} is not supported")


def run_shared_learner(
    *,
    shared_root: Path,
    stop_file: Path | None,
    rollout_steps: int,
    batch_size: int,
    actors: int,
    checkpoint_interval: int,
    max_checkpoints: int,
    total_timesteps: int,
    continuous: bool,
    seed: int = 7,
    torch_threads: int = 3,
    torch_interop_threads: int = 1,
    rollout_timeout_seconds: float = 3900.0,
    straggler_timeout_seconds: float = DEFAULT_STRAGGLER_TIMEOUT_SECONDS,
    warm_start_checkpoint: Path | None = None,
    warm_start_source_campaign: str | None = None,
) -> dict[str, Any]:
    """Train one PPO learner from all valid actor files in each policy round."""
    if rollout_steps < 1 or actors < 1 or batch_size < 1 or rollout_steps % batch_size:
        raise ValueError("each shared actor rollout must be positive and divisible by batch_size")
    if not 0 < straggler_timeout_seconds <= rollout_timeout_seconds:
        raise ValueError("straggler_timeout_seconds must be positive and no larger than rollout_timeout_seconds")
    if checkpoint_interval < rollout_steps * actors or checkpoint_interval % (rollout_steps * actors):
        raise ValueError("checkpoint_interval must be a multiple of rollout_steps times actors")
    if not continuous and total_timesteps < 1:
        raise ValueError("total_timesteps must be positive")
    import importlib

    torch = importlib.import_module("torch")
    contrib = importlib.import_module("sb3_contrib")
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(torch_interop_threads)
    torch.manual_seed(seed)
    store = SharedRoundStore(shared_root)
    store.heartbeat(status="STARTING", round=0, completed_timesteps=0, accepted_rollouts=0, rejected_rollouts=0)
    try:
        calibration = store.wait_for_contract(timeout=rollout_timeout_seconds, stop_file=stop_file)
    except SharedLearnerStopped:
        manifest = {
            "schema_version": 1, "artifact_type": "shared_learner", "algorithm": "MaskablePPO",
            "status": "CANCELLED", "rounds": [], "actors_per_round": {},
            "rejected_rollouts": [], "completed_timesteps": 0, "round": 0,
            "finished_unix_ns": time.time_ns(),
        }
        store.heartbeat(status="CANCELLED", round=0, completed_timesteps=0)
        _atomic_json(shared_root / "manifest.json", manifest)
        return manifest
    contract = calibration.get("contract")
    if not isinstance(contract, Mapping):
        raise SharedLearnerError("shared calibration contract is invalid")
    fingerprint = _safe_fingerprint(contract.get("task_contract_fingerprint"))
    if contract.get("actors") != actors or contract.get("rollout_steps") != rollout_steps:
        raise SharedLearnerError("learner campaign settings differ from actor contract")
    observation_payload = contract.get("observation_space")
    action_payload = contract.get("action_space")
    if not isinstance(observation_payload, Mapping) or not isinstance(action_payload, Mapping):
        raise SharedLearnerError("shared calibration spaces are missing")
    parsed_observation_space = _space_from_payload(observation_payload)
    parsed_action_space = _space_from_payload(action_payload)
    if not isinstance(parsed_observation_space, gym.spaces.Dict) or not isinstance(parsed_action_space, gym.spaces.Discrete):
        raise SharedLearnerError("shared MaskablePPO requires Dict observations and Discrete actions")
    observation_space = parsed_observation_space
    action_space = parsed_action_space
    environment = GhostDictEnv(observation_space, action_space)
    task_contract = contract.get("task_contract", {})
    task_contract = task_contract if isinstance(task_contract, Mapping) else {}
    recurrent_config = task_contract.get("recurrent_policy")
    recurrent_enabled = isinstance(recurrent_config, Mapping) and recurrent_config.get("name") == "maskable_recurrent_v1"
    if recurrent_enabled:
        from hk_agent.recurrent_sb3 import MaskableRecurrentPPO
        recurrent_settings = cast(Mapping[str, Any], recurrent_config)
        model = MaskableRecurrentPPO(
            "MultiInputLstmPolicy", environment, seed=seed, n_steps=rollout_steps,
            batch_size=batch_size, sequence_length=int(recurrent_settings["sequence_length"]),
            verbose=0, device="cpu",
            policy_kwargs={"lstm_hidden_size": int(recurrent_settings["hidden_size"]), "n_lstm_layers": int(recurrent_settings["layers"])},
            **_shared_ppo_parameters(task_contract),
        )
        model_class = MaskableRecurrentPPO
    else:
        model = contrib.MaskablePPO(
            "MultiInputPolicy", environment, seed=seed, n_steps=rollout_steps,
            batch_size=batch_size, verbose=0, device="cpu",
            **_shared_ppo_parameters(task_contract),
        )
        model_class = contrib.MaskablePPO
    checkpoints = shared_root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    previous = sorted(
        (path for path in checkpoints.glob("round-*.zip") if path.is_file()),
        key=lambda path: int(path.stem.removeprefix("round-"))
        if path.stem.removeprefix("round-").isdigit() else -1,
    )
    warm_start_warning: str | None = None
    if warm_start_checkpoint is not None:
        if previous:
            raise SharedLearnerError("warm-start campaigns must use a new shared checkpoint directory")
        try:
            source_manifest_path = warm_start_checkpoint.parent.parent / "manifest.json"
            try:
                source_manifest = json.loads(source_manifest_path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise SharedLearnerError(f"warm-start source manifest is unavailable: {error}") from error
            source_algorithm = source_manifest.get("algorithm") if isinstance(source_manifest, Mapping) else None
            source_model_class = _warm_start_model_class(source_algorithm, contrib.MaskablePPO)
            source_model = source_model_class.load(str(warm_start_checkpoint), device="cpu")
            validate_warm_start_spaces(
                source_model.observation_space, source_model.action_space,
                observation_space, action_space,
            )
            target_task = contract.get("task_contract", {})
            target_names_raw = target_task.get("action_names") if isinstance(target_task, Mapping) else None
            target_names = target_names_raw if isinstance(target_names_raw, list) and all(
                isinstance(item, str) for item in target_names_raw
            ) else None
            source_task = source_manifest.get("task_contract", {}) if isinstance(source_manifest, Mapping) else {}
            source_names_raw = source_task.get("action_names") if isinstance(source_task, Mapping) else None
            source_names: list[str] | None = None
            if isinstance(source_names_raw, list) and all(isinstance(item, str) for item in source_names_raw):
                source_names = source_names_raw
            elif target_names is not None:
                added_names = {"JUMP_MAX", "LEFT+JUMP_MAX", "RIGHT+JUMP_MAX"}
                source_names = [item for item in target_names if item not in added_names]
            compatible = compatible_warm_start_state(
                source_model.policy.state_dict(), model.policy.state_dict(),
                source_action_names=source_names, target_action_names=target_names,
            )
            target_state = model.policy.state_dict()
            target_state.update(compatible)
            model.policy.load_state_dict(target_state, strict=True)
            model.policy.optimizer.state.clear()
            model.num_timesteps = 0
            model._n_updates = 0
        except (SharedLearnerError, OSError, ValueError, TypeError, KeyError,
                RuntimeError, AttributeError, ImportError, EOFError, zipfile.BadZipFile) as error:
            warm_start_warning = f"Warm-start incompatível; a campanha começou do zero. Motivo: {error}"
            logging.getLogger(__name__).warning(warm_start_warning)
            warm_start_checkpoint = None
            warm_start_source_campaign = None
    completed = 0
    current_round = 0
    if previous:
        restored = _restore_latest_checkpoint(model_class, checkpoints, environment)
        if restored is None:
            raise SharedLearnerError("no valid shared-learner checkpoint is available to resume")
        model, current_round, completed = restored
    model._setup_learn(
        total_timesteps=max(1, total_timesteps),
        callback=lambda _locals, _globals: True,
        reset_num_timesteps=not bool(previous),
    )
    manifest_path = shared_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {
            "schema_version": 1, "artifact_type": "shared_learner", "algorithm": "MaskableRecurrentPPO" if recurrent_enabled else "MaskablePPO",
            "rounds": [], "actors_per_round": {}, "rejected_rollouts": [],
            "task_contract_fingerprint": fingerprint, "started_unix_ns": time.time_ns(),
        }
    manifest["status"] = "RUNNING"
    manifest["algorithm"] = "MaskableRecurrentPPO" if recurrent_enabled else "MaskablePPO"
    if warm_start_warning:
        manifest["warm_start_warning"] = warm_start_warning
    else:
        manifest.pop("warm_start_warning", None)
    manifest["task_contract_fingerprint"] = fingerprint
    manifest["task_contract"] = dict(contract.get("task_contract", {})) if isinstance(contract.get("task_contract"), Mapping) else {}
    manifest["rollout_steps"] = rollout_steps
    manifest["batch_size"] = batch_size
    manifest["ppo_hyperparameters"] = _shared_ppo_parameters(
        contract.get("task_contract", {}) if isinstance(contract.get("task_contract"), Mapping) else {}
    )
    if warm_start_checkpoint is not None:
        manifest["warm_start"] = {
            "source_campaign": warm_start_source_campaign,
            "checkpoint": str(warm_start_checkpoint),
            "checkpoint_sha256": _sha256_file(warm_start_checkpoint),
            "mode": "compatible_tensors_only",
            "optimizer_reset": True,
            "source_reward_version": "explore_v3",
        }
    manifest.setdefault("round_timings", [])
    entropy_target = float(model.ent_coef)
    existing_recovery = manifest.get("entropy_recovery")
    if (
        isinstance(existing_recovery, Mapping)
        and isinstance(existing_recovery.get("end_round_exclusive"), int)
        and existing_recovery["end_round_exclusive"] > current_round
    ):
        entropy_recovery_start = int(existing_recovery.get("start_round", current_round))
        entropy_recovery_end = int(existing_recovery["end_round_exclusive"])
        entropy_target = float(existing_recovery.get("target_ent_coef", entropy_target))
        manifest["entropy_recovery"] = dict(existing_recovery)
    else:
        entropy_recovery_start = current_round
        entropy_recovery_end = current_round + 10
        manifest["entropy_recovery"] = {
            "start_round": entropy_recovery_start,
            "end_round_exclusive": entropy_recovery_end,
            "recovery_ent_coef": 0.025,
            "target_ent_coef": entropy_target,
        }
    next_checkpoint = ((completed // checkpoint_interval) + 1) * checkpoint_interval

    def save_checkpoint(round_number: int) -> None:
        nonlocal next_checkpoint
        temporary = checkpoints / f".round-{round_number:08d}.pending.zip"
        target = checkpoints / f"round-{round_number:08d}.zip"
        model.save(temporary)
        os.replace(temporary, target)
        checkpoint_paths = sorted(
            checkpoints.glob("round-*.zip"),
            key=lambda path: int(path.stem.removeprefix("round-")),
        )
        newest = set(checkpoint_paths[-max_checkpoints:])
        permanent = {
            path for path in checkpoint_paths
            if path.stem.removeprefix("round-").isdigit()
            and int(path.stem.removeprefix("round-")) % 20 == 0
        }
        for stale in checkpoint_paths:
            if stale not in newest and stale not in permanent:
                stale.unlink(missing_ok=True)
        inventory = manifest.setdefault("checkpoints", [])
        if isinstance(inventory, list):
            inventory[:] = [str(path) for path in sorted(checkpoints.glob("round-*.zip"))]
        manifest["latest_checkpoint"] = str(target)
        manifest["completed_timesteps"] = completed
        manifest["round"] = round_number
        _atomic_json(manifest_path, manifest)
        next_checkpoint = ((completed // checkpoint_interval) + 1) * checkpoint_interval

    try:
        store.publish_policy(current_round, model.policy, fingerprint, replace_existing=bool(previous))
        while True:
            if stop_file is not None and stop_file.is_file():
                manifest["status"] = "CANCELLED"
                break
            if not continuous and completed >= total_timesteps:
                manifest["status"] = "COMPLETED"
                break
            store.heartbeat(
                status="WAITING", round=current_round, completed_timesteps=completed
            )
            rollouts, slots, wait_metrics = store.wait_rollouts(
                round_number=current_round, actors=actors, fingerprint=fingerprint,
                n_steps=rollout_steps, observation_space=observation_space,
                action_count=int(action_space.n), timeout=rollout_timeout_seconds,
                straggler_timeout=straggler_timeout_seconds, stop_file=stop_file,
            )
            if stop_file is not None and stop_file.is_file():
                manifest["status"] = "CANCELLED"
                break
            zero_fraction = zero_reward_fraction(rollouts)
            model.ent_coef = (
                0.025 if current_round < entropy_recovery_end else entropy_target
            )
            training_started = time.perf_counter()
            transitions = train_shared_round(model, rollouts)
            demonstration_rows = apply_context_demonstrations(
                model, shared_root, store, actor_slots=slots,
                round_number=current_round, manifest=manifest,
                enabled=os.environ.get("HK_AGENT_HELP_REQUESTS_ENABLED", "0") == "1",
            )
            training_seconds = time.perf_counter() - training_started
            completed += transitions
            current_round += 1
            store.publish_policy(current_round, model.policy, fingerprint)
            manifest_rounds = manifest.get("rounds")
            if isinstance(manifest_rounds, list):
                manifest_rounds.append(current_round - 1)
            if demonstration_rows:
                for row in demonstration_rows:
                    row["round"] = current_round - 1
            actor_rounds = manifest.get("actors_per_round")
            if isinstance(actor_rounds, dict):
                actor_rounds[str(current_round - 1)] = slots
            timings = manifest.get("round_timings")
            if isinstance(timings, list):
                timings.append({
                    "round": current_round - 1,
                    "training_seconds": training_seconds,
                    "transitions": transitions,
                    "completed_timesteps": completed,
                    "wait_rollouts_ms": wait_metrics["wait_rollouts_ms"],
                    "straggler_wait_ms": wait_metrics["straggler_wait_ms"],
                    "train_ms": round(training_seconds * 1000, 3),
                    "actors_received": slots,
                    "stale_rollouts": wait_metrics["stale_rollouts"],
                    "zero_reward_fraction": zero_fraction,
                    "ent_coef": float(model.ent_coef),
                })
            evaluation_rows = store.read_evaluations()
            manifest["evaluations"] = evaluation_rows
            manifest["evaluation_trend"] = [
                {"round": row.get("round"), "return": row.get("return"), "steps": row.get("steps")}
                for row in evaluation_rows
            ]
            progress_rows = manifest.setdefault("episode_progress", [])
            if isinstance(progress_rows, list):
                known_progress = {
                    (row.get("slot"), row.get("episode"))
                    for row in progress_rows if isinstance(row, Mapping)
                }
                for actor_slot in range(actors):
                    try:
                        actor_status = json.loads(store.actor_status_path(actor_slot).read_text("utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    episode = actor_status.get("episode_progress")
                    key = (actor_slot, episode.get("episode")) if isinstance(episode, Mapping) else None
                    if key is not None and key not in known_progress:
                        progress_rows.append({"slot": actor_slot, **dict(episode)})
                        known_progress.add(key)
                del progress_rows[:-500]
            existing_alerts = manifest.get("farm_alerts", [])
            alert_set = {item for item in existing_alerts if isinstance(item, str)} if isinstance(existing_alerts, list) else set()
            for actor_slot in range(actors):
                try:
                    actor_status = json.loads(store.actor_status_path(actor_slot).read_text("utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                alerts = actor_status.get("farm_alerts")
                if isinstance(alerts, list):
                    alert_set.update(item for item in alerts if isinstance(item, str))
            manifest["farm_alerts"] = sorted(alert_set)
            manifest["last_zero_reward_fraction"] = zero_fraction
            if zero_fraction > 0.90:
                alert_set.add("reward_zero_over_90_percent")
                manifest["farm_alerts"] = sorted(alert_set)
            manifest["completed_timesteps"] = completed
            manifest["round"] = current_round
            rejected_count = 0
            rejected_path = shared_root / "rejected.jsonl"
            if rejected_path.is_file():
                rejected_count = sum(1 for _ in rejected_path.open("rb"))
            store.heartbeat(
                status="TRAINING", round=current_round, completed_timesteps=completed,
                accepted_rollouts=len(slots),
                rejected_rollouts=rejected_count, last_training_seconds=training_seconds,
                last_wait_rollouts_ms=wait_metrics["wait_rollouts_ms"],
                last_straggler_wait_ms=wait_metrics["straggler_wait_ms"],
                last_train_ms=round(training_seconds * 1000, 3),
                last_actors_received=slots,
                stale_rollouts_total=store.stale_rollout_count(),
                evaluation_count=len(evaluation_rows),
                farm_alerts=manifest["farm_alerts"],
                zero_reward_fraction=zero_fraction,
            )
            if completed >= next_checkpoint:
                save_checkpoint(current_round)
            _atomic_json(manifest_path, manifest)
        if manifest.get("status") in {"CANCELLED", "COMPLETED"} and completed > 0:
            save_checkpoint(current_round)
    except BaseException as error:
        manifest.update({"status": "FAILED", "failure": f"{type(error).__name__}: {error}"})
        raise
    finally:
        manifest["completed_timesteps"] = completed
        manifest["round"] = current_round
        manifest["finished_unix_ns"] = time.time_ns()
        _atomic_json(manifest_path, manifest)
        store.heartbeat(status=manifest["status"], round=current_round, completed_timesteps=completed)
        environment.close()
    return cast(dict[str, Any], manifest)
