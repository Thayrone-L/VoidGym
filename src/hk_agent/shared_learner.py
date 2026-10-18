from __future__ import annotations

import hashlib
import json
import os
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
DEFAULT_STRAGGLER_TIMEOUT_SECONDS = 45.0
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
        if current.get("status") == "RESTARTING" and current.get("launch_id") == launch_id:
            return False
        if current.get("launch_id") not in (None, launch_id) and current.get("status") not in {
            "RESTARTING", "CANCELLED", "FAILED", "COMPLETED",
        }:
            return False
        _atomic_json(path, {
            **current, **values, "slot": slot, "launch_id": launch_id,
            "status": status, "heartbeat_unix_ns": time.time_ns(),
        })
        return True

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
        self, round_number: int, policy: Any, fingerprint: str, *, reuse_existing: bool = False
    ) -> dict[str, Any]:
        fingerprint = _safe_fingerprint(fingerprint)
        stem = f"round-{round_number:08d}"
        pointer_path = self.policy_root / f"{stem}.json"
        if pointer_path.exists():
            if reuse_existing:
                self.import_policy(round_number, policy, fingerprint)
                return cast(dict[str, Any], json.loads(pointer_path.read_text("utf-8")))
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
            try:
                os.link(pending_pointer, pointer_path)
            except FileExistsError as error:
                raise SharedLearnerError(
                    f"policy round {round_number} was already published"
                ) from error
        finally:
            pending_pointer.unlink(missing_ok=True)
        for old_artifact in self.policy_root.glob("round-*.safetensors"):
            try:
                old_round = int(old_artifact.name[6:14])
            except ValueError:
                continue
            if old_round < round_number - 2:
                old_artifact.unlink(missing_ok=True)
                old_manifest = old_artifact.with_name(
                    old_artifact.name.removesuffix(".safetensors") + ".policy.manifest.json"
                )
                old_manifest.unlink(missing_ok=True)
        return pointer

    def import_policy(self, round_number: int, policy: Any, fingerprint: str) -> dict[str, Any]:
        stem = f"round-{round_number:08d}"
        pointer = json.loads((self.policy_root / f"{stem}.json").read_text("utf-8"))
        if pointer.get("policy_round") != round_number:
            raise SharedLearnerError("policy round does not match request")
        if pointer.get("task_contract_fingerprint") != _safe_fingerprint(fingerprint):
            raise SharedLearnerError("policy task contract fingerprint differs")
        artifact = self.policy_root / str(pointer.get("artifact"))
        manifest = self.policy_root / str(pointer.get("manifest"))
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
        try:
            with np.load(artifact, allow_pickle=False) as packed:
                if set(packed.files) != required:
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
        key = (record.get("round"), record.get("slot"))
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(existing, Mapping) and (existing.get("round"), existing.get("slot")) == key:
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
            if not accepted and availability and all(value is False for value in availability):
                raise TimeoutError("all shared actor slots are unavailable before the first rollout")
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
    """Load actor arrays into a fresh masked Dict rollout buffer and update PPO once."""
    if not rollouts:
        raise SharedLearnerError("at least one actor rollout is required")
    import importlib
    torch = importlib.import_module("torch")
    from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer

    steps = len(rollouts[0]["rewards"])
    if any(len(data["rewards"]) != steps for data in rollouts):
        raise SharedLearnerError("actor rollout lengths differ")
    buffer = MaskableDictRolloutBuffer(
        buffer_size=steps,
        observation_space=model.observation_space,
        action_space=model.action_space,
        device=model.device,
        gae_lambda=model.gae_lambda,
        gamma=model.gamma,
        n_envs=len(rollouts),
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


def actor_join_round(learner_status: Mapping[str, Any]) -> int:
    """Return the current policy round so a restarted actor skips completed rounds."""
    value = learner_status.get("round", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return cast(int, value)


def run_deterministic_evaluation(
    model: Any, environment: Any, *, round_number: int, slot: int,
    max_steps: int, stop_file: Path | None = None,
    heartbeat: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Run one direct-policy episode; this path never calls the rollout collector."""
    from sb3_contrib.common.maskable.utils import get_action_masks

    try:
        environment.env_method("set_evaluation_mode", True)
    except (AttributeError, NotImplementedError):
        pass
    observation = environment.reset()
    episode_return = 0.0
    episode_progress: dict[str, Any] = {}
    completed = False
    steps = 0
    last_heartbeat = time.monotonic()
    try:
        for steps in range(1, max_steps + 1):
            if stop_file is not None and stop_file.is_file():
                break
            masks = get_action_masks(environment)
            actions, _ = model.predict(observation, deterministic=True, action_masks=masks)
            observation, rewards, dones, infos = environment.step(actions)
            episode_return += float(np.asarray(rewards).reshape(-1)[0])
            info = infos[0] if infos else {}
            if isinstance(info, Mapping) and isinstance(info.get("episode_progress"), Mapping):
                episode_progress = dict(info["episode_progress"])
            if bool(np.asarray(dones).reshape(-1)[0]):
                completed = True
                break
            if heartbeat is not None and time.monotonic() - last_heartbeat >= 5.0:
                heartbeat(steps)
                last_heartbeat = time.monotonic()
    finally:
        try:
            environment.env_method("set_evaluation_mode", False)
        except (AttributeError, NotImplementedError):
            pass
    model._last_obs = observation
    model._last_episode_starts = np.asarray([completed], dtype=np.bool_)
    return {
        "schema_version": 1,
        "round": round_number,
        "slot": slot,
        "deterministic": True,
        "steps": steps,
        "return": episode_return,
        "completed": completed,
        "rollout_submitted": False,
        "episode_progress": episode_progress,
        "finished_unix_ns": time.time_ns(),
    }


def _shared_ppo_parameters(task_contract: Mapping[str, Any]) -> dict[str, Any]:
    if task_contract.get("reward_version") not in {"explore_v1", "explore_v2", "explore_v3"}:
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
    torch.set_num_threads(config.torch_threads)
    torch.set_num_interop_threads(config.torch_interop_threads)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = contrib.MaskablePPO(
        "MultiInputPolicy",
        environment,
        seed=config.seed,
        n_steps=config.rollout_steps,
        batch_size=config.batch_size,
        verbose=0,
        device=config.training_device,
        **_shared_ppo_parameters(task_contract),
    )
    store = SharedRoundStore(shared_root)

    class ActorHeartbeatCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.round_number = 0
            self.last_heartbeat = 0.0
            self.resets = 0
            self.latest_episode_progress: dict[str, Any] = {}
            self.farm_alerts: set[str] = set()

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
                    farm_alerts=sorted(self.farm_alerts),
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
        "relaunches": int(bool(os.environ.get("HK_AGENT_RESTART_OF"))),
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
                status["status"] = "CANCELLED"
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
            if (
                task_contract.get("reward_version") == "explore_v3"
                and round_number > 0
                and evaluation_interval > 0
                and round_number % evaluation_interval == 0
                and not any(row.get("round") == round_number and row.get("slot") == slot for row in store.read_evaluations())
            ):
                store.publish_actor_status(slot=slot, launch_id=launch_id, status="EVALUATING", round=round_number)
                def publish_evaluation_heartbeat(steps: int, round_number: int = round_number) -> None:
                    store.publish_actor_status(
                        slot=slot, launch_id=launch_id, status="EVALUATING",
                        round=round_number, evaluation_steps=steps,
                    )
                evaluation = run_deterministic_evaluation(
                    model, model.get_env(), round_number=round_number, slot=slot,
                    max_steps=max(1, int(contract_config.get("horizon_steps", 3000))),
                    stop_file=stop_file,
                    heartbeat=publish_evaluation_heartbeat,
                )
                store.append_evaluation(evaluation)
                status["evaluations_completed"] = int(status.get("evaluations_completed", 0)) + 1
                _atomic_json(output_root / "manifest.json", status)
                if stop_file is not None and stop_file.is_file():
                    status["status"] = "CANCELLED"
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
            _append_actor_telemetry(shared_root, slot, {
                "round": round_number, "launch_id": launch_id,
                "wait_policy_ms": wait_policy_ms, "collect_ms": collect_ms,
                "publish_ms": publish_ms, "stale_rollouts": 0,
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
    rollout_timeout_seconds: float = 600.0,
    straggler_timeout_seconds: float = DEFAULT_STRAGGLER_TIMEOUT_SECONDS,
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
    model = contrib.MaskablePPO(
        "MultiInputPolicy", environment, seed=seed, n_steps=rollout_steps,
        batch_size=batch_size, verbose=0, device="cpu",
        **_shared_ppo_parameters(contract.get("task_contract", {}) if isinstance(contract.get("task_contract"), Mapping) else {}),
    )
    checkpoints = shared_root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    previous = sorted(checkpoints.glob("round-*.zip"))
    status_path = store.learner_status_path
    completed = 0
    current_round = 0
    if previous:
        model = contrib.MaskablePPO.load(str(previous[-1]), env=environment, device="cpu")
        try:
            prior = json.loads(status_path.read_text("utf-8"))
            completed = int(prior.get("completed_timesteps", 0))
            current_round = int(prior.get("round", 0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
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
            "schema_version": 1, "artifact_type": "shared_learner", "algorithm": "MaskablePPO",
            "rounds": [], "actors_per_round": {}, "rejected_rollouts": [],
            "task_contract_fingerprint": fingerprint, "started_unix_ns": time.time_ns(),
        }
    manifest["status"] = "RUNNING"
    manifest["algorithm"] = "MaskablePPO"
    manifest["task_contract_fingerprint"] = fingerprint
    manifest["rollout_steps"] = rollout_steps
    manifest["batch_size"] = batch_size
    manifest["ppo_hyperparameters"] = _shared_ppo_parameters(
        contract.get("task_contract", {}) if isinstance(contract.get("task_contract"), Mapping) else {}
    )
    manifest.setdefault("round_timings", [])
    next_checkpoint = ((completed // checkpoint_interval) + 1) * checkpoint_interval

    def save_checkpoint(round_number: int) -> None:
        nonlocal next_checkpoint
        temporary = checkpoints / f".round-{round_number:08d}.pending.zip"
        target = checkpoints / f"round-{round_number:08d}.zip"
        model.save(temporary)
        os.replace(temporary, target)
        checkpoint_paths = sorted(checkpoints.glob("round-*.zip"))
        for stale in checkpoint_paths[:-max_checkpoints]:
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
        store.publish_policy(current_round, model.policy, fingerprint, reuse_existing=True)
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
            training_started = time.perf_counter()
            transitions = train_shared_round(model, rollouts)
            training_seconds = time.perf_counter() - training_started
            completed += transitions
            current_round += 1
            store.publish_policy(current_round, model.policy, fingerprint)
            manifest_rounds = manifest.get("rounds")
            if isinstance(manifest_rounds, list):
                manifest_rounds.append(current_round - 1)
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


