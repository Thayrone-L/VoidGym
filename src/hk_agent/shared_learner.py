from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np

from hk_agent.policy_artifact import export_policy_artifact, import_policy_artifact

MAX_ROLLOUT_BYTES = 512 * 1024 * 1024
LEARNER_STALE_SECONDS = 300.0
SHARED_SCHEMA_VERSION = 1


class SharedLearnerError(RuntimeError):
    """Shared actor/learner protocol error."""


class SharedLearnerStopped(SharedLearnerError):
    """Cooperative stop requested while waiting for shared calibration."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
    try:
        with pending.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, json.dumps(value, indent=2, allow_nan=False).encode("utf-8") + b"\n")


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
        return gym.spaces.Box(
            low=np.asarray(payload.get("low"), dtype=np.dtype(dtype)),
            high=np.asarray(payload.get("high"), dtype=np.dtype(dtype)),
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

    def publish_policy(self, round_number: int, policy: Any, fingerprint: str) -> dict[str, Any]:
        fingerprint = _safe_fingerprint(fingerprint)
        stem = f"round-{round_number:08d}"
        artifact = self.policy_root / f"{stem}.safetensors"
        sidecar = self.policy_root / f"{stem}.policy.manifest.json"
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
        _atomic_json(self.policy_root / f"{stem}.json", pointer)
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
        stop_file: Path | None = None,
    ) -> tuple[list[dict[str, np.ndarray]], list[int]]:
        deadline = time.monotonic() + timeout
        accepted: dict[int, dict[str, np.ndarray]] = {}
        inspected: dict[int, tuple[int, int]] = {}
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
            time.sleep(0.2)
        if not accepted:
            if stop_file is not None and stop_file.is_file():
                return [], []
            raise TimeoutError("no valid actor rollouts arrived before quorum timeout")
        return [accepted[slot] for slot in sorted(accepted)], sorted(accepted)


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


def _shared_ppo_parameters(task_contract: Mapping[str, Any]) -> dict[str, Any]:
    if task_contract.get("reward_version") != "explore_v1":
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
    _, callback = model._setup_learn(
        total_timesteps=config.rollout_steps,
        callback=lambda _locals, _globals: True,
        reset_num_timesteps=True,
    )
    store = SharedRoundStore(shared_root)
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
            if not policy_path.is_file():
                time.sleep(0.25)
                continue
            store.import_policy(round_number, model.policy, fingerprint)
            if not model.collect_rollouts(
                model.get_env(), callback, model.rollout_buffer, n_rollout_steps=config.rollout_steps
            ):
                status["status"] = "CANCELLED"
                break
            store.publish_rollout(
                round_number=round_number,
                slot=slot,
                launch_id=launch_id,
                buffer=model.rollout_buffer,
                fingerprint=fingerprint,
            )
            round_number += 1
            status["rounds_published"] = int(status["rounds_published"]) + 1
            status["last_policy_round"] = round_number - 1
            _atomic_json(output_root / "manifest.json", status)
    except BaseException as error:
        status.update({"status": "FAILED", "failure": f"{type(error).__name__}: {error}"})
        raise
    finally:
        status["finished_unix_ns"] = time.time_ns()
        _atomic_json(output_root / "manifest.json", status)
        environment.close()
    return status


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
) -> dict[str, Any]:
    """Train one PPO learner from all valid actor files in each policy round."""
    if rollout_steps < 1 or actors < 1 or batch_size < 1 or rollout_steps * actors % batch_size:
        raise ValueError("shared rollout size must be positive and divisible by batch_size")
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
        store.publish_policy(round_number, model.policy, fingerprint)
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
        store.publish_policy(current_round, model.policy, fingerprint)
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
            rollouts, slots = store.wait_rollouts(
                round_number=current_round, actors=actors, fingerprint=fingerprint,
                n_steps=rollout_steps, observation_space=observation_space,
                action_count=int(action_space.n), timeout=rollout_timeout_seconds, stop_file=stop_file,
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
                })
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


