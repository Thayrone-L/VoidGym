from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from hk_agent.shade_run_lifecycle import ShadeRunTracker, ShadeRunUpdate

BUTTON_ORDER: tuple[str, ...] = (
    "LEFT",
    "RIGHT",
    "UP",
    "DOWN",
    "JUMP",
    "ATTACK",
    "DASH",
    "CAST",
    "SUPER_DASH",
    "DREAM_NAIL",
)
RAW_ACTIONS: tuple[tuple[str, ...], ...] = tuple(
    tuple(button for bit, button in enumerate(BUTTON_ORDER) if mask & (1 << bit))
    for mask in range(1 << len(BUTTON_ORDER))
)
ACTION_NAMES: tuple[str, ...] = tuple(
    "+".join(action) if action else "NOOP" for action in RAW_ACTIONS
)
_BUTTONS = frozenset(BUTTON_ORDER)
_PUBLIC_TERMINATION_REASONS = frozenset(
    {
        "player_death",
        "task_success",
        "time_limit",
        "connection_lost",
        "capture_failed",
        "reset_failed",
        "backend_termination",
        "backend_truncation",
    }
)


class FrameValidationError(RuntimeError):
    """Raised when a captured frame cannot become training experience."""


@dataclass(frozen=True, slots=True)
class PixelFrame:
    episode_id: str
    frame_id: int
    capture_monotonic_ns: int
    pixels: NDArray[np.uint8]
    scene_transition: bool = False
    duplicate_of: int | None = None
    last_applied_action_id: int | None = None
    player_dead: bool = False
    death_persisted: bool = False
    shade_pending: bool = False
    shade_recovered: bool = False
    run_failed: bool = False
    enemy_kills_since_last_frame: int = 0
    hp_lost_since_last_frame: int = 0
    geo_gained_since_last_frame: int = 0
    geo_spent_since_last_frame: int = 0


@dataclass(frozen=True, slots=True)
class AppliedAction:
    action_id: int
    requested_buttons: tuple[str, ...]
    applied_buttons: tuple[str, ...]
    lease_ms: int


@dataclass(frozen=True, slots=True)
class ResetResult:
    episode_id: str
    frame: PixelFrame


@dataclass(frozen=True, slots=True)
class BackendStep:
    frame: PixelFrame
    applied_action: AppliedAction
    terminated: bool = False
    truncated: bool = False
    termination_reason: str | None = None
    timing_ms: Mapping[str, float] | None = None


class EnvironmentBackend(Protocol):
    """Synchronous boundary to a process/transport implementation.

    An adapter for the asynchronous SupervisorServer should own its event-loop
    thread and expose only these blocking, one-command-at-a-time operations.
    """

    @property
    def run_id(self) -> str: ...

    @property
    def instance_id(self) -> str: ...

    def reset(self, *, seed: int | None, options: Mapping[str, Any] | None) -> ResetResult: ...

    def step(self, *, action_id: int, buttons: tuple[str, ...], lease_ms: int) -> BackendStep: ...

    def release_all(self) -> None: ...

    def close(self) -> None: ...


def extended_length_path(path: str, *, os_name: str = os.name) -> str:
    """Return a Windows extended-length path (\\\\?\\...) for deep experience trees.

    Per-launch runtime roots plus instance/episode identifiers exceed MAX_PATH
    (260 for files, 248 for directories), which made the experience writer fail
    with WinError 206. Extended-length paths lift that limit without requiring the
    system-wide LongPathsEnabled setting. Other platforms are returned unchanged.
    """
    if os_name != "nt":
        return path
    if path.startswith("\\\\?\\"):
        return path
    absolute = os.path.abspath(path)
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


class AtomicJsonlWriter:
    """Append complete JSON records durably, without a partially buffered line."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: Mapping[str, Any]) -> None:
        os.makedirs(extended_length_path(str(self.path.parent)), exist_ok=True)
        payload = (
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        descriptor = os.open(extended_length_path(str(self.path)), flags, 0o600)
        try:
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError(f"partial JSONL append: {written} of {len(payload)} bytes")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class HollowKnightEnv(gym.Env[dict[str, NDArray[np.generic]], int]):
    """Phase-2 visual environment with zero reward and raw discrete actions.

    The action index maps to :data:`RAW_ACTIONS`. Each non-NOOP action is one
    bitmask of raw game buttons held for the fixed ``lease_ms``; there are no
    route or gameplay macros. Opposite directions are neutralized. Observations
    contain ``pixels`` in ``[K,H,W,C]`` order and the previous ``K-1`` applied,
    normalized action indices in ``action_history``.
    """

    def __init__(
        self,
        backend: EnvironmentBackend,
        *,
        frame_height: int,
        frame_width: int,
        frame_stack: int = 4,
        lease_ms: int = 80,
        max_episode_steps: int | None = None,
        log_root: Path | None = None,
    ) -> None:
        super().__init__()
        if frame_height <= 0 or frame_width <= 0:
            raise ValueError("frame dimensions must be positive")
        if frame_stack < 1:
            raise ValueError("frame_stack must be at least one")
        if not 1 <= lease_ms <= 1_000:
            raise ValueError("lease_ms must be between 1 and 1000")
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")

        self.backend = backend
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame_stack = frame_stack
        self.lease_ms = lease_ms
        self.max_episode_steps = max_episode_steps
        self.log_root = log_root
        self.metadata = {"render_modes": []}
        self.action_space = gym.spaces.Discrete(len(RAW_ACTIONS))  # type: ignore[assignment]
        self.observation_space = gym.spaces.Dict(  # type: ignore[assignment]
            {
                "pixels": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(frame_stack, frame_height, frame_width, 3),
                    dtype=np.uint8,
                ),
                "action_history": gym.spaces.MultiDiscrete(
                    np.full(max(0, frame_stack - 1), len(RAW_ACTIONS), dtype=np.int64)
                ),
            }
        )
        self._frames: list[PixelFrame] = []
        self._actions: list[int] = []
        self._episode_id: str | None = None
        self._transition_id = 0
        self._next_action_id = 1
        self._episode_steps = 0
        self._episode_done = True
        self._closed = False
        self._seen_episode_ids: set[str] = set()
        self._shade_runs = ShadeRunTracker()
        self._shade_run_update: ShadeRunUpdate | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[np.generic]], dict[str, Any]]:
        super().reset(seed=seed)
        self._ensure_open()
        if self._episode_id is not None:
            self.backend.release_all()
        # From this point the old episode is unusable, even if backend.reset()
        # fails before a replacement episode is ready.
        self._episode_id = None
        self._episode_done = True
        self._frames = []
        self._actions = []
        try:
            result = self.backend.reset(seed=seed, options=options)
            if not result.episode_id:
                raise ValueError("backend returned an empty episode_id")
            if result.episode_id in self._seen_episode_ids:
                raise ValueError("backend reused an episode_id")
            frame = self._validate_frame(
                result.frame,
                previous=None,
                expected_action_id=None,
                expected_episode_id=result.episode_id,
            )
            self._shade_run_update = self._observe_shade_lifecycle(frame)
        except Exception:
            self._best_effort_release()
            raise

        self._episode_id = result.episode_id
        self._seen_episode_ids.add(result.episode_id)
        self._frames = [frame] * self.frame_stack
        self._actions = [0] * max(0, self.frame_stack - 1)
        self._transition_id = 0
        self._next_action_id = 1
        self._episode_steps = 0
        self._episode_done = False
        return self._observation(), self._info(frame_ids=self._frame_ids())

    def step(
        self, action: int, *, lease_ms: int | None = None
    ) -> tuple[dict[str, NDArray[np.generic]], float, bool, bool, dict[str, Any]]:
        self._ensure_open()
        if self._episode_id is None or self._episode_done:
            raise RuntimeError("reset() is required before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action index: {action!r}")
        requested_lease_ms = self.lease_ms if lease_ms is None else lease_ms
        if not 1 <= requested_lease_ms <= 2_000:
            raise ValueError("per-action lease_ms must be between 1 and 2000")

        requested_index = int(action)
        requested_buttons = RAW_ACTIONS[requested_index]
        normalized_buttons = self._normalize_buttons(requested_buttons)
        action_id = self._next_action_id
        self._next_action_id += 1
        started_ns = time.monotonic_ns()
        before_ids = self._frame_ids()
        previous = self._frames[-1]
        try:
            result = self.backend.step(
                action_id=action_id,
                buttons=normalized_buttons,
                lease_ms=requested_lease_ms,
            )
            self._validate_applied_action(
                result.applied_action, action_id, normalized_buttons, requested_lease_ms
            )
            frame = self._validate_frame(
                result.frame,
                previous=previous,
                expected_action_id=action_id,
                expected_episode_id=self._episode_id,
            )
            self._shade_run_update = self._observe_shade_lifecycle(frame)
            timing_ms = self._validate_timing(result.timing_ms)
            if type(result.terminated) is not bool or type(result.truncated) is not bool:
                raise ValueError("terminated and truncated must be bool values")
            if result.terminated and result.truncated:
                raise ValueError("a backend step cannot be both terminated and truncated")
            if (
                result.termination_reason is not None
                and result.termination_reason not in _PUBLIC_TERMINATION_REASONS
            ):
                raise ValueError("backend returned a non-allowlisted termination reason")
            self._validate_termination(
                result.terminated, result.truncated, result.termination_reason
            )
        except Exception as error:  # noqa: BLE001 - backend failures become truncations
            self._best_effort_release()
            self._episode_done = True
            self._best_effort_log_rejected(
                action_id=action_id,
                requested_index=requested_index,
                reason=type(error).__name__,
                detail=str(error),
            )
            info = self._info(frame_ids=before_ids)
            info.update({"termination_reason": "operational_failure", "valid": False})
            return self._observation(), 0.0, False, True, info

        next_frames = self._frames[1:] + [frame]
        next_actions = self._actions
        if self.frame_stack > 1:
            applied_index = self._buttons_to_action_index(result.applied_action.applied_buttons)
            next_actions = (self._actions + [applied_index])[-(self.frame_stack - 1) :]
        next_episode_steps = self._episode_steps + 1
        limit_reached = (
            self.max_episode_steps is not None
            and next_episode_steps >= self.max_episode_steps
            and not result.terminated
        )
        terminated = result.terminated
        truncated = result.truncated or limit_reached
        reason = result.termination_reason
        if limit_reached and reason is None:
            reason = "time_limit"
        if (terminated or truncated) and reason is None:
            reason = "backend_termination" if terminated else "backend_truncation"
        timing_ms["env_step"] = (time.monotonic_ns() - started_ns) / 1_000_000
        after_ids = [item.frame_id for item in next_frames]
        try:
            self._log_transition(
                before_ids=before_ids,
                after_ids=after_ids,
                requested_index=requested_index,
                requested_buttons=requested_buttons,
                requested_lease_ms=requested_lease_ms,
                applied=result.applied_action,
                terminated=terminated,
                truncated=truncated,
                reason=reason,
                timing_ms=timing_ms,
                frame=frame,
            )
        except Exception as error:  # noqa: BLE001 - persistence is an operational boundary
            self._best_effort_release()
            self._episode_done = True
            self._best_effort_log_rejected(
                action_id=action_id,
                requested_index=requested_index,
                reason="transition_log_failure",
                detail=str(error),
            )
            info = self._info(frame_ids=before_ids)
            info.update({"termination_reason": "transition_log_failure", "valid": False})
            return self._observation(), 0.0, False, True, info

        self._frames = next_frames
        self._actions = next_actions
        self._episode_steps = next_episode_steps
        self._episode_done = terminated or truncated
        self._transition_id += 1
        if self._episode_done:
            self._best_effort_release()
        info = self._info(frame_ids=after_ids)
        info.update({"termination_reason": reason, "valid": True})
        return self._observation(), 0.0, terminated, truncated, info

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.backend.release_all()
        finally:
            try:
                self.backend.close()
            finally:
                self._closed = True
                self._episode_done = True

    def _validate_frame(
        self,
        frame: PixelFrame,
        *,
        previous: PixelFrame | None,
        expected_action_id: int | None,
        expected_episode_id: str,
    ) -> PixelFrame:
        pixels = frame.pixels
        expected_shape = (self.frame_height, self.frame_width, 3)
        if not isinstance(pixels, np.ndarray):
            raise FrameValidationError("pixels must be a numpy array")
        if pixels.dtype != np.uint8 or pixels.shape != expected_shape:
            raise FrameValidationError(
                f"pixels must have dtype uint8 and shape {expected_shape}, got "
                f"{pixels.dtype} {pixels.shape}"
            )
        if frame.frame_id < 0 or frame.capture_monotonic_ns < 0:
            raise FrameValidationError("frame identifiers and timestamps must be non-negative")
        if type(frame.player_dead) is not bool:
            raise FrameValidationError("player_dead must be bool")
        if type(frame.death_persisted) is not bool:
            raise FrameValidationError("death_persisted must be bool")
        if frame.player_dead and not frame.death_persisted:
            raise FrameValidationError("player_dead requires death_persisted")
        for name, value in (
            ("shade_pending", frame.shade_pending),
            ("shade_recovered", frame.shade_recovered),
            ("run_failed", frame.run_failed),
        ):
            if type(value) is not bool:
                raise FrameValidationError(f"{name} must be bool")
        if frame.run_failed and not frame.player_dead:
            raise FrameValidationError("run_failed requires player_dead")
        for name, combat_value in (
            ("enemy_kills_since_last_frame", frame.enemy_kills_since_last_frame),
            ("hp_lost_since_last_frame", frame.hp_lost_since_last_frame),
            ("geo_gained_since_last_frame", frame.geo_gained_since_last_frame),
            ("geo_spent_since_last_frame", frame.geo_spent_since_last_frame),
        ):
            if (
                isinstance(combat_value, bool)
                or not isinstance(combat_value, int)
                or combat_value < 0
            ):
                raise FrameValidationError(f"{name} must be a non-negative integer")
        if frame.episode_id != expected_episode_id:
            raise FrameValidationError("frame does not belong to the active episode")
        if frame.scene_transition:
            raise FrameValidationError("frame captured during scene transition")
        if frame.duplicate_of is not None:
            raise FrameValidationError("duplicate frame")
        if previous is not None:
            if frame.frame_id <= previous.frame_id:
                raise FrameValidationError("frame_id must increase")
            if frame.capture_monotonic_ns < previous.capture_monotonic_ns:
                raise FrameValidationError("capture timestamp moved backwards")
        if expected_action_id is not None and frame.last_applied_action_id != expected_action_id:
            raise FrameValidationError("frame is not correlated with the requested action")
        return PixelFrame(
            episode_id=frame.episode_id,
            frame_id=frame.frame_id,
            capture_monotonic_ns=frame.capture_monotonic_ns,
            pixels=np.ascontiguousarray(pixels).copy(),
            scene_transition=False,
            duplicate_of=None,
            last_applied_action_id=frame.last_applied_action_id,
            player_dead=frame.player_dead,
            death_persisted=frame.death_persisted,
            shade_pending=frame.shade_pending,
            shade_recovered=frame.shade_recovered,
            run_failed=frame.run_failed,
            enemy_kills_since_last_frame=frame.enemy_kills_since_last_frame,
            hp_lost_since_last_frame=frame.hp_lost_since_last_frame,
            geo_gained_since_last_frame=frame.geo_gained_since_last_frame,
            geo_spent_since_last_frame=frame.geo_spent_since_last_frame,
        )

    def _validate_applied_action(
        self,
        applied: AppliedAction,
        expected_action_id: int,
        requested_buttons: tuple[str, ...],
        requested_lease_ms: int,
    ) -> None:
        if applied.action_id != expected_action_id:
            raise FrameValidationError("applied action_id does not match the request")
        if applied.requested_buttons != requested_buttons:
            raise FrameValidationError("backend changed the requested action record")
        if applied.lease_ms != requested_lease_ms:
            raise FrameValidationError("backend changed the action lease")
        if len(set(applied.applied_buttons)) != len(applied.applied_buttons):
            raise FrameValidationError("backend reported duplicate applied buttons")
        if not set(applied.applied_buttons).issubset(_BUTTONS):
            raise FrameValidationError("backend reported an unknown applied button")
        if applied.applied_buttons != requested_buttons:
            raise FrameValidationError("backend changed the applied action")

    @staticmethod
    def _validate_timing(timing: Mapping[str, float] | None) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, value in (timing or {}).items():
            if not isinstance(name, str) or not name:
                raise ValueError("timing names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("timings must be numeric")
            converted = float(value)
            if not math.isfinite(converted) or converted < 0:
                raise ValueError("timings must be finite and non-negative")
            result[name] = converted
        return result

    def _observation(self) -> dict[str, NDArray[np.generic]]:
        return {
            "pixels": np.stack([frame.pixels for frame in self._frames]),
            "action_history": np.asarray(self._actions, dtype=np.int64),
        }

    def _frame_ids(self) -> list[int]:
        return [frame.frame_id for frame in self._frames]

    def _info(self, *, frame_ids: Sequence[int]) -> dict[str, Any]:
        del frame_ids
        if not self._frames:
            return {}
        frame = self._frames[-1]
        if not (
            frame.player_dead
            or frame.death_persisted
            or frame.shade_pending
            or frame.shade_recovered
            or frame.run_failed
            or frame.enemy_kills_since_last_frame
            or frame.hp_lost_since_last_frame
            or frame.geo_gained_since_last_frame
            or frame.geo_spent_since_last_frame
        ):
            return {}
        lifecycle = self._shade_run_update
        return {
            "enemy_kills_since_last_frame": frame.enemy_kills_since_last_frame,
            "hp_lost_since_last_frame": frame.hp_lost_since_last_frame,
            "geo_gained_since_last_frame": frame.geo_gained_since_last_frame,
            "geo_spent_since_last_frame": frame.geo_spent_since_last_frame,
            "run_failed": lifecycle.run_failed if lifecycle is not None else frame.run_failed,
            "supervisor": {
                "player_dead": frame.player_dead,
                "death_persisted": frame.death_persisted,
                "shade_pending": frame.shade_pending,
                "shade_recovered": frame.shade_recovered,
                "run_failed": frame.run_failed,
                "enemy_kills_since_last_frame": frame.enemy_kills_since_last_frame,
                "hp_lost_since_last_frame": frame.hp_lost_since_last_frame,
                "geo_gained_since_last_frame": frame.geo_gained_since_last_frame,
                "geo_spent_since_last_frame": frame.geo_spent_since_last_frame,
                "gameplay_run_index": (
                    lifecycle.logical_run_index if lifecycle is not None else 1
                ),
                "recovery_required": (
                    lifecycle.recovery_required if lifecycle is not None else False
                ),
                "failed_gameplay_runs": (
                    lifecycle.failed_runs if lifecycle is not None else 0
                ),
            }
        }

    def _observe_shade_lifecycle(self, frame: PixelFrame) -> ShadeRunUpdate:
        return self._shade_runs.observe(
            player_dead=frame.player_dead,
            death_persisted=frame.death_persisted,
            shade_pending=frame.shade_pending,
            shade_recovered=frame.shade_recovered,
            run_failed=frame.run_failed,
        )

    def _log_transition(
        self,
        *,
        before_ids: list[int],
        after_ids: list[int],
        requested_index: int,
        requested_buttons: tuple[str, ...],
        requested_lease_ms: int,
        applied: AppliedAction,
        terminated: bool,
        truncated: bool,
        reason: str | None,
        timing_ms: Mapping[str, float],
        frame: PixelFrame,
    ) -> None:
        writer = self._transition_writer()
        if writer is None:
            return
        writer.append(
            {
                "schema_version": 1,
                "run_id": self.backend.run_id,
                "instance_id": self.backend.instance_id,
                "episode_id": self._episode_id,
                "transition_id": self._transition_id,
                "obs_before": {"frame_ids": before_ids},
                "requested_action": {
                    "index": requested_index,
                    "name": ACTION_NAMES[requested_index],
                    "buttons": list(requested_buttons),
                    "normalized_buttons": list(applied.requested_buttons),
                    "lease_ms": requested_lease_ms,
                },
                "applied_action": {
                    "action_id": applied.action_id,
                    "buttons": list(applied.applied_buttons),
                    "lease_ms": applied.lease_ms,
                },
                "obs_after": {"frame_ids": after_ids},
                "reward": {"total": 0.0, "components": {}},
                "terminated": terminated,
                "truncated": truncated,
                "termination_reason": reason,
                "supervisor": {
                    "player_dead": frame.player_dead,
                    "death_persisted": frame.death_persisted,
                    "shade_pending": frame.shade_pending,
                    "shade_recovered": frame.shade_recovered,
                    "run_failed": frame.run_failed,
                    "enemy_kills_since_last_frame": frame.enemy_kills_since_last_frame,
                    "hp_lost_since_last_frame": frame.hp_lost_since_last_frame,
                    "geo_gained_since_last_frame": frame.geo_gained_since_last_frame,
                    "geo_spent_since_last_frame": frame.geo_spent_since_last_frame,
                },
                "timing_ms": dict(timing_ms),
                "valid": True,
            }
        )

    def _log_rejected(
        self, *, action_id: int, requested_index: int, reason: str, detail: str
    ) -> None:
        if self.log_root is None:
            return
        AtomicJsonlWriter(
            self.log_root / "instances" / self.backend.instance_id / "operations.jsonl"
        ).append(
            {
                "schema_version": 1,
                "run_id": self.backend.run_id,
                "instance_id": self.backend.instance_id,
                "episode_id": self._episode_id,
                "action_id": action_id,
                "requested_action": requested_index,
                "event": "transition_rejected",
                "reason": reason,
                "detail": detail,
                "monotonic_ns": time.monotonic_ns(),
            }
        )

    def _best_effort_log_rejected(
        self, *, action_id: int, requested_index: int, reason: str, detail: str
    ) -> None:
        try:
            self._log_rejected(
                action_id=action_id,
                requested_index=requested_index,
                reason=reason,
                detail=detail,
            )
        except Exception:  # noqa: BLE001, S110 - the primary storage failure is preserved in info
            pass

    def _transition_writer(self) -> AtomicJsonlWriter | None:
        if self.log_root is None or self._episode_id is None:
            return None
        return AtomicJsonlWriter(
            self.log_root
            / "instances"
            / self.backend.instance_id
            / "episodes"
            / self._episode_id
            / "transitions.jsonl"
        )

    @staticmethod
    def _normalize_buttons(buttons: tuple[str, ...]) -> tuple[str, ...]:
        selected = set(buttons)
        if {"LEFT", "RIGHT"}.issubset(selected):
            selected.difference_update({"LEFT", "RIGHT"})
        if {"UP", "DOWN"}.issubset(selected):
            selected.difference_update({"UP", "DOWN"})
        return tuple(button for button in BUTTON_ORDER if button in selected)

    @staticmethod
    def _buttons_to_action_index(buttons: tuple[str, ...]) -> int:
        return sum(1 << BUTTON_ORDER.index(button) for button in buttons)

    @staticmethod
    def _validate_termination(terminated: bool, truncated: bool, reason: str | None) -> None:
        if not terminated and not truncated and reason is not None:
            raise ValueError("termination_reason requires an ended episode")
        terminal_reasons = {"player_death", "task_success", "backend_termination"}
        if terminated and reason is not None and reason not in terminal_reasons:
            raise ValueError("termination reason is not valid for terminated=True")
        truncation_reasons = {
            "time_limit",
            "connection_lost",
            "capture_failed",
            "reset_failed",
            "backend_truncation",
        }
        if truncated and reason is not None and reason not in truncation_reasons:
            raise ValueError("termination reason is not valid for truncated=True")

    def _best_effort_release(self) -> None:
        try:
            self.backend.release_all()
        except Exception:  # noqa: BLE001, S110 - emergency release is best effort
            pass

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("environment is closed")
