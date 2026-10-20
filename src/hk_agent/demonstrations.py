"""Opt-in, context-bound and integrity-checked human demonstrations."""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor

DEMONSTRATION_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class HelpTriggerConfig:
    failed_attempts: int = 100
    stalled_hours: float = 6.0
    stuck_minutes: float = 30.0
    repeated_stuck_count: int = 10
    requests_per_hour: int = 3

    def __post_init__(self) -> None:
        if (self.failed_attempts < 1 or self.stalled_hours <= 0 or self.stuck_minutes <= 0
                or self.repeated_stuck_count < 1 or self.requests_per_hour < 1):
            raise ValueError("help trigger thresholds must be positive")


class HelpRequestLedger:
    """Per-actor durable help requests; pending requests never block collection."""

    def __init__(self, root: Path, *, slot: int, config: HelpTriggerConfig | None = None) -> None:
        if slot < 0:
            raise ValueError("slot must be non-negative")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"slot-{slot}.json"
        self.slot = slot
        self.config = config or HelpTriggerConfig()
        self._state = self._load()

    def _load(self) -> dict[str, Any]:
        value = self._load_file(self.path)
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            value = {}
        stuck = value.get("stuck_by_place", {})
        requests = value.get("requests", [])
        return {
            "schema_version": 1,
            "stuck_by_place": dict(stuck) if isinstance(stuck, dict) else {},
            "requests": [dict(row) for row in requests if isinstance(row, dict)][-128:]
            if isinstance(requests, list) else [],
            "no_progress_since_unix": value.get("no_progress_since_unix", time.time()),
            "best_progress_by_scene": dict(value.get("best_progress_by_scene", {}))
            if isinstance(value.get("best_progress_by_scene", {}), dict) else {},
            "last_request_count_by_place": dict(value.get("last_request_count_by_place", {}))
            if isinstance(value.get("last_request_count_by_place", {}), dict) else {},
            "stuck_details": dict(value.get("stuck_details", {}))
            if isinstance(value.get("stuck_details", {}), dict) else {},
            "points_of_interest": list(value.get("points_of_interest", []))
            if isinstance(value.get("points_of_interest", []), list) else [],
            "neighbor_rewards": dict(value.get("neighbor_rewards", {}))
            if isinstance(value.get("neighbor_rewards", {}), dict) else {},
        }

    @staticmethod
    def _load_file(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f".{self.path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        temp.write_text(json.dumps(self._state, sort_keys=True, allow_nan=False), encoding="utf-8", newline="\n")
        os.replace(temp, self.path)

    def observe_progress(
        self, *, scene: str, progress: float, now: float | None = None, _lock_held: bool = False,
    ) -> None:
        if not scene or not np.isfinite(progress) or progress < 0:
            return
        now = time.time() if now is None else now
        previous = self._state.get("best_progress_by_scene", {})
        if not isinstance(previous, dict):
            previous = {}
        best = float(previous.get(scene, 0.0))
        new_record = progress > best
        if new_record:
            previous[scene] = float(progress)
            self._state["best_progress_by_scene"] = previous
            self._state["no_progress_since_unix"] = now
        changed = False
        resolved_requests: list[dict[str, Any]] = []
        requests = self._state.get("requests", [])
        if isinstance(requests, list):
            for row in requests:
                if not isinstance(row, dict) or row.get("status") not in {"pending", "demonstration_requested"}:
                    continue
                request_scene = row.get("scene")
                baseline = row.get("progress_at_request", 0.0)
                if request_scene != scene or (isinstance(baseline, (int, float)) and progress >= float(baseline) + 5.0):
                    row.update({"status": "resolved_solo", "resolved_unix": now,
                                "resolution_reason": "passou do local ou bateu novo recorde na sala"})
                    resolved_requests.append(dict(row))
                    changed = True
        if new_record or changed:
            def persist() -> None:
                self._refresh_requests()
                current_requests = self._state.get("requests", [])
                if isinstance(current_requests, list):
                    by_id = {row.get("request_id"): row for row in current_requests if isinstance(row, dict)}
                    for resolved in resolved_requests:
                        row = by_id.get(resolved.get("request_id"))
                        if row is not None and row.get("status") in {"pending", "demonstration_requested"}:
                            row.update(resolved)
                self._save()
            if _lock_held:
                persist()
            else:
                from hk_agent.run_journey import file_lock

                with file_lock(self.root / ".requests.lock"):
                    persist()

    def _refresh_requests(self) -> None:
        try:
            payload = json.loads(self.path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        rows = payload.get("requests", []) if isinstance(payload, dict) else []
        if isinstance(rows, list):
            self._state["requests"] = [dict(row) for row in rows if isinstance(row, dict)][-128:]

    def record_stuck(
        self, *, scene: str, cell: tuple[int, int], progress: float,
        thumbnail_path: str | None = None, now: float | None = None,
    ) -> dict[str, Any] | None:
        from hk_agent.run_journey import file_lock

        with file_lock(self.root / ".requests.lock"):
            return self._record_stuck_locked(
                scene=scene, cell=cell, progress=progress,
                thumbnail_path=thumbnail_path, now=now,
            )

    def _record_stuck_locked(
        self, *, scene: str, cell: tuple[int, int], progress: float,
        thumbnail_path: str | None, now: float | None,
    ) -> dict[str, Any] | None:
        if not scene or len(scene) > 256 or any(not 0 <= value < 12 for value in cell):
            return None
        self._refresh_requests()
        now = time.time() if now is None else now
        place = f"{scene}:c{cell[0]}r{cell[1]}"
        counts = self._state.setdefault("stuck_by_place", {})
        if not isinstance(counts, dict):
            counts = {}
            self._state["stuck_by_place"] = counts
        count = int(counts.get(place, 0)) + 1
        counts[place] = count
        self.observe_progress(scene=scene, progress=progress, now=now, _lock_held=True)
        since = self._state.get("no_progress_since_unix", now)
        elapsed_minutes = max(0.0, (now - float(since)) / 60.0) if isinstance(since, (int, float)) else 0.0
        details = self._state.setdefault("stuck_details", {})
        assert isinstance(details, dict)
        details[place] = {
            "scene": scene, "cell": [cell[0], cell[1]], "count": count,
            "last_stuck_unix": now, "stuck_minutes": elapsed_minutes,
            "point_of_interest": count >= 3,
        }
        if count >= 3:
            points = self._state.setdefault("points_of_interest", [])
            assert isinstance(points, list)
            if not any(isinstance(row, dict) and row.get("place") == place for row in points):
                points.append({"place": place, "scene": scene, "cell": [cell[0], cell[1]],
                               "label": "travei aqui", "created_unix": now})
        reason = None
        if count >= self.config.repeated_stuck_count:
            reason = "repeated_stuck"
        elif elapsed_minutes >= self.config.stuck_minutes:
            reason = "no_room_progress"
        requests = self._state.get("requests", [])
        assert isinstance(requests, list)
        last_counts = self._state.setdefault("last_request_count_by_place", {})
        last_count = int(last_counts.get(place, 0)) if isinstance(last_counts, dict) else 0
        if reason == "repeated_stuck" and count < last_count + self.config.repeated_stuck_count:
            reason = None
        hour_ago = now - 3600
        all_requests: list[dict[str, Any]] = []
        for ledger_path in self.root.glob("slot-*.json"):
            rows = self._load_file(ledger_path).get("requests", [])
            if isinstance(rows, list):
                all_requests.extend(row for row in rows if isinstance(row, dict))
        recent = [row for row in all_requests
                  if isinstance(row.get("created_unix"), (int, float))
                  and float(row["created_unix"]) >= hour_ago]
        from hk_agent.run_journey import atomic_json

        for ledger_path in sorted(self.root.glob("slot-*.json")):
            if ledger_path == self.path:
                other_state = self._state
            else:
                try:
                    other_state = self._load_file(ledger_path)
                except (OSError, json.JSONDecodeError):
                    continue
            other_requests = other_state.get("requests", []) if isinstance(other_state, dict) else []
            pending = next((row for row in other_requests if isinstance(row, dict)
                            and row.get("place") == place
                            and row.get("status") in {"pending", "demonstration_requested"}), None) if isinstance(other_requests, list) else None
            if pending is None:
                continue
            pending["stuck_count"] = int(pending.get("stuck_count", 0)) + 1
            pending["last_stuck_unix"] = now
            pending["progress_at_request"] = max(float(pending.get("progress_at_request", 0.0)), progress)
            if ledger_path == self.path:
                self._save()
            else:
                atomic_json(ledger_path, other_state)
                self._save()
            return dict(pending)
        if reason is None or len(recent) >= self.config.requests_per_hour:
            self._save()
            return None
        request = {
            "request_id": f"help-{self.slot}-{time.time_ns()}",
            "slot": self.slot, "scene": scene, "place": place,
            "context": f"traversal:{scene}:cell-{cell[0]}-{cell[1]}",
            "cell": [cell[0], cell[1]], "stuck_count": count,
            "reason": reason, "status": "pending", "created_unix": now,
            "last_stuck_unix": now, "progress_at_request": float(progress),
            "thumbnail_path": thumbnail_path,
            "message": (
                f"Instância {self.slot + 1} pede ajuda — "
                f"{'Passagem do Rei' if 'King_Passage' in scene else scene}, "
                f"trecho {cell[0]},{cell[1]} ({count} travadas)"
            ),
        }
        requests.append(request)
        self._state["requests"] = requests[-128:]
        if isinstance(last_counts, dict):
            last_counts[place] = count
        self._save()
        return dict(request)

    def stuck_count(self, *, scene: str, cell: tuple[int, int]) -> int:
        place = f"{scene}:c{cell[0]}r{cell[1]}"
        counts = self._state.get("stuck_by_place", {})
        value = counts.get(place, 0) if isinstance(counts, dict) else 0
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0

    def stuck_locations(self) -> list[dict[str, Any]]:
        details = self._state.get("stuck_details", {})
        if not isinstance(details, dict):
            return []
        now = time.time()
        rows = []
        for place, raw in details.items():
            if not isinstance(place, str) or not isinstance(raw, dict):
                continue
            row = dict(raw)
            stamp = row.get("last_stuck_unix")
            row["place"] = place
            row["stuck_minutes"] = max(0.0, (now - float(stamp)) / 60.0) if isinstance(stamp, (int, float)) else 0.0
            rows.append(row)
        return sorted(rows, key=lambda row: (int(row.get("count", 0)), float(row.get("last_stuck_unix", 0))), reverse=True)

    def reward_new_neighbor(self, *, scene: str, cell: tuple[int, int]) -> float:
        """Pay a small, lifetime-decaying reward for a new neighbor after repeated stuck episodes."""
        key = f"{scene}:c{cell[0]}r{cell[1]}"
        counts = self._state.setdefault("neighbor_rewards", {})
        if not isinstance(counts, dict):
            counts = {}
            self._state["neighbor_rewards"] = counts
        raw_visits = counts.get(key, 0)
        visits: int = int(raw_visits) if isinstance(raw_visits, int) and not isinstance(raw_visits, bool) else 0
        award: float = float(0.1 / (1.0 + visits) ** 0.5)
        counts[key] = visits + 1
        self._save()
        return award

    def resolve(self, request_id: str, *, status: str) -> dict[str, Any]:
        if status not in {"dismissed", "demonstration_requested", "discarded", "completed", "resolved_with_help"}:
            raise ValueError("invalid help request status")
        from hk_agent.run_journey import file_lock

        with file_lock(self.root / ".requests.lock"):
            self._state = self._load()
            for row in self._state["requests"]:
                if row.get("request_id") == request_id:
                    if row.get("status") not in {"pending", "demonstration_requested"}:
                        raise ValueError("help request is no longer pending")
                    row["status"] = status
                    row["updated_unix"] = time.time()
                    if status in {"completed", "resolved_with_help", "discarded", "dismissed"}:
                        row["resolved_unix"] = row["updated_unix"]
                    self._save()
                    return dict(row)
        raise FileNotFoundError("help request was not found")

    def snooze(self, request_id: str, *, minutes: float = 30.0, now: float | None = None) -> dict[str, Any]:
        if not np.isfinite(minutes) or not 1 <= minutes <= 1440:
            raise ValueError("snooze minutes must be between 1 and 1440")
        current = time.time() if now is None else now
        from hk_agent.run_journey import file_lock

        with file_lock(self.root / ".requests.lock"):
            self._state = self._load()
            for row in self._state["requests"]:
                if row.get("request_id") == request_id:
                    if row.get("status") not in {"pending", "demonstration_requested"}:
                        raise ValueError("help request is no longer open")
                    row["snoozed_until_unix"] = current + minutes * 60.0
                    row["snooze_count"] = int(row.get("snooze_count", 0)) + 1
                    self._save()
                    return dict(row)
        raise FileNotFoundError("help request was not found")

    def requests(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._state["requests"]]


def map_human_action(
    buttons: Sequence[str], duration_ms: int, actions: Sequence[Any], *,
    maximum_button_distance: int = 0,
) -> tuple[int | None, bool]:
    """Map one human input window to the nearest policy token and report exact coverage."""
    pressed = frozenset(str(button).upper() for button in buttons)
    if duration_ms < 1:
        raise ValueError("duration_ms must be positive")
    candidates: list[tuple[int, int, int]] = []
    for index, action in enumerate(actions):
        action_buttons = frozenset(str(button).upper() for button in action.buttons)
        distance = len(pressed.symmetric_difference(action_buttons))
        target_duration = int(action.resolve_lease_ms(80))
        duration_distance = abs(duration_ms - target_duration)
        candidates.append((distance, duration_distance, index))
    if not candidates:
        return None, False
    distance, duration_distance, index = min(candidates)
    return (index if distance <= maximum_button_distance else None,
            distance == 0 and duration_distance <= 40)


def augment_demonstration_batch(
    batch: dict[str, NDArray[Any]], actions: Sequence[Any], *, brightness_delta: float = 0.04,
) -> dict[str, NDArray[Any]]:
    """Add a physics-preserving horizontal mirror and slight brightness variants."""
    if not 0 <= brightness_delta <= 0.2:
        raise ValueError("brightness_delta must be in [0, 0.2]")
    names = {str(action.name): index for index, action in enumerate(actions)}
    mirrored: list[int] = []
    for action in actions:
        name = str(action.name).replace("LEFT", "__HORIZONTAL__").replace("RIGHT", "LEFT").replace("__HORIZONTAL__", "RIGHT")
        index = names.get(name)
        if index is None:
            mirrored = []
            break
        mirrored.append(index)
    if not mirrored:
        return batch
    result = {key: np.concatenate((value, value), axis=0) for key, value in batch.items()}
    result["sequence_count"] = np.asarray(2, dtype=np.int64)
    count = len(batch["actions"])
    result["pixels"][count:] = np.flip(batch["pixels"], axis=-2).copy()
    mapping = np.asarray(mirrored, dtype=np.int64)
    result["actions"][count:] = mapping[batch["actions"]]
    result["action_masks"][count:] = batch["action_masks"][:, mapping]
    if len(batch["action_history"]):
        result["action_history"][count:] = mapping[batch["action_history"]]
    if brightness_delta:
        # Deterministic per-sample exposure offsets avoid turning a single room
        # into an absolute coordinate lookup while keeping local geometry fixed.
        factors = np.linspace(1.0 - brightness_delta, 1.0 + brightness_delta, count, dtype=np.float32)
        pixels = result["pixels"][count:].astype(np.float32)
        result["pixels"][count:] = np.clip(pixels * factors.reshape((-1,) + (1,) * (pixels.ndim - 1)), 0, 255).astype(np.uint8)
    if result["pixels"].ndim == 5 and result["pixels"].shape[2] > 2 and result["pixels"].shape[3] > 2:
        pixels = result["pixels"][count:]
        padded = np.pad(pixels, ((0, 0), (0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
        height, width = pixels.shape[2:4]
        for index in range(count):
            dx, dy = (index % 3) - 1, ((index // 3) % 3) - 1
            pixels[index] = padded[index, :, 1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
    return result


def help_request(
    *, challenge: str, attempts_without_success: int, no_progress_hours: float,
    stuck_minutes: float, config: HelpTriggerConfig | None = None,
) -> str | None:
    config = config or HelpTriggerConfig()
    if attempts_without_success >= config.failed_attempts:
        return f"A IA pede ajuda: {challenge} após {attempts_without_success} tentativas. Pode mostrar como você faz?"
    if no_progress_hours >= config.stalled_hours:
        return f"A IA pede ajuda: sem progresso em {challenge} há {no_progress_hours:.1f} horas. Pode mostrar como você faz?"
    if stuck_minutes >= config.stuck_minutes:
        return f"A IA pede ajuda: presa em {challenge} há {stuck_minutes:.0f} minutos. Pode mostrar como você faz?"
    return None


class DemonstrationRecorder:
    """Writes human actions, masked PNG observations and knowledge outside the repository."""

    def __init__(self, root: Path, *, game_version: str, context: str) -> None:
        if not game_version or not context:
            raise ValueError("demonstration needs a game version and challenge context")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.game_version = game_version
        self.context = context
        self.frames: list[NDArray[np.uint8]] = []
        self.actions: list[int] = []
        self.masks: list[NDArray[np.bool_]] = []
        self.knowledge: list[NDArray[np.float32]] = []
        self.authorization: str | None = None
        self.frames_seen = 0
        self.representable_frames = 0
        self.unrepresentable_inputs: list[dict[str, Any]] = []

    def authorize(self, *, help_request_id: str | None = None, operator_forced: bool = False) -> None:
        if operator_forced:
            self.authorization = "operator_forced"
        elif isinstance(help_request_id, str) and help_request_id.strip():
            self.authorization = f"help_request:{help_request_id.strip()}"
        else:
            raise ValueError("demonstration recording requires a help request or operator force")

    def append(
        self, *, png_bytes: bytes, action: int,
        action_mask: NDArray[np.bool_], player_knowledge: NDArray[np.float32],
    ) -> None:
        if self.authorization is None:
            raise ValueError("demonstration recording has not been requested")
        if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("demonstration frame must be PNG")
        if action_mask.ndim != 1 or not action_mask.any() or not 0 <= action < len(action_mask):
            raise ValueError("demonstration action mask is malformed")
        if not action_mask[action]:
            raise ValueError("human action was illegal under the recorded action mask")
        if player_knowledge.ndim != 1 or not np.isfinite(player_knowledge).all():
            raise ValueError("player knowledge vector is malformed")
        self.frames_seen += 1
        self.representable_frames += 1
        self.frames.append(np.frombuffer(png_bytes, dtype=np.uint8).copy())
        self.actions.append(action)
        self.masks.append(action_mask.astype(np.bool_, copy=True))
        self.knowledge.append(player_knowledge.astype(np.float32, copy=True))

    def append_human(
        self, *, png_bytes: bytes, buttons: Sequence[str], duration_ms: int,
        action_templates: Sequence[Any], action_mask: NDArray[np.bool_],
        player_knowledge: NDArray[np.float32],
    ) -> tuple[int | None, bool]:
        action, representable = map_human_action(buttons, duration_ms, action_templates)
        if action is None or action >= len(action_mask) or not action_mask[action]:
            self.frames_seen += 1
            if not representable:
                self.unrepresentable_inputs.append({
                    "buttons": sorted({str(button).upper() for button in buttons}),
                    "duration_ms": duration_ms,
                    "reason": "no_exact_action" if action is None else "action_masked",
                })
            return None, False
        self.append(
            png_bytes=png_bytes, action=action, action_mask=action_mask,
            player_knowledge=player_knowledge,
        )
        if not representable:
            self.representable_frames -= 1
            self.unrepresentable_inputs.append({
                "buttons": sorted({str(button).upper() for button in buttons}),
                "duration_ms": duration_ms, "reason": "duration_mismatch",
            })
        return action, representable

    def finish(self) -> Path:
        if self.authorization is None:
            raise ValueError("demonstration recording has not been requested")
        if not self.frames:
            raise ValueError("empty human demonstration")
        target = self.root / f"demo-{time.time_ns()}.npz"
        offsets = np.cumsum([0, *(len(frame) for frame in self.frames)], dtype=np.int64)
        np.savez_compressed(
            target,
            frame_bytes=np.concatenate(self.frames),
            frame_offsets=offsets,
            actions=np.asarray(self.actions, dtype=np.int64),
            action_masks=np.stack(self.masks),
            player_knowledge=np.stack(self.knowledge),
        )
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest = {
            "schema_version": DEMONSTRATION_SCHEMA,
            "game_version": self.game_version,
            "context": self.context,
            "authorization": self.authorization,
            "samples": len(self.actions),
            "frames_seen": self.frames_seen,
            "representable_frames": self.representable_frames,
            "representable_percent": 100.0 * self.representable_frames / max(1, self.frames_seen),
            "unrepresentable_inputs": self.unrepresentable_inputs[:128],
            "artifact": target.name,
            "sha256": digest,
        }
        target.with_suffix(".json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        return target


def load_demonstration(
    artifact: Path, *, expected_game_version: str, expected_context: str,
) -> dict[str, NDArray[Any]]:
    manifest = json.loads(artifact.with_suffix(".json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if manifest.get("schema_version") != DEMONSTRATION_SCHEMA or manifest.get("sha256") != digest:
        raise ValueError("demonstration schema or integrity check failed")
    if not isinstance(manifest.get("authorization"), str) or not manifest["authorization"].strip():
        raise ValueError("demonstration has no recording authorization")
    if manifest.get("game_version") != expected_game_version:
        raise ValueError("demonstration game version is incompatible")
    if manifest.get("context") != expected_context:
        raise ValueError("demonstration belongs to a different challenge")
    with np.load(artifact, allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    actions = values.get("actions")
    masks = values.get("action_masks")
    if actions is None or masks is None or masks.ndim != 2 or len(actions) != len(masks):
        raise ValueError("demonstration sample schema is invalid")
    if not masks.any(axis=1).all() or not masks[np.arange(len(actions)), actions].all():
        raise ValueError("demonstration contains an unavailable action")
    return values


def demonstration_batch(
    artifact: Path, *, expected_game_version: str, expected_context: str,
    observation_space: Any, action_count: int,
) -> dict[str, NDArray[Any]]:
    """Decode authenticated PNG examples into the learner's exact Dict observation contract."""
    values = load_demonstration(
        artifact, expected_game_version=expected_game_version,
        expected_context=expected_context,
    )
    spaces = getattr(observation_space, "spaces", None)
    if not isinstance(spaces, dict) or set(spaces) != {"pixels", "action_history", "player_knowledge"}:
        raise ValueError("demonstration training requires pixels, action_history and player_knowledge observations")
    actions = values["actions"].astype(np.int64, copy=False)
    masks = values["action_masks"].astype(np.bool_, copy=False)
    if masks.shape[1] != action_count or np.any(actions >= action_count):
        raise ValueError("demonstration action contract differs from the learner")
    pixel_shape = tuple(spaces["pixels"].shape)
    if len(pixel_shape) != 4 or pixel_shape[-1] != 3:
        raise ValueError("learner pixels must use frame, height, width, RGB dimensions")
    frame_count, height, width, _channels = pixel_shape
    offsets = values["frame_offsets"].astype(np.int64, copy=False)
    raw = values["frame_bytes"].astype(np.uint8, copy=False)
    if offsets.shape != (len(actions) + 1,) or offsets[0] != 0 or offsets[-1] != len(raw):
        raise ValueError("demonstration PNG offsets are invalid")
    frames: list[NDArray[np.uint8]] = []
    for index in range(len(actions)):
        start, end = int(offsets[index]), int(offsets[index + 1])
        with Image.open(io.BytesIO(raw[start:end].tobytes())) as image:
            rgb = image.convert("RGB")
            if rgb.size != (width, height):
                rgb = rgb.resize((width, height), Image.Resampling.BILINEAR)
            frames.append(np.asarray(rgb, dtype=np.uint8).copy())
    pixels = np.empty((len(actions), *pixel_shape), dtype=np.uint8)
    history_size = int(np.prod(spaces["action_history"].shape))
    history = np.zeros((len(actions), history_size), dtype=np.int64)
    for index, frame in enumerate(frames):
        earliest = max(0, index - frame_count + 1)
        stack = frames[earliest:index + 1]
        if len(stack) < frame_count:
            stack = [stack[0]] * (frame_count - len(stack)) + stack
        pixels[index] = np.stack(stack)
        prior = actions[max(0, index - history_size):index]
        if len(prior):
            history[index, -len(prior):] = prior
    knowledge = values["player_knowledge"].astype(np.float32, copy=True)
    expected_knowledge_shape = tuple(spaces["player_knowledge"].shape)
    if knowledge.shape != (len(actions), *expected_knowledge_shape):
        raise ValueError("demonstration knowledge vector differs from the learner")
    # The task observation contains room-map cells. Keep skills/objectives, but
    # remove visited-cell, POI, and last-cell coordinates so imitation follows
    # visible geometry instead of memorizing where this demonstration occurred.
    from hk_agent.player_knowledge import GRID_HEIGHT, GRID_WIDTH, ITEM_FLAGS, PLACE_DESCRIPTORS

    grid = GRID_WIDTH * GRID_HEIGHT
    item_size = sum(1 + len(descriptor) for _flag, descriptor in ITEM_FLAGS)
    visited = item_size
    exits = visited + grid
    points = exits + 8 * 6
    last_cell = points + 8 * 5
    knowledge[:, visited:visited + grid] = 0
    knowledge[:, points:last_cell] = 0
    knowledge[:, last_cell:last_cell + grid] = 0
    places_start = last_cell + grid
    place_maps_start = places_start + 4 * len(PLACE_DESCRIPTORS)
    knowledge[:, place_maps_start:place_maps_start + grid * len(PLACE_DESCRIPTORS)] = 0
    return {
        "pixels": pixels,
        "action_history": history.reshape((len(actions), *spaces["action_history"].shape)),
        "player_knowledge": knowledge,
        "actions": actions,
        "action_masks": masks,
    }


def imitation_schedule(
    *, initial_weight: float, decay_rounds: int, round_number: int,
    samples_in_bank: int = 1, minimum_bank_size: int = 64,
) -> float:
    if initial_weight < 0 or decay_rounds < 1 or round_number < 0 or samples_in_bank < 0 or minimum_bank_size < 1:
        raise ValueError("invalid imitation schedule")
    scheduled = initial_weight * max(0.0, 1.0 - round_number / decay_rounds)
    if samples_in_bank < minimum_bank_size and initial_weight > 0:
        return max(scheduled, initial_weight * 0.1)
    return scheduled


def masked_behavioral_cloning_loss(
    logits: Tensor, actions: Tensor, action_masks: Tensor,
) -> Tensor:
    if logits.ndim != 2 or action_masks.shape != logits.shape or actions.shape != logits.shape[:1]:
        raise ValueError("behavioral cloning tensor shapes differ")
    if not action_masks.any(dim=1).all() or not action_masks.gather(1, actions.long().view(-1, 1)).all():
        raise ValueError("demonstration action is unavailable")
    masked_logits = logits.masked_fill(~action_masks.bool(), torch.finfo(logits.dtype).min)
    return torch.nn.functional.cross_entropy(masked_logits, actions.long())


def combine_ppo_and_imitation(ppo_loss: Tensor, imitation_loss: Tensor, *, weight: float) -> Tensor:
    if weight < 0:
        raise ValueError("imitation weight must be non-negative")
    return ppo_loss + weight * imitation_loss

