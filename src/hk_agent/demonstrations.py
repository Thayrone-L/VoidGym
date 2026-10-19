"""Opt-in, context-bound and integrity-checked human demonstrations."""
from __future__ import annotations

import hashlib
import io
import json
import time
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

    def __post_init__(self) -> None:
        if self.failed_attempts < 1 or self.stalled_hours <= 0 or self.stuck_minutes <= 0:
            raise ValueError("help trigger thresholds must be positive")


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
        self.frames.append(np.frombuffer(png_bytes, dtype=np.uint8).copy())
        self.actions.append(action)
        self.masks.append(action_mask.astype(np.bool_, copy=True))
        self.knowledge.append(player_knowledge.astype(np.float32, copy=True))

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
    knowledge = values["player_knowledge"].astype(np.float32, copy=False)
    expected_knowledge_shape = tuple(spaces["player_knowledge"].shape)
    if knowledge.shape != (len(actions), *expected_knowledge_shape):
        raise ValueError("demonstration knowledge vector differs from the learner")
    return {
        "pixels": pixels,
        "action_history": history.reshape((len(actions), *spaces["action_history"].shape)),
        "player_knowledge": knowledge,
        "actions": actions,
        "action_masks": masks,
    }


def imitation_schedule(*, initial_weight: float, decay_rounds: int, round_number: int) -> float:
    if initial_weight < 0 or decay_rounds < 1 or round_number < 0:
        raise ValueError("invalid imitation schedule")
    return initial_weight * max(0.0, 1.0 - round_number / decay_rounds)


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

