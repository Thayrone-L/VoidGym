"""Integrity-checked operator demonstrations for recurrent training."""
from __future__ import annotations

import hashlib
import io
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor

DEMONSTRATION_SCHEMA = 1


def context_artifact_directory(shared_root: Path, context: str) -> Path:
    digest = hashlib.sha256(context.encode("utf-8")).hexdigest()[:16]
    return shared_root / "demonstrations" / digest


@dataclass(frozen=True, slots=True)
class HelpTriggerConfig:
    failed_attempts: int = 100
    stalled_hours: float = 6.0
    stuck_minutes: float = 30.0

    def __post_init__(self) -> None:
        if min(self.failed_attempts, self.stalled_hours, self.stuck_minutes) <= 0:
            raise ValueError("help trigger thresholds must be positive")
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
        self.action_history: list[NDArray[np.int64] | None] = []
        self.authorization: str | None = None
        self.frames_seen = 0
        self.representable_frames = 0
        self.unrepresentable_inputs: list[dict[str, Any]] = []
        self.test_source = False

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
        action_history: NDArray[np.int64] | None = None,
        test_source: bool = False,
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
        self.action_history.append(
            action_history.astype(np.int64, copy=True) if action_history is not None else None
        )
        self.test_source = self.test_source or test_source

    def append_human(
        self, *, png_bytes: bytes, buttons: Sequence[str], duration_ms: int,
        action_templates: Sequence[Any], action_mask: NDArray[np.bool_],
        player_knowledge: NDArray[np.float32],
        action_history: NDArray[np.int64] | None = None,
        test_source: bool = False,
    ) -> tuple[int | None, bool]:
        self.test_source = self.test_source or test_source
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
            player_knowledge=player_knowledge, action_history=action_history,
            test_source=test_source,
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
        arrays: dict[str, NDArray[Any]] = {
            "frame_bytes": np.concatenate(self.frames),
            "frame_offsets": offsets,
            "actions": np.asarray(self.actions, dtype=np.int64),
            "action_masks": np.stack(self.masks),
            "player_knowledge": np.stack(self.knowledge),
        }
        if self.action_history and all(row is not None for row in self.action_history):
            arrays["action_history"] = np.stack([row for row in self.action_history if row is not None])
        np.savez_compressed(target, **cast(Any, arrays))
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest = {
            "schema_version": DEMONSTRATION_SCHEMA,
            "game_version": self.game_version,
            "context": self.context,
            "authorization": self.authorization,
            "test_source": self.test_source,
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


def demonstration_batch(  # noqa: C901 - validates the demonstration data contract
    artifact: Path, *, expected_game_version: str, expected_context: str,
    observation_space: Any, action_count: int,
    game_extras_mask: NDArray[np.bool_] | None = None,
) -> dict[str, NDArray[Any]]:
    """Decode authenticated PNGs into universal observations and named game extras."""
    values = load_demonstration(
        artifact, expected_game_version=expected_game_version,
        expected_context=expected_context,
    )
    spaces = getattr(observation_space, "spaces", None)
    if not isinstance(spaces, dict) or set(spaces) != {"pixels", "action_history", "game_extras"}:
        raise ValueError("demonstration training requires universal observations and game_extras")
    actions = values["actions"].astype(np.int64, copy=False)
    masks = values["action_masks"].astype(np.bool_, copy=False)
    if masks.shape[1] != action_count or np.any(actions >= action_count):
        raise ValueError("demonstration action contract differs from the learner")
    pixel_shape = tuple(spaces["pixels"].shape)
    if len(pixel_shape) != 3 or pixel_shape[0] % 3:
        raise ValueError("universal pixels must use channel, height, width dimensions")
    frame_count, height, width = pixel_shape[0] // 3, pixel_shape[1], pixel_shape[2]
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
    for index, _frame in enumerate(frames):
        earliest = max(0, index - frame_count + 1)
        stack = frames[earliest:index + 1]
        if len(stack) < frame_count:
            stack = [stack[0]] * (frame_count - len(stack)) + stack
        pixels[index] = np.stack(stack).transpose(0, 3, 1, 2).reshape(pixel_shape)
        prior = actions[max(0, index - history_size):index]
        if len(prior):
            history[index, -len(prior):] = prior
    recorded_history = values.get("action_history")
    if recorded_history is not None:
        if recorded_history.shape != (len(actions), history_size):
            raise ValueError("demonstration action history differs from the learner")
        history = recorded_history.astype(np.int64, copy=True)
    game_extras = values.get("game_extras", values.get("player_knowledge"))
    if game_extras is None:
        raise ValueError("demonstration game extras are missing")
    game_extras = game_extras.astype(np.float32, copy=True)
    expected_extras = tuple(spaces["game_extras"].shape)
    if game_extras.shape != (len(actions), *expected_extras):
        raise ValueError("demonstration game extras differ from the learner")
    if game_extras_mask is not None:
        if game_extras_mask.shape != expected_extras:
            raise ValueError("game extras mask differs from the learner")
        game_extras[:, ~game_extras_mask] = 0
    return {
        "pixels": pixels,
        "action_history": history.reshape((len(actions), *spaces["action_history"].shape)),
        "game_extras": game_extras,
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


def train_demonstration_update(
    policy: Any, optimizer: Any, batch: dict[str, NDArray[Any]], *,
    game_id: str, weight: float, max_grad_norm: float = 0.5,
    max_samples: int = 256,
) -> dict[str, float]:
    """Apply one masked BC update to a verified universal-policy demonstration."""
    if weight <= 0 or max_samples < 1:
        raise ValueError("imitation weight and sample limit must be positive")
    import torch

    count = min(len(batch["actions"]), max_samples)
    if count < 1:
        raise ValueError("demonstration batch is empty")
    device = next(policy.parameters()).device
    observations = {
        "pixels": torch.as_tensor(batch["pixels"][:count][None], device=device),
        "action_history": torch.as_tensor(batch["action_history"][:count][None], device=device),
        "game_extras": torch.as_tensor(batch["game_extras"][:count][None], device=device),
    }
    actions = torch.as_tensor(batch["actions"][:count][None], dtype=torch.long, device=device)
    masks = torch.as_tensor(batch["action_masks"][:count][None], dtype=torch.bool, device=device)
    starts = torch.zeros((1, count), dtype=torch.bool, device=device)
    starts[:, 0] = True
    privileged = torch.zeros((1, count, 4), dtype=torch.float32, device=device)
    output = policy(
        observations, policy.initial_state(1, device), starts, masks,
        privileged=privileged, game_id=game_id,
        game_extras=observations["game_extras"],
    )
    logits = output["logits"].reshape(-1, policy.action_count)
    loss = masked_behavioral_cloning_loss(
        logits, actions.reshape(-1), masks.reshape(-1, policy.action_count),
    )
    weighted_loss = weight * loss
    optimizer.zero_grad(set_to_none=True)
    weighted_loss.backward()  # type: ignore[no-untyped-call]
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()
    return {"loss": float(loss.detach().cpu()), "weight": weight,
            "samples": float(count)}


def apply_verified_demonstrations(
    root: Path, policy: Any, optimizer: Any, *, actions: Sequence[str],
    game_id: str, round_number: int, max_grad_norm: float = 0.5,
) -> dict[str, float]:
    """Train once on demonstrations authorized by resolved help-ledger entries."""
    from types import SimpleNamespace

    from voidgym.core.observation import universal_observation_space
    from voidgym.training.shared_rollouts import publish_json

    requests: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "help-requests").glob("slot-*.json")):
        rows = _read_json(path).get("requests", [])
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and isinstance(row.get("request_id"), str):
                    requests[row["request_id"]] = row
    records_path = root / "demonstration-training.json"
    records = _read_json(records_path)
    trained = set(records.get("trained_artifacts", []))
    extras_layer = policy.game_extras[game_id][0]
    extras_dim = int(extras_layer.in_features)
    space = universal_observation_space()
    observation_space = SimpleNamespace(spaces={
        "pixels": space.spaces["pixels"],
        "action_history": space.spaces["action_history"],
        "game_extras": SimpleNamespace(shape=(extras_dim,)),
    })
    totals = {"demo_active": 0.0, "demo_samples": 0.0, "demo_loss": 0.0}
    for artifact in sorted((root / "demonstrations").rglob("*.npz")):
        try:
            result = _train_one_demonstration(
            artifact, root, observation_space, policy, optimizer, requests,
            trained, actions=actions, game_id=game_id, round_number=round_number,
            max_grad_norm=max_grad_norm,
        )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if result is None:
            continue
        totals["demo_active"] += 1
        totals["demo_samples"] += result["samples"]
        totals["demo_loss"] += result["loss"]
        trained.add(result["sha256"])
    if trained != set(records.get("trained_artifacts", [])):
        publish_json(records_path, {"schema_version": 1,
                                    "trained_artifacts": sorted(trained)})
    return totals


def _train_one_demonstration(
    artifact: Path, root: Path, observation_space: Any, policy: Any, optimizer: Any,
    requests: dict[str, Any], trained: set[str], *, actions: Sequence[str],
    game_id: str, round_number: int, max_grad_norm: float,
) -> dict[str, Any] | None:
    metadata = json.loads(artifact.with_suffix(".json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    authorization = metadata.get("authorization")
    if digest in trained or not isinstance(authorization, str) or not authorization.startswith("help_request:"):
        return None
    request_id = authorization.removeprefix("help_request:")
    request = requests.get(request_id)
    if not isinstance(request, dict) or request.get("status") != "resolved_with_help":
        return None
    context, version = metadata.get("context"), metadata.get("game_version")
    if not isinstance(context, str) or not isinstance(version, str):
        raise TypeError("demonstration context or game version is missing")
    expected_folder = hashlib.sha256(context.encode("utf-8")).hexdigest()[:16]
    if artifact.parent.name != expected_folder or request.get("context") != context:
        raise ValueError("demonstration is not bound to its resolved help request")
    extras_dim = int(policy.game_extras[game_id][0].in_features)
    batch = demonstration_batch(
        artifact, expected_game_version=version, expected_context=context,
        observation_space=observation_space, action_count=len(actions),
        game_extras_mask=np.zeros(extras_dim, dtype=np.bool_),
    )
    weight = imitation_schedule(
        initial_weight=0.25, decay_rounds=20, round_number=round_number,
        samples_in_bank=len(batch["actions"]), minimum_bank_size=64,
    )
    update = train_demonstration_update(
        policy, optimizer, batch, game_id=game_id, weight=weight,
        max_grad_norm=max_grad_norm,
    )
    return {**update, "sha256": digest}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}

