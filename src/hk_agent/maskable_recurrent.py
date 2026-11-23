from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.distributions import Categorical


class RecurrentBoundary(StrEnum):
    """Events that require recurrent context to be discarded."""

    GAME_REOPENED = "game_reopened"
    SAVE_RELOADED = "save_reloaded"
    SAVE_CHANGED = "save_changed"
    DEATH = "death"
    PPO_CHUNK = "ppo_chunk"


@dataclass(slots=True)
class RecurrentState:
    hidden: Tensor
    cell: Tensor

    def detached(self) -> RecurrentState:
        return RecurrentState(self.hidden.detach(), self.cell.detach())


class RecurrentStateTracker:
    """Keep actor memory across PPO chunks; reset only at actual run discontinuities."""

    RESET_BOUNDARIES = frozenset(
        {
            RecurrentBoundary.GAME_REOPENED,
            RecurrentBoundary.SAVE_RELOADED,
            RecurrentBoundary.SAVE_CHANGED,
            RecurrentBoundary.DEATH,
        }
    )

    def __init__(self, layers: int, batch_size: int, hidden_size: int) -> None:
        if min(layers, batch_size, hidden_size) < 1:
            raise ValueError("recurrent dimensions must be positive")
        self.layers = layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self._state: RecurrentState | None = None
        self._episode_start = np.ones(batch_size, dtype=np.bool_)

    @property
    def episode_start(self) -> NDArray[np.bool_]:
        return self._episode_start.copy()

    def state(self, device: torch.device | str = "cpu") -> RecurrentState:
        if self._state is None:
            shape = (self.layers, self.batch_size, self.hidden_size)
            return RecurrentState(torch.zeros(shape, device=device), torch.zeros(shape, device=device))
        return RecurrentState(self._state.hidden.to(device), self._state.cell.to(device))

    def update(self, state: RecurrentState) -> None:
        expected = (self.layers, self.batch_size, self.hidden_size)
        if tuple(state.hidden.shape) != expected or tuple(state.cell.shape) != expected:
            raise ValueError("recurrent state shape differs from tracker")
        if not torch.isfinite(state.hidden).all() or not torch.isfinite(state.cell).all():
            raise ValueError("recurrent state contains non-finite values")
        self._state = state.detached()
        self._episode_start[:] = False

    def boundary(self, event: RecurrentBoundary) -> None:
        if event in self.RESET_BOUNDARIES:
            self._state = None
            self._episode_start[:] = True


class MaskableRecurrentActorCritic(nn.Module):
    """LSTM core and categorical policy/value heads over existing extracted features."""

    def __init__(
        self,
        feature_size: int,
        action_count: int,
        hidden_size: int = 256,
        layers: int = 1,
    ) -> None:
        super().__init__()
        if min(feature_size, action_count, hidden_size, layers) < 1:
            raise ValueError("network dimensions must be positive")
        self.feature_size = feature_size
        self.action_count = action_count
        self.hidden_size = hidden_size
        self.layers = layers
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers=layers)
        self.policy_head = nn.Linear(hidden_size, action_count)
        self.value_head = nn.Linear(hidden_size, 1)

    def initial_state(self, batch_size: int, *, device: torch.device | str | None = None) -> RecurrentState:
        shape = (self.layers, batch_size, self.hidden_size)
        target = device or next(self.parameters()).device
        return RecurrentState(torch.zeros(shape, device=target), torch.zeros(shape, device=target))

    def forward_sequence(
        self,
        features: Tensor,
        state: RecurrentState,
        episode_starts: Tensor,
        action_masks: Tensor,
    ) -> tuple[Tensor, Tensor, RecurrentState]:
        if features.ndim != 3 or features.shape[-1] != self.feature_size:
            raise ValueError("features must have shape [time, batch, feature_size]")
        time_steps, batch_size, _ = features.shape
        if tuple(episode_starts.shape) != (time_steps, batch_size):
            raise ValueError("episode_starts must have shape [time, batch]")
        if tuple(action_masks.shape) != (time_steps, batch_size, self.action_count):
            raise ValueError("action_masks must have shape [time, batch, actions]")
        if not torch.isfinite(features).all() or not torch.isfinite(episode_starts).all():
            raise ValueError("sequence data contains non-finite values")
        if torch.any(action_masks.sum(dim=-1) == 0):
            raise ValueError("each timestep must have at least one legal action")
        hidden, cell = state.hidden, state.cell
        if tuple(hidden.shape) != (self.layers, batch_size, self.hidden_size) or tuple(cell.shape) != tuple(hidden.shape):
            raise ValueError("initial LSTM state shape differs from sequence batch")
        outputs: list[Tensor] = []
        for timestep in range(time_steps):
            keep = (1.0 - episode_starts[timestep].to(features.dtype)).view(1, batch_size, 1)
            out, (hidden, cell) = self.lstm(
                features[timestep : timestep + 1], (hidden * keep, cell * keep)
            )
            outputs.append(out)
        latent = torch.cat(outputs, dim=0)
        logits = self.policy_head(latent)
        logits = logits.masked_fill(~action_masks.to(torch.bool), torch.finfo(logits.dtype).min)
        values = self.value_head(latent).squeeze(-1)
        return logits, values, RecurrentState(hidden, cell)

    def act(
        self,
        features: Tensor,
        state: RecurrentState,
        episode_start: Tensor,
        action_masks: Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, RecurrentState]:
        logits, values, next_state = self.forward_sequence(
            features.unsqueeze(0), state, episode_start.reshape(1, -1), action_masks.unsqueeze(0)
        )
        distribution = Categorical(logits=logits[0])
        actions = logits[0].argmax(-1) if deterministic else distribution.sample()  # type: ignore[no-untyped-call]
        return actions, values[0], distribution.log_prob(actions), next_state  # type: ignore[no-untyped-call]

    def evaluate_actions(
        self,
        features: Tensor,
        state: RecurrentState,
        episode_starts: Tensor,
        action_masks: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, RecurrentState]:
        logits, values, next_state = self.forward_sequence(features, state, episode_starts, action_masks)
        distribution = Categorical(logits=logits)
        return values, distribution.log_prob(actions), distribution.entropy(), next_state  # type: ignore[no-untyped-call]


def compatible_warm_start(
    target: nn.Module, source_state: dict[str, Tensor]
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """Copy only exact-name/shape feature extractor weights; leave new recurrence/heads fresh."""
    target_state = target.state_dict()
    compatible = {
        name: value for name, value in source_state.items()
        if name in target_state and tuple(value.shape) == tuple(target_state[name].shape)
        and name.startswith(("features_extractor.", "pi_features_extractor.", "vf_features_extractor."))
    }
    target.load_state_dict(compatible, strict=False)
    return {
        name: (tuple(source_state[name].shape), tuple(target_state[name].shape))
        for name in compatible
    }

@dataclass(frozen=True, slots=True)
class RecurrentRollout:
    features: NDArray[np.float32]
    actions: NDArray[np.int64]
    action_masks: NDArray[np.bool_]
    episode_starts: NDArray[np.float32]
    initial_hidden: NDArray[np.float32]
    initial_cell: NDArray[np.float32]
    values: NDArray[np.float32]
    log_probs: NDArray[np.float32]
    advantages: NDArray[np.float32]
    returns: NDArray[np.float32]
    hidden_states: NDArray[np.float32]
    cell_states: NDArray[np.float32]

    def validate(self) -> None:
        if self.features.ndim != 3:
            raise ValueError("rollout features must be [time, batch, feature]")
        steps, batch_size, feature_size = self.features.shape
        if min(steps, batch_size, feature_size) < 1:
            raise ValueError("rollout dimensions must be non-empty")
        for name in ("actions", "episode_starts", "values", "log_probs", "advantages", "returns"):
            if getattr(self, name).shape != (steps, batch_size):
                raise ValueError(f"rollout {name} shape is invalid")
        if self.action_masks.ndim != 3 or self.action_masks.shape[:2] != (steps, batch_size):
            raise ValueError("rollout action_masks shape is invalid")
        if self.action_masks.dtype != np.bool_ or not self.action_masks.any(axis=-1).all():
            raise ValueError("rollout action_masks must be non-empty boolean masks")
        action_count = self.action_masks.shape[-1]
        if np.any(self.actions < 0) or np.any(self.actions >= action_count):
            raise ValueError("rollout actions are out of range")
        if not self.action_masks[np.arange(steps)[:, None], np.arange(batch_size), self.actions].all():
            raise ValueError("rollout includes an unavailable action")
        if self.initial_hidden.ndim != 3 or self.initial_hidden.shape != self.initial_cell.shape:
            raise ValueError("rollout initial states must be [layers, batch, hidden]")
        layers, state_batch, hidden_size = self.initial_hidden.shape
        if state_batch != batch_size or min(layers, hidden_size) < 1:
            raise ValueError("rollout initial state dimensions differ")
        recurrent_shape = (steps, layers, batch_size, hidden_size)
        if self.hidden_states.shape != recurrent_shape or self.cell_states.shape != recurrent_shape:
            raise ValueError("rollout per-step states have invalid shapes")
        for name in (
            "features", "episode_starts", "initial_hidden", "initial_cell", "values",
            "log_probs", "advantages", "returns", "hidden_states", "cell_states",
        ):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"rollout {name} contains non-finite values")

    def save(self, path: str) -> None:
        self.validate()
        np.savez_compressed(
            path,
            features=self.features,
            actions=self.actions,
            action_masks=self.action_masks,
            episode_starts=self.episode_starts,
            initial_hidden=self.initial_hidden,
            initial_cell=self.initial_cell,
            values=self.values,
            log_probs=self.log_probs,
            advantages=self.advantages,
            returns=self.returns,
            hidden_states=self.hidden_states,
            cell_states=self.cell_states,
        )

    @classmethod
    def load(cls, path: str) -> RecurrentRollout:
        with np.load(path, allow_pickle=False) as archive:
            expected = {
                "features", "actions", "action_masks", "episode_starts", "initial_hidden",
                "initial_cell", "values", "log_probs", "advantages", "returns",
                "hidden_states", "cell_states",
            }
            if set(archive.files) != expected:
                raise ValueError("rollout artifact field inventory is invalid")
            rollout = cls(**{name: archive[name].copy() for name in expected})
        rollout.validate()
        return rollout


def train_recurrent_ppo_sequences(
    policy: MaskableRecurrentActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: RecurrentRollout,
    *,
    sequence_length: int = 128,
    epochs: int = 4,
    clip_range: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
    max_grad_norm: float = 0.5,
) -> dict[str, float]:
    """Train masked PPO on truncated BPTT sequences with saved initial states."""
    rollout.validate()
    if sequence_length < 1 or epochs < 1 or not 0 < clip_range < 1:
        raise ValueError("invalid recurrent PPO hyperparameters")
    started = __import__("time").perf_counter()
    losses: list[float] = []
    steps, batch_size, _ = rollout.features.shape
    device = next(policy.parameters()).device
    for _ in range(epochs):
        chunks = [
            (env, start, min(start + sequence_length, steps))
            for env in np.random.permutation(batch_size)
            for start in range(0, steps, sequence_length)
        ]
        np.random.shuffle(chunks)
        for env, start, end in chunks:
            features = torch.as_tensor(rollout.features[start:end, env : env + 1], device=device)
            starts = torch.as_tensor(rollout.episode_starts[start:end, env : env + 1], device=device)
            masks = torch.as_tensor(rollout.action_masks[start:end, env : env + 1], device=device)
            actions = torch.as_tensor(rollout.actions[start:end, env : env + 1], device=device)
            old_log_prob = torch.as_tensor(rollout.log_probs[start:end, env : env + 1], device=device)
            advantages = torch.as_tensor(rollout.advantages[start:end, env : env + 1], device=device)
            returns = torch.as_tensor(rollout.returns[start:end, env : env + 1], device=device)
            hidden = torch.as_tensor(rollout.hidden_states[start, :, env : env + 1], device=device)
            cell = torch.as_tensor(rollout.cell_states[start, :, env : env + 1], device=device)
            new_values, new_log_prob, entropy, _ = policy.evaluate_actions(
                features, RecurrentState(hidden, cell), starts, masks, actions
            )
            ratio = torch.exp(new_log_prob - old_log_prob)
            policy_loss = -torch.minimum(
                advantages * ratio,
                advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range),
            ).mean()
            value_loss = torch.nn.functional.mse_loss(new_values, returns)
            loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    elapsed = __import__("time").perf_counter() - started
    return {"mean_loss": float(np.mean(losses)), "train_seconds": elapsed, "sequences": float(len(losses))}




def import_recurrent_extractor_warm_start(
    policy: nn.Module,
    artifact_path: str,
    manifest_path: str,
    *,
    expected_fingerprint: str,
    expected_artifact_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_manifest_id: str | None = None,
) -> dict[str, Any]:
    """Validate a portable prior policy and copy only exact-shape visual/knowledge extractors."""
    import hashlib
    import json
    from pathlib import Path

    from hk_agent.policy_artifact import _safetensors_api, validate_policy_manifest

    artifact = Path(artifact_path).resolve(strict=True)
    sidecar = Path(manifest_path).resolve(strict=True)
    payload = sidecar.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
        raise ValueError("warm-start manifest differs from its frozen identity")
    manifest = validate_policy_manifest(json.loads(payload.decode("utf-8")))
    if manifest["compatibility_fingerprint"] != expected_fingerprint:
        raise ValueError("warm-start policy compatibility fingerprint differs")
    if expected_manifest_id is not None and manifest["manifest_id"] != expected_manifest_id:
        raise ValueError("warm-start policy manifest ID differs")
    artifact_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if artifact_digest != manifest["artifact"]["sha256"] or artifact.stat().st_size != manifest["artifact"]["size_bytes"]:
        raise ValueError("warm-start policy artifact differs from its manifest")
    if expected_artifact_sha256 is not None and artifact_digest != expected_artifact_sha256:
        raise ValueError("warm-start policy differs from its frozen identity")
    known_sources = {
        "stable_baselines3.common.policies.MultiInputActorCriticPolicy",
        "sb3_contrib.common.maskable.policies.MaskableMultiInputActorCriticPolicy",
    }
    if manifest["policy_class"] not in known_sources:
        raise ValueError("warm-start source is not a compatible feed-forward policy")
    safe_open, _ = _safetensors_api()
    with safe_open(str(artifact), framework="pt", device="cpu") as reader:
        metadata = reader.metadata()
        if metadata != {
            "format": "voidgym-policy-v1",
            "compatibility_fingerprint": expected_fingerprint,
            "tensor_schema_sha256": manifest["tensor_schema_sha256"],
        }:
            raise ValueError("warm-start safetensors metadata differs from its manifest")
        names = reader.keys()
        source_state = {name: reader.get_tensor(name) for name in names}
    copied = compatible_warm_start(policy, source_state)
    if not copied:
        raise ValueError("warm-start has no compatible feature extractor tensors")
    return {
        "source_policy_class": manifest["policy_class"],
        "manifest_id": manifest["manifest_id"],
        "artifact_sha256": artifact_digest,
        "compatibility_fingerprint": expected_fingerprint,
        "copied_tensors": sorted(copied),
        "fresh_recurrent_and_heads": True,
    }
