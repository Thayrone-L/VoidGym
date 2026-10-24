"""Universal masked recurrent actor-critic with optional game and critic inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from voidgym.core.actions import UNIVERSAL_ACTIONS
from voidgym.core.observation import UNIVERSAL_ACTION_HISTORY_KEY, UNIVERSAL_IMAGE_KEY
from voidgym.core.semantics import SEM_GRID, Sem
from voidgym.models.aux_heads import HeroMotionHead, InverseDynamicsHead
from voidgym.models.encoder import ImpalaEncoder
from voidgym.models.perception import SemanticHead


class UniversalPolicy(nn.Module):
    """Shared perception, recurrent core, named actor heads and asymmetric critic.

    Recurrent states use ``(layers, batch, hidden)`` tensors like the v1 PPO.
    """

    def __init__(self, action_count: int = len(UNIVERSAL_ACTIONS), hidden_size: int = 256,
                 layers: int = 1, privileged_dim: int = 4) -> None:
        super().__init__()
        self.action_count = action_count
        self.hidden_size = hidden_size
        self.layers = layers
        self.encoder = ImpalaEncoder(256)
        self.perception = SemanticHead(32)
        self.inverse_dynamics = InverseDynamicsHead(256, action_count)
        self.hero_motion = HeroMotionHead(256)
        self.history_embedding = nn.Embedding(256, 64)
        self.history_projection = nn.Linear(64 * 3, 64)
        self.game_extras = nn.ModuleDict()
        self.core = nn.LSTM(256 + 64 + 256, hidden_size, layers, batch_first=True)
        self.actor = nn.ModuleDict({"explore": nn.Linear(hidden_size, action_count)})
        self.privileged_projection = nn.Linear(privileged_dim, 64)
        self.critic = nn.Sequential(nn.Linear(hidden_size + 64, 128), nn.ReLU(), nn.Linear(128, 1))

    def add_game_extras(self, game_id: str, input_dim: int) -> None:
        """Register a zero-output adapter so new game inputs preserve the policy initially."""
        if not game_id or input_dim < 1:
            raise ValueError("game_id and positive input_dim are required")
        if game_id in self.game_extras:
            raise ValueError(f"extras adapter already exists for {game_id}")
        adapter = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 256))
        last = cast(nn.Linear, adapter[-1])
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.game_extras[game_id] = adapter

    def add_expert(self, name: str) -> None:
        if not name or name in self.actor:
            raise ValueError("expert name must be new and nonempty")
        self.actor[name] = nn.Linear(self.hidden_size, self.action_count)

    def forward(self, obs: Mapping[str, Tensor], state: tuple[Tensor, Tensor] | None = None,
                episode_starts: Tensor | None = None, action_mask: Tensor | None = None,
                *, expert: str = "explore", game_id: str | None = None,
                game_extras: Tensor | None = None, privileged: Tensor | None = None
                ) -> dict[str, Tensor | tuple[Tensor, Tensor]]:
        image, sequence = _sequence(obs[UNIVERSAL_IMAGE_KEY], 4)
        history, _ = _sequence(obs[UNIVERSAL_ACTION_HISTORY_KEY], 2)
        batch, steps = image.shape[:2]
        feature_map, vector = self._encode_sequence(image)
        history_vector = self.history_projection(
            self.history_embedding(history.long().clamp(0, 255)).flatten(start_dim=-2))
        extra_vector = vector.new_zeros((batch, steps, 256))
        if game_extras is not None:
            if game_id is None or game_id not in self.game_extras:
                raise ValueError("register a game extras adapter before passing game_extras")
            if game_extras.ndim == 2 and steps == 1:
                game_extras = game_extras.unsqueeze(1)
            extra_vector = self.game_extras[game_id](game_extras.to(vector.dtype))
        recurrent_input = torch.cat((vector, history_vector, extra_vector), dim=-1)
        hidden = state or self.initial_state(batch, vector.device, vector.dtype)
        starts = _starts(episode_starts, batch, steps, vector.device)
        outputs: list[Tensor] = []
        h, c = hidden
        for index in range(steps):
            keep = (~starts[:, index]).to(vector.dtype).view(1, batch, 1)
            h, c = h * keep, c * keep
            current, (h, c) = self.core(recurrent_input[:, index:index + 1], (h, c))
            outputs.append(current)
        latent = torch.cat(outputs, dim=1)
        if expert not in self.actor:
            raise ValueError(f"unknown actor expert: {expert}")
        logits = self.actor[expert](latent)
        if action_mask is not None:
            mask, _ = _sequence(action_mask, 2)
            if mask.shape != logits.shape:
                raise ValueError("action_mask must match logits shape")
            if not mask.bool().any(dim=-1).all():
                raise ValueError("each action mask must allow at least one action")
            logits = logits.masked_fill(~mask.bool(), -1e9)
        privileged_vector = latent.new_zeros((batch, steps, 64))
        if privileged is not None:
            if privileged.ndim == 2 and steps == 1:
                privileged = privileged.unsqueeze(1)
            privileged_vector = F.relu(self.privileged_projection(privileged.to(latent.dtype)))
        values = self.critic(torch.cat((latent, privileged_vector), dim=-1)).squeeze(-1)
        semantic = self.perception(feature_map.flatten(0, 1)).reshape(
            batch, steps, len(Sem), *SEM_GRID)
        motion, on_ground = self.hero_motion(vector)
        output: dict[str, Tensor | tuple[Tensor, Tensor]] = {
            "logits": logits, "values": values, "state": (h, c),
            "semantic_logits": semantic, "features": vector,
            "motion": motion, "on_ground_logits": on_ground,
        }
        if not sequence:
            output = {key: (value[:, 0] if isinstance(value, Tensor) and value.ndim > 1 else value)
                      for key, value in output.items()}
        return output

    def initial_state(self, batch: int, device: torch.device,
                      dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor]:
        shape = (self.layers, batch, self.hidden_size)
        return torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)

    def _encode_sequence(self, image: Tensor) -> tuple[Tensor, Tensor]:
        batch, steps = image.shape[:2]
        feature_map, vector = self.encoder(image.flatten(0, 1))
        return (feature_map.reshape(batch, steps, *feature_map.shape[1:]),
                vector.reshape(batch, steps, -1))


def _sequence(value: Tensor, expected_ndim: int) -> tuple[Tensor, bool]:
    if value.ndim == expected_ndim:
        return value.unsqueeze(1), False
    if value.ndim == expected_ndim + 1:
        return value, True
    raise ValueError("observation tensor has an unexpected rank")


def _starts(value: Tensor | None, batch: int, steps: int, device: torch.device) -> Tensor:
    if value is None:
        return torch.zeros((batch, steps), dtype=torch.bool, device=device)
    starts, _ = _sequence(value, 1)
    if starts.shape != (batch, steps):
        raise ValueError("episode_starts must have shape (B,) or (B,T)")
    return starts.bool()
