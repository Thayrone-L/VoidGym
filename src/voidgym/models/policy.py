"""Política recorrente universal com máscara e crítico privilegiado assimétrico."""

import torch
from torch import nn

from voidgym.models.aux_heads import HeroMotionHead, InverseDynamicsHead
from voidgym.models.encoder import ImpalaEncoder
from voidgym.models.perception import SemanticHead


class UniversalPolicy(nn.Module):
    def __init__(self, action_count: int = 43, history_size: int = 43, hidden: int = 256,
                 privileged_size: int = 4) -> None:
        super().__init__()
        self.encoder = ImpalaEncoder()
        self.perception = SemanticHead()
        self.action_history = nn.Embedding(history_size, 64)
        self.core = nn.LSTM(320, hidden, batch_first=True)
        self.actors = nn.ModuleDict({"explore": nn.Linear(hidden, action_count)})
        self.critics = nn.Sequential(nn.Linear(hidden + 64, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.privileged = nn.Sequential(nn.Linear(privileged_size, 64), nn.ReLU())
        self.extras = nn.ModuleDict()
        self.inverse_dynamics = InverseDynamicsHead(action_count)
        self.hero_motion = HeroMotionHead()

    def add_game_extras(self, game_id: str, input_size: int, output_size: int) -> None:
        layer = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, output_size))
        final = layer[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.extras[game_id] = layer

    def forward(self, pixels: torch.Tensor, action_history: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None,
                mask: torch.Tensor | None = None, privileged: torch.Tensor | None = None,
                expert: str = "explore", game_id: str | None = None,
                episode_starts: torch.Tensor | None = None, compute_auxiliary: bool = True) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        sequence_input = pixels.ndim == 5
        if sequence_input:
            batch, time, channels, height, width = pixels.shape
            features_flat, vector_flat = self.encoder(pixels.reshape(batch * time, channels, height, width))
            features = features_flat.reshape(batch, time, *features_flat.shape[1:])
            vector = vector_flat.reshape(batch, time, -1)
        else:
            features, vector = self.encoder(pixels)
            vector = vector.unsqueeze(1)
        if action_history.ndim == 2:
            hist = self.action_history(action_history.long()).mean(dim=-2).unsqueeze(1)
        else:
            hist = self.action_history(action_history.long()).mean(dim=-2)
        hist = hist.expand(-1, vector.shape[1], -1)
        if game_id and game_id in self.extras and vector.shape[1] == 1:
            vector = vector + self.extras[game_id](vector)
        recurrent_input = torch.cat((vector, hist), dim=-1)
        if sequence_input and episode_starts is not None:
            hidden = state
            outputs = []
            for time_index in range(recurrent_input.shape[1]):
                if hidden is not None:
                    keep = (1.0 - episode_starts[:, time_index].float()).view(1, -1, 1)
                    hidden = (hidden[0] * keep, hidden[1] * keep)
                output, hidden = self.core(recurrent_input[:, time_index:time_index + 1], hidden)
                outputs.append(output)
            recurrent = torch.cat(outputs, dim=1)
            assert hidden is not None
            next_state = hidden
        else:
            recurrent, next_state = self.core(recurrent_input, state)
        logits = self.actors[expert](recurrent)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e9)
        critic_input = recurrent
        if privileged is not None:
            privileged_features = self.privileged(privileged.float())
            if privileged_features.ndim == 2:
                privileged_features = privileged_features.unsqueeze(1)
            critic_input = torch.cat((recurrent, privileged_features), dim=-1)
        else:
            critic_input = torch.cat((recurrent, recurrent.new_zeros((*recurrent.shape[:-1], 64))), dim=-1)
        if compute_auxiliary and sequence_input:
            sem = self.perception(features.reshape(-1, *features.shape[2:])).reshape(
                batch, time, 10, 27, 48,
            )
        elif compute_auxiliary:
            sem = self.perception(features)
        else:
            sem = vector.new_empty((*vector.shape[:2], 0, 27, 48))
        motion = self.hero_motion(vector) if compute_auxiliary else vector.new_empty((*vector.shape[:2], 0))
        return logits, self.critics(critic_input).squeeze(-1), next_state, sem, motion
