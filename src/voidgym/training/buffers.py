"""Recurrent rollout storage, including oracle labels and per-transition discontinuities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from voidgym.core.semantics import IGNORE, SEM_GRID


@dataclass(slots=True)
class SequenceBatch:
    observations: dict[str, Tensor]
    actions: Tensor
    old_log_probs: Tensor
    old_values: Tensor
    returns: Tensor
    advantages: Tensor
    action_masks: Tensor
    episode_starts: Tensor
    initial_state: tuple[Tensor, Tensor]
    sem: Tensor
    hero_motion: Tensor
    discontinuity: Tensor
    privileged: Tensor
    has_labels: Tensor


class RecurrentRolloutBuffer:
    """Fixed-length on-policy data split into contiguous environment sequences."""

    def __init__(self, steps: int, envs: int, sequence_length: int = 32) -> None:
        if min(steps, envs, sequence_length) < 1:
            raise ValueError("steps, envs and sequence_length must be positive")
        self.steps, self.envs, self.sequence_length = steps, envs, sequence_length
        self.observations: list[dict[str, Tensor]] = []
        self.actions: list[Tensor] = []
        self.rewards: list[Tensor] = []
        self.dones: list[Tensor] = []
        self.values: list[Tensor] = []
        self.log_probs: list[Tensor] = []
        self.masks: list[Tensor] = []
        self.starts: list[Tensor] = []
        self.states: list[tuple[Tensor, Tensor]] = []
        self.sems: list[Tensor] = []
        self.motions: list[Tensor] = []
        self.discontinuities: list[Tensor] = []
        self.privileged: list[Tensor] = []
        self.has_labels: list[Tensor] = []
        self.returns: Tensor | None = None
        self.advantages: Tensor | None = None

    def add(self, observations: dict[str, Tensor], actions: Tensor, rewards: Tensor,
            dones: Tensor, values: Tensor, log_probs: Tensor, action_masks: Tensor,
            episode_starts: Tensor, state: tuple[Tensor, Tensor],
            infos: list[dict[str, Any]]) -> None:
        if len(self.observations) >= self.steps:
            raise RuntimeError("rollout buffer is full")
        self.observations.append({key: value.detach().cpu().clone() for key, value in observations.items()})
        self.actions.append(actions.detach().cpu().long().reshape(self.envs))
        self.rewards.append(rewards.detach().cpu().float().reshape(self.envs))
        self.dones.append(dones.detach().cpu().bool().reshape(self.envs))
        self.values.append(values.detach().cpu().float().reshape(self.envs))
        self.log_probs.append(log_probs.detach().cpu().float().reshape(self.envs))
        self.masks.append(action_masks.detach().cpu().bool())
        self.starts.append(episode_starts.detach().cpu().bool().reshape(self.envs))
        self.states.append((state[0].detach().cpu().clone(), state[1].detach().cpu().clone()))
        sems, motions, discontinuities, privileged, has_labels = [], [], [], [], []
        for env_index, info in enumerate(infos):
            labels = info.get("training_labels")
            sem = labels.get("sem") if isinstance(labels, dict) else None
            valid = isinstance(sem, np.ndarray) and sem.shape == SEM_GRID
            sems.append(torch.as_tensor(sem, dtype=torch.uint8) if valid else
                        torch.full(SEM_GRID, IGNORE, dtype=torch.uint8))
            velocity = labels.get("hero_velocity") if isinstance(labels, dict) else None
            grounded = labels.get("on_ground") if isinstance(labels, dict) else None
            if velocity is None or grounded is None:
                motion = torch.full((3,), float("nan"))
                valid = False
            else:
                motion = torch.tensor([float(velocity[0]), float(velocity[1]), float(grounded)])
            motions.append(motion)
            signals = info.get("signals", {})
            discontinuities.append(bool(self.dones[-1][env_index] or signals.get("room_changed")
                                        or signals.get("died")))
            privileged.append(_privileged(info))
            has_labels.append(valid)
        self.sems.append(torch.stack(sems))
        self.motions.append(torch.stack(motions))
        self.discontinuities.append(torch.tensor(discontinuities, dtype=torch.bool))
        self.privileged.append(torch.stack(privileged))
        self.has_labels.append(torch.tensor(has_labels, dtype=torch.bool))

    def compute_returns(self, last_values: Tensor, *, gamma: float, gae_lambda: float) -> None:
        count = len(self.rewards)
        if count != self.steps:
            raise RuntimeError("rollout buffer is not full")
        rewards, dones = torch.stack(self.rewards), torch.stack(self.dones)
        values = torch.stack(self.values)
        advantages = torch.zeros_like(rewards)
        next_advantage = torch.zeros(self.envs)
        next_value = last_values.detach().cpu().reshape(self.envs)
        for index in reversed(range(count)):
            not_done = (~dones[index]).float()
            delta = rewards[index] + gamma * next_value * not_done - values[index]
            next_advantage = delta + gamma * gae_lambda * not_done * next_advantage
            advantages[index] = next_advantage
            next_value = values[index]
        self.advantages = advantages
        self.returns = advantages + values

    def sequences(self, device: torch.device) -> list[SequenceBatch]:
        if self.advantages is None or self.returns is None:
            raise RuntimeError("compute_returns must be called before sampling")
        batches: list[SequenceBatch] = []
        observations = {key: torch.stack([step[key] for step in self.observations])
                        for key in self.observations[0]}
        for env in range(self.envs):
            for start in range(0, self.steps, self.sequence_length):
                end = min(self.steps, start + self.sequence_length)
                h, c = self.states[start]
                batches.append(SequenceBatch(
                    observations={key: value[start:end, env].unsqueeze(0).to(device)
                                  for key, value in observations.items()},
                    actions=torch.stack(self.actions[start:end])[:, env].unsqueeze(0).to(device),
                    old_log_probs=torch.stack(self.log_probs[start:end])[:, env].unsqueeze(0).to(device),
                    old_values=torch.stack(self.values[start:end])[:, env].unsqueeze(0).to(device),
                    returns=self.returns[start:end, env].unsqueeze(0).to(device),
                    advantages=self.advantages[start:end, env].unsqueeze(0).to(device),
                    action_masks=torch.stack(self.masks[start:end])[:, env].unsqueeze(0).to(device),
                    episode_starts=torch.stack(self.starts[start:end])[:, env].unsqueeze(0).to(device),
                    initial_state=(h[:, env:env + 1].to(device), c[:, env:env + 1].to(device)),
                    sem=torch.stack(self.sems[start:end])[:, env].unsqueeze(0).to(device),
                    hero_motion=torch.stack(self.motions[start:end])[:, env].unsqueeze(0).to(device),
                    discontinuity=torch.stack(self.discontinuities[start:end])[:, env].unsqueeze(0).to(device),
                    privileged=torch.stack(self.privileged[start:end])[:, env].unsqueeze(0).to(device),
                    has_labels=torch.stack(self.has_labels[start:end])[:, env].unsqueeze(0).to(device),
                ))
        return batches


def _privileged(info: dict[str, Any]) -> Tensor:
    explicit = info.get("privileged")
    if explicit is not None:
        value = torch.as_tensor(explicit, dtype=torch.float32).flatten()
        if value.numel() == 4:
            return value
    signals = info.get("signals", {})
    position = signals.get("player_pos_room", (.5, .5))
    return torch.tensor((float(position[0]), float(position[1]),
                         float(signals.get("progress", 0.0)),
                         float(signals.get("hp", 0.0)) / 10.0), dtype=torch.float32)
