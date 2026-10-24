"""Recurrent clipped PPO with semantic, dynamics and motion auxiliary losses."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch
from torch import Tensor
from torch.distributions import Categorical

from voidgym.models.perception import semantic_loss
from voidgym.models.policy import UniversalPolicy
from voidgym.training.augmentations import augment_sequence
from voidgym.training.buffers import SequenceBatch


class RecurrentPPOTrainer:
    def __init__(self, policy: UniversalPolicy, *, learning_rate: float = 2.5e-4,
                 clip: float = .2, entropy_coef: float = .01, value_coef: float = .5,
                 c_sem: float = .5, c_inv: float = .1, c_mot: float = .1,
                 augment: bool = True, max_grad_norm: float = .5,
                 semantic_weights: Tensor | None = None,
                 lr_encoder_scale: float = 1.0,
                 game_id: str | None = None) -> None:
        if not 0 < lr_encoder_scale <= 10:
            raise ValueError("lr_encoder_scale must be in (0, 10]")
        self.policy = policy
        self.game_id = game_id
        encoder_ids = {id(parameter) for parameter in policy.encoder.parameters()}
        encoder_parameters = [parameter for parameter in policy.parameters()
                              if id(parameter) in encoder_ids]
        other_parameters = [parameter for parameter in policy.parameters()
                            if id(parameter) not in encoder_ids]
        self.optimizer = torch.optim.Adam([
            {"params": encoder_parameters, "lr": learning_rate * lr_encoder_scale},
            {"params": other_parameters, "lr": learning_rate},
        ], eps=1e-5)
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.c_sem, self.c_inv, self.c_mot = c_sem, c_inv, c_mot
        self.augment = augment
        self.max_grad_norm = max_grad_norm
        self.semantic_weights = (semantic_weights.to(next(policy.parameters()).device)
                                 if semantic_weights is not None else None)

    def update(self, sequences: Iterable[SequenceBatch], *, epochs: int = 1,
               sequence_batch_size: int = 8) -> dict[str, float]:
        rows: list[dict[str, float]] = []
        sequences = list(sequences)
        for _ in range(epochs):
            order = torch.randperm(len(sequences)).tolist()
            for start in range(0, len(order), sequence_batch_size):
                group = [sequences[index] for index in order[start:start + sequence_batch_size]]
                rows.append(self._update_group(group))
        if not rows:
            return {}
        return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}

    def _update_group(self, group: list[SequenceBatch]) -> dict[str, float]:
        batch = _collate(group)
        observations = batch.observations
        sem = batch.sem
        if self.augment:
            observations = dict(observations)
            observations["pixels"], shifted = augment_sequence(observations["pixels"], sem)
            assert shifted is not None
            sem = shifted
        self.policy.train()
        game_extras = observations.get("game_extras")
        output = self.policy(
            observations, batch.initial_state, batch.episode_starts, batch.action_masks,
            privileged=batch.privileged,
            game_id=self.game_id if game_extras is not None else None,
            game_extras=game_extras,
        )
        logits = output["logits"]
        values = output["values"]
        features = output["features"]
        semantic_logits = output["semantic_logits"]
        motion = output["motion"]
        ground = output["on_ground_logits"]
        assert isinstance(logits, Tensor) and isinstance(values, Tensor)
        assert isinstance(features, Tensor) and isinstance(semantic_logits, Tensor)
        assert isinstance(motion, Tensor) and isinstance(ground, Tensor)
        distribution = Categorical(logits=logits)
        new_log_prob = cast(Tensor, distribution.log_prob(batch.actions))  # type: ignore[no-untyped-call]
        ratio = torch.exp(new_log_prob - batch.old_log_probs)
        advantage = batch.advantages
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        policy_loss = -torch.minimum(advantage * ratio,
                                     advantage * ratio.clamp(1 - self.clip, 1 + self.clip)).mean()
        value_loss = torch.nn.functional.mse_loss(values, batch.returns)
        entropy = cast(Tensor, distribution.entropy()).mean()  # type: ignore[no-untyped-call]
        sem_targets = sem.reshape(-1, *sem.shape[-2:])
        sem_logits = semantic_logits.flatten(0, 1)
        sem_loss = semantic_loss(sem_logits, sem_targets, self.semantic_weights)
        inverse_loss = self._inverse_loss(features, batch)
        motion_loss = self.policy.hero_motion.loss(motion, ground, batch.hero_motion,
                                                   batch.has_labels)
        loss = (policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy +
                self.c_sem * sem_loss + self.c_inv * inverse_loss + self.c_mot * motion_loss)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        approximate_kl = ((ratio - 1) - new_log_prob + batch.old_log_probs).mean()
        variance = batch.returns.var(unbiased=False)
        explained = 1 - (batch.returns - values.detach()).var(unbiased=False) / variance.clamp_min(1e-8)
        return {"loss": float(loss.detach()), "L_ppo": float(policy_loss.detach()),
                "L_semantic": float(sem_loss.detach()),
                "L_inverse_dynamics": float(inverse_loss.detach()),
                "L_hero_motion": float(motion_loss.detach()),
                "entropy": float(entropy.detach()), "kl": float(approximate_kl.detach()),
                "explained_variance": float(explained.detach())}

    def _inverse_loss(self, features: Tensor, batch: SequenceBatch) -> Tensor:
        if features.shape[1] < 2:
            return features.sum() * 0.0
        valid = ((~batch.discontinuity[:, :-1]) & (~batch.discontinuity[:, 1:]) &
                 batch.has_labels[:, :-1] & batch.has_labels[:, 1:])
        logits = cast(Tensor, self.policy.inverse_dynamics(features[:, :-1], features[:, 1:]))
        actions = batch.actions[:, :-1]
        if not valid.any():
            return logits.sum() * 0.0
        return torch.nn.functional.cross_entropy(logits[valid], actions[valid])


def _collate(group: list[SequenceBatch]) -> SequenceBatch:
    lengths = {item.actions.shape[1] for item in group}
    if len(lengths) != 1:
        raise ValueError("sequence minibatches must have equal lengths")
    def cat(field: str) -> Tensor:
        return torch.cat(cast(list[Tensor], [getattr(item, field) for item in group]), dim=0)
    observations = {key: torch.cat([item.observations[key] for item in group])
                    for key in group[0].observations}
    return SequenceBatch(
        observations, cat("actions"), cat("old_log_probs"), cat("old_values"), cat("returns"),
        cat("advantages"), cat("action_masks"), cat("episode_starts"),
        (torch.cat([item.initial_state[0] for item in group], dim=1),
         torch.cat([item.initial_state[1] for item in group], dim=1)),
        cat("sem"), cat("hero_motion"), cat("discontinuity"), cat("privileged"), cat("has_labels"),
    )
