# mypy: ignore-errors
"""SB3 adapter. sb3-contrib exposes no typed extension points for recurrent mask buffers;
runtime integration tests exercise these overrides against the pinned sb3-contrib release.
The independently typed sequence model and rollout schema live in maskable_recurrent.py.
"""
from __future__ import annotations

from collections.abc import Generator
from typing import Any, ClassVar, NamedTuple

import numpy as np
import torch
from gymnasium import spaces
from numpy.typing import NDArray
from sb3_contrib.common.maskable.distributions import make_masked_proba_distribution
from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer
from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import CombinedExtractor
from torch import Tensor


class MaskableRecurrentDictRolloutBufferSamples(NamedTuple):
    observations: dict[str, Tensor]
    actions: Tensor
    old_values: Tensor
    old_log_prob: Tensor
    advantages: Tensor
    returns: Tensor
    lstm_states: RNNStates
    episode_starts: Tensor
    mask: Tensor
    action_masks: Tensor


class MaskableRecurrentDictRolloutBuffer(RecurrentDictRolloutBuffer):
    """Recurrent Dict buffer that keeps masks and starts BPTT at bounded chunks."""

    sequence_length = 128

    def reset(self) -> None:
        super().reset()
        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("recurrent masked policy requires Discrete actions")
        self.mask_dims = int(self.action_space.n)
        self.action_masks = np.ones(
            (self.buffer_size, self.n_envs, self.mask_dims), dtype=np.bool_
        )

    def add(
        self,
        *args: Any,
        lstm_states: RNNStates,
        action_masks: NDArray[np.bool_],
        **kwargs: Any,
    ) -> None:
        if action_masks.shape != (self.n_envs, self.mask_dims) or not action_masks.any(axis=1).all():
            raise ValueError("recurrent action masks have invalid shape or empty rows")
        self.action_masks[self.pos] = action_masks
        super().add(*args, lstm_states=lstm_states, **kwargs)

    def get(self, batch_size: int | None = None) -> Generator[MaskableRecurrentDictRolloutBufferSamples, None, None]:
        if not self.full:
            raise AssertionError("rollout buffer must be full before sampling")
        if not self.generator_ready:
            for name in ("hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf"):
                self.__dict__[name] = self.__dict__[name].swapaxes(1, 2)
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)
            for name in (
                "actions", "values", "log_probs", "advantages", "returns",
                "hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf",
                "episode_starts", "action_masks",
            ):
                self.__dict__[name] = self.swap_and_flatten(self.__dict__[name])
            self.generator_ready = True
        total = self.buffer_size * self.n_envs
        if batch_size is None:
            batch_size = total
        split = np.random.randint(total)
        indices = np.concatenate((np.arange(total)[split:], np.arange(total)[:split]))
        env_change = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        env_change[:: self.sequence_length, :] = 1.0
        env_change = self.swap_and_flatten(env_change)
        for start in range(0, total, batch_size):
            yield self._get_samples(indices[start : start + batch_size], env_change)

    def _get_samples(
        self, batch_inds: NDArray[np.int64], env_change: NDArray[np.float32], env: Any = None
    ) -> MaskableRecurrentDictRolloutBufferSamples:
        sample = super()._get_samples(batch_inds, env_change, env)
        masks = self.pad(self.action_masks[batch_inds]).reshape((-1, self.mask_dims))
        return MaskableRecurrentDictRolloutBufferSamples(*sample, action_masks=self.to_torch(masks))


class IntDimensionCombinedExtractor(CombinedExtractor):
    """Normalize Gym's NumPy dimensions before passing them to ``torch.nn.LSTM``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._features_dim = int(self._features_dim)


class MaskableRecurrentActorCriticPolicy(RecurrentMultiInputActorCriticPolicy):
    """sb3-contrib recurrent policy with MaskableCategorical at sampling and update time."""

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, *args: Any, **kwargs: Any):
        if not isinstance(action_space, spaces.Discrete):
            raise TypeError("MaskableRecurrentPPO currently supports Discrete actions only")
        kwargs.setdefault("features_extractor_class", IntDimensionCombinedExtractor)
        super().__init__(observation_space, action_space, *args, **kwargs)
        self.action_dist = make_masked_proba_distribution(action_space)

    def _get_action_dist_from_latent(self, latent_pi: Tensor) -> Any:  # type: ignore[override]
        logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(
        self,
        obs: Any,
        lstm_states: RNNStates,
        episode_starts: Tensor,
        deterministic: bool = False,
        action_masks: NDArray[np.bool_] | Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, RNNStates]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features
        latent_pi, states_pi = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, states_vf = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
            states_vf = (states_pi[0].detach(), states_pi[1].detach())
        else:
            latent_vf = self.critic(vf_features)
            states_vf = states_pi
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions.reshape((-1, *self.action_space.shape)), values, log_prob, RNNStates(states_pi, states_vf)

    def evaluate_actions(
        self,
        obs: Any,
        actions: Tensor,
        lstm_states: RNNStates,
        episode_starts: Tensor,
        action_masks: NDArray[np.bool_] | Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features
        latent_pi, _ = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        distribution.apply_masking(action_masks)
        return self.value_net(latent_vf), distribution.log_prob(actions), distribution.entropy()

    def predict_masked(
        self,
        observation: Any,
        state: tuple[NDArray[np.float32], ...] | None,
        episode_start: NDArray[np.bool_],
        action_masks: NDArray[np.bool_],
        deterministic: bool = False,
    ) -> tuple[NDArray[np.int64], tuple[NDArray[np.float32], ...]]:
        obs_tensor, _ = self.obs_to_tensor(observation)
        batch = next(iter(obs_tensor.values())).shape[0] if isinstance(obs_tensor, dict) else obs_tensor.shape[0]
        if state is None:
            shape = (self.lstm_hidden_state_shape[0], batch, self.lstm_hidden_state_shape[2])
            zeros = np.zeros(shape, dtype=np.float32)
            state = (zeros, zeros, zeros, zeros)
        tensors = tuple(torch.as_tensor(item, device=self.device) for item in state)
        rnn = RNNStates((tensors[0], tensors[1]), (tensors[2], tensors[3]))
        starts = torch.as_tensor(episode_start, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions, _, _, next_state = self(obs_tensor, rnn, starts, deterministic, action_masks)
        next_arrays = tuple(item.cpu().numpy() for pair in (next_state.pi, next_state.vf) for item in pair)
        return actions.cpu().numpy().reshape(-1), next_arrays


class MaskableRecurrentPPO(RecurrentPPO):
    """Recurrent PPO subclass; masks and bounded BPTT apply in sampling and updates."""

    def __init__(self, *args: Any, sequence_length: int = 128, **kwargs: Any) -> None:
        if not 1 <= sequence_length <= 2048:
            raise ValueError("sequence_length must be in [1, 2048]")
        self.sequence_length = sequence_length
        super().__init__(*args, **kwargs)

    policy_aliases: ClassVar[dict[str, type[MaskableRecurrentActorCriticPolicy]]] = {
        "MultiInputLstmPolicy": MaskableRecurrentActorCriticPolicy,
        "MlpLstmPolicy": MaskableRecurrentActorCriticPolicy,
    }

    def _setup_model(self) -> None:
        super()._setup_model()
        lstm = self.policy.lstm_actor
        state_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
        self.rollout_buffer = MaskableRecurrentDictRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            state_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.rollout_buffer.sequence_length = self.sequence_length

    def collect_rollouts(
        self,
        env: Any,
        callback: BaseCallback,
        rollout_buffer: Any,
        n_rollout_steps: int,
    ) -> bool:
        from sb3_contrib.common.maskable.utils import get_action_masks
        from stable_baselines3.common.utils import obs_as_tensor

        if not isinstance(rollout_buffer, MaskableRecurrentDictRolloutBuffer):
            raise TypeError("MaskableRecurrentPPO requires its recurrent masked buffer")
        if self._last_obs is None or self._last_lstm_states is None:
            raise RuntimeError("recurrent collector has no initialized observation/state")
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()
        n_steps = 0
        lstm_states = self._last_lstm_states
        while n_steps < n_rollout_steps:
            masks = get_action_masks(env)
            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                starts = torch.as_tensor(self._last_episode_starts, dtype=torch.float32, device=self.device)
                actions, values, log_probs, next_states = self.policy(
                    obs_tensor, lstm_states, starts, action_masks=masks
                )
            # Keep Discrete actions one-dimensional (n_envs,), as SB3's VecEnv
            # expects; adding a trailing singleton makes each env receive array([a]).
            action_array = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(action_array)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            rollout_buffer.add(
                self._last_obs,
                action_array,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=lstm_states,
                action_masks=masks,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones
            lstm_states = next_states
            self._last_lstm_states = next_states
            n_steps += 1
        with torch.no_grad():
            starts = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
            last_values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), next_states.vf, starts)
        rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=dones)
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        import torch.nn.functional as F
        from stable_baselines3.common.utils import explained_variance

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        losses: list[float] = []
        for _ in range(self.n_epochs):
            for data in self.rollout_buffer.get(self.batch_size):
                actions = data.actions.long().flatten()
                mask = data.mask > 1e-8
                values, log_prob, entropy = self.policy.evaluate_actions(
                    data.observations,
                    actions,
                    data.lstm_states,
                    data.episode_starts,
                    data.action_masks,
                )
                advantages = data.advantages
                if self.normalize_advantage and advantages.numel() > 1:
                    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)
                ratio = torch.exp(log_prob - data.old_log_prob)
                policy_loss = -torch.minimum(
                    advantages * ratio,
                    advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range),
                )[mask].mean()
                value_loss = F.mse_loss(data.returns[mask], values.flatten()[mask])
                entropy_loss = -entropy[mask].mean()
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                losses.append(float(loss.detach().cpu()))
            self._n_updates += 1
        self.logger.record("train/loss", float(np.mean(losses)))
        self.logger.record("train/explained_variance", explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        ))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
