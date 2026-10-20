from __future__ import annotations

import numpy as np
import pytest
import torch

from hk_agent.maskable_recurrent import (
    MaskableRecurrentActorCritic,
    RecurrentBoundary,
    RecurrentStateTracker,
    compatible_warm_start,
)


def test_maskable_lstm_never_selects_an_unavailable_action() -> None:
    policy = MaskableRecurrentActorCritic(feature_size=5, action_count=4, hidden_size=8)
    features = torch.randn(7, 2, 5)
    starts = torch.zeros(7, 2)
    masks = torch.tensor([True, False, True, False]).expand(7, 2, 4).clone()
    logits, values, _ = policy.forward_sequence(
        features, policy.initial_state(2), starts, masks
    )
    actions = torch.distributions.Categorical(logits=logits).sample()
    selected_masks = masks.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    assert selected_masks.all()
    assert values.shape == (7, 2)
    assert torch.isfinite(logits).all()


def test_hidden_state_persists_across_ppo_chunks_and_resets_at_real_boundaries() -> None:
    tracker = RecurrentStateTracker(layers=1, batch_size=2, hidden_size=4)
    state = tracker.state()
    tracker.update(type(state)(torch.ones_like(state.hidden), torch.ones_like(state.cell)))
    tracker.boundary(RecurrentBoundary.PPO_CHUNK)
    assert not tracker.episode_start.any()
    assert torch.equal(tracker.state().hidden, torch.ones_like(state.hidden))
    tracker.boundary(RecurrentBoundary.SAVE_RELOADED)
    assert tracker.episode_start.all()
    assert torch.equal(tracker.state().hidden, torch.zeros_like(state.hidden))


def test_episode_start_resets_lstm_state_inside_sequence() -> None:
    torch.manual_seed(17)
    policy = MaskableRecurrentActorCritic(3, 2, hidden_size=6)
    features = torch.randn(4, 1, 3)
    masks = torch.ones(4, 1, 2, dtype=torch.bool)
    starts = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    state = policy.initial_state(1)
    logits, _, _ = policy.forward_sequence(features, state, starts, masks)
    isolated, _, _ = policy.forward_sequence(
        features[2:], policy.initial_state(1), torch.tensor([[1.0], [0.0]]), masks[2:]
    )
    torch.testing.assert_close(logits[2:], isolated)


def test_recurrent_sequence_rejects_empty_or_malformed_action_masks() -> None:
    policy = MaskableRecurrentActorCritic(2, 3, hidden_size=4)
    features = torch.zeros(2, 1, 2)
    starts = torch.zeros(2, 1)
    state = policy.initial_state(1)
    with pytest.raises(ValueError, match="at least one legal"):
        policy.forward_sequence(features, state, starts, torch.zeros(2, 1, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="action_masks"):
        policy.forward_sequence(features, state, starts, torch.ones(2, 1, 2, dtype=torch.bool))


def test_warm_start_copies_only_compatible_feature_extractor_tensors() -> None:
    source = torch.nn.Module()
    source.features_extractor = torch.nn.Linear(3, 4)
    source.lstm = torch.nn.Linear(4, 4)
    target = torch.nn.Module()
    target.features_extractor = torch.nn.Linear(3, 4)
    target.lstm = torch.nn.Linear(4, 4)
    with torch.no_grad():
        source.features_extractor.weight.fill_(0.75)
        source.lstm.weight.fill_(0.25)
    copied = compatible_warm_start(target, source.state_dict())
    assert set(copied) == {"features_extractor.weight", "features_extractor.bias"}
    torch.testing.assert_close(target.features_extractor.weight, source.features_extractor.weight)
    assert not torch.equal(target.lstm.weight, source.lstm.weight)


def test_tracker_returns_valid_numpy_episode_flags() -> None:
    tracker = RecurrentStateTracker(layers=1, batch_size=1, hidden_size=2)
    flags = tracker.episode_start
    assert isinstance(flags, np.ndarray)
    assert flags.dtype == np.bool_
    assert flags.tolist() == [True]


def test_delayed_cue_requires_lstm_memory_and_is_learnable() -> None:
    torch.manual_seed(23)
    np.random.seed(23)
    policy = MaskableRecurrentActorCritic(2, 2, hidden_size=12)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.02)
    cue = torch.randint(0, 2, (64,))
    features = torch.zeros(6, 64, 2)
    features[0, torch.arange(64), cue] = 1.0
    masks = torch.ones(6, 64, 2, dtype=torch.bool)
    starts = torch.zeros(6, 64)
    starts[0] = 1.0
    labels = cue
    for _ in range(90):
        logits, _, _ = policy.forward_sequence(features, policy.initial_state(64), starts, masks)
        loss = torch.nn.functional.cross_entropy(logits[-1], labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits, _, _ = policy.forward_sequence(features, policy.initial_state(64), starts, masks)
    assert (logits[-1].argmax(-1) == labels).float().mean().item() >= 0.95
    # At the decision time the feed-forward input is identical for both labels.
    assert torch.count_nonzero(features[-1]).item() == 0
    assert torch.unique(labels).numel() == 2


def test_recurrent_rollout_serialization_and_bptt_update(tmp_path) -> None:
    from hk_agent.maskable_recurrent import RecurrentRollout, train_recurrent_ppo_sequences

    steps, batch, feature_size, actions, hidden = 4, 2, 3, 3, 5
    features = np.random.default_rng(2).normal(size=(steps, batch, feature_size)).astype(np.float32)
    masks = np.ones((steps, batch, actions), dtype=np.bool_)
    selected = np.zeros((steps, batch), dtype=np.int64)
    rollout = RecurrentRollout(
        features=features,
        actions=selected,
        action_masks=masks,
        episode_starts=np.zeros((steps, batch), dtype=np.float32),
        initial_hidden=np.zeros((1, batch, hidden), dtype=np.float32),
        initial_cell=np.zeros((1, batch, hidden), dtype=np.float32),
        values=np.zeros((steps, batch), dtype=np.float32),
        log_probs=np.full((steps, batch), -1.1, dtype=np.float32),
        advantages=np.ones((steps, batch), dtype=np.float32),
        returns=np.ones((steps, batch), dtype=np.float32),
        hidden_states=np.zeros((steps, 1, batch, hidden), dtype=np.float32),
        cell_states=np.zeros((steps, 1, batch, hidden), dtype=np.float32),
    )
    path = tmp_path / "sequence.npz"
    rollout.save(str(path))
    loaded = RecurrentRollout.load(str(path))
    np.testing.assert_array_equal(loaded.action_masks, masks)
    np.testing.assert_array_equal(loaded.hidden_states, rollout.hidden_states)
    policy = MaskableRecurrentActorCritic(feature_size, actions, hidden_size=hidden)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    metrics = train_recurrent_ppo_sequences(policy, optimizer, loaded, sequence_length=2, epochs=1)
    assert metrics["sequences"] == 4.0
    assert metrics["train_seconds"] >= 0.0



def test_sb3_maskable_recurrent_ppo_collects_and_updates_a_rollout() -> None:
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv

    from hk_agent.recurrent_sb3 import MaskableRecurrentPPO

    class TinyMaskedEnv(gym.Env):
        observation_space = gym.spaces.Dict({"state": gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)})
        action_space = gym.spaces.Discrete(3)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return {"state": np.zeros(3, dtype=np.float32)}, {}

        def step(self, action):
            assert np.asarray(action).ndim == 0
            return {"state": np.full(3, action / 2.0, dtype=np.float32)}, 0.0, False, False, {}

        def action_masks(self):
            return np.array([True, False, True], dtype=np.bool_)

    vector = DummyVecEnv([TinyMaskedEnv])
    model = MaskableRecurrentPPO(
        "MultiInputLstmPolicy", vector, n_steps=8, batch_size=8, n_epochs=1, sequence_length=4,
        policy_kwargs={"lstm_hidden_size": 8}, device="cpu", verbose=0,
    )
    model.learn(total_timesteps=8)
    assert model.num_timesteps == 8
    from hk_agent.recurrent_sb3 import MaskableRecurrentDictRolloutBuffer
    assert isinstance(model.rollout_buffer, MaskableRecurrentDictRolloutBuffer)
    assert model.rollout_buffer.sequence_length == 4
    vector.close()


def test_sb3_recurrent_policy_normalizes_numpy_observation_dimensions() -> None:
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv

    from hk_agent.recurrent_sb3 import MaskableRecurrentPPO

    class NumpyDimensionEnv(gym.Env):
        observation_space = gym.spaces.Dict({
            "state": gym.spaces.MultiDiscrete(np.array([2, 3], dtype=np.int64)),
        })
        action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return {"state": np.array([0, 0], dtype=np.int64)}, {}

        def step(self, action):
            return {"state": np.array([0, 0], dtype=np.int64)}, 0.0, False, False, {}

        def action_masks(self):
            return np.ones(2, dtype=np.bool_)

    vector = DummyVecEnv([NumpyDimensionEnv])
    model = MaskableRecurrentPPO(
        "MultiInputLstmPolicy", vector, n_steps=4, batch_size=4, n_epochs=1,
        policy_kwargs={"lstm_hidden_size": 8}, device="cpu", verbose=0,
    )
    assert type(model.policy.lstm_actor.input_size) is int
    vector.close()


def test_shared_learner_trains_from_recurrent_state_rollouts() -> None:
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv

    from hk_agent.recurrent_sb3 import MaskableRecurrentPPO
    from hk_agent.shared_learner import train_shared_round

    class TinyMaskedEnv(gym.Env):
        observation_space = gym.spaces.Dict({"state": gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)})
        action_space = gym.spaces.Discrete(3)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return {"state": np.zeros(3, dtype=np.float32)}, {}

        def step(self, action):
            return {"state": np.zeros(3, dtype=np.float32)}, 0.0, False, False, {}

        def action_masks(self):
            return np.array([True, False, True], dtype=np.bool_)

    env = DummyVecEnv([TinyMaskedEnv])
    model = MaskableRecurrentPPO(
        "MultiInputLstmPolicy", env, n_steps=4, batch_size=4, n_epochs=1,
        sequence_length=2, policy_kwargs={"lstm_hidden_size": 8}, device="cpu", verbose=0,
    )
    model._setup_learn(total_timesteps=8, callback=lambda _locals, _globals: True, reset_num_timesteps=True)
    steps, hidden = 4, 8
    rollouts = []
    for slot in range(2):
        rollouts.append({
            "obs__state": np.full((steps, 3), slot / 2, dtype=np.float32),
            "actions": np.zeros((steps,), dtype=np.int64),
            "rewards": np.zeros((steps,), dtype=np.float32),
            "episode_starts": np.zeros((steps,), dtype=np.float32),
            "values": np.zeros((steps,), dtype=np.float32),
            "log_probs": np.full((steps,), -0.7, dtype=np.float32),
            "advantages": np.ones((steps,), dtype=np.float32),
            "returns": np.ones((steps,), dtype=np.float32),
            "action_masks": np.tile(np.array([True, False, True]), (steps, 1)),
            "lstm_hidden_pi": np.zeros((steps, 1, hidden), dtype=np.float32),
            "lstm_cell_pi": np.zeros((steps, 1, hidden), dtype=np.float32),
            "lstm_hidden_vf": np.zeros((steps, 1, hidden), dtype=np.float32),
            "lstm_cell_vf": np.zeros((steps, 1, hidden), dtype=np.float32),
        })
    assert train_shared_round(model, rollouts) == steps * 2
    assert model.num_timesteps == steps * 2
    assert model.rollout_buffer.sequence_length == 2
    env.close()


def test_self_imitation_updates_recurrent_policy_with_action_masks() -> None:
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv

    from hk_agent.recurrent_sb3 import MaskableRecurrentPPO
    from hk_agent.self_imitation import train_sil_update

    class TinyMaskedEnv(gym.Env):
        observation_space = gym.spaces.Dict({"state": gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)})
        action_space = gym.spaces.Discrete(3)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return {"state": np.zeros(3, dtype=np.float32)}, {}

        def step(self, action):
            return {"state": np.zeros(3, dtype=np.float32)}, 0.0, False, False, {}

        def action_masks(self):
            return np.array([True, False, True], dtype=np.bool_)

    env = DummyVecEnv([TinyMaskedEnv])
    model = MaskableRecurrentPPO(
        "MultiInputLstmPolicy", env, n_steps=4, batch_size=4, n_epochs=1,
        sequence_length=4, policy_kwargs={"lstm_hidden_size": 8}, device="cpu", verbose=0,
    )
    count, hidden = 4, 8
    sample = {
        "obs__state": np.zeros((count, 3), dtype=np.float32),
        "actions": np.zeros(count, dtype=np.int64),
        "action_masks": np.tile(np.array([True, False, True]), (count, 1)),
        "episode_starts": np.zeros(count, dtype=np.float32),
        "returns": np.ones(count, dtype=np.float32),
        "log_probs": np.full(count, -0.69, dtype=np.float32),
        "lstm_hidden_pi": np.zeros((1, hidden), dtype=np.float32),
        "lstm_cell_pi": np.zeros((1, hidden), dtype=np.float32),
        "lstm_hidden_vf": np.zeros((1, hidden), dtype=np.float32),
        "lstm_cell_vf": np.zeros((1, hidden), dtype=np.float32),
    }
    result = train_sil_update(model, [sample])
    assert result["active"] is True
    assert result["positive_samples"] > 0
    sample["actions"][:] = 1
    assert train_sil_update(model, [sample])["reason"] == "invalid_action_mask"
    env.close()


def test_masked_distribution_matches_sb3_maskable_categorical() -> None:
    from sb3_contrib.common.maskable.distributions import MaskableCategorical

    logits = torch.tensor([[0.2, -0.4, 1.1, 0.7], [-0.1, 0.5, 0.2, -0.7]])
    masks = torch.tensor([[True, False, True, False], [False, True, True, True]])
    our_logits = logits.masked_fill(~masks, torch.finfo(logits.dtype).min)
    ours = torch.distributions.Categorical(logits=our_logits)
    reference = MaskableCategorical(logits=logits, masks=masks)
    torch.testing.assert_close(ours.probs, reference.probs)
    assert torch.equal(ours.probs.argmax(-1), reference.probs.argmax(-1))


def test_recurrent_chunk_state_persists_and_checkpoint_reloads(tmp_path) -> None:
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv

    from hk_agent.recurrent_sb3 import MaskableRecurrentPPO

    class PersistentEnv(gym.Env):
        observation_space = gym.spaces.Dict({"state": gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)})
        action_space = gym.spaces.Discrete(3)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return {"state": np.full(3, 0.2, dtype=np.float32)}, {}

        def step(self, action):
            return {"state": np.full(3, 0.4, dtype=np.float32)}, 0.0, False, False, {}

        def action_masks(self):
            return np.array([True, False, True], dtype=np.bool_)

    env = DummyVecEnv([PersistentEnv])
    model = MaskableRecurrentPPO(
        "MultiInputLstmPolicy", env, n_steps=4, batch_size=4, n_epochs=1,
        sequence_length=2, policy_kwargs={"lstm_hidden_size": 8}, device="cpu", verbose=0,
    )
    model.learn(total_timesteps=4)
    first_state = model._last_lstm_states.pi[0].detach().clone()
    assert torch.count_nonzero(first_state).item() > 0
    model.learn(total_timesteps=4, reset_num_timesteps=False)
    second_state = model._last_lstm_states.pi[0].detach().clone()
    assert torch.count_nonzero(second_state).item() > 0
    assert not torch.equal(first_state, second_state)
    path = tmp_path / "recurrent.zip"
    model.save(path)
    restored = MaskableRecurrentPPO.load(path, env=env, device="cpu")
    assert restored.sequence_length == 2
    for name, tensor in model.policy.state_dict().items():
        torch.testing.assert_close(restored.policy.state_dict()[name], tensor)
    env.close()


def test_shared_rollout_roundtrip_preserves_mask_and_lstm_states(tmp_path) -> None:
    import gymnasium as gym
    from sb3_contrib.common.recurrent.type_aliases import RNNStates

    from hk_agent.recurrent_sb3 import MaskableRecurrentDictRolloutBuffer
    from hk_agent.shared_learner import SharedRoundStore

    obs_space = gym.spaces.Dict({"state": gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)})
    action_space = gym.spaces.Discrete(3)
    buffer = MaskableRecurrentDictRolloutBuffer(
        buffer_size=2, observation_space=obs_space, action_space=action_space,
        hidden_state_shape=(2, 1, 1, 4), device="cpu", n_envs=1,
    )
    buffer.reset()
    states = RNNStates(
        (torch.zeros(1, 1, 4), torch.zeros(1, 1, 4)),
        (torch.zeros(1, 1, 4), torch.zeros(1, 1, 4)),
    )
    for idx in range(2):
        buffer.add(
            {"state": np.full((1, 3), idx / 2, dtype=np.float32)},
            np.array([[0]], dtype=np.int64), np.array([0.0], dtype=np.float32),
            np.array([idx == 0], dtype=np.float32), torch.zeros(1), torch.zeros(1),
            lstm_states=states, action_masks=np.array([[True, False, True]], dtype=np.bool_),
        )
    store = SharedRoundStore(tmp_path)
    fingerprint = "a" * 64
    store.publish_rollout(
        round_number=1, slot=0, launch_id="test-actor", buffer=buffer, fingerprint=fingerprint
    )
    loaded = store.read_rollout(
        round_number=1, slot=0, fingerprint=fingerprint, n_steps=2,
        observation_space=obs_space, action_count=3,
    )
    for key in ("lstm_hidden_pi", "lstm_cell_pi", "lstm_hidden_vf", "lstm_cell_vf"):
        np.testing.assert_array_equal(loaded[key], np.zeros((2, 1, 4), dtype=np.float32))
    np.testing.assert_array_equal(loaded["action_masks"], np.tile([True, False, True], (2, 1)))
