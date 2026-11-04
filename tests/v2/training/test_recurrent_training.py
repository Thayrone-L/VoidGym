from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from voidgym.core.actions import UNIVERSAL_ACTIONS
from voidgym.core.observation import UNIVERSAL_ACTION_HISTORY_KEY, UNIVERSAL_IMAGE_KEY
from voidgym.core.semantics import SEM_GRID
from voidgym.models.policy import UniversalPolicy
from voidgym.training.augmentations import random_shift
from voidgym.training.buffers import RecurrentRolloutBuffer
from voidgym.training.checkpoints import load_universal, save_universal
from voidgym.training.recurrent_ppo import RecurrentPPOTrainer


def _observation(count: int = 1) -> dict[str, torch.Tensor]:
    return {UNIVERSAL_IMAGE_KEY: torch.zeros((count, 12, 108, 192), dtype=torch.uint8),
            UNIVERSAL_ACTION_HISTORY_KEY: torch.zeros((count, 3), dtype=torch.long)}


def test_random_shift_moves_image_and_semantic_target_together(monkeypatch: pytest.MonkeyPatch) -> None:
    values = iter((torch.tensor([4]), torch.tensor([4])))
    monkeypatch.setattr(torch, "randint", lambda *_args, **_kwargs: next(values))
    images = torch.zeros((1, 2, 12, 108, 192), dtype=torch.uint8)
    images[:, :, :, 8:12, 8:12] = 255
    sem = torch.zeros((1, 2, *SEM_GRID), dtype=torch.uint8)
    sem[:, :, 2, 2] = 5
    shifted, moved = random_shift(images, sem)
    assert moved is not None
    assert torch.equal(shifted[:, :, :, 4:8, 4:8], torch.full((1, 2, 12, 4, 4), 255))
    assert torch.all(moved[:, :, 1, 1] == 5)


def test_buffer_labels_returns_and_recurrent_ppo_update_are_finite() -> None:
    device = torch.device("cpu")
    model = UniversalPolicy()
    trainer = RecurrentPPOTrainer(model, augment=False, c_sem=.5, c_inv=.1, c_mot=.1)
    buffer = RecurrentRolloutBuffer(steps=2, envs=1, sequence_length=2)
    state = model.initial_state(1, device)
    for step in range(2):
        obs = _observation()
        with torch.no_grad():
            output = model(obs, state=state, episode_starts=torch.tensor([step == 0]))
            distribution = torch.distributions.Categorical(logits=output["logits"])
            action = torch.tensor([0])
            log_prob = distribution.log_prob(action)
        info = {"training_labels": {"sem": np.zeros(SEM_GRID, np.uint8),
                                    "hero_velocity": np.zeros(2, np.float32), "on_ground": 1},
                "signals": {"player_pos_room": (.2, .3), "progress": .2, "hp": 5}}
        buffer.add(obs, action, torch.tensor([1.0]), torch.tensor([False]), output["values"],
                   log_prob, torch.ones((1, len(UNIVERSAL_ACTIONS)), dtype=torch.bool),
                   torch.tensor([step == 0]), state, [info])
        state = output["state"]
    buffer.compute_returns(torch.zeros(1), gamma=.995, gae_lambda=.95)
    assert buffer.sems[0].shape == (1, *SEM_GRID)
    metrics = trainer.update(buffer.sequences(device), epochs=1)
    assert metrics and all(np.isfinite(value) for value in metrics.values())


def test_gradient_accumulation_keeps_one_optimizer_step_per_effective_batch() -> None:
    device = torch.device("cpu")
    model = UniversalPolicy()
    trainer = RecurrentPPOTrainer(model, augment=False)
    buffer = RecurrentRolloutBuffer(steps=2, envs=2, sequence_length=2)
    state = model.initial_state(2, device)
    info = {"training_labels": {"sem": np.zeros(SEM_GRID, np.uint8),
                                 "hero_velocity": np.zeros(2, np.float32), "on_ground": 1},
            "signals": {"player_pos_room": (.2, .3), "progress": .2, "hp": 5}}
    for step in range(2):
        obs = _observation(2)
        starts = torch.full((2,), step == 0, dtype=torch.bool)
        with torch.no_grad():
            output = model(obs, state=state, episode_starts=starts)
            actions = torch.zeros(2, dtype=torch.long)
            log_probs = torch.distributions.Categorical(logits=output["logits"]).log_prob(actions)
        buffer.add(obs, actions, torch.ones(2), torch.zeros(2, dtype=torch.bool),
                   output["values"], log_probs,
                   torch.ones((2, len(UNIVERSAL_ACTIONS)), dtype=torch.bool), starts,
                   state, [info, info])
        state = output["state"]
    buffer.compute_returns(torch.zeros(2), gamma=.995, gae_lambda=.95)
    original_step = trainer.optimizer.step
    calls = 0

    def counted_step(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_step(*args, **kwargs)

    trainer.optimizer.step = counted_step
    metrics = trainer.update(buffer.sequences(device), epochs=1, sequence_batch_size=2,
                             microbatch_size=1)
    assert calls == 1
    assert metrics and all(np.isfinite(value) for value in metrics.values())


def test_component_checkpoint_roundtrip_and_action_table_guard(tmp_path) -> None:
    model = UniversalPolicy()
    checkpoint = tmp_path / "universal-test"
    save_universal(checkpoint, model, ["A", "B"], steps=12, metrics={"score": .5})
    restored = UniversalPolicy()
    manifest = load_universal(checkpoint, restored, ["A", "B"])
    assert manifest["trained_steps"] == 12
    with pytest.raises(ValueError, match="action table"):
        load_universal(checkpoint, restored, ["B", "A"])


def test_truncation_bootstraps_value_without_crossing_gae_boundary() -> None:
    buffer = RecurrentRolloutBuffer(steps=1, envs=1, sequence_length=1)
    state = (torch.zeros((1, 1, 2)), torch.zeros((1, 1, 2)))
    buffer.add(
        _observation(), torch.tensor([0]), torch.tensor([1.0]), torch.tensor([True]),
        torch.tensor([2.0]), torch.tensor([0.0]),
        torch.ones((1, len(UNIVERSAL_ACTIONS)), dtype=torch.bool),
        torch.tensor([False]), state,
        [{"signals": {"truncated": True}, "ppo_bootstrap_value": torch.tensor(7.0)}],
    )
    buffer.compute_returns(torch.tensor([99.0]), gamma=0.9, gae_lambda=0.95)
    torch.testing.assert_close(buffer.returns, torch.tensor([[7.3]]))
    torch.testing.assert_close(buffer.advantages, torch.tensor([[5.3]]))


def test_actor_policy_path_can_skip_auxiliary_heads() -> None:
    model = UniversalPolicy().eval()
    model.perception.forward = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("semantic head called"))
    model.semantic_skip.forward = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("semantic skip called"))
    model.hero_motion.forward = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("motion head called"))
    with torch.no_grad():
        output = model(_observation(), compute_auxiliary=False)
    assert {"logits", "values", "state"} <= set(output)
    assert "semantic_logits" not in output and "motion" not in output


def test_missing_oracle_marks_all_auxiliary_targets_invalid() -> None:
    model = UniversalPolicy()
    buffer = RecurrentRolloutBuffer(steps=1, envs=1, sequence_length=1)
    state = model.initial_state(1, torch.device("cpu"))
    buffer.add(_observation(), torch.tensor([0]), torch.tensor([0.0]), torch.tensor([False]),
               torch.tensor([0.0]), torch.tensor([0.0]),
               torch.ones((1, len(UNIVERSAL_ACTIONS)), dtype=torch.bool),
               torch.tensor([True]), state, [{}])
    sample = buffer.sems[0]
    assert sample.unique().tolist() == [255]
    assert not buffer.has_labels[0].any()


def test_actor_keeps_live_environment_across_stuck_slices(monkeypatch) -> None:
    from voidgym.training import actor as actor_module

    class FakePolicy:
        def __init__(self):
            self.calls = 0

        def __call__(self, _obs, state, *_args, **kwargs):
            assert kwargs["compute_auxiliary"] is False
            self.calls += 1
            value = 2.0 if self.calls % 2 else 7.0
            return {"logits": torch.zeros((1, 1)), "values": torch.tensor([value]),
                    "state": state}

        @staticmethod
        def initial_state(batch, device):
            return (torch.zeros((1, batch, 2), device=device),
                    torch.zeros((1, batch, 2), device=device))

    class FakeEnv:
        def __init__(self):
            self.actions = ("NOOP",)
            self.adapter = SimpleNamespace(game_id="hollow_knight",
                                           frame_rgb=lambda: np.zeros((108, 192, 3), np.uint8))
            self.steps = 0
            self.reset_calls = 0

        def action_masks(self):
            return np.ones(1, dtype=np.bool_)

        def step(self, _action):
            self.steps += 1
            return _observation_numpy(), 0.0, False, True, {
                "signals": {"truncated": True, "termination_reason": "stuck"},
                "training_labels": None,
            }

        def reset(self, *, seed=None):
            self.reset_calls += 1
            return _observation_numpy(), {"signals": {}, "training_labels": None}

    class Writer:
        def __init__(self):
            self.rows = []

        def append(self, row):
            self.rows.append(row)

    monkeypatch.setattr(
        actor_module, "_forward",
        lambda policy, *args, **kwargs: policy(*args, compute_auxiliary=False, **kwargs),
    )
    env, policy, writer = FakeEnv(), FakePolicy(), Writer()
    buffer = RecurrentRolloutBuffer(steps=2, envs=1, sequence_length=1)
    state = policy.initial_state(1, torch.device("cpu"))
    actor_module._collect_steps(
        env, policy, buffer, writer, actor_id="test", policy_round=0,
        observation=_observation_numpy(), info={"signals": {}, "training_labels": None},
        state=state, episode_start=True, total_steps=0, episode_return=0.0,
        round_returns=[], device=torch.device("cpu"), seed=1, stop_file=None,
    )
    assert env.steps == 2
    assert env.reset_calls == 0
    assert [float(value.item()) for value in buffer.bootstrap_values] == [7.0, 7.0]
    assert writer.rows[0]["truncated"] and not writer.rows[0]["done"]


def _observation_numpy() -> dict[str, np.ndarray]:
    return {"pixels": np.zeros((12, 108, 192), np.uint8),
            "action_history": np.zeros(3, np.int64)}
