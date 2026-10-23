from __future__ import annotations

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


def _observation() -> dict[str, torch.Tensor]:
    return {UNIVERSAL_IMAGE_KEY: torch.zeros((1, 12, 108, 192), dtype=torch.uint8),
            UNIVERSAL_ACTION_HISTORY_KEY: torch.zeros((1, 3), dtype=torch.long)}


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


def test_component_checkpoint_roundtrip_and_action_table_guard(tmp_path) -> None:
    model = UniversalPolicy()
    checkpoint = tmp_path / "universal-test"
    save_universal(checkpoint, model, ["A", "B"], steps=12, metrics={"score": .5})
    restored = UniversalPolicy()
    manifest = load_universal(checkpoint, restored, ["A", "B"])
    assert manifest["trained_steps"] == 12
    with pytest.raises(ValueError, match="action table"):
        load_universal(checkpoint, restored, ["B", "A"])


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
