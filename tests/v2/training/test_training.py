"""Perdas auxiliares, aumento acoplado e publicação de checkpoint."""

import numpy as np
import torch

from voidgym.models.policy import UniversalPolicy
from voidgym.training.buffers import TrainingBuffer
from voidgym.training.checkpoints import load_universal, save_universal
from voidgym.training.recurrent_ppo import RecurrentPPO, augment_sequence, auxiliary_objective


def test_shift_moves_image_and_semantics_together() -> None:
    image = torch.zeros(1, 3, 108, 192, dtype=torch.uint8)
    image[..., 40:44, 40:44] = 255
    sem = torch.zeros(1, 27, 48, dtype=torch.uint8)
    sem[..., 10, 10] = 2
    moved, moved_sem = augment_sequence(image, sem, 4, 0)
    assert moved[..., 40:44, 44:48].all()
    assert moved_sem[..., 10, 11].item() == 2


def test_checkpoint_saves_in_parts_and_checks_action_table(tmp_path) -> None:
    from voidgym.core.actions import UNIVERSAL_ACTIONS

    table = tuple(action.name for action in UNIVERSAL_ACTIONS)
    model = UniversalPolicy(len(table))
    path = save_universal(model, tmp_path / "universal-test", 12)
    loaded = load_universal(path, table)
    assert loaded.actors["explore"].weight.shape == model.actors["explore"].weight.shape
    try:
        load_universal(path, tuple(reversed(table)))
    except ValueError:
        pass
    else:
        raise AssertionError("incompatible action table was accepted")


def test_missing_labels_make_finite_zero_auxiliary_losses() -> None:
    semantic = torch.randn(2, 10, 27, 48, requires_grad=True)
    inverse = torch.randn(2, 43, requires_grad=True)
    motion = torch.randn(2, 3, requires_grad=True)
    total, metrics = auxiliary_objective(
        semantic, torch.full((2, 27, 48), 255), inverse, torch.zeros(2, dtype=torch.long),
        motion, torch.full((2, 3), float("nan")), torch.ones(2, dtype=torch.bool),
    )
    assert torch.isfinite(total)
    assert all(value.item() == 0 for value in metrics.values())
    total.backward()


def test_training_buffer_marks_missing_and_partial_ground_truth() -> None:
    buffer = TrainingBuffer()
    image = np.zeros((12, 108, 192), dtype=np.uint8)
    buffer.add(image, 3, {"discontinuity": True})
    buffer.add(image, 4, {"training_labels": {"sem": np.zeros((27, 48), dtype=np.uint8)}})
    assert buffer.labeled == [False, True]
    assert np.all(buffer.semantic[0] == 255)
    assert np.all(np.isnan(buffer.hero_motion[0]))
    assert np.all(np.isnan(buffer.hero_motion[1]))
    assert buffer.discontinuity == [True, False]
    assert np.all(np.isnan(buffer.privileged[0]))


def test_optimizer_runs_one_finite_update() -> None:
    model = UniversalPolicy()
    trainer = RecurrentPPO(model)
    loss = sum(parameter.square().mean() for parameter in model.parameters())
    assert np.isfinite(trainer.update(loss))
