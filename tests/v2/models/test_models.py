"""Contratos dos modelos universais e isolamento das entradas privilegiadas."""

import torch

from voidgym.models.perception import semantic_loss
from voidgym.models.policy import UniversalPolicy


def test_encoder_policy_and_semantic_shapes() -> None:
    model = UniversalPolicy()
    image = torch.zeros(2, 12, 108, 192, dtype=torch.uint8)
    history = torch.zeros(2, 3, dtype=torch.long)
    logits, value, state, sem, motion = model(image, history)
    assert logits.shape == (2, 1, 43)
    assert value.shape == (2, 1)
    assert state[0].shape == (1, 2, 256)
    assert sem.shape == (2, 10, 27, 48)
    assert motion.shape == (2, 1, 3)


def test_semantic_loss_ignores_255() -> None:
    logits = torch.randn(1, 10, 27, 48, requires_grad=True)
    labels = torch.full((1, 27, 48), 255)
    loss = semantic_loss(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()


def test_privileged_value_does_not_change_actor() -> None:
    model = UniversalPolicy().eval()
    image = torch.zeros(1, 12, 108, 192, dtype=torch.uint8)
    history = torch.zeros(1, 3, dtype=torch.long)
    mask = torch.ones(1, 43, dtype=torch.bool)
    no_priv = model(image, history, mask=mask)[0]
    yes_priv = model(image, history, mask=mask, privileged=torch.ones(1, 4))[0]
    torch.testing.assert_close(no_priv, yes_priv)


def test_auxiliary_gradients_reach_encoder_but_not_actor() -> None:
    model = UniversalPolicy()
    image = torch.randint(0, 255, (2, 12, 108, 192), dtype=torch.uint8)
    features, vector = model.encoder(image)
    (model.perception(features).mean() + model.hero_motion(vector).mean()).backward()
    assert model.encoder.blocks[0].conv.weight.grad is not None
    assert model.actors["explore"].weight.grad is None


def test_policy_accepts_recurrent_sequences() -> None:
    model = UniversalPolicy()
    pixels = torch.zeros(2, 4, 12, 108, 192, dtype=torch.uint8)
    history = torch.zeros(2, 4, 3, dtype=torch.long)
    mask = torch.ones(2, 4, 43, dtype=torch.bool)
    logits, value, _, semantic, motion = model(pixels, history, mask=mask)
    assert logits.shape == (2, 4, 43)
    assert value.shape == (2, 4)
    assert semantic.shape == (2, 4, 10, 27, 48)
    assert motion.shape == (2, 4, 3)


def test_episode_start_resets_recurrent_memory() -> None:
    model = UniversalPolicy().eval()
    pixels = torch.zeros(1, 4, 12, 108, 192, dtype=torch.uint8)
    pixels[:, 0] = 30
    pixels[:, 1] = 90
    pixels[:, 2:] = 170
    history = torch.zeros(1, 4, 3, dtype=torch.long)
    reset = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)
    with torch.no_grad():
        full = model(pixels, history, episode_starts=reset)[0]
        suffix = model(pixels[:, 2:], history[:, 2:])[0]
    torch.testing.assert_close(full[:, 2:], suffix)
