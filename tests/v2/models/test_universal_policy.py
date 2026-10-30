from __future__ import annotations

import torch

from voidgym.core.actions import UNIVERSAL_ACTIONS
from voidgym.core.observation import FRAME_STACK, UNIVERSAL_ACTION_HISTORY_KEY, UNIVERSAL_IMAGE_KEY
from voidgym.core.semantics import IGNORE, SEM_GRID, Sem
from voidgym.models.encoder import ImpalaEncoder
from voidgym.models.perception import SemanticHead, semantic_loss
from voidgym.models.policy import UniversalPolicy


def _obs(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        UNIVERSAL_IMAGE_KEY: torch.randint(0, 256, (batch, FRAME_STACK * 3, 108, 192), dtype=torch.uint8),
        UNIVERSAL_ACTION_HISTORY_KEY: torch.zeros((batch, 3), dtype=torch.int64),
    }


def test_encoder_and_semantic_head_shapes_and_ignore_loss() -> None:
    encoder = ImpalaEncoder()
    features, vector = encoder(_obs()[UNIVERSAL_IMAGE_KEY])
    assert features.shape == (2, 32, 14, 24)
    assert vector.shape == (2, 256)
    logits = SemanticHead()(features)
    assert logits.shape == (2, len(Sem), *SEM_GRID)
    targets = torch.full((2, *SEM_GRID), IGNORE, dtype=torch.uint8)
    loss = semantic_loss(logits, targets)
    assert torch.isfinite(loss) and loss.item() == 0
    targets[:, 4, 5] = int(Sem.PLAYER)
    assert torch.isfinite(semantic_loss(logits, targets))


def test_inverse_and_motion_auxiliary_losses_backpropagate_to_features_only() -> None:
    policy = UniversalPolicy()
    image_features, vector = policy.encoder(_obs()[UNIVERSAL_IMAGE_KEY])
    sem_loss = semantic_loss(policy.perception(image_features),
                             torch.zeros((2, *SEM_GRID), dtype=torch.long))
    inv_logits = policy.inverse_dynamics(vector, vector.detach())
    inv_loss = policy.inverse_dynamics.loss(inv_logits, torch.tensor([1, 2]),
                                            torch.tensor([False, True]))
    motion, ground = policy.hero_motion(vector)
    motion_loss = policy.hero_motion.loss(motion, ground, torch.zeros((2, 3)))
    (sem_loss + inv_loss + motion_loss).backward()
    assert policy.encoder.blocks[0].conv.weight.grad is not None
    assert policy.inverse_dynamics.net[0].weight.grad is not None
    assert policy.hero_motion.net[0].weight.grad is not None
    assert all(param.grad is None for param in policy.actor.parameters())


def test_asymmetric_critic_extras_and_action_mask() -> None:
    policy = UniversalPolicy()
    obs = _obs(1)
    output_zero = policy(obs, privileged=torch.zeros((1, 4)))
    output_ones = policy(obs, privileged=torch.ones((1, 4)))
    assert torch.equal(torch.softmax(output_zero["logits"], -1),
                       torch.softmax(output_ones["logits"], -1))
    assert not torch.equal(output_zero["values"], output_ones["values"])
    policy.add_game_extras("test_game", 3)
    with_extras = policy(obs, game_id="test_game", game_extras=torch.randn(1, 3))
    assert torch.equal(output_zero["logits"], with_extras["logits"])
    mask = torch.ones((1, len(UNIVERSAL_ACTIONS)), dtype=torch.bool)
    mask[:, 1] = False
    masked = policy(obs, action_mask=mask)
    assert masked["logits"][0, 1].item() == -1e9
    assert len(policy.actor) == 1 and "explore" in policy.actor
