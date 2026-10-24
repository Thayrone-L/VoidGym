from __future__ import annotations

import numpy as np
import torch

from voidgym.models.policy import UniversalPolicy
from voidgym.training.self_imitation import (
    SelfImitationBank,
    sil_guard_reason,
    sil_positive_advantage,
    train_sil_update,
)


def test_bank_persists_positive_universal_sequences(tmp_path) -> None:
    bank = SelfImitationBank(tmp_path / "sil", max_steps=32)
    count = 32
    rollout = {
        "obs__pixels": np.zeros((count, 12, 108, 192), dtype=np.uint8),
        "obs__action_history": np.zeros((count, 3), dtype=np.int64),
        "actions": np.zeros(count, dtype=np.int64),
        "action_masks": np.ones((count, 2), dtype=np.bool_),
        "episode_starts": np.zeros(count, dtype=np.bool_),
        "values": np.zeros(count, dtype=np.float32),
        "returns": np.ones(count, dtype=np.float32),
        "log_probs": np.zeros(count, dtype=np.float32),
        "state_h": np.zeros((count, 1, 16), dtype=np.float32),
        "state_c": np.zeros((count, 1, 16), dtype=np.float32),
        "privileged": np.zeros((count, 4), dtype=np.float32),
    }

    assert bank.ingest([rollout], round_number=1, gamma=.995) == 1
    sample = SelfImitationBank(tmp_path / "sil").load()[0]
    assert sample["obs__pixels"].shape == (32, 12, 108, 192)
    assert sample["state_h"].shape == (1, 16)
    assert sample["privileged"].shape == (32, 4)


def test_universal_policy_sil_update_is_masked_and_finite() -> None:
    count, actions = 2, 3
    model = UniversalPolicy(action_count=actions, hidden_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    sample = {
        "obs__pixels": np.zeros((count, 12, 108, 192), dtype=np.uint8),
        "obs__action_history": np.zeros((count, 3), dtype=np.int64),
        "actions": np.zeros(count, dtype=np.int64),
        "action_masks": np.ones((count, actions), dtype=np.bool_),
        "episode_starts": np.zeros(count, dtype=np.bool_),
        "returns": np.ones(count, dtype=np.float32),
        "log_probs": np.zeros(count, dtype=np.float32),
        "state_h": np.zeros((1, 16), dtype=np.float32),
        "state_c": np.zeros((1, 16), dtype=np.float32),
        "privileged": np.zeros((count, 4), dtype=np.float32),
    }

    result = train_sil_update(
        model, optimizer, [sample], entropy_floor=0, kl_limit=100,
    )
    assert result["active"] is True
    assert np.isfinite(result["weight"])
    assert result["positive_samples"] == count


def test_sil_advantage_and_collapse_guards() -> None:
    assert torch.equal(
        sil_positive_advantage(torch.tensor([1., 2.]), torch.tensor([1., 3.])),
        torch.zeros(2),
    )
    assert sil_guard_reason(.01, 0, entropy_floor=.02, kl_limit=.05) == "entropy_below_floor"
    assert sil_guard_reason(.5, .06, entropy_floor=.02, kl_limit=.05) == "kl_above_limit"
    assert sil_guard_reason(.5, .01, entropy_floor=.02, kl_limit=.05) is None
