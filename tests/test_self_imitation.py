from __future__ import annotations

import numpy as np

from hk_agent.self_imitation import SelfImitationBank, sil_guard_reason, sil_positive_advantage


def _rollout() -> dict[str, np.ndarray]:
    count = 64
    row: dict[str, np.ndarray] = {
        "obs__pixels": np.zeros((count, 2), dtype=np.uint8),
        "actions": np.zeros(count, dtype=np.int64),
        "action_masks": np.ones((count, 2), dtype=np.bool_),
        "episode_starts": np.zeros(count, dtype=np.float32),
        "values": np.zeros(count, dtype=np.float32),
        "returns": np.ones(count, dtype=np.float32),
        "log_probs": np.zeros(count, dtype=np.float32),
    }
    for name in ("lstm_hidden_pi", "lstm_cell_pi", "lstm_hidden_vf", "lstm_cell_vf"):
        row[name] = np.zeros((count, 1, 3), dtype=np.float32)
    return row


def test_bank_persists_and_only_stores_positive_advantage(tmp_path) -> None:
    bank = SelfImitationBank(tmp_path / "self-imitation", max_steps=32)
    rollout = _rollout()
    assert bank.ingest([rollout], round_number=4, gamma=0.99) == 2
    assert bank.summary()["sample_count"] == 1
    restarted = SelfImitationBank(tmp_path / "self-imitation", max_steps=32)
    sample = restarted.load()[0]
    assert sample["actions"].shape == (32,)
    assert sample["action_masks"].all()
    rollout["returns"][:] = -1
    assert restarted.ingest([rollout], round_number=5, gamma=0.99) == 0
    assert restarted.summary()["sample_count"] == 1


def test_bank_respects_step_and_byte_caps(tmp_path) -> None:
    bank = SelfImitationBank(tmp_path / "self-imitation", max_steps=32, max_bytes=1_000_000)
    bank.ingest([_rollout()], round_number=1, gamma=0.99)
    bank.ingest([_rollout()], round_number=2, gamma=0.99)
    assert bank.summary()["step_count"] <= 32
    assert bank.summary()["sample_count"] == 1


def test_sil_advantage_is_zero_when_return_does_not_beat_value() -> None:
    import torch

    result = sil_positive_advantage(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]))
    assert torch.equal(result, torch.zeros(2))


def test_sil_entropy_and_kl_guards() -> None:
    assert sil_guard_reason(0.01, 0.0, entropy_floor=0.02, kl_limit=0.05) == "entropy_below_floor"
    assert sil_guard_reason(0.5, 0.06, entropy_floor=0.02, kl_limit=0.05) == "kl_above_limit"
    assert sil_guard_reason(0.5, 0.01, entropy_floor=0.02, kl_limit=0.05) is None
