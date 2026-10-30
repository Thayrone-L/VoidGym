from pathlib import Path

import pytest
import torch

from voidgym.training.buffers import SequenceBatch
from voidgym.training.shared_rollouts import load_rollout, publish_rollout


def _sequence() -> SequenceBatch:
    return SequenceBatch(
        observations={
            "pixels": torch.zeros((1, 4, 108, 192), dtype=torch.uint8),
            "action_history": torch.zeros((1, 3), dtype=torch.int64),
            "game_extras": torch.tensor([[0.2, 0.8]], dtype=torch.float32),
        },
        actions=torch.tensor([[1]]), old_log_probs=torch.tensor([[-0.4]]),
        old_values=torch.tensor([[0.1]]), returns=torch.tensor([[0.5]]),
        advantages=torch.tensor([[0.4]]), action_masks=torch.tensor([[[True, True]]]),
        episode_starts=torch.tensor([[True]]),
        initial_state=(torch.zeros((1, 1, 256)), torch.zeros((1, 1, 256))),
        sem=torch.full((1, 27, 48), 255, dtype=torch.uint8),
        hero_motion=torch.full((1, 3), float("nan")),
        discontinuity=torch.zeros((1, 1), dtype=torch.bool),
        privileged=torch.zeros((1, 1, 4)), has_labels=torch.zeros((1, 1), dtype=torch.bool),
    )


def test_rollout_round_trip_and_contract_validation(tmp_path: Path) -> None:
    actions = ("left", "right")
    path = publish_rollout(tmp_path, round_id=3, actor_id="actor-a", policy_round=2,
                           action_table=actions, sequences=[_sequence()])
    loaded = load_rollout(path, expected_round=3, expected_actor="actor-a",
                          expected_policy_round=2, action_table=actions,
                          device=torch.device("cpu"))
    assert torch.equal(loaded[0].observations["pixels"], _sequence().observations["pixels"])
    assert torch.allclose(loaded[0].observations["game_extras"], torch.tensor([[0.2, 0.8]]))
    with pytest.raises(ValueError, match="contract"):
        load_rollout(path, expected_round=4, expected_actor="actor-a",
                     expected_policy_round=2, action_table=actions,
                     device=torch.device("cpu"))
