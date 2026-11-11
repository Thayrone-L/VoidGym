from __future__ import annotations

import io
import json

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from voidgym.training.demonstrations import (
    DemonstrationRecorder,
    apply_verified_demonstrations,
    context_artifact_directory,
    demonstration_batch,
)
from voidgym.models.policy import UniversalPolicy
from voidgym.training.demonstrations import train_demonstration_update


def test_verified_demonstration_becomes_universal_observations(tmp_path) -> None:
    output = io.BytesIO()
    Image.new("RGB", (16, 12), color=(80, 120, 160)).save(output, format="PNG")
    context = "traversal:Tutorial_01:cell-2-3"
    recorder = DemonstrationRecorder(
        context_artifact_directory(tmp_path, context),
        game_version="test-v1", context=context,
    )
    recorder.authorize(help_request_id="help-req-1")
    mask = np.ones(3, dtype=np.bool_)
    extras = np.asarray([1., 2., 3., 4.], dtype=np.float32)
    for action in (0, 1):
        recorder.append(
            png_bytes=output.getvalue(), action=action, action_mask=mask,
            player_knowledge=extras,
            action_history=np.asarray([0, 0, action], dtype=np.int64),
        )
    artifact = recorder.finish()
    observation_space = gym.spaces.Dict({
        "pixels": gym.spaces.Box(0, 255, shape=(12, 108, 192), dtype=np.uint8),
        "action_history": gym.spaces.MultiDiscrete(np.full(3, 256, dtype=np.int64)),
        "game_extras": gym.spaces.Box(-10, 10, shape=(4,), dtype=np.float32),
    })

    batch = demonstration_batch(
        artifact, expected_game_version="test-v1",
        expected_context=context, observation_space=observation_space,
        action_count=3, game_extras_mask=np.asarray([True, False, True, True]),
    )
    assert batch["pixels"].shape == (2, 12, 108, 192)
    assert batch["action_history"].shape == (2, 3)
    assert batch["game_extras"].tolist() == [[1., 0., 3., 4.], [1., 0., 3., 4.]]
    assert batch["actions"].tolist() == [0, 1]

    ledger = tmp_path / "help-requests" / "slot-0.json"
    ledger.parent.mkdir(parents=True)
    row = {"request_id": "help-req-1", "context": context, "status": "pending"}
    ledger.write_text(json.dumps({"requests": [row]}), encoding="utf-8")
    policy = UniversalPolicy(action_count=3, hidden_size=16)
    policy.add_game_extras("hollow_knight", 4)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    kwargs = {"actions": ("LEFT", "RIGHT", "NOOP"), "game_id": "hollow_knight", "round_number": 1}
    assert apply_verified_demonstrations(tmp_path, policy, optimizer, **kwargs)["demo_active"] == 0
    row["status"] = "resolved_with_help"
    ledger.write_text(json.dumps({"requests": [row]}), encoding="utf-8")
    trained = apply_verified_demonstrations(tmp_path, policy, optimizer, **kwargs)
    assert trained["demo_active"] == 1
    assert trained["demo_samples"] == 2
    assert np.isfinite(trained["demo_loss"])
    assert apply_verified_demonstrations(tmp_path, policy, optimizer, **kwargs)["demo_active"] == 0


def test_demonstration_update_trains_the_universal_actor_head() -> None:
    policy = UniversalPolicy(action_count=3, hidden_size=16)
    policy.add_game_extras("hollow_knight", 4)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    batch = {
        "pixels": np.zeros((2, 12, 108, 192), dtype=np.uint8),
        "action_history": np.zeros((2, 3), dtype=np.int64),
        "game_extras": np.zeros((2, 4), dtype=np.float32),
        "actions": np.asarray([0, 1], dtype=np.int64),
        "action_masks": np.ones((2, 3), dtype=np.bool_),
    }

    result = train_demonstration_update(
        policy, optimizer, batch, game_id="hollow_knight", weight=.1,
    )
    assert np.isfinite(result["loss"])
    assert result["samples"] == 2
