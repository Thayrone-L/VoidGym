from __future__ import annotations

import io

import gymnasium as gym
import numpy as np
from PIL import Image

from voidgym.training.demonstrations import DemonstrationRecorder, demonstration_batch


def test_verified_demonstration_becomes_universal_observations(tmp_path) -> None:
    output = io.BytesIO()
    Image.new("RGB", (16, 12), color=(80, 120, 160)).save(output, format="PNG")
    recorder = DemonstrationRecorder(
        tmp_path / "demos", game_version="test-v1", context="test-challenge",
    )
    recorder.authorize(operator_forced=True)
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
        expected_context="test-challenge", observation_space=observation_space,
        action_count=3, game_extras_mask=np.asarray([True, False, True, True]),
    )
    assert batch["pixels"].shape == (2, 12, 108, 192)
    assert batch["action_history"].shape == (2, 3)
    assert batch["game_extras"].tolist() == [[1., 0., 3., 4.], [1., 0., 3., 4.]]
    assert batch["actions"].tolist() == [0, 1]
