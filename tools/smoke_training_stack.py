from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hk_agent.first_training import FirstTrainingConfig, train_ppo


class SyntheticPixelEnvironment(gym.Env[dict[str, np.ndarray[Any, Any]], int]):
    def __init__(self) -> None:
        self.metadata = {"render_modes": []}
        self.action_space = gym.spaces.Discrete(4)  # type: ignore[assignment]
        self.observation_space = gym.spaces.Dict(  # type: ignore[assignment]
            {
                "pixels": gym.spaces.Box(0, 255, (12, 90, 160), np.uint8),
                "action_history": gym.spaces.MultiDiscrete(np.full(3, 1024)),
            }
        )
        self.steps = 0

    def _observation(self) -> dict[str, np.ndarray[Any, Any]]:
        return {
            "pixels": np.zeros((12, 90, 160), dtype=np.uint8),
            "action_history": np.zeros(3, dtype=np.int64),
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.steps = 0
        return self._observation(), {}

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray[Any, Any]], float, bool, bool, dict[str, Any]]:
        del action
        self.steps += 1
        return self._observation(), 0.0, False, self.steps >= 32, {}

    def task_contract(self) -> dict[str, Any]:
        return {"diagnostic_training": True, "synthetic": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "training" / "synthetic-smoke",
    )
    args = parser.parse_args()
    environment = SyntheticPixelEnvironment()
    result = train_ppo(
        environment,
        task=environment,
        config=FirstTrainingConfig(total_timesteps=128, checkpoint_interval=128),
        output_root=args.output,
    )
    return 0 if result["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
