from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hk_agent.shared_learner import GhostDictEnv, _space_from_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Load and predict with a shared learner checkpoint")
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    calibration = json.loads((args.shared_root / "calibration.json").read_text("utf-8"))["contract"]
    observation_space = _space_from_payload(calibration["observation_space"])
    action_space = _space_from_payload(calibration["action_space"])
    if not isinstance(action_space, gym.spaces.Discrete):
        raise TypeError("shared learner checkpoint expects a Discrete action space")
    import sb3_contrib

    environment = GhostDictEnv(observation_space, action_space)
    try:
        model = sb3_contrib.MaskablePPO.load(str(args.checkpoint), env=environment, device="cpu")
        observation, _ = environment.reset()
        action, _ = model.predict(
            observation, deterministic=True,
            action_masks=np.ones((1, int(action_space.n)), dtype=np.bool_),
        )
        print(json.dumps({
            "status": "PASSED", "checkpoint": str(args.checkpoint),
            "action": np.asarray(action).tolist(),
        }))
    finally:
        environment.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
