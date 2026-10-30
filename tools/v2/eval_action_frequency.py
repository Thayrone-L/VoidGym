"""Count deterministic policy actions over a fixed batch of platform-gym rooms."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from voidgym.envs.adapter_env import AdapterEnv
from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter
from voidgym.models.policy import UniversalPolicy
from voidgym.training.checkpoints import load_universal


def evaluate(checkpoint: Path, *, levels: Iterable[int], episodes: int, seed: int,
             max_steps: int, device_name: str = "cpu",
             progress_path: Path | None = None) -> dict[str, Any]:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device(device_name)
    model = UniversalPolicy().to(device)
    action_table = tuple(PlatformGymAdapter(level=0).action_table())
    checkpoint_manifest = load_universal(checkpoint, model, action_table)
    model.eval()
    attack_indices = [i for i, name in enumerate(action_table)
                      if "ATTACK" in name or "NAIL_ART" in name]
    output: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "trained_steps": checkpoint_manifest.get("trained_steps"),
        "split": "test",
        "episodes_per_level": episodes,
        "max_steps_per_room": max_steps,
        "action_table": list(action_table),
        "attack_actions": [action_table[index] for index in attack_indices],
        "old_level_mapping_after_86g": {"0": 0, "1": 1, "2": 2, "3": 3,
                                         "4": 4, "5": 5, "7": 6, "8": 7},
        "levels": {},
    }
    for level in levels:
        envs: list[AdapterEnv] = []
        observations: list[dict[str, np.ndarray]] = []
        for episode in range(episodes):
            episode_seed = seed + level * 10_000 + episode * 10
            env = AdapterEnv(
                PlatformGymAdapter(level=level, seed=episode_seed, split="test"),
                lambda _before, _after: 0.0,
            )
            observation, _ = env.reset(seed=episode_seed)
            envs.append(env)
            observations.append(observation)

        active = np.ones(episodes, dtype=np.bool_)
        starts = torch.ones(episodes, dtype=torch.bool, device=device)
        state = model.initial_state(episodes, device)
        counts: Counter[str] = Counter()
        attack_available = 0
        total_actions = 0
        successes = 0
        completed = 0
        episode_steps = [0] * episodes
        no_op = action_table.index("NOOP")
        level_steps = max_steps
        try:
            for _step in range(level_steps):
                if not active.any():
                    break
                tensors = {
                    key: torch.as_tensor(np.stack([obs[key] for obs in observations]), device=device)
                    for key in observations[0]
                }
                masks_np = np.stack([env.action_masks() for env in envs])
                masks_np[~active] = False
                masks_np[~active, no_op] = True
                masks = torch.as_tensor(masks_np, device=device)
                attack_available += int(masks_np[active][:, attack_indices].any(axis=1).sum())
                with torch.inference_mode():
                    result = model(tensors, state, starts, masks)
                    logits = result["logits"]
                    next_state = result["state"]
                    assert isinstance(logits, torch.Tensor) and isinstance(next_state, tuple)
                    selected = logits.argmax(-1).cpu().numpy()
                state = next_state
                starts_next = np.zeros(episodes, dtype=np.bool_)
                for index, env in enumerate(envs):
                    if not active[index]:
                        starts_next[index] = True
                        continue
                    action = int(selected[index])
                    counts[action_table[action]] += 1
                    total_actions += 1
                    episode_steps[index] += 1
                    observation, _, terminated, truncated, info = env.step(action)
                    observations[index] = observation
                    if terminated or truncated:
                        signals = info.get("signals", {})
                        success = (bool(signals.get("terminated")) if level == 8 else
                                   bool(signals.get("room_changed")))
                        successes += int(success)
                        completed += 1
                        active[index] = False
                        starts_next[index] = True
                starts = torch.as_tensor(starts_next, device=device)
        finally:
            for env in envs:
                env.close()
        output["levels"][str(level)] = {
            "actions": dict(counts),
            "total_actions": total_actions,
            "attack_mask_available_steps": attack_available,
            "attack_mask_available_fraction": attack_available / max(1, total_actions),
            "episodes_completed": completed,
            "episodes_succeeded": successes,
            "mean_steps": float(np.mean(episode_steps)),
            "episodes_at_step_limit": int(sum(count >= level_steps for count in episode_steps)),
        }
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(f"level {level}: {total_actions} actions, success={successes}/{episodes}", flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=list(range(9)))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=85_000)
    parser.add_argument("--max-steps", type=int, default=40,
                        help="bounded action-frequency sample per room")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1 or any(not 0 <= level <= 8 for level in args.levels):
        parser.error("episodes/max-steps must be positive and levels must be in [0, 8]")
    result = evaluate(args.checkpoint, levels=args.levels, episodes=args.episodes,
                      seed=args.seed, max_steps=args.max_steps, device_name=args.device,
                      progress_path=args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
