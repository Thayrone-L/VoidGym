"""Measure held-out semantic perception with frames balanced across platform levels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from voidgym.core.semantics import IGNORE, SEM_GRID, Sem
from voidgym.envs.adapter_env import AdapterEnv
from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter
from voidgym.models.policy import UniversalPolicy
from voidgym.training.checkpoints import load_universal

from eval_platform import _save_confusion, _save_perception


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "relatorios" / "85_eval")
    parser.add_argument("--frames", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=185_000)
    parser.add_argument("--max-steps", type=int, default=1500)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(3)
    device = torch.device("cpu")
    model = UniversalPolicy().to(device).eval()
    action_table = tuple(PlatformGymAdapter(level=0, split="test").action_table())
    manifest = load_universal(args.checkpoint, model, action_table)
    args.output.mkdir(parents=True, exist_ok=True)
    confusion = np.zeros((len(Sem), len(Sem)), dtype=np.int64)
    previews: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
    target_per_level = (args.frames + 7) // 8
    counts: dict[int, int] = {}

    for level in range(8):
        collected = 0
        episode = 0
        while collected < target_per_level:
            seed = args.seed + level * 10_000 + episode * 10
            adapter = PlatformGymAdapter(level=level, seed=seed, split="test")
            env = AdapterEnv(adapter, lambda _before, _after: 0.0)
            observation, info = env.reset(seed=seed)
            state = model.initial_state(1, device)
            starts = torch.ones(1, dtype=torch.bool)
            for _ in range(args.max_steps):
                tensors = {key: torch.as_tensor(value[None])
                           for key, value in observation.items()}
                mask = torch.as_tensor(env.action_masks()[None])
                with torch.no_grad():
                    output = model(tensors, state, starts, mask)
                    logits = output["logits"]
                    semantic_logits = output["semantic_logits"]
                    next_state = output["state"]
                    assert isinstance(logits, Tensor)
                    assert isinstance(semantic_logits, Tensor)
                    assert isinstance(next_state, tuple)
                    action = int(logits.argmax(-1).item())
                    predicted = semantic_logits.argmax(dim=-3).squeeze(0).cpu().numpy()
                state = next_state
                starts = torch.zeros(1, dtype=torch.bool)
                labels = info.get("training_labels", {})
                target = np.asarray(labels.get("sem", np.full(SEM_GRID, IGNORE)), dtype=np.uint8)
                if target.shape == SEM_GRID:
                    valid = target != IGNORE
                    np.add.at(confusion, (target[valid].astype(int), predicted[valid].astype(int)), 1)
                    if len(previews) < 12 and (collected % max(1, target_per_level // 2) == 0):
                        previews.append((adapter.frame_rgb().copy(), predicted.copy(), target.copy(), level))
                    collected += 1
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated or collected >= target_per_level:
                    break
            env.close()
            episode += 1
        counts[level] = collected
        print(f"Percepção: nível {level}, {collected} quadros.", flush=True)

    _save_confusion(args.output / "confusion.csv", confusion)
    _save_perception(args.output / "percepcao.png", previews)
    report = {"checkpoint": str(args.checkpoint), "trained_steps": manifest.get("trained_steps"),
              "requested_frames": args.frames, "sampled_frames": sum(counts.values()),
              "valid_pixels": int(confusion.sum()), "frames_per_level": counts,
              "split": "test", "mode": "deterministic",
              "iou": {item.name: _iou(confusion, index) for index, item in enumerate(Sem)}}
    (args.output / "perception_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _iou(confusion: np.ndarray, index: int) -> float | None:
    union = int(confusion[index].sum() + confusion[:, index].sum() - confusion[index, index])
    return float(confusion[index, index] / union) if union else None


def main() -> int:
    result = evaluate(_args())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
