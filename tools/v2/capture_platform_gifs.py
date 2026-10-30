"""Create deterministic review GIFs for the requested platform-gym levels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eval_platform import _episode
from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter
from voidgym.models.policy import UniversalPolicy
from voidgym.training.checkpoints import load_universal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=85_000)
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3, 4, 6, 7])
    parser.add_argument("--max-steps", type=int, default=1500)
    args = parser.parse_args()

    torch.set_num_threads(3)
    device = torch.device("cpu")
    model = UniversalPolicy().to(device).eval()
    action_table = tuple(PlatformGymAdapter(level=0, split="test").action_table())
    checkpoint = load_universal(args.checkpoint, model, action_table)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "gifs").mkdir(exist_ok=True)
    results = []
    for level in args.levels:
        seed = args.seed + level * 10_000
        result = _episode(
            model, level, seed, stochastic=False, max_steps=args.max_steps,
            confusion=None, preview_target=[], preview_limit=0,
            perception_count=[0], perception_limit=0, output_root=args.output,
            capture=True,
        )
        result["level"] = level
        result["seed"] = seed
        results.append(result)
        print(f"GIF review level {level}: success={result['success']} steps={result['steps']}",
              flush=True)

    manifest = {
        "checkpoint": str(args.checkpoint),
        "trained_steps": checkpoint.get("trained_steps"),
        "split": "test",
        "mode": "deterministic",
        "levels": results,
        "gif_dir": str(args.output / "gifs"),
    }
    (args.output / "gif_review_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


