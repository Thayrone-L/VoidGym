from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hk_agent.shared_learner import run_shared_learner


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a file-based shared MaskablePPO learner")
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--actors", type=int, required=True)
    parser.add_argument("--rollout-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--checkpoint-interval", type=int, required=True)
    parser.add_argument("--max-checkpoints", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=4096)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--torch-threads", type=int, default=3)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--rollout-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--straggler-timeout-seconds", type=float, default=45.0)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, str(args.torch_threads))
    result = run_shared_learner(
        shared_root=args.shared_root,
        stop_file=args.stop_file,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        actors=args.actors,
        checkpoint_interval=args.checkpoint_interval,
        max_checkpoints=args.max_checkpoints,
        total_timesteps=args.timesteps,
        continuous=args.continuous,
        seed=args.seed,
        torch_threads=args.torch_threads,
        torch_interop_threads=args.torch_interop_threads,
        rollout_timeout_seconds=args.rollout_timeout_seconds,
        straggler_timeout_seconds=args.straggler_timeout_seconds,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") in {"COMPLETED", "CANCELLED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
