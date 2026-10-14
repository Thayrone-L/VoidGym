from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hk_agent.config import load_config
from hk_agent.gym_env import RAW_ACTIONS, HollowKnightEnv
from hk_agent.menu_bootstrap import MenuBootstrapPlan
from hk_agent.process_owner import IsolatedEpisodeProcessOwner, ProcessOwnerConfig
from hk_agent.random_baseline import run_random_baseline
from hk_agent.supervisor_backend import SupervisorEnvironmentBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a short zero-reward random baseline under transactional isolation"
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.local.toml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--height", type=int, default=338)
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--lease-ms", type=int, default=80)
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "artifacts" / "baseline"
    )
    parser.add_argument(
        "--bootstrap-plan",
        type=Path,
        default=PROJECT_ROOT / "configs" / "menu-bootstrap" / "continue-kings-pass-v1.json",
        help="declarative operational menu plan run before BEGIN_EPISODE",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually launch the isolated game episode; omission is a safe dry-run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps < 1 or args.steps > 10_000:
        raise SystemExit("--steps must be between 1 and 10000")
    plan = {
        "mode": "execute" if args.execute else "dry-run",
        "policy": "random_action_space_sample",
        "seed": args.seed,
        "step_budget": args.steps,
        "action_count": len(RAW_ACTIONS),
        "reward": 0.0,
        "menu_bootstrap": str(args.bootstrap_plan),
        "transactional_launcher": str(
            PROJECT_ROOT / "tools" / "start-isolated-hollow-knight.ps1"
        ),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    config = load_config(args.config.resolve(strict=True))
    output_root = args.output_root.resolve(strict=False)
    bootstrap_plan = MenuBootstrapPlan.load(args.bootstrap_plan.resolve(strict=True))
    owner = IsolatedEpisodeProcessOwner(
        ProcessOwnerConfig(
            game_executable=config.game_executable,
            personal_root=config.personal_save_root,
            experimental_root=config.experimental_save_root,
            backup_root=config.backup_root,
            transactional_launcher=PROJECT_ROOT
            / "tools"
            / "start-isolated-hollow-knight.ps1",
            state_root=output_root / "episodes",
        )
    )
    backend = SupervisorEnvironmentBackend(
        game_profile=config.game_profile,
        reset_process=owner.start_episode,
        shutdown_process=owner.shutdown,
        menu_bootstrap_plan=bootstrap_plan,
        menu_bootstrap_artifact_root=output_root / "bootstrap",
    )
    environment = HollowKnightEnv(
        backend,
        frame_height=args.height,
        frame_width=args.width,
        frame_stack=args.frame_stack,
        lease_ms=args.lease_ms,
        max_episode_steps=args.steps,
        log_root=output_root / "experience",
    )
    try:
        result = run_random_baseline(
            environment,
            seed=args.seed,
            step_budget=args.steps,
            manifest_path=output_root / "baseline-manifest.json",
        )
    finally:
        owner.close()
    print(json.dumps({**plan, "result": result}, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
