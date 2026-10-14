from __future__ import annotations

import argparse
import importlib.util
import json
import secrets
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hk_agent.campaign_source import (
    CampaignBootstrapSequence,
    CampaignOrigin,
    ResolvedCampaignBootstrap,
    campaign_origin_provenance,
    resolve_campaign_bootstrap,
    validate_save_precondition,
)
from hk_agent.config import load_config
from hk_agent.first_training import FirstTrainingConfig, atomic_json, sha256_file, train_ppo
from hk_agent.gym_env import HollowKnightEnv
from hk_agent.launch_options import windowed_game_arguments
from hk_agent.menu_bootstrap import MenuBootstrapPlan
from hk_agent.process_owner import IsolatedEpisodeProcessOwner, ProcessOwnerConfig
from hk_agent.provenance import fingerprint_directory
from hk_agent.route_memory import TopologicalRouteMemory
from hk_agent.scene_change_task import (
    ReachSceneChangeTask,
    SceneChangeTaskConfig,
    route_detector_signature,
)
from hk_agent.supervisor_backend import SupervisorEnvironmentBackend
from hk_agent.training_task import ChannelsFirstObservation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the pixel-only reach_scene_change_v1 curriculum"
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config.training.local.toml"
    )
    parser.add_argument(
        "--bootstrap-plan",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "menu-bootstrap"
        / "continue-kings-pass-v1.json",
    )
    parser.add_argument(
        "--new-game-bootstrap-plan",
        type=Path,
        default=PROJECT_ROOT / "configs" / "menu-bootstrap" / "new-game-kings-pass-v1.json",
    )
    parser.add_argument(
        "--save-source", choices=("existing_save", "new_game"), default="existing_save"
    )
    parser.add_argument("--save-slot", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--experimental-save-root", type=Path)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--timesteps", type=int, default=4096)
    parser.add_argument("--checkpoint-interval", type=int, default=1024)
    parser.add_argument("--max-checkpoints", type=int, default=5)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lease-ms", type=int, default=80)
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=540)
    parser.add_argument("--height", type=int, default=338)
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--time-cost", type=float, default=-0.002)
    parser.add_argument("--novelty-shaping-budget", type=float, default=0.0)
    parser.add_argument("--success-reward", type=float, default=10.0)
    parser.add_argument("--enemy-kill-reward", type=float, default=2.0)
    parser.add_argument("--enemy-kill-reward-budget", type=float, default=6.0)
    parser.add_argument("--shade-recovery-reward", type=float, default=2.0)
    parser.add_argument("--damage-penalty", type=float, default=-0.25)
    parser.add_argument("--geo-gain-reward", type=float, default=0.01)
    parser.add_argument("--geo-spend-reward", type=float, default=0.02)
    parser.add_argument("--death-penalty", type=float, default=-2.0)
    parser.add_argument("--run-failed-penalty", type=float, default=-4.0)
    parser.add_argument("--policy-height", type=int, default=90)
    parser.add_argument("--policy-width", type=int, default=160)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "training" / "reach-scene-change-v1",
    )
    parser.add_argument(
        "--route-memory",
        type=Path,
        help="persistent topological memory (defaults to a save-scoped artifact)",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    return parser.parse_args()


def _provenance(
    args: argparse.Namespace,
    project_config: Any,
    resolved_bootstrap: ResolvedCampaignBootstrap,
    origin_provenance: dict[str, object],
) -> dict[str, Any]:
    bootstrap_plan = resolved_bootstrap.plan
    game_executable = project_config.game_executable.resolve(strict=True)
    mod_dll = (
        game_executable.parent
        / "Hollow Knight_Data"
        / "Managed"
        / "Mods"
        / "HkAgentBridge"
        / "HkAgentBridge.dll"
    ).resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    bootstrap_path = resolved_bootstrap.base_path
    if bootstrap_plan.readiness is None:
        raise RuntimeError("the bootstrap plan has no readiness reference")
    reference_path = bootstrap_plan.readiness.reference_image.resolve(strict=True)
    source_path = (PROJECT_ROOT / "src" / "hk_agent" / "scene_change_task.py").resolve(
        strict=True
    )
    route_memory_source = (PROJECT_ROOT / "src" / "hk_agent" / "route_memory.py").resolve(
        strict=True
    )
    temporal_actions_source = (
        PROJECT_ROOT / "src" / "hk_agent" / "temporal_actions.py"
    ).resolve(strict=True)
    gym_env_source = (PROJECT_ROOT / "src" / "hk_agent" / "gym_env.py").resolve(
        strict=True
    )
    files = {
        "config": config_path,
        "bootstrap_plan": bootstrap_path,
        "readiness_reference": reference_path,
        "game_executable": game_executable,
        "mod_dll": mod_dll,
        "curriculum_source": source_path,
        "route_memory_source": route_memory_source,
        "temporal_actions_source": temporal_actions_source,
        "environment_source": gym_env_source,
    }
    return {
        "game_profile": project_config.game_profile,
        "campaign_origin": origin_provenance,
        "experimental_save_snapshot": fingerprint_directory(
            project_config.experimental_save_root
        ),
        "files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in files.items()
        },
        "observation_dimensions": {
            "capture": {
                "height": args.height,
                "width": args.width,
                "frame_stack": args.frame_stack,
                "channels_per_frame": 3,
            },
            "policy": {
                "height": args.policy_height,
                "width": args.policy_width,
                "channels": args.frame_stack * 3,
                "action_history_length": max(0, args.frame_stack - 1),
            },
        },
    }


def build_task(
    *,
    project_config: Any,
    bootstrap_plan: MenuBootstrapPlan,
    bootstrap_plan_provider: Callable[[], MenuBootstrapPlan] | None,
    output_root: Path,
    task_config: SceneChangeTaskConfig,
    args: argparse.Namespace,
) -> tuple[ReachSceneChangeTask, IsolatedEpisodeProcessOwner]:
    owner = IsolatedEpisodeProcessOwner(
        ProcessOwnerConfig(
            game_executable=project_config.game_executable,
            personal_root=project_config.personal_save_root,
            experimental_root=project_config.experimental_save_root,
            backup_root=project_config.backup_root,
            redirected_save_mode=args.experimental_save_root is not None,
            transactional_launcher=PROJECT_ROOT / "tools" / "start-isolated-hollow-knight.ps1",
            game_arguments=windowed_game_arguments(args.window_width, args.window_height),
            state_root=output_root / "episodes",
            maximum_episode_seconds=None if args.continuous else 3600.0,
        )
    )
    try:
        backend = SupervisorEnvironmentBackend(
            game_profile=project_config.game_profile,
            reset_process=owner.start_episode,
            shutdown_process=owner.shutdown,
            menu_bootstrap_plan=(bootstrap_plan if bootstrap_plan_provider is None else None),
            menu_bootstrap_plan_provider=bootstrap_plan_provider,
            menu_bootstrap_artifact_root=output_root / "bootstrap",
            bridge_timeout_seconds=30.0,
            process_timeout_seconds=30.0,
        )
        environment = HollowKnightEnv(
            backend,
            frame_height=args.height,
            frame_width=args.width,
            frame_stack=args.frame_stack,
            lease_ms=args.lease_ms,
            max_episode_steps=task_config.horizon_steps,
            log_root=output_root / "experience",
        )
        return (
            ReachSceneChangeTask(
                environment,
                config=task_config,
                task_log_path=output_root / "task-transitions.jsonl",
                route_memory=TopologicalRouteMemory(
                    args.route_memory,
                    scope={
                        "game_profile": project_config.game_profile,
                        "save_source": args.save_source,
                        "save_slot": args.save_slot,
                        "campaign_id": str(args.output_root.resolve(strict=False)),
                        "detector_signature": route_detector_signature(task_config),
                    },
                ),
            ),
            owner,
        )
    except BaseException:
        owner.close()
        raise


def main() -> int:
    args = parse_args()
    if args.continuous and args.stop_file is None:
        raise SystemExit("--continuous requires --stop-file for cooperative shutdown")
    origin = CampaignOrigin(args.save_source, args.save_slot)
    resolved_bootstrap = resolve_campaign_bootstrap(
        continue_plan_path=args.bootstrap_plan,
        new_game_plan_path=args.new_game_bootstrap_plan,
        origin=origin,
    )
    bootstrap_plan = resolved_bootstrap.plan
    resume_bootstrap = (
        resolve_campaign_bootstrap(
            continue_plan_path=args.bootstrap_plan,
            new_game_plan_path=args.new_game_bootstrap_plan,
            origin=CampaignOrigin("existing_save", args.save_slot),
        )
        if origin.source == "new_game"
        else None
    )
    bootstrap_sequence = CampaignBootstrapSequence(
        bootstrap_plan, resume_bootstrap.plan if resume_bootstrap is not None else None
    )
    origin_provenance = campaign_origin_provenance(resolved_bootstrap, resume_bootstrap)
    training_config = FirstTrainingConfig(
        seed=args.seed,
        total_timesteps=args.timesteps,
        checkpoint_interval=args.checkpoint_interval,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        continuous=args.continuous,
        max_checkpoints=args.max_checkpoints,
        resume_checkpoint=args.resume_checkpoint,
    )
    training_config.validate()
    task_config = SceneChangeTaskConfig(
        horizon_steps=None if args.continuous else args.horizon,
        time_cost=args.time_cost,
        novelty_shaping_budget=args.novelty_shaping_budget,
        success_reward=args.success_reward,
        enemy_kill_reward=args.enemy_kill_reward,
        enemy_kill_reward_budget=args.enemy_kill_reward_budget,
        shade_recovery_reward=args.shade_recovery_reward,
        damage_penalty=args.damage_penalty,
        geo_gain_reward=args.geo_gain_reward,
        geo_spend_reward=args.geo_spend_reward,
        death_penalty=args.death_penalty,
        run_failed_penalty=args.run_failed_penalty,
        terminate_on_scene_change=not args.continuous,
    )
    task_config.validate()
    if args.route_memory is None:
        args.route_memory = (
            args.output_root.resolve(strict=False)
            / f"route-memory-{args.save_source}-slot-{args.save_slot}.json"
        )
    dependencies = {
        "stable_baselines3": importlib.util.find_spec("stable_baselines3") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
    }
    plan: dict[str, object] = {
        "mode": "execute" if args.execute else "dry-run",
        "curriculum": "reach_scene_change_v1",
        "claim": (
            "pixel-only stable visual-regime-change diagnostic; room transition "
            "not yet validated against a holdout corpus"
        ),
        "algorithm": "PPO",
        "seed": args.seed,
        "duration_mode": "continuous" if args.continuous else "limited",
        "timesteps": None if args.continuous else args.timesteps,
        "checkpoint_interval": args.checkpoint_interval,
        "max_checkpoints": args.max_checkpoints,
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        "dependencies": dependencies,
        "bootstrap_plan": str(resolved_bootstrap.base_path),
        "campaign_origin": origin_provenance,
        "will_launch_game": args.execute,
        "stop_file": str(args.stop_file) if args.stop_file else None,
        "route_memory": str(args.route_memory),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    if not all(dependencies.values()):
        raise SystemExit("training dependencies are absent; install .[training] first")

    project_config = load_config(args.config.resolve(strict=True), allow_training=True)
    if args.experimental_save_root is not None:
        project_config = replace(
            project_config,
            experimental_save_root=args.experimental_save_root.resolve(strict=True),
        )
    if not project_config.training_enabled:
        raise SystemExit("the selected configuration does not enable training")
    validate_save_precondition(project_config.experimental_save_root, origin)
    provenance = _provenance(
        args, project_config, resolved_bootstrap, origin_provenance
    )
    run_root = args.output_root.resolve(strict=False) / (
        time.strftime("run-%Y%m%dT%H%M%S") + f"-{secrets.token_hex(3)}"
    )
    training_root = run_root / "training"
    if args.stop_file is not None and args.stop_file.is_file():
        now = time.time_ns()
        result: dict[str, Any] = {
            "schema_version": 2,
            "status": "CANCELLED",
            "algorithm": "PPO",
            "seed": training_config.seed,
            "duration_mode": "continuous" if training_config.continuous else "limited",
            "total_timesteps": (
                None if training_config.continuous else training_config.total_timesteps
            ),
            "checkpoint_interval": training_config.checkpoint_interval,
            "max_checkpoints": training_config.max_checkpoints,
            "completed_timesteps": 0,
            "checkpoints": [],
            "pruned_checkpoints": [],
            "provenance": provenance,
            "started_unix_ns": now,
            "finished_unix_ns": now,
        }
        atomic_json(training_root / "manifest.json", result)
        print(json.dumps({**plan, "run_root": str(run_root), "result": result}, indent=2))
        return 0
    task, owner = build_task(
        project_config=project_config,
        bootstrap_plan=bootstrap_plan,
        bootstrap_plan_provider=bootstrap_sequence,
        output_root=training_root,
        task_config=task_config,
        args=args,
    )
    wrapped = ChannelsFirstObservation(
        task, target_height=args.policy_height, target_width=args.policy_width
    )
    try:
        result = train_ppo(
            wrapped,
            task=task,
            config=training_config,
            output_root=training_root,
            provenance=provenance,
            stop_file=args.stop_file,
        )
    finally:
        owner.close()
    print(json.dumps({**plan, "run_root": str(run_root), "result": result}, indent=2))
    return 0 if result["status"] in {"COMPLETED", "CANCELLED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
