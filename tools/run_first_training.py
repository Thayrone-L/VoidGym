from __future__ import annotations

import argparse
import importlib.util
import json
import os
import secrets
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

# Native math runtimes read these variables while NumPy/Torch are imported.
# Pre-parse the manual runner's thread budget before importing project modules.
_thread_parser = argparse.ArgumentParser(add_help=False)
_thread_parser.add_argument("--torch-threads", type=int, choices=range(1, 9), default=3)
_thread_args, _ = _thread_parser.parse_known_args()
for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_env, str(_thread_args.torch_threads))
del _thread_args, _thread_env, _thread_parser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hk_agent.campaign_source import (
    CampaignBootstrapSequence,
    CampaignOrigin,
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
from hk_agent.supervisor_backend import SupervisorEnvironmentBackend
from hk_agent.training_task import (
    ChannelsFirstObservation,
    VisualMotionTask,
    VisualMotionTaskConfig,
    resolve_calibrated_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diagnostic PPO against Hollow Knight")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.training.local.toml")
    parser.add_argument(
        "--bootstrap-plan",
        type=Path,
        default=PROJECT_ROOT / "configs" / "menu-bootstrap" / "continue-kings-pass-v1.json",
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
    parser.add_argument("--game-install", type=Path)
    parser.add_argument("--instance-data-root", type=Path)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timesteps", type=int, default=256)
    parser.add_argument("--checkpoint-interval", type=int, default=128)
    parser.add_argument("--max-checkpoints", type=int, default=5)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--warm-start-policy", type=Path)
    parser.add_argument("--warm-start-policy-manifest", type=Path)
    parser.add_argument("--warm-start-policy-sha256")
    parser.add_argument("--warm-start-policy-manifest-sha256")
    parser.add_argument("--warm-start-policy-manifest-id")
    parser.add_argument("--calibration-samples", type=int, default=5)
    parser.add_argument("--lease-ms", type=int, default=80)
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=540)
    parser.add_argument("--height", type=int, default=338)
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--training-device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--torch-threads", type=int, choices=range(1, 9), default=3)
    parser.add_argument("--torch-interop-threads", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--visual-change-scale", type=float, default=0.1)
    parser.add_argument("--time-cost", type=float, default=-0.001)
    parser.add_argument("--success-reward", type=float, default=1.0)
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
        default=PROJECT_ROOT / "artifacts" / "training" / "visual-motion-v1",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    return parser.parse_args()


def build_task(
    *,
    project_config: Any,
    bootstrap_plan: MenuBootstrapPlan,
    bootstrap_plan_provider: Callable[[], MenuBootstrapPlan] | None,
    output_root: Path,
    task_config: VisualMotionTaskConfig,
    args: argparse.Namespace,
    phase: str,
) -> tuple[VisualMotionTask, IsolatedEpisodeProcessOwner]:
    phase_root = output_root / phase
    owner = IsolatedEpisodeProcessOwner(
        ProcessOwnerConfig(
            game_executable=(args.game_install / project_config.game_executable.name) if args.game_install else project_config.game_executable,
            personal_root=project_config.personal_save_root,
            experimental_root=project_config.experimental_save_root,
            backup_root=project_config.backup_root,
            redirected_save_mode=args.experimental_save_root is not None or args.instance_data_root is not None,
            transactional_launcher=PROJECT_ROOT / "tools" / "start-isolated-hollow-knight.ps1",
            game_arguments=windowed_game_arguments(args.window_width, args.window_height, log_file=str(args.log_file) if args.log_file else None),
            state_root=phase_root / "episodes",
            maximum_episode_seconds=None if args.continuous else 3600.0,
            instance_data_root=args.instance_data_root,
            expected_product_name=(project_config.instance_product_names[int(os.environ["HK_AGENT_SLOT_ID"])] if args.instance_data_root is not None and os.environ.get("HK_AGENT_SLOT_ID") is not None else None),
            other_instance_data_roots=tuple(root for root in project_config.instance_data_roots if root.resolve(strict=False) != args.instance_data_root.resolve(strict=False)) if args.instance_data_root else (),
        )
    )
    try:
        backend = SupervisorEnvironmentBackend(
            game_profile=project_config.game_profile,
            reset_process=owner.start_episode,
            shutdown_process=owner.shutdown,
            menu_bootstrap_plan=(bootstrap_plan if bootstrap_plan_provider is None else None),
            menu_bootstrap_plan_provider=bootstrap_plan_provider,
            menu_bootstrap_artifact_root=phase_root / "bootstrap",
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
            log_root=phase_root / "experience",
        )
        return (
            VisualMotionTask(
                environment,
                config=task_config,
                task_log_path=phase_root / "task-transitions.jsonl",
            ),
            owner,
        )
    except BaseException:
        owner.close()
        raise


def calibrate(
    task: VisualMotionTask, *, seed: int, samples: int, stop_file: Path | None = None
) -> tuple[list[float], list[float], bool]:
    if samples < 5 or samples > 32:
        raise ValueError("calibration samples must be between 5 and 32")
    task.reset(seed=seed)
    noop: list[float] = []
    active: list[float] = []
    for _ in range(samples):
        if stop_file is not None and stop_file.is_file():
            return noop, active, True
        _, _, terminated, truncated, info = task.step(0)
        if terminated or truncated or info.get("valid") is not True:
            raise RuntimeError("NOOP calibration ended before its sample budget")
        noop.append(float(info["motion_fraction"]))
    for index in range(samples):
        if stop_file is not None and stop_file.is_file():
            return noop, active, True
        # Alternating raw directions measures controllable motion without a route.
        action = 1 if index % 2 == 0 else 2
        _, _, terminated, truncated, info = task.step(action)
        if terminated or truncated or info.get("valid") is not True:
            raise RuntimeError("active calibration ended before its sample budget")
        active.append(float(info["motion_fraction"]))
    return noop, active, stop_file is not None and stop_file.is_file()


def cancelled_result(
    config: FirstTrainingConfig, output_root: Path, provenance: dict[str, Any]
) -> dict[str, Any]:
    now = time.time_ns()
    result: dict[str, Any] = {
        "schema_version": 2,
        "status": "CANCELLED",
        "algorithm": "PPO",
        "seed": config.seed,
        "duration_mode": "continuous" if config.continuous else "limited",
        "total_timesteps": None if config.continuous else config.total_timesteps,
        "checkpoint_interval": config.checkpoint_interval,
        "max_checkpoints": config.max_checkpoints,
        "completed_timesteps": 0,
        "checkpoints": [],
        "pruned_checkpoints": [],
        "provenance": provenance,
        "started_unix_ns": now,
        "finished_unix_ns": now,
    }
    atomic_json(output_root / "manifest.json", result)
    return result


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
        continuous=args.continuous,
        max_checkpoints=args.max_checkpoints,
        resume_checkpoint=args.resume_checkpoint,
        warm_start_policy=args.warm_start_policy,
        warm_start_policy_manifest=args.warm_start_policy_manifest,
        warm_start_policy_sha256=args.warm_start_policy_sha256,
        warm_start_policy_manifest_sha256=args.warm_start_policy_manifest_sha256,
        warm_start_policy_manifest_id=args.warm_start_policy_manifest_id,
        training_device=args.training_device,
        torch_threads=args.torch_threads,
        torch_interop_threads=args.torch_interop_threads,
    )
    training_config.validate()
    dependency_state = {
        "stable_baselines3": importlib.util.find_spec("stable_baselines3") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
    }
    plan: dict[str, object] = {
        "mode": "execute" if args.execute else "dry-run",
        "diagnostic_training": True,
        "claim": "motor-signal diagnostic; not game progress or task success",
        "algorithm": "PPO",
        "seed": args.seed,
        "duration_mode": "continuous" if args.continuous else "limited",
        "timesteps": None if args.continuous else args.timesteps,
        "checkpoint_interval": args.checkpoint_interval,
        "max_checkpoints": args.max_checkpoints,
        "runtime_resources": {
            "training_device": args.training_device,
            "torch_threads": args.torch_threads,
            "torch_interop_threads": args.torch_interop_threads,
        },
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        "warm_start_policy": (str(args.warm_start_policy) if args.warm_start_policy else None),
        "warm_start_policy_manifest": (
            str(args.warm_start_policy_manifest) if args.warm_start_policy_manifest else None
        ),
        "calibration_samples_per_group": args.calibration_samples,
        "dependencies": dependency_state,
        "bootstrap_plan": str(resolved_bootstrap.base_path),
        "campaign_origin": origin_provenance,
        "stop_file": str(args.stop_file) if args.stop_file else None,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    if not all(dependency_state.values()):
        raise SystemExit("training dependencies are absent; install .[training] first")

    project_config = load_config(args.config.resolve(strict=True), allow_training=True)
    if args.game_install is not None or args.instance_data_root is not None:
        slot = int(os.environ.get("HK_AGENT_SLOT_ID", "-1"))
        if slot < 0 or slot >= len(project_config.install_roots) or args.game_install is None or args.instance_data_root is None:
            raise ValueError("managed instance arguments require a valid reserved slot")
        if args.game_install.resolve(strict=True) != project_config.install_roots[slot] or args.instance_data_root.resolve(strict=False) != project_config.instance_data_roots[slot]:
            raise ValueError("managed instance arguments do not match the validated slot")
    elif os.environ.get("HK_AGENT_SLOT_ID") is not None:
        raise ValueError("execute slot requires validated managed instance arguments")
    if args.experimental_save_root is not None:
        project_config = replace(
            project_config,
            experimental_save_root=args.experimental_save_root.resolve(strict=True),
        )
    if not project_config.training_enabled:
        raise SystemExit("the selected configuration does not enable training")
    validate_save_precondition(project_config.experimental_save_root, origin)
    if bootstrap_plan.readiness is None:
        raise SystemExit("the bootstrap plan has no readiness reference")
    game_executable = ((args.game_install / project_config.game_executable.name) if args.game_install else project_config.game_executable).resolve(strict=True)
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
    reference_path = bootstrap_plan.readiness.reference_image.resolve(strict=True)
    training_task_path = (PROJECT_ROOT / "src" / "hk_agent" / "training_task.py").resolve(
        strict=True
    )
    action_masking_path = (PROJECT_ROOT / "src" / "hk_agent" / "action_masking.py").resolve(
        strict=True
    )
    temporal_actions_path = (PROJECT_ROOT / "src" / "hk_agent" / "temporal_actions.py").resolve(
        strict=True
    )
    gym_env_path = (PROJECT_ROOT / "src" / "hk_agent" / "gym_env.py").resolve(strict=True)
    policy_runtime_path = (PROJECT_ROOT / "src" / "hk_agent" / "first_training.py").resolve(
        strict=True
    )
    policy_artifact_path = (PROJECT_ROOT / "src" / "hk_agent" / "policy_artifact.py").resolve(
        strict=True
    )
    training_lock_path = (PROJECT_ROOT / "requirements-training.lock").resolve(strict=True)
    provenance: dict[str, Any] = {
        "launch_id": os.environ.get("HK_AGENT_LAUNCH_ID"),
        "slot_id": os.environ.get("HK_AGENT_SLOT_ID"),
        "game_install": str(args.game_install) if args.game_install else str(game_executable.parent),
        "instance_data_root": str(args.instance_data_root) if args.instance_data_root else None,
        "product_name": os.environ.get("HK_AGENT_PRODUCT_NAME"),
        "log_file": str(args.log_file) if args.log_file else None,
        "game_profile": project_config.game_profile,
        "campaign_origin": origin_provenance,
        "files": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "bootstrap_plan": {
                "path": str(bootstrap_path),
                "sha256": sha256_file(bootstrap_path),
            },
            "readiness_reference": {
                "path": str(reference_path),
                "sha256": sha256_file(reference_path),
            },
            "game_executable": {
                "path": str(game_executable),
                "sha256": sha256_file(game_executable),
            },
            "mod_dll": {"path": str(mod_dll), "sha256": sha256_file(mod_dll)},
            "curriculum_source": {
                "path": str(training_task_path),
                "sha256": sha256_file(training_task_path),
            },
            "action_masking_source": {
                "path": str(action_masking_path),
                "sha256": sha256_file(action_masking_path),
            },
            "temporal_actions_source": {
                "path": str(temporal_actions_path),
                "sha256": sha256_file(temporal_actions_path),
            },
            "environment_source": {
                "path": str(gym_env_path),
                "sha256": sha256_file(gym_env_path),
            },
            "policy_runtime_source": {
                "path": str(policy_runtime_path),
                "sha256": sha256_file(policy_runtime_path),
            },
            "policy_artifact_source": {
                "path": str(policy_artifact_path),
                "sha256": sha256_file(policy_artifact_path),
            },
            "training_lock": {
                "path": str(training_lock_path),
                "sha256": sha256_file(training_lock_path),
            },
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
            },
        },
    }
    run_root = args.output_root.resolve(strict=False) / (
        time.strftime("run-%Y%m%dT%H%M%S") + f"-{secrets.token_hex(3)}"
    )
    base_task_config = VisualMotionTaskConfig(
        horizon_steps=None if args.continuous else args.horizon,
        visual_change_scale=args.visual_change_scale,
        time_cost=args.time_cost,
        diagnostic_target_bonus=args.success_reward,
        enemy_kill_reward=args.enemy_kill_reward,
        enemy_kill_reward_budget=args.enemy_kill_reward_budget,
        shade_recovery_reward=args.shade_recovery_reward,
        damage_penalty=args.damage_penalty,
        geo_gain_reward=args.geo_gain_reward,
        geo_spend_reward=args.geo_spend_reward,
        death_penalty=args.death_penalty,
        run_failed_penalty=args.run_failed_penalty,
    )
    base_task_config.validate()

    if args.stop_file is not None and args.stop_file.is_file():
        result = cancelled_result(training_config, run_root / "training", provenance)
        print(json.dumps({**plan, "run_root": str(run_root), "result": result}, indent=2))
        return 0

    calibration_task, calibration_owner = build_task(
        project_config=project_config,
        bootstrap_plan=bootstrap_plan,
        bootstrap_plan_provider=bootstrap_sequence,
        output_root=run_root,
        task_config=base_task_config,
        args=args,
        phase="calibration",
    )
    try:
        noop_motion, active_motion, cancelled = calibrate(
            calibration_task,
            seed=args.seed,
            samples=args.calibration_samples,
            stop_file=args.stop_file,
        )
        if not cancelled:
            calibrated = resolve_calibrated_config(
                base_task_config,
                noop_motion=noop_motion,
                active_motion=active_motion,
            )
            atomic_json(
                run_root / "calibration.json",
                {
                    "schema_version": 1,
                    "status": "PASSED",
                    "noop_motion": noop_motion,
                    "active_motion": active_motion,
                    "resolved_motion_target_fraction": calibrated.motion_target_fraction,
                },
            )
    finally:
        try:
            calibration_task.close()
        finally:
            calibration_owner.close()

    if cancelled:
        result = cancelled_result(training_config, run_root / "training", provenance)
        print(json.dumps({**plan, "run_root": str(run_root), "result": result}, indent=2))
        return 0

    task, owner = build_task(
        project_config=project_config,
        bootstrap_plan=bootstrap_plan,
        bootstrap_plan_provider=bootstrap_sequence,
        output_root=run_root,
        task_config=calibrated,
        args=args,
        phase="training",
    )
    wrapped = ChannelsFirstObservation(
        task, target_height=args.policy_height, target_width=args.policy_width
    )
    try:
        result = train_ppo(
            wrapped,
            task=task,
            config=training_config,
            output_root=run_root / "training",
            provenance=provenance,
            stop_file=args.stop_file,
        )
    finally:
        owner.close()
    print(json.dumps({**plan, "run_root": str(run_root), "result": result}, indent=2))
    return 0 if result["status"] in {"COMPLETED", "CANCELLED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
