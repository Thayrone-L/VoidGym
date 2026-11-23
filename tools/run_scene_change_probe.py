from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, replace
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
from hk_agent.gym_env import HollowKnightEnv
from hk_agent.launch_options import windowed_game_arguments
from hk_agent.menu_bootstrap import MenuBootstrapPlan
from hk_agent.process_owner import IsolatedEpisodeProcessOwner, ProcessOwnerConfig
from hk_agent.provenance import fingerprint_directory
from hk_agent.route_memory import TopologicalRouteMemory
from hk_agent.scene_change_probe import (
    SceneChangeProbeConfig,
    atomic_json,
    completed_probe_result,
    run_scene_change_probe,
    sha256_file,
)
from hk_agent.scene_change_task import (
    SCENE_CHANGE_ACTION_NAMES,
    ReachSceneChangeTask,
    SceneChangeTaskConfig,
    route_detector_signature,
)
from hk_agent.supervisor_backend import SupervisorEnvironmentBackend


def provenance(
    args: argparse.Namespace,
    project_config: Any,
    resolved_bootstrap: ResolvedCampaignBootstrap,
    origin_provenance: dict[str, object],
) -> dict[str, Any]:
    bootstrap_plan = resolved_bootstrap.plan
    game_executable = ((args.game_install / project_config.game_executable.name) if args.game_install else project_config.game_executable).resolve(strict=True)
    mod_dll = (
        game_executable.parent
        / "Hollow Knight_Data"
        / "Managed"
        / "Mods"
        / "HkAgentBridge"
        / "HkAgentBridge.dll"
    ).resolve(strict=True)
    if bootstrap_plan.readiness is None:
        raise RuntimeError("the bootstrap plan has no readiness reference")
    paths = {
        "config": args.config.resolve(strict=True),
        "bootstrap_plan": resolved_bootstrap.base_path,
        "readiness_reference": bootstrap_plan.readiness.reference_image.resolve(strict=True),
        "game_executable": game_executable,
        "mod_dll": mod_dll,
        "curriculum_source": (
            PROJECT_ROOT / "src" / "hk_agent" / "scene_change_task.py"
        ).resolve(strict=True),
        "route_memory_source": (
            PROJECT_ROOT / "src" / "hk_agent" / "route_memory.py"
        ).resolve(strict=True),
        "temporal_actions_source": (
            PROJECT_ROOT / "src" / "hk_agent" / "temporal_actions.py"
        ).resolve(strict=True),
        "environment_source": (
            PROJECT_ROOT / "src" / "hk_agent" / "gym_env.py"
        ).resolve(strict=True),
        "probe_source": (
            PROJECT_ROOT / "src" / "hk_agent" / "scene_change_probe.py"
        ).resolve(strict=True),
        "probe_runner": Path(__file__).resolve(strict=True),
    }
    return {
        "game_profile": project_config.game_profile,
        "game_install": str(args.game_install) if args.game_install else str(game_executable.parent),
        "instance_data_root": str(args.instance_data_root) if args.instance_data_root else None,
        "product_name": os.environ.get("HK_AGENT_PRODUCT_NAME"),
        "log_file": str(args.log_file) if args.log_file else None,
        "campaign_origin": origin_provenance,
        "experimental_save_snapshot": fingerprint_directory(
            project_config.experimental_save_root
        ),
        "files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "observation_dimensions": {
            "capture": {
                "height": args.height,
                "width": args.width,
                "frame_stack": args.frame_stack,
                "channels_per_frame": 3,
            },
            "policy": None,
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
            game_executable=(args.game_install / project_config.game_executable.name) if args.game_install else project_config.game_executable,
            personal_root=project_config.personal_save_root,
            experimental_root=project_config.experimental_save_root,
            backup_root=project_config.backup_root,
            redirected_save_mode=args.experimental_save_root is not None or args.instance_data_root is not None,
            transactional_launcher=PROJECT_ROOT / "tools" / "start-isolated-hollow-knight.ps1",
            game_arguments=windowed_game_arguments(args.window_width, args.window_height, log_file=str(args.log_file) if args.log_file else None),
            state_root=output_root / "episodes",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded random probe for reach_scene_change_v1 without a learner"
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
    parser.add_argument("--game-install", type=Path)
    parser.add_argument("--instance-data-root", type=Path)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--action-repeat", type=int, default=3)
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
        default=PROJECT_ROOT / "artifacts" / "probes" / "reach-scene-change-v1",
    )
    parser.add_argument(
        "--route-memory",
        type=Path,
        help="persistent topological memory (defaults to a save-scoped artifact)",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--bridge-smoke-seconds", type=int,
                        help=argparse.SUPPRESS)
    return parser.parse_args()


def run_bridge_smoke(task: ReachSceneChangeTask, *, seconds: int, stop_file: Path | None) -> dict[str, Any]:
    """Exercise the real bridge with deterministic, slot-specific actions."""
    import hashlib
    import struct
    import zlib

    import numpy as np

    from hk_agent.win_job import process_identity

    if not 10 <= seconds <= 600:
        raise ValueError("bridge smoke duration must be between 10 and 600 seconds")
    slot = os.environ.get("HK_AGENT_SLOT_ID")
    if slot not in {"0", "1"}:
        raise RuntimeError("bridge smoke requires a reserved slot 0 or 1")
    task_action_name = "LEFT" if slot == "0" else "JUMP_TAP"
    action_index = SCENE_CHANGE_ACTION_NAMES.index(task_action_name)
    observation, _ = task.reset(seed=1_000 + int(slot))
    backend = task.environment.backend
    endpoint = getattr(backend, "endpoint", None)
    supervisor = getattr(backend, "_supervisor", None)
    accepted = getattr(supervisor, "accepted", None)
    if endpoint is None or not accepted:
        raise RuntimeError("bridge did not complete HELLO before smoke actions")
    bridge = accepted[-1]
    identity = process_identity(os.getpid())
    # On Windows the venv's Scripts\python.exe is a redirector that starts the
    # base interpreter as a child; record the parent so the launcher PID the
    # panel holds can be matched to this real runner process.
    identity["parent_pid"] = os.getppid()
    evidence_path = os.environ.get("HK_AGENT_SMOKE_EVIDENCE_FILE")
    if not evidence_path:
        raise RuntimeError("HK_AGENT_SMOKE_EVIDENCE_FILE is missing")
    evidence_root = Path(evidence_path).parent
    frames_root = evidence_root / "smoke-frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    def save_png(path: Path, rgb: Any) -> None:
        array = np.asarray(rgb, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("smoke capture must be an RGB image")
        height, width, _ = array.shape

        def chunk(kind: bytes, content: bytes) -> bytes:
            return (struct.pack(">I", len(content)) + kind + content
                    + struct.pack(">I", zlib.crc32(kind + content) & 0xFFFFFFFF))

        rows = b"".join(b"\0" + array[row].tobytes() for row in range(height))
        data = (b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
        path.write_bytes(data)

    save_png(frames_root / "initial.png", observation["pixels"][-1])
    endpoint_record = {
        "host": endpoint.host,
        "port": endpoint.port,
        "run_id": endpoint.run_id,
        "token_sha256": hashlib.sha256(endpoint.token.encode("utf-8")).hexdigest(),
        "game_profile": endpoint.game_profile,
        "instance_id": bridge.instance_id,
        "session_id": bridge.session_id,
        "game_version": bridge.game_version,
        "persistent_data_path": bridge.persistent_data_path,
        "product_name": bridge.product_name,
        "runner_process": identity,
        "slot_id": int(slot),
        "started_unix_ns": time.time_ns(),
        "steps": 0,
        "action": task_action_name,
        # Declared from the first write so readers never see a partial schema;
        # after-action.png only exists once the first step has completed.
        "frame_paths": [str(frames_root / "initial.png"), str(frames_root / "after-action.png")],
    }
    atomic_json(Path(evidence_path), endpoint_record)

    started = time.monotonic()
    last_evidence = started
    completed_steps = 0
    action_counts: dict[str, int] = {task_action_name: 0}
    while time.monotonic() - started < seconds:
        if stop_file is not None and stop_file.is_file():
            break
        observation, _, terminated, truncated, info = task.step(action_index)
        if info.get("valid") is not True:
            raise RuntimeError(f"bridge smoke transition invalid: {info.get('termination_reason')}")
        completed_steps += 1
        action_counts[task_action_name] += 1
        if completed_steps == 1:
            save_png(frames_root / "after-action.png", observation["pixels"][-1])
        if time.monotonic() - last_evidence >= 2:
            endpoint_record.update({
                "steps": completed_steps,
                "last_action_unix_ns": time.time_ns(),
                "elapsed_seconds": time.monotonic() - started,
            })
            atomic_json(Path(evidence_path), endpoint_record)
            last_evidence = time.monotonic()
        if terminated or truncated:
            raise RuntimeError("gameplay ended before the cooperative smoke stop")
    result = {
        "slot_id": int(slot),
        "action": task_action_name,
        "action_counts": action_counts,
        "steps": completed_steps,
        "elapsed_seconds": time.monotonic() - started,
        "stopped_cooperatively": bool(stop_file is not None and stop_file.is_file()),
        "frame_paths": [str(frames_root / "initial.png"), str(frames_root / "after-action.png")],
    }
    endpoint_record.update(result)
    endpoint_record["finished_unix_ns"] = time.time_ns()
    atomic_json(Path(evidence_path), endpoint_record)
    return result


def main() -> int:
    args = parse_args()
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
    probe_config = SceneChangeProbeConfig(
        seed=args.seed,
        episodes=args.episodes,
        horizon_steps=args.horizon,
        action_repeat=args.action_repeat,
    )
    probe_config.validate()
    task_config = SceneChangeTaskConfig(
        horizon_steps=args.horizon,
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
    )
    task_config.validate()
    if args.route_memory is None and args.instance_data_root is not None:
        # Two instances must never share the default route-memory file; the
        # panel always passes an exclusive per-slot --route-memory.
        raise ValueError("instance launches require an exclusive --route-memory path")
    if args.route_memory is None:
        args.route_memory = (
            args.output_root.resolve(strict=False)
            / f"route-memory-{args.save_source}-slot-{args.save_slot}.json"
        )
    plan: dict[str, Any] = {
        "mode": "execute" if args.execute else "dry-run",
        "probe": "reach_scene_change_v1_random_probe",
        "learner": None,
        "checkpoints": False,
        "will_launch_game": args.execute,
        "probe_config": asdict(probe_config),
        "task_config": asdict(task_config),
        "bootstrap_plan": str(resolved_bootstrap.base_path),
        "campaign_origin": origin_provenance,
        "stop_file": str(args.stop_file) if args.stop_file else None,
        "route_memory": str(args.route_memory),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    if args.bridge_smoke_seconds is not None and not 10 <= args.bridge_smoke_seconds <= 600:
        raise ValueError("bridge smoke duration must be between 10 and 600 seconds")

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
    validate_save_precondition(project_config.experimental_save_root, origin)
    provenance_record = provenance(
        args, project_config, resolved_bootstrap, origin_provenance
    )
    run_root = args.output_root.resolve(strict=False) / (
        time.strftime("run-%Y%m%dT%H%M%S") + f"-{secrets.token_hex(3)}"
    )
    manifest_path = run_root / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "scene_change_probe",
        "status": "RUNNING",
        "gate_passed": False,
        "claim": (
            "random pixel-only detector probe; not training and not proof of room transition"
        ),
        "probe_config": asdict(probe_config),
        "task_contract": None,
        "provenance": provenance_record,
        "episodes": [],
        "death_episodes": 0,
        "episode_resets_started": 0,
        "operational_failures": 0,
        "started_unix_ns": time.time_ns(),
    }
    atomic_json(manifest_path, manifest)
    task = None
    owner = None
    try:
        if args.stop_file is not None and args.stop_file.is_file():
            manifest.update(
                {
                    "status": "CANCELLED",
                    "gate_passed": False,
                    "positive_events": 0,
                    "death_episodes": 0,
                    "episode_resets_started": 0,
                    "operational_failures": 0,
                }
            )
        else:
            task, owner = build_task(
                project_config=project_config,
                bootstrap_plan=bootstrap_plan,
                bootstrap_plan_provider=bootstrap_sequence,
                output_root=run_root,
                task_config=task_config,
                args=args,
            )
            manifest["task_contract"] = task.task_contract()
            atomic_json(manifest_path, manifest)
            if args.bridge_smoke_seconds is not None:
                manifest["bridge_smoke"] = run_bridge_smoke(
                    task, seconds=args.bridge_smoke_seconds, stop_file=args.stop_file
                )
                summaries: list[dict[str, Any]] = []
                positive_events = 0
            else:
                summaries, positive_events = run_scene_change_probe(
                    task,
                    config=probe_config,
                    artifact_root=run_root,
                    stop_file=args.stop_file,
                )
            if args.stop_file is not None and args.stop_file.is_file():
                manifest.update(
                    {
                        "status": "CANCELLED",
                        "gate_passed": False,
                        "positive_events": positive_events,
                        "death_episodes": sum(
                            item.get("ended_by_death") is True for item in summaries
                        ),
                        "episode_resets_started": len(summaries),
                        "operational_failures": 0,
                        "episodes": summaries,
                    }
                )
            elif args.bridge_smoke_seconds is not None:
                manifest.update({"status": "COMPLETED", "positive_events": 0,
                                 "death_episodes": 0, "episode_resets_started": 1,
                                 "operational_failures": 0})
            else:
                manifest.update(completed_probe_result(summaries, positive_events))
    except BaseException as error:
        manifest.update(
            {
                "status": "FAILED",
                "gate_passed": False,
                "operational_failures": 1,
                "failure": f"{type(error).__name__}: {error}",
            }
        )
        raise
    finally:
        teardown_failures: list[str] = []
        if task is not None:
            try:
                task.close()
            except Exception as error:  # noqa: BLE001 - teardown must reach the manifest
                teardown_failures.append(f"task.close: {type(error).__name__}: {error}")
        if owner is not None:
            try:
                owner.close()
            except Exception as error:  # noqa: BLE001 - teardown must reach the manifest
                teardown_failures.append(f"owner.close: {type(error).__name__}: {error}")
        if teardown_failures and manifest["status"] != "FAILED":
            manifest.update(
                {
                    "status": "FAILED",
                    "gate_passed": False,
                    "operational_failures": 1,
                    "failure": "; ".join(teardown_failures),
                }
            )
        manifest["finished_unix_ns"] = time.time_ns()
        atomic_json(manifest_path, manifest)
    print(json.dumps({**plan, "run_root": str(run_root), "result": manifest}, indent=2))
    return 0 if manifest["status"] in {"COMPLETED", "CANCELLED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
