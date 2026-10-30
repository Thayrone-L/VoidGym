"""Low-priority recurrent PPO pre-training on the procedurally generated platform gym."""

from __future__ import annotations

import argparse
import atexit
import csv
import ctypes
import copy
import faulthandler
from functools import partial
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from voidgym.core.config import config_from_dict, save_campaign_config
from voidgym.core.experience_store import ExperienceWriter
from voidgym.core.hardware import HardwareProfile, detect_hardware, plan_resources, save_resource_profile
from voidgym.core.hardware import memory_commit_stats
from voidgym.envs.adapter_env import AdapterEnv
from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter
from voidgym.gyms.platform_gym.curriculum import Curriculum
from voidgym.gyms.platform_gym.vector_env import PlatformSubprocVecEnv
from voidgym.gyms.platform_gym.vec_worker import make_platform_env
from voidgym.models.policy import UniversalPolicy
from voidgym.training.buffers import RecurrentRolloutBuffer
from voidgym.training.checkpoints import load_universal, save_universal
from voidgym.training.checkpoints import prune_checkpoints
from voidgym.models.perception import load_class_weights
from voidgym.training.recurrent_ppo import RecurrentPPOTrainer

_ACTIVE_RUN_ROOT: Path | None = None
_ACTIVE_RUN_STARTED: str | None = None
_ACTIVE_RUN_MONOTONIC: float | None = None
_ACTIVE_STOP_REQUESTED = False
_ACTIVE_RUN_INITIALIZED = False
_ACTIVE_STOP_REASON: str | None = None
_RUN_CLEANUPS: list[Callable[[], None]] = []
_FAULT_HANDLE: Any = None


def _register_cleanup(callback: Callable[[], None]) -> None:
    _RUN_CLEANUPS.append(callback)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_terminal_status(state: str, reason: str) -> None:
    if _ACTIVE_RUN_ROOT is None:
        return
    payload: dict[str, Any] = {
        "status": state, "reason": reason, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "started_at": _ACTIVE_RUN_STARTED,
        "duration_seconds": (time.monotonic() - _ACTIVE_RUN_MONOTONIC
                             if _ACTIVE_RUN_MONOTONIC is not None else None),
    }
    heartbeat = _ACTIVE_RUN_ROOT / "heartbeat.json"
    if heartbeat.is_file():
        payload["last_heartbeat"] = json.loads(heartbeat.read_text(encoding="utf-8"))
    _atomic_json(_ACTIVE_RUN_ROOT / "status.json", payload)


def _ensure_csv_header(path: Path, fields: list[str]) -> None:
    if not path.is_file():
        return
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        previous_fields = reader.fieldnames or []
        if previous_fields == fields:
            return
        previous_rows = list(reader)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in previous_rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    temporary.replace(path)


def _mark_unclean_exit() -> None:
    if _ACTIVE_RUN_ROOT is None or not _ACTIVE_RUN_INITIALIZED:
        return
    status_path = _ACTIVE_RUN_ROOT / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    if status.get("status") == "RUNNING" or (
        (_ACTIVE_RUN_ROOT / "heartbeat.json").exists() and status.get("status") not in
        {"COMPLETED", "STOPPED", "FAILED"}
    ):
        _write_terminal_status("FAILED", "process exited without writing terminal status")


atexit.register(_mark_unclean_exit)


def _priority_below_normal() -> None:
    if os.name == "nt":
        try:
            import psutil  # type: ignore[import-untyped]
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except Exception:  # noqa: BLE001 - Win32 API is the fallback on restricted installs
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)


def _tensor_observation(observations: list[dict[str, np.ndarray]] | dict[str, np.ndarray],
                        device: torch.device) -> dict[str, torch.Tensor]:
    if isinstance(observations, dict):
        batched = observations
    else:
        batched = {key: np.stack([item[key] for item in observations])
                   for key in observations[0]}
    return {key: torch.as_tensor(value, device=device) for key, value in batched.items()}


def _enable_main_faulthandler(root: Path) -> None:
    global _FAULT_HANDLE
    if _FAULT_HANDLE is not None and not _FAULT_HANDLE.closed:
        faulthandler.disable()
        _FAULT_HANDLE.close()
    _FAULT_HANDLE = (root / "faulthandler-main.log").open("a", encoding="utf-8")
    faulthandler.enable(file=_FAULT_HANDLE, all_threads=True)

    def close_handler() -> None:
        global _FAULT_HANDLE
        faulthandler.disable()
        if _FAULT_HANDLE is not None:
            _FAULT_HANDLE.close()
            _FAULT_HANDLE = None

    _register_cleanup(close_handler)


def _measure_worker_private_gb(root: Path, seed: int) -> float:
    """Spawn one torch-free environment and measure its private resident footprint."""
    probe = PlatformSubprocVecEnv([
        partial(make_platform_env, level=0, seed=seed, worker_index="probe",
                diagnostics_dir=root),
    ], start_method="spawn")
    try:
        probe.reset()
        probe._refresh_worker_memory()
        measured = probe.worker_startup[0].get("private_bytes")
        if measured is None:
            measured = probe.worker_startup[0].get("rss_bytes")
        if measured is None or int(measured) <= 0:
            raise RuntimeError("cannot measure private memory of a spawned environment worker")
        return int(measured) / 2**30
    finally:
        probe.close()


def _write_runtime_versions(root: Path, profile: HardwareProfile) -> None:
    from importlib.metadata import PackageNotFoundError, version

    def installed(package: str) -> str | None:
        try:
            return version(package)
        except PackageNotFoundError:
            return None

    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        driver_version = driver.stdout.strip() if driver.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        driver_version = None
    _atomic_json(root / "versions.json", {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "nvidia_driver": driver_version,
        "numpy": np.__version__,
        "pillow": installed("Pillow"),
        "stable_baselines3": installed("stable-baselines3"),
        "paging_file_total_gb": profile.paging_file_total_gb,
        "commit_used_gb": profile.commit_used_gb,
        "commit_limit_gb": profile.commit_limit_gb,
        "commit_available_gb": profile.commit_available_gb,
        "worker_private_gb": profile.worker_private_gb,
    })


def _memory_telemetry(device: torch.device,
                      workers: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        root_process = psutil.Process(os.getpid())
        processes = [root_process, *root_process.children(recursive=True)]
        worker_by_pid = {int(row["pid"]): row for row in workers if row.get("pid")}
        process_rows: list[dict[str, Any]] = []
        for process in processes:
            try:
                rss = process.memory_info().rss
                full = process.memory_full_info()
                process_rows.append({
                    "pid": process.pid,
                    "role": "learner" if process.pid == os.getpid() else "environment_worker",
                    "rss_bytes": int(rss),
                    "private_bytes": int(getattr(full, "uss", rss)),
                    "torch_imported": worker_by_pid.get(process.pid, {}).get("torch_imported"),
                })
            except (psutil.Error, OSError):
                continue
        ram_total = int(memory.total)
        ram_available = int(memory.available)
    except (ImportError, OSError):
        ram_total = ram_available = 0
        process_rows = []
    telemetry: dict[str, Any] = {
        "ram_total_bytes": ram_total or None,
        "ram_available_bytes": ram_available or None,
        **memory_commit_stats(),
        "processes": process_rows,
        "workers": workers,
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device))
        if device.type == "cuda" else 0,
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device))
        if device.type == "cuda" else 0,
    }
    telemetry["process_rss_total_bytes"] = sum(int(row["rss_bytes"]) for row in process_rows)
    telemetry["process_private_total_bytes"] = sum(int(row["private_bytes"])
                                                   for row in process_rows)
    return telemetry


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="platform-pretrain")
    parser.add_argument("--total-steps", type=int, default=20_000_000)
    parser.add_argument("--envs", type=int)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=.995)
    parser.add_argument("--gae-lambda", type=float, default=.95)
    parser.add_argument("--ent-coef", type=float, default=.01)
    parser.add_argument("--clip", type=float, default=.2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--c-sem", type=float, default=.5)
    parser.add_argument("--c-inv", type=float, default=.1)
    parser.add_argument("--c-mot", type=float, default=.1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--hours-limit", type=float)
    parser.add_argument("--stop-at-curriculum-level", type=int, choices=range(1, 9),
                        help="stop after a completed round once the curriculum reaches this level")
    parser.add_argument("--goal-window-episodes", type=int, default=100)
    parser.add_argument("--goal-success-levels-0-7", "--goal-success-levels-0-6",
                        dest="goal_success_levels_0_7", type=float, default=.70)
    parser.add_argument("--goal-success-level-8", "--goal-success-level-7",
                        dest="goal_success_level_8", type=float, default=.30)
    parser.add_argument("--no-improvement-steps", type=int, default=3_000_000)
    parser.add_argument("--quick-eval-interval-steps", type=int, default=1_000_000,
                        help="run quick evaluation whenever this many new steps are crossed")
    parser.add_argument("--quick-eval-episodes-per-level", type=int, default=50)
    parser.add_argument("--microbatch-sequences", type=int, default=16,
                        help="sequences per CUDA microbatch; effective batch is unchanged")
    parser.add_argument("--priority", choices=("below_normal",), default="below_normal")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--allow-cpu", action="store_true",
                        help="confirm CPU training when an NVIDIA GPU is present")
    parser.add_argument("--runtime-root", type=Path, default=PROJECT_ROOT / "runtime",
                        help="runtime directory for profiles, checkpoints and experience")
    parser.add_argument("--seed", type=int, default=8500)
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args(argv)


def _run_impl(args: argparse.Namespace) -> Path:
    if (args.total_steps < 1 or args.rollout_steps < 1 or args.seq_len < 1
            or args.quick_eval_interval_steps < 1 or args.quick_eval_episodes_per_level < 1
            or args.microbatch_sequences < 1 or args.goal_window_episodes < 1
            or args.no_improvement_steps < 1
            or not 0 <= args.goal_success_levels_0_7 <= 1
            or not 0 <= args.goal_success_level_8 <= 1):
        raise ValueError("step counts and sequence length must be positive")
    _priority_below_normal()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    nvidia_smi = _nvidia_smi_available()
    profile = detect_hardware(args.runtime_root)
    if nvidia_smi and not torch.cuda.is_available() and not args.allow_cpu:
        _cpu_fallback_warning("nvidia-smi sees an NVIDIA GPU, but this PyTorch build has no CUDA")
        raise SystemExit("CPU training not started; rerun with --allow-cpu to confirm")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    free_cores = max(2, (profile.physical_cores or 1) - 5)
    overrides: dict[str, Any] = {}
    if args.device is not None:
        overrides["train_device"] = args.device
    if args.envs is not None:
        overrides["gym_envs"] = args.envs
    plan = plan_resources(profile, "pretrain_platform", overrides)
    env_count = plan.gym_envs
    if (nvidia_smi and plan.train_device == "cpu" and not args.allow_cpu):
        _cpu_fallback_warning("ResourcePlan selected CPU despite an NVIDIA GPU being present")
        raise SystemExit("CPU training not started; rerun with --allow-cpu to confirm")
    if (args.device == "cuda" and (profile.gpu_memory_free_gb or 0.0) < 6.0):
        raise SystemExit("CUDA requested but ResourcePlan requires at least 6 GiB of free VRAM")
    root: Path = Path(args.runtime_root) / "v2" / "pretrain" / str(args.run_name)
    root.mkdir(parents=True, exist_ok=True)
    _enable_main_faulthandler(root)
    if profile.commit_available_gb is not None:
        worker_private_gb = _measure_worker_private_gb(root, args.seed)
        profile = replace(profile, worker_private_gb=worker_private_gb)
        plan = plan_resources(profile, "pretrain_platform", overrides)
        env_count = plan.gym_envs
    requested_batch_size = args.batch_size or plan.batch_size
    required_sequences = max(1, requested_batch_size // args.seq_len)
    minimum_rollout_steps = math.ceil(required_sequences / env_count) * args.seq_len
    args.rollout_steps = max(args.rollout_steps, minimum_rollout_steps)
    device = torch.device(plan.train_device)
    threads = min(free_cores, max(1, plan.torch_threads_learner))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    global _ACTIVE_RUN_INITIALIZED
    _ACTIVE_RUN_INITIALIZED = True
    status_path = root / "status.json"
    if status_path.exists():
        (root / "status.previous.json").write_bytes(status_path.read_bytes())
    _atomic_json(status_path, {
        "status": "RUNNING", "started_at": _ACTIVE_RUN_STARTED,
    })
    config = config_from_dict({
        "schema_version": 1, "campaign_id": args.run_name, "kind": "pretrain_platform",
        "game_id": "platform_gym",
        "ppo": {"learning_rate": args.lr, "gamma": args.gamma, "gae_lambda": args.gae_lambda,
                "clip_range": args.clip, "entropy_coef": args.ent_coef, "epochs": args.n_epochs,
                "batch_size": args.batch_size or plan.batch_size, "sequence_length": args.seq_len},
        "aux": {"c_sem": args.c_sem, "c_inv": args.c_inv, "c_mot": args.c_mot,
                "augment_random_shift": not args.no_augment},
        "resources": {"gym_envs": env_count, "train_device": plan.train_device},
        "storage": {"root": str(root / "experience")},
    })
    save_campaign_config(config, root / "config.json")
    save_resource_profile(profile, plan, root)
    _write_runtime_versions(root, profile)
    (root / "runtime_plan.json").write_text(json.dumps({
        "planned_envs": env_count, "free_physical_cores_reserved": free_cores,
        "torch_threads": threads, "train_device": plan.train_device,
        "actor_device": str(device if device.type == "cuda" else torch.device("cpu")),
        "priority": args.priority,
        "cuda_available": torch.cuda.is_available(), "nvidia_smi": nvidia_smi,
        "free_vram_gb": profile.gpu_memory_free_gb,
        "commit_available_gb_at_plan": profile.commit_available_gb,
        "worker_private_gb": profile.worker_private_gb,
        "planned_commit_headroom_gb": plan.commit_headroom_gb,
        "learner_commit_reserve_gb": 5.5,
        "worker_memory_safety_factor": 3.0,
        "warnings": list(plan.warnings),
        "effective_batch_size": args.batch_size or plan.batch_size,
        "microbatch_sequences": args.microbatch_sequences,
        "rollout_steps_per_environment": args.rollout_steps,
    }, indent=2) + "\n", encoding="utf-8")
    curriculum_path = root / "curriculum.json"
    curriculum = Curriculum(seed=args.seed)
    if args.resume and (args.resume / "curriculum.json").is_file():
        curriculum = Curriculum.load(args.resume / "curriculum.json")
    environments = PlatformSubprocVecEnv(
        [partial(make_platform_env, level=0, seed=args.seed + index, worker_index=index,
                 diagnostics_dir=root) for index in range(env_count)],
        start_method="spawn",
    )
    _register_cleanup(environments.close)
    environments.seed(args.seed)
    observations = environments.reset()
    infos = list(environments.reset_infos)
    action_table = environments.get_attr("actions")[0]
    current_levels = [0] * env_count
    policy = UniversalPolicy().to(device)
    actor_device = device if device.type == "cuda" else torch.device("cpu")
    actor_policy = copy.deepcopy(policy).to(actor_device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    trainer = RecurrentPPOTrainer(policy, learning_rate=args.lr, clip=args.clip,
                                 entropy_coef=args.ent_coef, c_sem=args.c_sem,
                                 c_inv=args.c_inv, c_mot=args.c_mot,
                                 augment=not args.no_augment,
                                 semantic_weights=load_class_weights(
                                     PROJECT_ROOT / "configs" / "v2" / "semantic_weights_50000.json"))
    global_step = 0
    if args.resume:
        manifest = load_universal(args.resume, policy, action_table,
                                  optimizer=trainer.optimizer)
        global_step = int(manifest.get("trained_steps", 0))
    state = actor_policy.initial_state(env_count, actor_device)
    episode_starts = torch.ones(env_count, dtype=torch.bool, device=actor_device)
    episode_had_exit = [False] * env_count
    experience_writer = ExperienceWriter(root / "experience", "platform_gym", args.run_name,
                                        source="platform_gym", block_steps=1024,
                                        action_table=action_table)
    _register_cleanup(experience_writer.close)
    log_path = root / "log.csv"
    log_mode = "a" if log_path.exists() and global_step else "w"
    fieldnames = ["steps", "L_ppo", "L_semantic", "L_inverse_dynamics", "L_hero_motion",
                  "entropy", "kl", "explained_variance", "steps_per_second", "curriculum_level",
                  "success_rate_100", "quick_eval_success",
                  *[f"success_rate_{args.goal_window_episodes}_level_{level}" for level in range(9)],
                  *[f"success_rate_level_{level}" for level in range(9)]]
    if log_mode == "a":
        _ensure_csv_header(log_path, fieldnames)
    started = time.monotonic()
    started_wall = time.time()
    started_step = global_step
    last_checkpoint = global_step
    last_eval_interval = global_step // args.quick_eval_interval_steps
    stop_requested = [False]
    phase_totals: defaultdict[str, float] = defaultdict(float)
    profile_path = root / "phase_profile.csv"
    profile_fields = ["steps", "round_seconds", "collection_seconds", "inference_seconds",
                      "gae_seconds", "update_seconds", "quick_eval_seconds",
                      "effective_batch_size", "microbatch_sequences",
                      "gradient_accumulation_steps", "ram_available_gb",
                      "commit_used_gb", "commit_limit_gb", "commit_available_gb",
                      "process_rss_total_gb", "process_private_total_gb",
                      "worker_private_total_gb", "workers_torch_imported",
                      "cuda_allocated_mb", "cuda_reserved_mb",
                      "cuda_peak_allocated_mb", "cuda_peak_reserved_mb"]
    profile_mode = "a" if profile_path.exists() and global_step else "w"
    if profile_mode == "a":
        _ensure_csv_header(profile_path, profile_fields)
    profile_stream = profile_path.open(profile_mode, newline="", encoding="utf-8")
    _register_cleanup(profile_stream.close)
    profile_writer = csv.DictWriter(profile_stream, fieldnames=profile_fields)
    if profile_mode == "w":
        profile_writer.writeheader()
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def request_stop(_signum: int, _frame: Any) -> None:
        global _ACTIVE_STOP_REQUESTED
        _ACTIVE_STOP_REQUESTED = True
        stop_requested[0] = True

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)
        def restore_sigterm() -> None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        _register_cleanup(restore_sigterm)

    def restore_sigint() -> None:
        signal.signal(signal.SIGINT, previous_sigint)

    _register_cleanup(restore_sigint)
    with log_path.open(log_mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if log_mode == "w":
            writer.writeheader()
        while (global_step < args.total_steps and not stop_requested[0]
               and (args.stop_at_curriculum_level is None
                    or curriculum.level < args.stop_at_curriculum_level)):
            if args.hours_limit and time.monotonic() - started >= args.hours_limit * 3600:
                break
            round_started = time.perf_counter()
            phase_round: defaultdict[str, float] = defaultdict(float)
            actor_policy.load_state_dict({key: value.detach().cpu()
                                          for key, value in policy.state_dict().items()})
            buffer = RecurrentRolloutBuffer(args.rollout_steps, env_count, args.seq_len)
            for _ in range(args.rollout_steps):
                collection_started = time.perf_counter()
                obs_tensor = _tensor_observation(observations, actor_device)
                masks_np = np.stack(environments.env_method("action_masks"))
                masks = torch.as_tensor(masks_np, device=actor_device)
                before_state = tuple(item.clone() for item in state)
                inference_started = time.perf_counter()
                with torch.no_grad():
                    output = actor_policy(obs_tensor, state, episode_starts, masks,
                                    privileged=_privileged_tensor(infos, actor_device))
                    logits, values, next_state = output["logits"], output["values"], output["state"]
                    assert isinstance(logits, torch.Tensor) and isinstance(values, torch.Tensor)
                    assert isinstance(next_state, tuple)
                    distribution = Categorical(logits=logits)
                    actions = distribution.sample()  # type: ignore[no-untyped-call]
                    log_probs = distribution.log_prob(actions)  # type: ignore[no-untyped-call]
                if actor_device.type == "cuda":
                    torch.cuda.synchronize(actor_device)
                inference_seconds = time.perf_counter() - inference_started
                phase_round["inference_seconds"] += inference_seconds
                sampled_frames = (observations["pixels"][:, -3:, :, :].transpose(0, 2, 3, 1)
                                  if random.random() < .01 else None)
                next_observations_raw, rewards, dones, next_infos_raw = environments.step(
                    actions.detach().cpu().numpy().astype(np.int64)
                )
                next_observations = {key: np.array(value, copy=True)
                                     for key, value in next_observations_raw.items()}
                next_infos = list(next_infos_raw)
                dones = np.asarray(dones, dtype=np.bool_)
                truncations = np.asarray([bool(item.get("TimeLimit.truncated", False))
                                          for item in next_infos], dtype=np.bool_)
                for index, info in enumerate(next_infos):
                    signals = info.get("signals", {})
                    episode_had_exit[index] |= bool(signals.get("room_changed"))
                    if sampled_frames is not None:
                        labels = infos[index].get("training_labels", {})
                        velocity = labels.get("hero_velocity", (float("nan"), float("nan")))
                        motion = np.asarray((*velocity, labels.get("on_ground", float("nan"))),
                                            dtype=np.float32)
                        experience_writer.append({
                            "frame": sampled_frames[index], "action": int(actions[index].item()),
                            "action_mask": masks[index].detach().cpu().numpy(),
                            "reward_universal": float(rewards[index]),
                            "done": bool(dones[index] and not truncations[index]),
                            "truncated": bool(truncations[index]),
                            "discontinuity": bool(dones[index] or signals.get("room_changed") or signals.get("died")),
                            "sem": labels.get("sem", np.full((27, 48), 255, dtype=np.uint8)),
                            "hero_motion": motion,
                            "privileged": _privileged_tensor([infos[index]], actor_device)[0]
                            .detach().cpu().numpy(),
                            "scene": f"platform_gym_level_{current_levels[index]}",
                            "tags": ["pretrain_sample", f"step:{global_step}"]})
                    if dones[index]:
                        curriculum.record(episode_had_exit[index], current_levels[index])
                        episode_had_exit[index] = False
                        level = curriculum.sample_level(args.seed + global_step + index)
                        environments.env_method("set_level", level, indices=[index])
                        reset_result = environments.env_method(
                            "reset", seed=args.seed + global_step + index + 1, indices=[index]
                        )[0]
                        reset_obs, reset_info = reset_result
                        for key, value in next_observations.items():
                            value[index] = reset_obs[key]
                        next_infos[index] = reset_info
                        current_levels[index] = level
                buffer.add(obs_tensor, actions, torch.as_tensor(rewards, device=device),
                           torch.as_tensor(dones, device=device), values, log_probs,
                           masks, episode_starts, (before_state[0], before_state[1]), infos)
                observations, infos = next_observations, next_infos
                episode_starts = torch.as_tensor(dones, dtype=torch.bool, device=actor_device)
                state = next_state
                global_step += env_count
                phase_round["collection_seconds"] += max(
                    0.0, time.perf_counter() - collection_started - inference_seconds)
            with torch.no_grad():
                last_output = actor_policy(_tensor_observation(observations, actor_device), state,
                                           episode_starts,
                                           privileged=_privileged_tensor(infos, actor_device))
            gae_started = time.perf_counter()
            buffer.compute_returns(last_output["values"], gamma=args.gamma, gae_lambda=args.gae_lambda)
            phase_round["gae_seconds"] += time.perf_counter() - gae_started
            requested_sequences = (args.batch_size or plan.batch_size) // args.seq_len
            available_sequences = env_count * ((args.rollout_steps + args.seq_len - 1) // args.seq_len)
            sequence_batch_size = max(1, min(requested_sequences, available_sequences))
            update_started = time.perf_counter()
            metrics = trainer.update(
                buffer.sequences(torch.device("cpu")), epochs=args.n_epochs,
                sequence_batch_size=sequence_batch_size,
                microbatch_size=min(args.microbatch_sequences, sequence_batch_size),
            )
            phase_round["update_seconds"] += time.perf_counter() - update_started
            quick_eval: float | str = ""
            quick_eval_seconds = 0.0
            if global_step // args.quick_eval_interval_steps > last_eval_interval:
                actor_policy.load_state_dict({key: value.detach().cpu()
                                              for key, value in policy.state_dict().items()})
                eval_started = time.perf_counter()
                quick_eval = _quick_eval(
                    actor_policy, args.seed + global_step,
                    episodes_per_level=args.quick_eval_episodes_per_level,
                    max_level=curriculum.level,
                )
                quick_eval_seconds = time.perf_counter() - eval_started
                last_eval_interval = global_step // args.quick_eval_interval_steps
            round_seconds = time.perf_counter() - round_started
            phase_round["quick_eval_seconds"] += quick_eval_seconds
            for phase_name, phase_value in phase_round.items():
                phase_totals[phase_name] += phase_value
            memory_sample = _memory_telemetry(device, environments.worker_startup)
            commit_available = memory_sample.get("commit_available_bytes")
            if commit_available is not None and commit_available < 3 * 2**30:
                _atomic_json(root / "memory_guard_failure.json", {
                    "steps": global_step,
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "commit_available_gb": _bytes_to_gib(commit_available),
                    "memory_sample": memory_sample,
                })
                raise RuntimeError(
                    "commit disponível caiu para "
                    f"{_bytes_to_gib(commit_available):.2f} GiB; mínimo: 3 GiB"
                )
            private_worker_bytes = [
                row.get("private_bytes") for row in memory_sample["processes"]
                if row.get("role") == "environment_worker" and row.get("private_bytes") is not None
            ]
            effective_batch_size = sequence_batch_size * args.seq_len
            microbatch_sequences = min(args.microbatch_sequences, sequence_batch_size)
            accumulation_steps = (sequence_batch_size + microbatch_sequences - 1) // microbatch_sequences
            profile_writer.writerow({
                "steps": global_step, "round_seconds": round_seconds,
                "collection_seconds": phase_round["collection_seconds"],
                "inference_seconds": phase_round["inference_seconds"],
                "gae_seconds": phase_round["gae_seconds"],
                "update_seconds": phase_round["update_seconds"],
                "quick_eval_seconds": quick_eval_seconds,
                "effective_batch_size": effective_batch_size,
                "microbatch_sequences": microbatch_sequences,
                "gradient_accumulation_steps": accumulation_steps,
                "ram_available_gb": _bytes_to_gib(memory_sample.get("ram_available_bytes")),
                "commit_used_gb": _bytes_to_gib(memory_sample.get("commit_used_bytes")),
                "commit_limit_gb": _bytes_to_gib(memory_sample.get("commit_limit_bytes")),
                "commit_available_gb": _bytes_to_gib(memory_sample.get("commit_available_bytes")),
                "process_rss_total_gb": _bytes_to_gib(memory_sample.get("process_rss_total_bytes")),
                "process_private_total_gb": _bytes_to_gib(
                    memory_sample.get("process_private_total_bytes")),
                "worker_private_total_gb": _bytes_to_gib(sum(int(value or 0)
                                                               for value in private_worker_bytes)),
                "workers_torch_imported": sum(1 for row in environments.worker_startup
                                               if row.get("torch_imported") is True),
                "cuda_allocated_mb": memory_sample["cuda_allocated_bytes"] / 2**20,
                "cuda_reserved_mb": memory_sample["cuda_reserved_bytes"] / 2**20,
                "cuda_peak_allocated_mb": (torch.cuda.max_memory_allocated(device) / 2**20
                                             if device.type == "cuda" else 0.0),
                "cuda_peak_reserved_mb": (torch.cuda.max_memory_reserved(device) / 2**20
                                           if device.type == "cuda" else 0.0),
            })
            profile_stream.flush()
            elapsed = max(.001, time.monotonic() - started)
            row: dict[str, Any] = {key: metrics.get(key, 0.0) for key in fieldnames[1:8]}
            row.update({"steps": global_step,
                        "steps_per_second": (global_step - started_step) / elapsed,
                        "curriculum_level": curriculum.level,
                        "success_rate_100": (sum(curriculum.current) / len(curriculum.current)
                                             if curriculum.current else 0.0),
                        "quick_eval_success": quick_eval})
            row.update({f"success_rate_level_{level}": curriculum.success_rate(level)
                        for level in range(9)})
            window_rates = {level: curriculum.window_success_rate(level, args.goal_window_episodes)
                            for level in range(9)}
            row.update({f"success_rate_{args.goal_window_episodes}_level_{level}":
                        ("" if window_rates[level] is None else window_rates[level])
                        for level in range(9)})
            writer.writerow(row)
            stream.flush()
            _atomic_json(root / "heartbeat.json", {
                "steps": global_step, "curriculum_level": curriculum.level,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "success_by_level": {str(level): curriculum.success_rate(level)
                                     for level in range(9)},
                "success_100_by_level": {str(level): window_rates[level]
                                         for level in range(9)},
                "system_memory": memory_sample,
            })
            stalled_level = curriculum.update_progress_watch(
                global_step, window=args.goal_window_episodes,
                levels_0_7_target=args.goal_success_levels_0_7,
                level_8_target=args.goal_success_level_8,
                no_improvement_steps=args.no_improvement_steps,
            )
            if curriculum.training_goal_met(
                    window=args.goal_window_episodes,
                    levels_0_7_target=args.goal_success_levels_0_7,
                    level_8_target=args.goal_success_level_8):
                global _ACTIVE_STOP_REASON
                _ACTIVE_STOP_REASON = "all curriculum levels met rolling success targets"
                print(_ACTIVE_STOP_REASON, flush=True)
                break
            if stalled_level is not None:
                _ACTIVE_STOP_REASON = (f"level {stalled_level} had no rolling-success improvement "
                                       f"for {args.no_improvement_steps} steps")
                print(_ACTIVE_STOP_REASON, flush=True)
                break
            if args.hours_limit and time.monotonic() - started >= args.hours_limit * 3600:
                _ACTIVE_STOP_REASON = f"time limit reached ({args.hours_limit:g} hours)"
                break
            if global_step - last_checkpoint >= 100_000:
                _save_checkpoint(root, policy, trainer, curriculum, global_step, metrics, action_table)
                prune_checkpoints(root)
                last_checkpoint = global_step
        _save_checkpoint(root, policy, trainer, curriculum, global_step, {}, action_table)
    curriculum.save(curriculum_path)
    profile_seconds = sum(phase_totals.values())
    (root / "phase_profile.json").write_text(json.dumps({
        "phase_seconds": dict(phase_totals), "measured_seconds": profile_seconds,
        "phase_fractions": {key: value / profile_seconds for key, value in phase_totals.items()}
        if profile_seconds else {},
        "throughput_steps_per_second": (global_step - started_step)
        / max(.001, time.monotonic() - started),
        "cuda_peak_allocated_mb": (torch.cuda.max_memory_allocated(device) / 2**20
                                    if device.type == "cuda" else 0.0),
        "cuda_peak_reserved_mb": (torch.cuda.max_memory_reserved(device) / 2**20
                                  if device.type == "cuda" else 0.0),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"steps": global_step, "elapsed_seconds": time.time() - started_wall,
                      "steps_per_second": (global_step - started_step)
                      / max(.001, time.monotonic() - started),
                      "device": str(device), "torch_threads": threads, "envs": env_count,
                      "peak_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                      "run_root": str(root)}, indent=2))
    return root


def run(args: argparse.Namespace) -> Path:
    """Run training with persistent heartbeat and terminal status reporting."""
    global _ACTIVE_RUN_ROOT, _ACTIVE_RUN_STARTED, _ACTIVE_RUN_MONOTONIC
    global _ACTIVE_STOP_REQUESTED, _ACTIVE_RUN_INITIALIZED, _ACTIVE_STOP_REASON
    root = Path(args.runtime_root) / "v2" / "pretrain" / str(args.run_name)
    heartbeat = root / "heartbeat.json"
    status = root / "status.json"
    previous_status: dict[str, Any] = {}
    if status.is_file():
        try:
            previous_status = json.loads(status.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_status = {}
    if heartbeat.exists() and previous_status.get("status") in {None, "RUNNING"}:
        warning = "Previous run stopped without a terminal status.\n"
        print(warning.rstrip(), file=sys.stderr)
        root.mkdir(parents=True, exist_ok=True)
        with (root / "stderr.log").open("a", encoding="utf-8") as stream:
            stream.write(warning)
    _ACTIVE_RUN_ROOT = root
    _ACTIVE_RUN_STARTED = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _ACTIVE_RUN_MONOTONIC = time.monotonic()
    _ACTIVE_STOP_REQUESTED = False
    _ACTIVE_STOP_REASON = None
    _ACTIVE_RUN_INITIALIZED = False
    _RUN_CLEANUPS.clear()
    try:
        result = _run_impl(args)
    except BaseException as error:
        if _ACTIVE_RUN_INITIALIZED:
            with (root / "stderr.log").open("a", encoding="utf-8") as stream:
                traceback.print_exc(file=stream)
                stream.flush()
            _write_terminal_status("FAILED", f"{type(error).__name__}: {error}")
        raise
    else:
        _write_terminal_status("STOPPED" if _ACTIVE_STOP_REQUESTED else "COMPLETED",
                               "termination signal received" if _ACTIVE_STOP_REQUESTED
                               else (_ACTIVE_STOP_REASON or "training loop finished"))
        return result
    finally:
        for cleanup in reversed(_RUN_CLEANUPS):
            try:
                cleanup()
            except Exception as cleanup_error:  # noqa: BLE001 - preserve original training failure
                print(f"Resource cleanup failed: {cleanup_error}", file=sys.stderr)
        _RUN_CLEANUPS.clear()
        _ACTIVE_RUN_ROOT = None
        _ACTIVE_RUN_MONOTONIC = None
        _ACTIVE_RUN_INITIALIZED = False


def _nvidia_smi_available() -> bool:
    try:
        result = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True,
                                text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "GPU 0:" in result.stdout


def _cpu_fallback_warning(reason: str) -> None:
    print(f"\033[91mAVISO: {reason}. O pré-treino não será iniciado em CPU sem confirmação.\033[0m",
          file=sys.stderr)


def _bytes_to_gib(value: Any) -> float | None:
    return float(value) / 2**30 if isinstance(value, (int, float)) else None


def _privileged_tensor(infos: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    rows = []
    for info in infos:
        signals = info.get("signals", {})
        position = signals.get("player_pos_room", (.5, .5))
        rows.append([float(position[0]), float(position[1]), float(signals.get("progress", 0)),
                     float(signals.get("hp", 0)) / 10.0])
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _quick_eval(policy: UniversalPolicy, seed: int, *, episodes_per_level: int = 50,
                max_level: int = 6, max_steps: int = 300) -> float:
    """Run a brief deterministic held-out evaluation once each million steps."""
    device = next(policy.parameters()).device
    successes = 0
    episodes = 0
    policy.eval()
    for level in range(min(9, max_level + 1)):
        for episode in range(episodes_per_level):
            adapter = PlatformGymAdapter(level=level, seed=seed + level * 100 + episode * 10,
                                         split="test")
            env = AdapterEnv(adapter, lambda _before, _after: 0.0)
            observation, _ = env.reset(seed=seed + level * 100 + episode * 10)
            state = policy.initial_state(1, device)
            starts = torch.ones(1, dtype=torch.bool, device=device)
            success = False
            for _ in range(max_steps):
                tensors = {key: torch.as_tensor(value[None], device=device)
                           for key, value in observation.items()}
                mask = torch.as_tensor(env.action_masks()[None], device=device)
                with torch.no_grad():
                    output = policy(tensors, state, starts, mask)
                logits, next_state = output["logits"], output["state"]
                assert isinstance(logits, torch.Tensor) and isinstance(next_state, tuple)
                action = int(logits.argmax(-1).item())
                observation, _, terminated, truncated, info = env.step(action)
                signals = info.get("signals", {})
                success |= (bool(signals.get("terminated")) if level == 8 else
                            bool(signals.get("room_changed")))
                state = next_state
                starts = torch.tensor([terminated or truncated], device=device)
                if success or terminated or truncated:
                    break
            successes += int(success)
            episodes += 1
            env.close()
    return successes / max(1, episodes)


def _save_checkpoint(root: Path, policy: UniversalPolicy, trainer: RecurrentPPOTrainer,
                     curriculum: Curriculum, steps: int, metrics: dict[str, float],
                     action_table: tuple[str, ...]) -> None:
    milestone = steps >= 1_000_000 and steps % 1_000_000 < 512
    name = f"universal-{steps:09d}" + ("-permanent" if milestone else "")
    checkpoint = root / name
    save_universal(checkpoint, policy, action_table, steps=steps,
                   metrics=metrics, optimizer=trainer.optimizer,
                   curriculum=curriculum.to_dict())


def main() -> int:
    args = _parse()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
