"""Low-priority recurrent PPO pre-training on the procedurally generated platform gym."""

from __future__ import annotations

import argparse
import csv
import ctypes
import copy
from functools import partial
import json
import os
import random
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.distributions import Categorical
from stable_baselines3.common.vec_env import SubprocVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from voidgym.core.config import config_from_dict, save_campaign_config
from voidgym.core.experience_store import ExperienceWriter
from voidgym.core.hardware import detect_hardware, plan_resources, save_resource_profile
from voidgym.envs.adapter_env import AdapterEnv
from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter
from voidgym.gyms.platform_gym.curriculum import Curriculum
from voidgym.gyms.platform_gym.reward import CombinedPlatformReward
from voidgym.models.policy import UniversalPolicy
from voidgym.training.buffers import RecurrentRolloutBuffer
from voidgym.training.checkpoints import load_universal, save_universal
from voidgym.training.checkpoints import prune_checkpoints
from voidgym.models.perception import load_class_weights
from voidgym.training.recurrent_ppo import RecurrentPPOTrainer


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
    parser.add_argument("--stop-at-curriculum-level", type=int, choices=range(1, 8),
                        help="stop after a completed round once the curriculum reaches this level")
    parser.add_argument("--priority", choices=("below_normal",), default="below_normal")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--allow-cpu", action="store_true",
                        help="confirm CPU training when an NVIDIA GPU is present")
    parser.add_argument("--runtime-root", type=Path, default=PROJECT_ROOT / "runtime",
                        help="runtime directory for profiles, checkpoints and experience")
    parser.add_argument("--seed", type=int, default=8500)
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    if args.total_steps < 1 or args.rollout_steps < 1 or args.seq_len < 1:
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
    plan = plan_resources(profile, "pretrain_platform", overrides)
    if args.envs is not None:
        overrides["gym_envs"] = args.envs
        plan = plan_resources(profile, "pretrain_platform", overrides)
    env_count = plan.gym_envs
    if (nvidia_smi and plan.train_device == "cpu" and not args.allow_cpu):
        _cpu_fallback_warning("ResourcePlan selected CPU despite an NVIDIA GPU being present")
        raise SystemExit("CPU training not started; rerun with --allow-cpu to confirm")
    if (args.device == "cuda" and (profile.gpu_memory_free_gb or 0.0) < 6.0):
        raise SystemExit("CUDA requested but ResourcePlan requires at least 6 GiB of free VRAM")
    device = torch.device(plan.train_device)
    threads = min(free_cores, max(1, plan.torch_threads_learner))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    root: Path = Path(args.runtime_root) / "v2" / "pretrain" / str(args.run_name)
    root.mkdir(parents=True, exist_ok=True)
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
    (root / "runtime_plan.json").write_text(json.dumps({
        "planned_envs": env_count, "free_physical_cores_reserved": free_cores,
        "torch_threads": threads, "train_device": plan.train_device,
        "actor_device": str(device if device.type == "cuda" else torch.device("cpu")),
        "priority": args.priority,
        "cuda_available": torch.cuda.is_available(), "nvidia_smi": nvidia_smi,
        "free_vram_gb": profile.gpu_memory_free_gb,
    }, indent=2) + "\n", encoding="utf-8")
    curriculum_path = root / "curriculum.json"
    curriculum = Curriculum(seed=args.seed)
    if args.resume and (args.resume / "curriculum.json").is_file():
        curriculum = Curriculum.load(args.resume / "curriculum.json")
    environments = SubprocVecEnv(
        [partial(_make_env, level=0, seed=args.seed + index) for index in range(env_count)],
        start_method="spawn",
    )
    environments.seed(args.seed)
    observations = cast(dict[str, np.ndarray], environments.reset())
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
    log_path = root / "log.csv"
    log_mode = "a" if log_path.exists() and global_step else "w"
    fieldnames = ["steps", "L_ppo", "L_semantic", "L_inverse_dynamics", "L_hero_motion",
                  "entropy", "kl", "explained_variance", "steps_per_second", "curriculum_level",
                  "success_rate_100", "quick_eval_success"]
    started = time.monotonic()
    started_wall = time.time()
    started_step = global_step
    last_checkpoint = global_step
    last_eval_million = global_step // 1_000_000
    stop_requested = [False]
    phase_totals: defaultdict[str, float] = defaultdict(float)
    profile_path = root / "phase_profile.csv"
    profile_fields = ["steps", "round_seconds", "collection_seconds", "inference_seconds",
                      "gae_seconds", "update_seconds", "quick_eval_seconds",
                      "cuda_peak_allocated_mb", "cuda_peak_reserved_mb"]
    profile_mode = "a" if profile_path.exists() and global_step else "w"
    profile_stream = profile_path.open(profile_mode, newline="", encoding="utf-8")
    profile_writer = csv.DictWriter(profile_stream, fieldnames=profile_fields)
    if profile_mode == "w":
        profile_writer.writeheader()
    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda _signum, _frame: stop_requested.__setitem__(0, True))
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
                next_observations_raw, rewards, dones, next_infos = environments.step(
                    actions.detach().cpu().numpy().astype(np.int64)
                )
                next_observations = {
                    key: np.array(value, copy=True)
                    for key, value in cast(dict[str, np.ndarray], next_observations_raw).items()
                }
                next_infos = list(next_infos)
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
                        curriculum.record(episode_had_exit[index])
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
            # Keep RTX 3060 activation memory bounded while avoiding the former
            # four-sequence microbatches, which multiplied optimizer calls.
            sequence_batch_size = max(1, min(16, requested_sequences, available_sequences))
            update_started = time.perf_counter()
            metrics = trainer.update(buffer.sequences(device), epochs=args.n_epochs,
                                     sequence_batch_size=sequence_batch_size)
            phase_round["update_seconds"] += time.perf_counter() - update_started
            quick_eval: float | str = ""
            quick_eval_seconds = 0.0
            if global_step // 1_000_000 > last_eval_million:
                actor_policy.load_state_dict({key: value.detach().cpu()
                                              for key, value in policy.state_dict().items()})
                eval_started = time.perf_counter()
                quick_eval = _quick_eval(actor_policy, args.seed + global_step)
                quick_eval_seconds = time.perf_counter() - eval_started
                last_eval_million = global_step // 1_000_000
            round_seconds = time.perf_counter() - round_started
            phase_round["quick_eval_seconds"] += quick_eval_seconds
            for phase_name, phase_value in phase_round.items():
                phase_totals[phase_name] += phase_value
            profile_writer.writerow({
                "steps": global_step, "round_seconds": round_seconds,
                "collection_seconds": phase_round["collection_seconds"],
                "inference_seconds": phase_round["inference_seconds"],
                "gae_seconds": phase_round["gae_seconds"],
                "update_seconds": phase_round["update_seconds"],
                "quick_eval_seconds": quick_eval_seconds,
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
            writer.writerow(row)
            stream.flush()
            if global_step - last_checkpoint >= 250_000:
                _save_checkpoint(root, policy, trainer, curriculum, global_step, metrics, action_table)
                prune_checkpoints(root)
                last_checkpoint = global_step
        _save_checkpoint(root, policy, trainer, curriculum, global_step, {}, action_table)
    signal.signal(signal.SIGINT, previous_sigint)
    curriculum.save(curriculum_path)
    experience_writer.close()
    environments.close()
    profile_stream.close()
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


def _make_env(level: int, seed: int) -> AdapterEnv:
    return AdapterEnv(PlatformGymAdapter(level=level, seed=seed), CombinedPlatformReward())


def _privileged_tensor(infos: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    rows = []
    for info in infos:
        signals = info.get("signals", {})
        position = signals.get("player_pos_room", (.5, .5))
        rows.append([float(position[0]), float(position[1]), float(signals.get("progress", 0)),
                     float(signals.get("hp", 0)) / 10.0])
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _quick_eval(policy: UniversalPolicy, seed: int, *, episodes_per_level: int = 2,
                max_steps: int = 300) -> float:
    """Run a brief deterministic held-out evaluation once each million steps."""
    device = next(policy.parameters()).device
    successes = 0
    episodes = 0
    policy.eval()
    for level in range(7):
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
                success |= bool(signals.get("room_changed"))
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
                   curriculum={"level": curriculum.level, "seed": curriculum.seed,
                               "episodes": curriculum.episodes,
                               "current": list(curriculum.current)})


def main() -> int:
    args = _parse()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
