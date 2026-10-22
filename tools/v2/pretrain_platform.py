"""Pré-treino PPO/A2C recorrente no ginásio, configurável e retomável."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from voidgym.core.config import AuxConfig, CampaignConfig, PPOConfig, ResourceOverrides, campaign_to_json
from voidgym.core.hardware import detect_hardware, plan_resources
from voidgym.core.paths import runtime_root
from voidgym.core.experience_store import ExperienceWriter
from voidgym.core.actions import UNIVERSAL_ACTIONS
from voidgym.envs.adapter_env import AdapterEnv
from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter
from voidgym.gyms.platform_gym.curriculum import Curriculum
from voidgym.models.policy import UniversalPolicy
from voidgym.training.checkpoints import load_universal, save_universal
from voidgym.training.reward import CombinedTrainingReward
from voidgym.training.recurrent_ppo import augment_sequence, ppo_objective
from voidgym.training.buffers import TrainingBuffer
from voidgym.models.perception import semantic_loss


def _priority() -> None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.windll.kernel32
        kernel.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.SetPriorityClass.restype = wintypes.BOOL
        kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00004000)


def _prune_checkpoints(run: Path) -> None:
    checkpoints = sorted(run.glob("universal-*"), key=lambda path: path.name)
    keep = {path for path in checkpoints[-5:]}
    keep.update(path for path in checkpoints if int(path.name.rsplit("-", 1)[1]) % 1_000_000 == 0)
    for obsolete in checkpoints:
        if obsolete not in keep:
            import shutil
            shutil.rmtree(obsolete)


def _training_seed(seed: int) -> int:
    """Keep every tenth reserved style seed out of the training split."""
    return seed + 1 if seed % 10 == 0 else seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="platform-pretrain")
    parser.add_argument("--total-steps", type=int, default=20_000_000)
    parser.add_argument("--envs", type=int)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2048)
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
    parser.add_argument("--resume")
    parser.add_argument("--hours-limit", type=float)
    parser.add_argument("--priority", default="below_normal")
    parser.add_argument("--amp", action="store_true", help="reserved until fp32 comparison is complete")
    args = parser.parse_args()
    if args.amp:
        parser.error("AMP remains disabled until a 200k-step fp32 comparison is recorded")
    _priority()
    profile = detect_hardware()
    resources = plan_resources(profile, "pretrain_platform", ResourceOverrides(
        gym_envs=args.envs, batch_size=args.batch_size,
    ))
    env_count = args.envs or resources.gym_envs
    threads = resources.torch_threads_learner
    torch.set_num_threads(threads)
    device = resources.train_device
    config = CampaignConfig(campaign_id=args.run_name, aux=AuxConfig(
        c_sem=args.c_sem, c_inv=args.c_inv, c_mot=args.c_mot,
    ), ppo=PPOConfig(
        learning_rate=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_range=args.clip, ent_coef=args.ent_coef, n_epochs=args.n_epochs,
        batch_size=args.batch_size, seq_len=args.seq_len, rollout_steps=args.rollout_steps,
    ), resources=ResourceOverrides(gym_envs=env_count, train_device=device,
                                   torch_threads_learner=threads, batch_size=args.batch_size)).validate()
    run = runtime_root() / "pretrain" / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(campaign_to_json(config), encoding="utf-8")
    (run / "run_arguments.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (run / "hardware.json").write_text(json.dumps(profile.__dict__, indent=2), encoding="utf-8")
    log_path = run / "log.csv"
    new_log = not log_path.exists()
    log = log_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(log, fieldnames=[
        "steps", "L_ppo", "L_semantic", "L_inverse_dynamics", "L_hero_motion",
        "entropy", "KL", "explained_variance", "steps_per_second", "level", "success_rate",
        "success_by_level",
    ])
    if new_log:
        writer.writeheader()
    table = tuple(a.name for a in UNIVERSAL_ACTIONS)
    model = load_universal(args.resume, table, device) if args.resume else UniversalPolicy(len(table)).to(device)
    actor = copy.deepcopy(model).to("cpu").eval()
    semantic_weight_path = Path(__file__).parents[2] / "src/voidgym/models/semantic_weights.json"
    semantic_weights = None
    if semantic_weight_path.exists():
        weight_data = json.loads(semantic_weight_path.read_text(encoding="utf-8"))["weights_by_class"]
        semantic_weights = torch.tensor([weight_data[name] for name in (
            "EMPTY", "SOLID", "PLATFORM", "HAZARD", "ENEMY", "PLAYER", "BREAKABLE",
            "PICKUP", "INTERACTIVE", "EXIT",
        )], dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    reward_models = [CombinedTrainingReward() for _ in range(env_count)]
    envs = [
        AdapterEnv(PlatformGymAdapter(split="train"), reward_models[i])
        for i in range(env_count)
    ]
    experience = ExperienceWriter(
        run / "store", "platform_gym", args.run_name, source="platform_gym", action_table=table,
    )
    curriculum = Curriculum(run / "curriculum.json")
    curriculum.load()
    env_levels = [curriculum.sample_level(i) for i in range(env_count)]
    observations = [
        env.reset(seed=_training_seed(i), options={"level": env_levels[i]})[0]
        for i, env in enumerate(envs)
    ]
    current_labels = [env._info().get("training_labels", {}) for env in envs]
    hiddens: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * env_count
    episode_success: list[bool] = []
    successes_by_level: dict[int, list[bool]] = {}
    started = time.monotonic()
    steps = int(json.loads((Path(args.resume) / "manifest.json").read_text(encoding="utf-8"))["steps"]) if args.resume else 0
    last_metrics: dict[str, object] = {}
    max_seconds = args.hours_limit * 3600 if args.hours_limit else float("inf")
    next_quick_eval = (steps // 1_000_000 + 1) * 1_000_000
    stop_requested = False

    def _request_stop(signum: int, frame: object) -> None:
        nonlocal stop_requested
        del signum, frame
        stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)
    while steps < args.total_steps and time.monotonic() - started < max_seconds and not stop_requested:
        # The collector is a single synchronous actor; a large intra-op pool on
        # each tiny batch adds heavy thread startup/coordination overhead.
        torch.set_num_threads(1)
        actor.load_state_dict({key: value.detach().cpu() for key, value in model.state_dict().items()})
        batch_obs: list[dict[str, np.ndarray]] = []
        batch_masks: list[np.ndarray] = []
        training_buffer = TrainingBuffer()
        old_logs: list[torch.Tensor] = []
        old_values: list[torch.Tensor] = []
        rewards: list[float] = []
        bootstrap_values: list[float] = []
        for _ in range(min(args.rollout_steps, max(1, (args.total_steps - steps + env_count - 1) // env_count))):
            for i, env in enumerate(envs):
                obs = observations[i]
                mask_np = env.action_masks()
                image = torch.as_tensor(obs["pixels"], device="cpu").unsqueeze(0)
                history = torch.as_tensor(obs["action_history"], device="cpu").unsqueeze(0)
                mask = torch.as_tensor(mask_np, device="cpu").unsqueeze(0)
                privileged = np.asarray([
                    *env.adapter.signals()["player_pos_room"],
                    float(env.adapter.signals()["progress"]),
                    float(env.adapter.signals()["hp"]) / max(1, env.adapter.signals()["max_hp"]),
                ], dtype=np.float32)
                privileged_tensor = torch.as_tensor(privileged, device="cpu").unsqueeze(0)
                with torch.no_grad():
                    logits, value, next_state, _, _ = actor(image, history, hiddens[i], mask, privileged_tensor)
                    dist = Categorical(logits=logits[:, -1])
                    action_t = dist.sample()  # type: ignore[no-untyped-call]
                next_obs, reward, terminated, truncated, step_info = env.step(int(action_t.item()))
                done = terminated or truncated
                if random.random() < .01:
                    training_sample = current_labels[i]
                    hero_velocity = np.asarray(training_sample.get("hero_velocity", (np.nan, np.nan)), dtype=np.float32)
                    experience.append({
                        "frame": obs["pixels"], "action": int(action_t.item()), "action_mask": mask_np,
                        "reward_universal": reward_models[i].last.universal,
                        "reward_game": reward_models[i].last.game, "done": terminated,
                        "truncated": truncated, "sem": training_sample.get("sem", np.full((27, 48), 255)),
                        "hero_motion": np.asarray((*hero_velocity, training_sample.get("on_ground", np.nan))),
                        "scene": str(step_info.get("signals", {}).get("room_id", "")),
                    })
                if done:
                    bootstrap_values.append(0.0)
                else:
                    next_image = torch.as_tensor(next_obs["pixels"], device="cpu").unsqueeze(0)
                    next_history = torch.as_tensor(next_obs["action_history"], device="cpu").unsqueeze(0)
                    with torch.no_grad():
                        next_value = actor(next_image, next_history, next_state)[1][:, -1]
                    bootstrap_values.append(float(next_value.item()))
                batch_obs.append(obs)
                batch_masks.append(mask_np)
                training_buffer.add(obs["pixels"], int(action_t.item()), {
                    "training_labels": current_labels[i], "privileged": privileged,
                    "discontinuity": done,
                })
                old_logs.append(dist.log_prob(action_t).detach().squeeze())  # type: ignore[no-untyped-call]
                old_values.append(value[:, -1].detach().squeeze())
                rewards.append(reward)
                hiddens[i] = next_state if not (terminated or truncated) else None
                if terminated or truncated:
                    success = bool(terminated)
                    episode_success.append(success)
                    successes_by_level.setdefault(env_levels[i], []).append(success)
                    curriculum.record(success)
                    curriculum.save()
                    env_levels[i] = curriculum.sample_level(steps + i)
                    observations[i] = env.reset(
                        seed=_training_seed(steps + i), options={"level": env_levels[i]},
                    )[0]
                    current_labels[i] = env._info().get("training_labels", {})
                else:
                    observations[i] = next_obs
                    current_labels[i] = step_info.get("training_labels", {})
                steps += 1
        if not batch_obs:
            break
        time_steps = len(batch_obs) // env_count
        imgs = torch.as_tensor(np.stack([o["pixels"] for o in batch_obs]), device=device).reshape(
            time_steps, env_count, 12, 108, 192,
        ).transpose(0, 1)
        histories = torch.as_tensor(np.stack([o["action_history"] for o in batch_obs]), device=device).reshape(
            time_steps, env_count, 3,
        ).transpose(0, 1)
        masks = torch.as_tensor(np.stack(batch_masks), device=device).reshape(
            time_steps, env_count, -1,
        ).transpose(0, 1)
        priv = torch.as_tensor(np.stack(training_buffer.privileged), device=device).reshape(
            time_steps, env_count, 4,
        ).transpose(0, 1)
        actions_t = torch.tensor(training_buffer.actions, device=device).reshape(
            time_steps, env_count,
        ).transpose(0, 1)
        old_log = torch.stack(old_logs).reshape(time_steps, env_count).transpose(0, 1)
        old_value = torch.stack(old_values).reshape(time_steps, env_count).transpose(0, 1)
        rew = torch.tensor(rewards, device=device).reshape(time_steps, env_count).transpose(0, 1)
        done_t = torch.tensor(training_buffer.discontinuity, device=device, dtype=torch.float32).reshape(
            time_steps, env_count,
        ).transpose(0, 1)
        episode_starts = torch.zeros_like(done_t)
        episode_starts[:, 0] = 1.0
        episode_starts[:, 1:] = done_t[:, :-1]
        next_values = torch.tensor(bootstrap_values, device=device).reshape(time_steps, env_count).transpose(0, 1)
        labels_t = torch.as_tensor(np.stack(training_buffer.semantic), device=device).reshape(
            time_steps, env_count, 27, 48,
        ).transpose(0, 1)
        motion_t = torch.as_tensor(np.stack(training_buffer.hero_motion), device=device).reshape(
            time_steps, env_count, 3,
        ).transpose(0, 1)
        advantages = torch.zeros_like(rew)
        gae = torch.zeros(env_count, device=device)
        for t in range(time_steps - 1, -1, -1):
            continuation = 1.0 - done_t[:, t]
            delta = rew[:, t] + args.gamma * next_values[:, t] * continuation - old_value[:, t]
            gae = delta + args.gamma * args.gae_lambda * continuation * gae
            advantages[:, t] = gae
        returns = advantages + old_value
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        torch.set_num_threads(threads)
        ppo_metric = sem_metric = inv_metric = mot_metric = ent_metric = kl_metric = 0.0
        count = 0
        for _epoch in range(args.n_epochs):
            recurrent_state = None
            for start in range(0, time_steps, args.seq_len):
                end = min(start + args.seq_len, time_steps)
                img_seq, sem_seq = augment_sequence(
                    imgs[:, start:end], labels_t[:, start:end], random.randint(-4, 4), random.randint(-4, 4),
                    brightness=random.uniform(-.1, .1), contrast=random.uniform(.9, 1.1),
                )
                logits, value, recurrent_state, sem_logits, motion_pred = model(
                    img_seq, histories[:, start:end], recurrent_state, masks[:, start:end],
                    priv[:, start:end], episode_starts=episode_starts[:, start:end],
                )
                dist = Categorical(logits=logits)
                new_log = dist.log_prob(actions_t[:, start:end])  # type: ignore[no-untyped-call]
                ppo_loss, metrics = ppo_objective(
                    new_log.reshape(-1), old_log[:, start:end].reshape(-1),
                    advantages[:, start:end].reshape(-1), value.reshape(-1),
                    returns[:, start:end].reshape(-1), dist.entropy().reshape(-1),  # type: ignore[no-untyped-call]
                    args.clip, args.ent_coef,
                )
                sem_loss = semantic_loss(sem_logits.flatten(0, 1), sem_seq.flatten(0, 1), semantic_weights)
                vectors = model.encoder(img_seq.reshape(-1, 12, 108, 192))[1].reshape(
                    env_count, end - start, 256,
                )
                inv_logits = model.inverse_dynamics(vectors[:, :-1], vectors[:, 1:])
                inv_targets = actions_t[:, start:end - 1]
                inv_valid = ~done_t[:, start:end - 1].bool()
                inv_loss = torch.nn.functional.cross_entropy(inv_logits[inv_valid], inv_targets[inv_valid]) if inv_valid.any() else inv_logits.sum() * 0
                valid_motion = torch.isfinite(motion_t[:, start:end]).all(-1)
                mot_loss = torch.nn.functional.mse_loss(motion_pred[valid_motion], motion_t[:, start:end][valid_motion]) if valid_motion.any() else motion_pred.sum() * 0
                loss = ppo_loss + args.c_sem * sem_loss + args.c_inv * inv_loss + args.c_mot * mot_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                optimizer.step()
                recurrent_state = tuple(hidden.detach() for hidden in recurrent_state)
                ppo_metric += float(ppo_loss.detach()); sem_metric += float(sem_loss.detach())
                inv_metric += float(inv_loss.detach()); mot_metric += float(mot_loss.detach())
                ent_metric += float(metrics["entropy"]); kl_metric += float(metrics["KL"])
                count += 1
        rate = steps / max(time.monotonic() - started, 1e-9)
        current_history = list(curriculum.results[curriculum.level])[-100:]
        success_rate = float(np.mean(current_history)) if current_history else 0.0
        level_rates = {
            str(level): float(np.mean(successes[-100:]))
            for level, successes in successes_by_level.items()
        }
        writer.writerow({"steps": steps, "L_ppo": ppo_metric/count, "L_semantic": sem_metric/count,
                         "L_inverse_dynamics": inv_metric/count, "L_hero_motion": mot_metric/count,
                         "entropy": ent_metric/count, "KL": kl_metric/count,
                         "explained_variance": float(1 - torch.var(returns - old_value) / (torch.var(returns) + 1e-8)),
                         "steps_per_second": rate, "level": curriculum.level, "success_rate": success_rate,
                         "success_by_level": json.dumps(level_rates, sort_keys=True)})
        last_metrics = {
            "L_ppo": ppo_metric/count, "L_semantic": sem_metric/count,
            "L_inverse_dynamics": inv_metric/count, "L_hero_motion": mot_metric/count,
            "entropy": ent_metric/count, "KL": kl_metric/count,
            "explained_variance": float(1 - torch.var(returns - old_value) / (torch.var(returns) + 1e-8)),
            "steps_per_second": rate, "level": curriculum.level, "success_rate": success_rate,
            "success_by_level": level_rates,
        }
        log.flush()
        if steps % 250_000 < env_count * args.rollout_steps:
            save_universal(model, run / f"universal-{steps:09d}", steps, {"level": curriculum.level}, last_metrics)
            _prune_checkpoints(run)
        print(json.dumps({"steps": steps, "steps_per_second": rate, "L_semantic": sem_metric/count}), flush=True)
        if steps >= next_quick_eval:
            checkpoint = run / f"universal-{steps:09d}"
            if not checkpoint.exists():
                save_universal(model, checkpoint, steps, {"level": curriculum.level}, last_metrics)
                _prune_checkpoints(run)
            evaluation_dir = run / "quick-evaluation" / f"steps-{steps:09d}"
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("eval_platform.py")),
                 str(checkpoint), "--quick", "--output", str(evaluation_dir)],
                capture_output=True, text=True, check=False,
            )
            if completed.returncode:
                failure_path = run / "quick-evaluation" / "failures.jsonl"
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                with failure_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"steps": steps, "stderr": completed.stderr}) + "\n")
            else:
                print(json.dumps({"quick_evaluation_steps": steps, "results": completed.stdout}), flush=True)
            next_quick_eval += 1_000_000
    save_universal(model, run / f"universal-{steps:09d}", steps, {"level": curriculum.level}, last_metrics)
    _prune_checkpoints(run)
    log.close()
    experience.flush()
    for env in envs:
        env.close()
    if torch.cuda.is_available():
        print(json.dumps({"cuda_peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
