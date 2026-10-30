"""Recurrent universal-policy actor loop for one game adapter."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.distributions import Categorical

from voidgym.core.experience_store import ExperienceWriter
from voidgym.core.observation import fit_to_universal
from voidgym.envs.adapter_env import AdapterEnv
from voidgym.models.policy import UniversalPolicy
from voidgym.training.buffers import RecurrentRolloutBuffer, _privileged
from voidgym.training.checkpoints import load_universal
from voidgym.training.shared_rollouts import publish_rollout


def _latest_policy(root: Path, *, minimum_round: int, timeout: float,
                   stop_file: Path | None) -> tuple[int, Path] | None:
    deadline = time.monotonic() + timeout
    latest_path = root / "latest.json"
    while time.monotonic() < deadline:
        if stop_file is not None and stop_file.is_file():
            return None
        try:
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
            policy_round = payload.get("round")
            relative = payload.get("policy")
            if type(policy_round) is int and policy_round >= minimum_round and isinstance(relative, str):
                path = (root / relative).resolve(strict=True)
                path.relative_to(root.resolve(strict=True))
                if path.is_dir() and (path / "manifest.json").is_file():
                    return policy_round, path
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for policy round {minimum_round}")


def _to_tensors(observation: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value[None], device=device) for key, value in observation.items()}


def _forward(policy: UniversalPolicy, observation: dict[str, torch.Tensor],
             state: tuple[torch.Tensor, torch.Tensor], starts: torch.Tensor,
             mask: torch.Tensor, privileged: torch.Tensor,
             game_id: str) -> dict[str, Any]:
    extras = observation.get("game_extras")
    return cast(dict[str, Any], policy(
        observation, state, starts, mask, privileged=privileged,
        game_id=game_id if extras is not None else None, game_extras=extras,
    ))


def _latest_rgb(env: AdapterEnv) -> np.ndarray:
    return fit_to_universal(env.adapter.frame_rgb())


def _experience_row(env: AdapterEnv, observation: dict[str, np.ndarray], action: int,
                    reward: float, done: bool, truncated: bool, info: dict[str, Any],
                    action_mask: np.ndarray, actor_id: str, policy_round: int) -> dict[str, Any]:
    labels = info.get("training_labels")
    labels = labels if isinstance(labels, dict) else {}
    sem = labels.get("sem")
    if not isinstance(sem, np.ndarray) or sem.shape != (27, 48):
        sem = np.full((27, 48), 255, dtype=np.uint8)
    velocity = labels.get("hero_velocity")
    grounded = labels.get("on_ground")
    motion = (np.asarray([velocity[0], velocity[1], grounded], dtype=np.float32)
              if velocity is not None and grounded is not None
              else np.full(3, np.nan, dtype=np.float32))
    signals = info.get("signals", {})
    return {
        "frame": _latest_rgb(env), "action": action, "action_mask": action_mask,
        "action_history": observation["action_history"],
        "reward_universal": reward, "reward_game": reward,
        "done": done, "truncated": truncated,
        "scene": str(signals.get("scene_name", "")),
        "tags": [actor_id, f"policy_round:{policy_round}"],
        "sem": sem, "hero_motion": motion,
        "interact_prompt": int(bool(labels.get("interact_prompt", False))),
        "interact_prompt_valid": bool(labels.get("interact_prompt_valid", False)),
        "privileged": _privileged(info).numpy(),
    }


def _collect_steps(
    env: AdapterEnv, policy: UniversalPolicy, buffer: RecurrentRolloutBuffer,
    writer: ExperienceWriter, *, actor_id: str, policy_round: int,
    observation: dict[str, np.ndarray], info: dict[str, Any],
    state: tuple[torch.Tensor, torch.Tensor], episode_start: bool,
    total_steps: int, episode_return: float, round_returns: list[float],
    device: torch.device, seed: int, stop_file: Path | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], tuple[torch.Tensor, torch.Tensor],
           bool, int, float, bool]:
    for _ in range(buffer.steps):
        if stop_file is not None and stop_file.is_file():
            return observation, info, state, episode_start, total_steps, episode_return, True
        masks = np.asarray(env.action_masks(), dtype=np.bool_)
        obs_t = _to_tensors(observation, device)
        mask_t = torch.as_tensor(masks[None], device=device)
        critic_context = _privileged(info)[None].to(device)
        starts = torch.tensor([episode_start], dtype=torch.bool, device=device)
        with torch.no_grad():
            output = _forward(policy, obs_t, state, starts, mask_t,
                              critic_context, env.adapter.game_id)
            logits, values = output["logits"], output["values"]
            next_state = output["state"]
            if not isinstance(logits, torch.Tensor) or not isinstance(values, torch.Tensor):
                raise TypeError("universal policy returned invalid actor tensors")
            distribution = Categorical(logits=logits)
            action_t = distribution.sample()  # type: ignore[no-untyped-call]
            log_prob = distribution.log_prob(action_t)  # type: ignore[no-untyped-call]
        action = int(action_t.item())
        old_state = (state[0].detach().clone(), state[1].detach().clone())
        next_observation, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        buffer.add(
            obs_t, action_t, torch.tensor([reward], dtype=torch.float32, device=device),
            torch.tensor([done], dtype=torch.bool, device=device), values.reshape(1),
            log_prob.reshape(1), mask_t, starts, old_state, [next_info],
        )
        writer.append(_experience_row(env, observation, action, reward, terminated,
                                      truncated, next_info, masks, actor_id, policy_round))
        total_steps += 1
        episode_return += reward
        if done:
            round_returns.append(episode_return)
            episode_return = 0.0
            observation, info = env.reset(seed=seed + total_steps)
            state, episode_start = policy.initial_state(1, device), True
        else:
            observation, info, state, episode_start = next_observation, next_info, next_state, False
    return observation, info, state, episode_start, total_steps, episode_return, False


def _publish_round(
    env: AdapterEnv, policy: UniversalPolicy, buffer: RecurrentRolloutBuffer,
    writer: ExperienceWriter, *, shared_root: Path, actor_id: str, round_id: int,
    policy_round: int, action_table: tuple[str, ...], gamma: float, gae_lambda: float,
    observation: dict[str, np.ndarray],
    info: dict[str, Any], state: tuple[torch.Tensor, torch.Tensor], episode_start: bool,
    device: torch.device, wait_timeout: float, stop_file: Path | None,
) -> tuple[Path, tuple[int, Path] | None]:
    with torch.no_grad():
        output = _forward(
            policy, _to_tensors(observation, device), state,
            torch.tensor([episode_start], dtype=torch.bool, device=device),
            torch.as_tensor(env.action_masks()[None], device=device),
            _privileged(info)[None].to(device), env.adapter.game_id,
        )
    last = output["values"]
    if not isinstance(last, torch.Tensor):
        raise TypeError("universal policy returned invalid bootstrap value")
    buffer.compute_returns(last, gamma=gamma, gae_lambda=gae_lambda)
    rollout_file = publish_rollout(
        shared_root, round_id=round_id, actor_id=actor_id,
        policy_round=policy_round, action_table=action_table,
        sequences=buffer.sequences(torch.device("cpu")),
    )
    writer.flush()
    latest = _latest_policy(shared_root, minimum_round=round_id,
                            timeout=wait_timeout, stop_file=stop_file)
    return rollout_file, latest


def run_actor(
              env: AdapterEnv, policy: UniversalPolicy, *, shared_root: Path,
              actor_id: str, action_table: tuple[str, ...], universal_checkpoint: Path,
              experience_root: Path, campaign_id: str, rollout_steps: int = 512,
              sequence_length: int = 32, gamma: float = .995, gae_lambda: float = .95,
              device: str = "cpu", seed: int = 7, wait_timeout: float = 3600.0,
              stop_file: Path | None = None, semantic_labels: bool = True) -> dict[str, Any]:
    if rollout_steps < 1 or sequence_length < 1 or rollout_steps % sequence_length:
        raise ValueError("rollout_steps must be a positive multiple of sequence_length")
    if tuple(env.actions) != action_table:
        raise ValueError("actor action table differs from the campaign contract")
    selected_device = torch.device(device)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy.to(selected_device).eval()
    base_manifest = load_universal(universal_checkpoint, policy, action_table)
    policy_round = int((base_manifest.get("metrics") or {}).get("shared_round", 0))
    latest = _latest_policy(shared_root, minimum_round=policy_round,
                            timeout=wait_timeout, stop_file=stop_file)
    if latest is None:
        return {"status": "CANCELLED", "actor_id": actor_id, "rounds": 0}
    policy_round, policy_path = latest
    load_universal(policy_path, policy, action_table)
    observation, info = env.reset(seed=seed)
    state = policy.initial_state(1, selected_device)
    episode_start = True
    round_id = policy_round + 1
    total_steps = 0
    episode_return = 0.0
    round_returns: list[float] = []
    writer = ExperienceWriter(experience_root, env.adapter.game_id, campaign_id,
                              source="actor", block_steps=min(rollout_steps, 1024),
                              action_table=action_table)
    try:
        while stop_file is None or not stop_file.is_file():
            buffer = RecurrentRolloutBuffer(rollout_steps, 1, sequence_length)
            (observation, info, state, episode_start, total_steps,
             episode_return, stopped) = _collect_steps(
                env, policy, buffer, writer, actor_id=actor_id, policy_round=policy_round,
                observation=observation, info=info, state=state, episode_start=episode_start,
                total_steps=total_steps, episode_return=episode_return,
                round_returns=round_returns, device=selected_device, seed=seed,
                stop_file=stop_file,
            )
            if stopped:
                break
            rollout_file, latest = _publish_round(
                env, policy, buffer, writer, shared_root=shared_root, actor_id=actor_id,
                round_id=round_id, policy_round=policy_round, action_table=action_table,
                gamma=gamma, gae_lambda=gae_lambda, observation=observation, info=info,
                state=state, episode_start=episode_start, device=selected_device,
                wait_timeout=wait_timeout, stop_file=stop_file,
            )
            if latest is None:
                break
            policy_round, policy_path = latest
            load_universal(policy_path, policy, action_table)
            policy.eval()
            round_id += 1
            if len(round_returns) > 100:
                del round_returns[:-100]
    finally:
        writer.close()
        env.close()
    return {"status": "CANCELLED" if stop_file is not None and stop_file.is_file() else "COMPLETED",
            "actor_id": actor_id, "completed_steps": total_steps,
            "last_policy_round": policy_round, "last_rollout": str(rollout_file) if "rollout_file" in locals() else None,
            "recent_episode_returns": round_returns}
