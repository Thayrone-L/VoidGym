"""Ator de referência mínimo usado pelo harness E2E sem jogo real."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from tests.v2.fake_adapter import FakeAdapter
from voidgym.core.experience_store import ExperienceWriter
from voidgym.envs.adapter_env import AdapterEnv


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _wait(path: Path, stop: Path, timeout: float = 180.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if stop.exists():
            return False
        if path.is_file():
            return True
        time.sleep(0.02)
    raise TimeoutError(f"tempo esgotado esperando {path.name}")


def run(root: Path, actor_id: str, first_round: int, last_round: int,
        pause_after: int | None, adapter_name: str = "fake",
        policy_checkpoint: Path | None = None) -> None:
    policy_dir = root / "policy"
    rollout_dir = root / "rollouts"
    stop = root / "STOP"
    adapter: Any
    reward_fn: Callable[[Any, Any], float]
    if adapter_name == "platform_gym":
        from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter
        from voidgym.gyms.platform_gym.reward import CombinedPlatformReward
        adapter = PlatformGymAdapter(level=0)
        reward_fn = CombinedPlatformReward()
    else:
        adapter = FakeAdapter()
        reward_fn = lambda _before, _after: -0.01
    env = AdapterEnv(adapter, reward_fn)
    observation, _ = env.reset(seed=first_round)
    policy_model = None
    recurrent_state = None
    episode_start = True
    if policy_checkpoint is not None:
        import torch
        from voidgym.models.policy import UniversalPolicy
        from voidgym.training.checkpoints import load_universal

        policy_model = UniversalPolicy().eval()
        load_universal(policy_checkpoint, policy_model, env.actions)
        recurrent_state = policy_model.initial_state(1, torch.device("cpu"))
    with ExperienceWriter(root / "experience", adapter.game_id, "e2e",
                          source="test", block_steps=1,
                          action_table=env.actions) as writer:
        for round_id in range(first_round, last_round + 1):
            policy_version = 0
            action = 2
            if round_id > 1 and not _wait(policy_dir / f"round-{round_id - 1}.json", stop):
                return
            if round_id > 1:
                policy = json.loads((policy_dir / f"round-{round_id - 1}.json").read_text(encoding="utf-8"))
                policy_version = int(policy["updates"])
                action = int(policy["action"])
            if policy_model is not None:
                import torch

                tensors = {key: torch.as_tensor(value[None]) for key, value in observation.items()}
                mask = torch.as_tensor(env.action_masks()[None])
                with torch.no_grad():
                    prediction = policy_model(tensors, recurrent_state,
                                              torch.tensor([episode_start]), mask)
                logits, recurrent_state = prediction["logits"], prediction["state"]
                if not isinstance(logits, torch.Tensor) or not isinstance(recurrent_state, tuple):
                    raise TypeError("universal policy returned invalid actor outputs")
                action = int(logits.argmax(-1).item())
                episode_start = False
            observation, reward, terminated, truncated, info = env.step(action)
            writer.append({
                "frame": env.adapter.frame_rgb().copy(),
                "action": action,
                "action_mask": np.asarray(env.action_masks(), dtype=np.bool_),
                "reward_universal": reward,
                "done": terminated,
                "truncated": truncated,
                "scene": "synthetic",
                "tags": [actor_id, f"policy_version:{policy_version}"],
            })
            writer.flush()
            _write_json(rollout_dir / f"round-{round_id}-{actor_id}.json", {
                "actor_id": actor_id,
                "round": round_id,
                "policy_version": policy_version,
                "signals": info["signals"],
            })
            if pause_after == round_id and not _wait(root / "resume" / actor_id, stop):
                return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fake", "hollow_knight"), default="fake")
    parser.add_argument("--shared-root", type=Path)
    parser.add_argument("--experience-root", type=Path)
    parser.add_argument("--campaign-id")
    parser.add_argument("--actor-id")
    parser.add_argument("--universal-checkpoint", type=Path)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--wait-timeout", type=float, default=3600.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    v2_args, remaining = parser.parse_known_args()
    if v2_args.mode == "hollow_knight":
        required = (v2_args.shared_root, v2_args.experience_root,
                    v2_args.campaign_id, v2_args.universal_checkpoint)
        if any(value is None for value in required):
            parser.error("hollow_knight requires shared, experience, campaign and checkpoint paths")
        from tools.run_first_training import parse_args as parse_hk_args

        hk_args = parse_hk_args(remaining)
        if not hk_args.execute:
            parser.error("hollow_knight actor only runs with --execute")
        from voidgym.adapters.hollow_knight.runtime import create_hollow_knight_env
        from voidgym.models.policy import UniversalPolicy
        from voidgym.training.actor import run_actor
        from hk_agent.player_knowledge import PlayerKnowledge

        slot_id = int(os.environ.get("HK_AGENT_SLOT_ID", "-1"))
        actor_id = v2_args.actor_id or f"actor-slot-{slot_id if slot_id >= 0 else 'unknown'}"
        shared = v2_args.shared_root.resolve()
        env, owner = create_hollow_knight_env(
            hk_args, shared_root=shared, actor_id=actor_id,
            slot_id=slot_id, request_semantic_labels=True
        )
        try:
            extras_dim = int(PlayerKnowledge.space().shape[0])
            policy = UniversalPolicy(action_count=len(env.actions))
            policy.add_game_extras("hollow_knight", extras_dim)
            run_actor(
                env, policy, shared_root=shared, actor_id=actor_id,
                action_table=tuple(env.actions),
                universal_checkpoint=v2_args.universal_checkpoint.resolve(strict=True),
                experience_root=v2_args.experience_root,
                campaign_id=v2_args.campaign_id,
                rollout_steps=hk_args.rollout_steps,
                sequence_length=v2_args.sequence_length, gamma=hk_args.gamma or 0.995,
                gae_lambda=hk_args.gae_lambda or 0.95,
                device=v2_args.device, seed=hk_args.seed,
                wait_timeout=v2_args.wait_timeout, stop_file=hk_args.stop_file,
            )
        finally:
            owner.close()
        return 0

    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--first-round", type=int, required=True)
    parser.add_argument("--last-round", type=int, required=True)
    parser.add_argument("--pause-after", type=int)
    parser.add_argument("--adapter", choices=("fake", "platform_gym"), default="fake")
    parser.add_argument("--policy-checkpoint", type=Path)
    if v2_args.actor_id is not None:
        remaining = ["--actor-id", v2_args.actor_id, *remaining]
    args = parser.parse_args(remaining)
    if args.first_round < 1 or args.last_round < args.first_round:
        parser.error("round range is invalid")
    run(args.root, args.actor_id, args.first_round, args.last_round, args.pause_after,
        args.adapter, args.policy_checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
