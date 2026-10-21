"""Ator mínimo de ciclo de vida usado pelo e2e e pelo supervisor da v2."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from voidgym.core.experience_store import ExperienceWriter
from voidgym.envs.adapter_env import AdapterEnv
from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter
from voidgym.gyms.platform_gym.reward import GymReward
from voidgym.core.reward import UniversalReward


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--actor-id", required=True)
    parser.add_argument("--game", choices=("fake", "platform_gym"), default="fake")
    args = parser.parse_args()
    root = args.root
    state = root / "state"
    requests = root / "requests"
    actor_dir = root / "actors" / args.actor_id
    actor_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(actor_dir / "pid", str(os.getpid()))
    writer = ExperienceWriter(
        root / "runtime",
        args.game,
        f"e2e-{args.actor_id}",
        source="actor",
        action_table=("NOOP", "RIGHT"),
    )
    game_reward = GymReward()
    env = AdapterEnv(PlatformGymAdapter(), UniversalReward()) if args.game == "platform_gym" else None
    progress = json.loads((state / "progress.json").read_text(encoding="utf-8"))
    last_round = int(progress.get("completed_rounds", 0))
    try:
        while not (state / "stop").exists():
            request = requests / f"{args.actor_id}.txt"
            if not request.exists():
                time.sleep(0.02)
                continue
            try:
                round_number = int(request.read_text(encoding="utf-8").strip())
            except ValueError:
                time.sleep(0.02)
                continue
            if round_number <= last_round:
                time.sleep(0.02)
                continue
            _collect_round(writer, root, args.actor_id, round_number, env, game_reward)
            _atomic_text(root / "acks" / f"{args.actor_id}-{round_number}.txt", "ready")
            last_round = round_number
        return 0
    finally:
        writer.flush()


def _collect_round(writer: ExperienceWriter, root: Path, actor_id: str, round_number: int, env: AdapterEnv | None = None, game_reward: GymReward | None = None) -> None:
    rollout = root / "rollouts" / actor_id / f"round-{round_number:06d}.json"
    if rollout.exists():
        return
    mask: NDArray[np.bool_]
    if env is None:
        frame = np.zeros((108, 192, 3), dtype=np.uint8)
        frame[48:56, 40 + round_number : 48 + round_number] = 255
        action, mask, labels, reward_universal, reward_game, scene = 1, np.asarray([True, True], dtype=np.bool_), {}, 0.1, 0.0, "fake-room"
    else:
        env.reset(seed=round_number)
        if game_reward is not None:
            game_reward.reset()
        before = dict(env.adapter.signals())
        action = env.adapter.action_table().index("RIGHT")
        _, reward, _, _, info = env.step(action)
        frame = env.adapter.frame_rgb()
        mask = env.action_masks()
        labels = info.get("training_labels", {})
        reward_universal = float(reward)
        reward_game = game_reward(before, info["signals"]) if game_reward else 0.0
        scene = str(info["signals"]["room_id"])
    step = {"frame": frame, "action": action, "action_mask": mask,
            "reward_universal": reward_universal, "reward_game": reward_game,
            "scene": scene, "source": "actor"}
    if labels.get("sem") is not None:
        step["sem"] = labels["sem"]
    if labels.get("hero_velocity") is not None:
        velocity = np.asarray(labels["hero_velocity"], dtype=np.float32).reshape(-1)
        step["hero_motion"] = np.asarray([velocity[0], velocity[1], labels.get("on_ground", 0)], dtype=np.float32)
    writer.append(step)
    writer.end_episode()
    _atomic_text(rollout, json.dumps({"actor": actor_id, "round": round_number, "steps": 1}))


if __name__ == "__main__":
    raise SystemExit(main())
