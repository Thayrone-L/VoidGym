"""Ator mínimo de ciclo de vida usado pelo e2e e pelo supervisor da v2."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from voidgym.core.experience_store import ExperienceWriter


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--actor-id", required=True)
    args = parser.parse_args()
    root = args.root
    state = root / "state"
    requests = root / "requests"
    actor_dir = root / "actors" / args.actor_id
    actor_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(actor_dir / "pid", str(os.getpid()))
    writer = ExperienceWriter(
        root / "runtime",
        "fake",
        f"e2e-{args.actor_id}",
        source="actor",
        action_table=("NOOP", "RIGHT"),
    )
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
            _collect_round(writer, root, args.actor_id, round_number)
            _atomic_text(root / "acks" / f"{args.actor_id}-{round_number}.txt", "ready")
            last_round = round_number
        return 0
    finally:
        writer.flush()


def _collect_round(writer: ExperienceWriter, root: Path, actor_id: str, round_number: int) -> None:
    rollout = root / "rollouts" / actor_id / f"round-{round_number:06d}.json"
    if rollout.exists():
        return
    frame = np.zeros((108, 192, 3), dtype=np.uint8)
    frame[48:56, 40 + round_number : 48 + round_number] = 255
    writer.append(
        {
            "frame": frame,
            "action": 1,
            "action_mask": np.asarray([True, True]),
            "reward_universal": 0.1,
            "reward_game": 0.0,
            "scene": "fake-room",
            "source": "actor",
        }
    )
    writer.end_episode()
    _atomic_text(rollout, json.dumps({"actor": actor_id, "round": round_number, "steps": 1}))


if __name__ == "__main__":
    raise SystemExit(main())
