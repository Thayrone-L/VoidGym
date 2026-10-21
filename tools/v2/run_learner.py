"""Learner mínimo para validar publicação, salvamento e retomada no e2e."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    args = parser.parse_args()
    root = args.root
    state = root / "state"
    progress_path = state / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    completed = int(progress.get("completed_rounds", 0))
    while not (state / "stop").exists() and completed < args.rounds:
        request_path = root / "requests" / "learner.txt"
        if not request_path.exists():
            time.sleep(0.02)
            continue
        try:
            round_number = int(request_path.read_text(encoding="utf-8").strip())
        except ValueError:
            time.sleep(0.02)
            continue
        if round_number <= completed:
            time.sleep(0.02)
            continue
        if not _both_rollouts_exist(root, round_number):
            time.sleep(0.02)
            continue
        checkpoint = root / "checkpoints" / f"round-{round_number:06d}.json"
        _atomic_json(checkpoint, {"round": round_number, "learner": "fake", "trained": True})
        completed = round_number
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress["completed_rounds"] = completed
        _atomic_json(progress_path, progress)
        _atomic_json(root / "acks" / f"learner-{round_number}.json", {"round": round_number})
    return 0


def _both_rollouts_exist(root: Path, round_number: int) -> bool:
    return all(
        (root / "rollouts" / actor / f"round-{round_number:06d}.json").exists()
        for actor in ("actor-1", "actor-2")
    )


if __name__ == "__main__":
    raise SystemExit(main())
