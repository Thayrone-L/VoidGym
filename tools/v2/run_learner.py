"""Learner de referência mínimo com checkpoint/resume para o harness E2E."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"registro inválido: {path}")
    return value


def _write(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def run(root: Path, actors: tuple[str, ...], checkpoint: Path,
        stop_after_round: int) -> None:
    rollout_dir = root / "rollouts"
    policy_dir = root / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    last_round = 0
    updates = 0
    policy_parameter = 0.0
    if checkpoint.is_file():
        saved = _read(checkpoint)
        last_round = int(saved["round"])
        updates = int(saved["updates"])
        policy_parameter = float(saved["policy_parameter"])
    for round_id in range(last_round + 1, stop_after_round + 1):
        deadline = time.monotonic() + 180.0
        rows: list[dict[str, Any]] = []
        while len(rows) < len(actors):
            for actor_id in actors:
                path = rollout_dir / f"round-{round_id}-{actor_id}.json"
                if path.is_file() and not any(row["actor_id"] == actor_id for row in rows):
                    row = _read(path)
                    if row.get("round") != round_id:
                        raise ValueError(f"rollout de {actor_id} tem rodada incorreta")
                    rows.append(row)
            if len(rows) == len(actors):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"atores não publicaram a rodada {round_id}")
            time.sleep(0.02)
        updates += 1
        positions = [float(row["signals"]["position"]) for row in rows]
        policy_parameter += sum(positions) / len(positions) / 1000.0
        saved = {"round": round_id, "updates": updates,
                 "actors": list(actors), "rollouts": rows,
                 "policy_parameter": policy_parameter}
        _write(checkpoint, saved)
        _write(policy_dir / f"round-{round_id}.json", {
            "round": round_id, "updates": updates,
            "policy_parameter": policy_parameter,
            "action": 2 if policy_parameter >= 0 else 1,
        })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--actors", default="actor-a,actor-b")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stop-after-round", type=int, required=True)
    args = parser.parse_args()
    actors = tuple(item for item in args.actors.split(",") if item)
    if not actors or args.stop_after_round < 1:
        parser.error("at least one actor and a positive target round are required")
    run(args.root, actors, args.checkpoint, args.stop_after_round)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
