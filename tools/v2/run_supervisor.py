"""Supervisor de teste: inicia o learner e dois atores, recupera falhas e retoma."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from voidgym.core.config import CampaignConfig, save_config
from voidgym.core.hardware import detect_hardware, plan_resources, write_hardware_plan


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _spawn(script: str, root: Path, *args: str) -> subprocess.Popen[bytes]:
    command = [sys.executable, str(Path(__file__).with_name(script)), "--root", str(root), *args]
    return subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2])


def _stop(processes: dict[str, subprocess.Popen[bytes]], state: Path) -> None:
    _write_text(state / "stop", "stop")
    for process in processes.values():
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    state = root / "state"
    root.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    (state / "stop").unlink(missing_ok=True)
    progress_path = state / "progress.json"
    if not args.resume or not progress_path.exists():
        _write_text(progress_path, json.dumps({"completed_rounds": 0, "actor_restarts": 0}))
    _write_files(root)
    processes: dict[str, subprocess.Popen[bytes]] = {
        "learner": _spawn("run_learner.py", root, "--rounds", str(args.rounds)),
        "actor-1": _spawn("run_actor.py", root, "--actor-id", "actor-1"),
        "actor-2": _spawn("run_actor.py", root, "--actor-id", "actor-2"),
    }
    try:
        _run_rounds(root, args.rounds, progress_path, processes)
        return 0
    except (TimeoutError, RuntimeError) as error:
        print(f"Supervisor v2: {error}", file=sys.stderr)
        return 1
    finally:
        _stop(processes, state)


def _write_files(root: Path) -> None:
    config = CampaignConfig(campaign_id="e2e", kind="evaluation").validate()
    save_config(config, root / "config.json")
    profile = detect_hardware(root)
    plan = plan_resources(profile, "evaluation")
    write_hardware_plan(profile, plan, root)


def _run_rounds(
    root: Path,
    rounds: int,
    progress_path: Path,
    processes: dict[str, subprocess.Popen[bytes]],
) -> None:
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    completed = int(progress.get("completed_rounds", 0))
    for round_number in range(completed + 1, rounds + 1):
        for actor in ("actor-1", "actor-2"):
            _ensure_running(actor, processes, root, progress_path)
            _write_text(root / "requests" / f"{actor}.txt", str(round_number))
        _wait_for_actors(root, round_number, processes, progress_path)
        _write_text(root / "requests" / "learner.txt", str(round_number))
        _wait_for_learner(root, round_number, processes["learner"])
        time.sleep(0.25)


def _ensure_running(
    actor: str,
    processes: dict[str, subprocess.Popen[bytes]],
    root: Path,
    progress_path: Path,
) -> None:
    process = processes[actor]
    if process.poll() is not None:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress["actor_restarts"] = int(progress.get("actor_restarts", 0)) + 1
        _write_text(progress_path, json.dumps(progress))
        processes[actor] = _spawn("run_actor.py", root, "--actor-id", actor)


def _wait_for_actors(
    root: Path,
    round_number: int,
    processes: dict[str, subprocess.Popen[bytes]],
    progress_path: Path,
) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        ready = True
        for actor in ("actor-1", "actor-2"):
            _ensure_running(actor, processes, root, progress_path)
            ready &= (root / "acks" / f"{actor}-{round_number}.txt").exists()
        if ready:
            return
        time.sleep(0.02)
    raise TimeoutError(f"atores não concluíram a rodada {round_number}")


def _wait_for_learner(root: Path, round_number: int, learner: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 30
    ack = root / "acks" / f"learner-{round_number}.json"
    while time.monotonic() < deadline:
        if ack.exists():
            return
        if learner.poll() is not None:
            raise RuntimeError(f"learner terminou antes da rodada {round_number}")
        time.sleep(0.02)
    raise TimeoutError(f"learner não concluiu a rodada {round_number}")


if __name__ == "__main__":
    raise SystemExit(main())
