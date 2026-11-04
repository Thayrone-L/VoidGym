"""Crash-resume supervisor for the v2 Hollow Knight shared learner."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from voidgym.training.runtime_monitor import publish_role_state


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)


def _checkpoint_files_ok(path: Path) -> bool:
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or type(manifest.get("trained_steps")) is not int:
            return False
        required = ["encoder.pt", "perception.pt", "semantic_skip.pt", "aux_heads.pt",
                    "core.pt", "critic.pt"]
        required.extend(f"actor/{name}.pt" for name in manifest.get("experts", []))
        required.extend(f"game_extras/{name}.pt" for name in manifest.get("game_ids", []))
        return all((path / name).is_file() and (path / name).stat().st_size > 0
                   for name in required)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def latest_checkpoint(root: Path, fallback: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    try:
        latest = json.loads((root / "latest.json").read_text(encoding="utf-8"))
        relative = latest.get("policy") if isinstance(latest, dict) else None
        if isinstance(relative, str):
            current = (root / relative).resolve(strict=True)
            current.relative_to(root.resolve(strict=True))
            if _checkpoint_files_ok(current):
                steps = int(json.loads((current / "manifest.json").read_text("utf-8"))["trained_steps"])
                return current
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        pass
    for path in (root / "policies").glob("universal-*"):
        if _checkpoint_files_ok(path):
            try:
                steps = int(json.loads((path / "manifest.json").read_text("utf-8"))["trained_steps"])
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
            candidates.append((steps, path))
    if candidates:
        selected = max(candidates, key=lambda item: item[0])[1]
        publish_role_state(root / "latest.json", {
            "schema_version": 1,
            "round": int(json.loads((selected / "manifest.json").read_text("utf-8"))
                          .get("metrics", {}).get("shared_round", 0)),
            "policy": selected.relative_to(root).as_posix(),
            "trained_steps": int(json.loads((selected / "manifest.json").read_text("utf-8"))["trained_steps"]),
        })
        return selected
    if _checkpoint_files_ok(fallback):
        return fallback
    raise RuntimeError("no complete universal checkpoint is available for learner resume")


def supervise(command: Sequence[str], *, shared_root: Path, stop_file: Path,
              fallback_checkpoint: Path, max_restarts: int = 3,
              window_hours: float = 24.0,
              popen_factory: Any = subprocess.Popen) -> int:
    if max_restarts != 3 or window_hours != 24.0:
        raise ValueError("HK learner recovery is fixed at 3 restarts per rolling 24 hours")
    shared_root.mkdir(parents=True, exist_ok=True)
    history_path = shared_root / "learner-restarts.jsonl"
    history: list[str] = []
    if history_path.is_file():
        for line in history_path.read_text("utf-8").splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict) and isinstance(row.get("time"), str):
                    history.append(row["time"])
            except json.JSONDecodeError:
                continue
    deadline = datetime.now(UTC) - timedelta(hours=24)
    history = [item for item in history if _timestamp(item) >= deadline]
    restart_count = 0
    fault_path = shared_root / "faulthandler-supervisor.log"
    with fault_path.open("a", encoding="utf-8") as faults:
        faulthandler.enable(file=faults, all_threads=True)
        try:
            attempt = 0
            while True:
                attempt += 1
                now = datetime.now().astimezone().isoformat()
                publish_role_state(shared_root / "supervisor.json", {
                    "status": "RUNNING", "attempt": attempt, "restart_count": restart_count,
                    "updated_unix_ns": time.time_ns(),
                })
                if attempt > 1:
                    publish_role_state(shared_root / "status.json", {
                        "status": "RUNNING", "supervisor_attempt": attempt,
                        "restart_count": restart_count, "started_at": now,
                    })
                stdout_path = shared_root / f"learner-attempt-{attempt}.stdout.log"
                stderr_path = shared_root / f"learner-attempt-{attempt}.stderr.log"
                with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
                    process = popen_factory(list(command), cwd=PROJECT_ROOT, env=os.environ.copy(),
                                            stdin=subprocess.DEVNULL, stdout=stdout,
                                            stderr=stderr)
                    return_code = int(process.wait())
                state = {}
                try:
                    loaded = json.loads((shared_root / "status.json").read_text("utf-8"))
                    state = loaded if isinstance(loaded, dict) else {}
                except (OSError, json.JSONDecodeError):
                    pass
                terminal = state.get("status")
                if return_code == 0 and terminal in {"COMPLETED", "CANCELLED", "STOPPED"}:
                    publish_role_state(shared_root / "supervisor.json", {
                        "status": terminal, "attempt": attempt,
                        "restart_count": restart_count, "finished_at": datetime.now().astimezone().isoformat(),
                    })
                    return 0
                if stop_file.is_file() and terminal in {"CANCELLED", "STOPPED"}:
                    return 0
                now_dt = datetime.now(UTC)
                history = [item for item in history if _timestamp(item) >= now_dt - timedelta(hours=24)]
                history.append(now_dt.isoformat())
                with history_path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps({"time": history[-1], "attempt": attempt,
                                             "exit_code": return_code,
                                             "status": terminal}, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                if stop_file.is_file():
                    publish_role_state(shared_root / "status.json", {
                        "status": "CANCELLED", "finished_at": datetime.now().astimezone().isoformat(),
                        "last_learner_status": terminal,
                    })
                    return 0
                if len(history) > 3:
                    failure = {
                        "status": "FAILED", "reason": "learner restart limit reached (3 per 24 hours)",
                        "exit_code": return_code, "restart_count_24h": len(history),
                        "finished_at": datetime.now().astimezone().isoformat(),
                    }
                    publish_role_state(shared_root / "status.json", failure)
                    publish_role_state(shared_root / "learner.json", failure)
                    publish_role_state(shared_root / "supervisor.json", failure)
                    return return_code or 1
                resume = latest_checkpoint(shared_root, fallback_checkpoint)
                publish_role_state(shared_root / "supervisor.json", {
                    "status": "RESTARTING", "attempt": attempt,
                    "restart_count": len(history), "resume_checkpoint": str(resume),
                    "updated_unix_ns": time.time_ns(),
                })
                restart_count = len(history)
        finally:
            faulthandler.disable()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--fallback-checkpoint", type=Path, required=True)
    parser.add_argument("--command-json", type=Path, required=True)
    args = parser.parse_args(argv)
    command_data = json.loads(args.command_json.read_text("utf-8"))
    command = command_data.get("argv") if isinstance(command_data, dict) else None
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        parser.error("command-json must contain a nonempty argv string list")
    return supervise(command, shared_root=args.shared_root.resolve(),
                     stop_file=args.stop_file.resolve(),
                     fallback_checkpoint=args.fallback_checkpoint.resolve(strict=True))


if __name__ == "__main__":
    raise SystemExit(main())
