"""Crash-resume supervisor for the v2 Hollow Knight shared learner."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
import faulthandler
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from voidgym.training.runtime_monitor import publish_role_state
from voidgym.training.memory_history import MemoryHistorySampler


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


def _round_value(value: Any) -> int | None:
    """Accept nonnegative integral JSON integers and decimal values."""
    if type(value) is int:
        return value if value >= 0 else None
    if type(value) is float and math.isfinite(value) and value >= 0 and value.is_integer():
        return int(value)
    return None


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


def _latest_progress(root: Path) -> dict[str, Any]:
    """Read the freshest learner/checkpoint progress without changing runtime files."""
    candidates: list[dict[str, Any]] = []
    for filename in ("learner.json", "latest.json"):
        try:
            payload = json.loads((root / filename).read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        steps = payload.get("completed_timesteps", payload.get("trained_steps"))
        if type(steps) is not int or steps < 0:
            continue
        row: dict[str, Any] = {"completed_timesteps": steps}
        round_id = payload.get("round", payload.get("metrics", {}).get("shared_round", 0)
                              if isinstance(payload.get("metrics"), dict) else 0)
        parsed_round = _round_value(round_id)
        if parsed_round is not None:
            row["round"] = parsed_round
        relative = payload.get("policy")
        if isinstance(relative, str):
            try:
                checkpoint = (root / relative).resolve(strict=True)
                checkpoint.relative_to(root.resolve(strict=True))
                if _checkpoint_files_ok(checkpoint):
                    row["policy"] = checkpoint.relative_to(root).as_posix()
            except (OSError, ValueError):
                pass
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            row["metrics"] = metrics
        candidates.append(row)
    policies = root / "policies"
    for path in policies.glob("universal-*") if policies.is_dir() else ():
        if not _checkpoint_files_ok(path):
            continue
        try:
            manifest = json.loads((path / "manifest.json").read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics = manifest.get("metrics") if isinstance(manifest, dict) else None
        row = {
            "completed_timesteps": int(manifest["trained_steps"]),
            "policy": path.relative_to(root).as_posix(),
            "metrics": metrics if isinstance(metrics, dict) else {},
        }
        round_id = metrics.get("shared_round", 0) if isinstance(metrics, dict) else 0
        parsed_round = _round_value(round_id)
        if parsed_round is not None:
            row["round"] = parsed_round
        candidates.append(row)
    if not candidates:
        return {}
    best = max(candidates, key=lambda row: (row["completed_timesteps"], row.get("round", 0)))
    # If the learner heartbeat is ahead of the last checkpoint, preserve its real
    # step/round while still reporting the newest complete policy available.
    latest_policy = max((row for row in candidates if "policy" in row),
                        key=lambda row: (row["completed_timesteps"], row.get("round", 0)),
                        default=None)
    result = dict(best)
    if latest_policy is not None:
        result["policy"] = latest_policy["policy"]
        if best["completed_timesteps"] == latest_policy["completed_timesteps"]:
            result["metrics"] = latest_policy.get("metrics", result.get("metrics", {}))
    result.setdefault("round", 0)
    result.setdefault("metrics", {})
    return result


def supervise(command: Sequence[str], *, shared_root: Path, stop_file: Path,
              fallback_checkpoint: Path, max_restarts: int = 3,
              window_hours: float = 24.0,
              popen_factory: Any = subprocess.Popen) -> int:
    if not 0 <= max_restarts <= 3 or window_hours != 24.0:
        raise ValueError("HK learner recovery allows 0–3 restarts in a rolling 24 hours")
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
                    state_path = shared_root / "learner.json"
                    if not state_path.is_file():  # compatibility with older learner commands
                        state_path = shared_root / "status.json"
                    loaded = json.loads(state_path.read_text("utf-8"))
                    state = loaded if isinstance(loaded, dict) else {}
                except (OSError, json.JSONDecodeError):
                    pass
                terminal = state.get("status")
                if return_code == 0 and terminal in {"COMPLETED", "CANCELLED", "STOPPED"}:
                    final = {**state, **_latest_progress(shared_root),
                             "status": terminal, "exit_code": return_code,
                             "finished_at": datetime.now().astimezone().isoformat()}
                    publish_role_state(shared_root / "status.json", final)
                    publish_role_state(shared_root / "supervisor.json", {
                        "status": terminal, "attempt": attempt,
                        "restart_count": restart_count, "finished_at": datetime.now().astimezone().isoformat(),
                    })
                    return 0
                if stop_file.is_file() and terminal in {"CANCELLED", "STOPPED"}:
                    final = {**state, **_latest_progress(shared_root),
                             "status": terminal, "exit_code": return_code,
                             "finished_at": datetime.now().astimezone().isoformat()}
                    publish_role_state(shared_root / "status.json", final)
                    publish_role_state(shared_root / "supervisor.json", {
                        "status": terminal, "attempt": attempt,
                        "restart_count": restart_count,
                        "finished_at": datetime.now().astimezone().isoformat(),
                    })
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
                        **state, **_latest_progress(shared_root),
                        "status": "CANCELLED", "finished_at": datetime.now().astimezone().isoformat(),
                        "last_learner_status": terminal,
                    })
                    return 0
                if len(history) > max_restarts:
                    failure = {
                        "status": "FAILED",
                        "reason": f"learner restart limit reached ({max_restarts} per 24 hours)",
                        "exit_code": return_code, "restart_count_24h": len(history),
                        "finished_at": datetime.now().astimezone().isoformat(),
                    }
                    status_path = shared_root / "status.json"
                    current_status = {}
                    try:
                        current = json.loads(status_path.read_text("utf-8"))
                        current_status = current if isinstance(current, dict) else {}
                    except (OSError, json.JSONDecodeError):
                        pass
                    publish_role_state(status_path, {
                        **current_status, **_latest_progress(shared_root), **failure,
                    })
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
    parser.add_argument("--max-restarts", type=int, default=3, choices=range(4))
    args = parser.parse_args(argv)
    command_data = json.loads(args.command_json.read_text("utf-8"))
    command = command_data.get("argv") if isinstance(command_data, dict) else None
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        parser.error("command-json must contain a nonempty argv string list")
    shared_root = args.shared_root.resolve()
    memory_sampler = MemoryHistorySampler(shared_root)
    memory_sampler.start()
    try:
        return supervise(command, shared_root=shared_root,
                         stop_file=args.stop_file.resolve(),
                         fallback_checkpoint=args.fallback_checkpoint.resolve(strict=True),
                         max_restarts=args.max_restarts)
    finally:
        memory_sampler.stop()


if __name__ == "__main__":
    raise SystemExit(main())
