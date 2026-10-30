"""Crash-aware supervisor for long platform pre-training campaigns."""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def latest_valid_checkpoint(root: Path, fallback: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for checkpoint in root.glob("universal-*"):
        manifest_path = checkpoint / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            steps = int(manifest["trained_steps"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        required = ("encoder.pt", "core.pt", "perception.pt", "aux_heads.pt")
        if all((checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0
               for name in required):
            candidates.append((steps, checkpoint))
    return max(candidates, default=(0, fallback), key=lambda item: item[0])[1]


def supervise(command_builder: Callable[[Path, int], Sequence[str]], *,
              run_root: Path, initial_checkpoint: Path, max_restarts: int = 3,
              window_hours: float = 24.0,
              popen_factory: Callable[..., Any] = subprocess.Popen) -> int:
    """Restart only unclean exits, bounded by the restart count and rolling window."""
    run_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "status.json"
    history_path = run_root / "supervisor-history.json"
    history: list[str] = []
    if history_path.is_file():
        try:
            history = list(json.loads(history_path.read_text(encoding="utf-8")).get("crashes", []))
        except (OSError, json.JSONDecodeError, AttributeError, TypeError):
            history = []
    window_start = datetime.now(UTC) - timedelta(hours=window_hours)
    history = [value for value in history if _parse_time(value) >= window_start]
    restarts = 0
    resume = initial_checkpoint
    attempt = 0
    while True:
        attempt += 1
        _atomic_json(status_path, {
            "status": "RUNNING", "supervisor_attempt": attempt,
            "started_at": datetime.now().astimezone().isoformat(),
        })
        stdout_path = run_root / f"supervisor-attempt-{attempt}.stdout.log"
        stderr_path = run_root / f"supervisor-attempt-{attempt}.stderr.log"
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            process = popen_factory(
                list(command_builder(resume, attempt)), cwd=PROJECT_ROOT,
                stdout=stdout, stderr=stderr, env=os.environ.copy(),
            )
            return_code = int(process.wait())
        status = _read_status(status_path)
        if status.get("status") in {"COMPLETED", "STOPPED", "FAILED"}:
            return return_code

        now = datetime.now(UTC)
        window_start = now - timedelta(hours=window_hours)
        history = [value for value in history if _parse_time(value) >= window_start]
        history.append(now.isoformat())
        _atomic_json(history_path, {"crashes": history})
        _atomic_json(status_path, {
            "status": "CRASHED", "exit_code": return_code,
            "crashed_at": now.astimezone().isoformat(), "attempt": attempt,
            "restart_count": restarts,
        })
        if restarts >= max_restarts or len(history) >= 3:
            _atomic_json(status_path, {
                "status": "FAILED", "reason": "restart limit reached after unclean exits",
                "exit_code": return_code, "finished_at": now.astimezone().isoformat(),
                "crash_count_in_window": len(history), "max_restarts": max_restarts,
            })
            return return_code or 1
        resume = latest_valid_checkpoint(run_root, resume)
        restarts += 1


def _read_status(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _parse_time(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value)
        return result.replace(tzinfo=UTC) if result.tzinfo is None else result
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--hours-limit", type=float, default=24.0)
    parser.add_argument("--total-steps", type=int, default=100_000_000)
    parser.add_argument("--envs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--microbatch-sequences", type=int, default=16)
    parser.add_argument("--quick-eval-interval-steps", type=int, default=1_000_000)
    parser.add_argument("--quick-eval-episodes-per-level", type=int, default=50)
    parser.add_argument("--stop-at-curriculum-level", type=int, choices=range(1, 8))
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--window-hours", type=float, default=24.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=8500)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    run_root = args.runtime_root / "v2" / "pretrain" / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    fault_path = run_root / "faulthandler-supervisor.log"
    with fault_path.open("a", encoding="utf-8") as fault_file:
        faulthandler.enable(file=fault_file, all_threads=True)

        def command_builder(resume: Path, _attempt: int) -> Sequence[str]:
            command = [
                sys.executable, "-X", "faulthandler", "-u",
                str(PROJECT_ROOT / "tools" / "v2" / "pretrain_launcher.py"),
                "--run-name", args.run_name, "--runtime-root", str(args.runtime_root),
                "--resume", str(resume), "--hours-limit", str(args.hours_limit),
                "--total-steps", str(args.total_steps), "--device", args.device,
                "--microbatch-sequences", str(args.microbatch_sequences),
                "--quick-eval-interval-steps", str(args.quick_eval_interval_steps),
                "--quick-eval-episodes-per-level", str(args.quick_eval_episodes_per_level),
                "--seed", str(args.seed),
            ]
            if args.envs is not None:
                command.extend(("--envs", str(args.envs)))
            if args.batch_size is not None:
                command.extend(("--batch-size", str(args.batch_size)))
            if args.stop_at_curriculum_level is not None:
                command.extend(("--stop-at-curriculum-level", str(args.stop_at_curriculum_level)))
            return command

        try:
            return supervise(command_builder, run_root=run_root,
                             initial_checkpoint=args.resume,
                             max_restarts=args.max_restarts, window_hours=args.window_hours)
        finally:
            faulthandler.disable()


if __name__ == "__main__":
    raise SystemExit(main())
