"""Periodic host-level memory snapshots for long-running training jobs."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from voidgym.core.hardware import memory_commit_stats


class MemoryHistorySampler:
    """Append the ten largest private-memory processes every five minutes."""

    def __init__(self, run_root: Path, *, interval_seconds: float = 300.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.path = run_root / "memory-history.jsonl"
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="memory-history", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=15)

    def _sample(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        error: str | None = None
        try:
            import psutil  # type: ignore[import-untyped]

            for process in psutil.process_iter(("pid", "name")):
                try:
                    private = int(process.memory_full_info().uss)
                    working_set = int(process.memory_info().rss)
                    rows.append({
                        "name": process.info.get("name") or "",
                        "pid": int(process.info["pid"]),
                        "private_bytes": private,
                        "working_set_bytes": working_set,
                    })
                except (psutil.Error, OSError, KeyError, TypeError):
                    continue
        except ImportError as exc:
            error = f"psutil unavailable: {exc}"
        rows.sort(key=lambda row: row["private_bytes"], reverse=True)
        return {
            "recorded_at": datetime.now().astimezone().isoformat(),
            "processes_top_private": rows[:10],
            "memory": memory_commit_stats(),
            "sampler_error": error,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            record = self._sample()
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError:
                # Telemetry must not terminate or interfere with the learner.
                pass
            self._stop.wait(self.interval_seconds)
