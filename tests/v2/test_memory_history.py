from __future__ import annotations

import json
from pathlib import Path
import time

from voidgym.training.memory_history import MemoryHistorySampler


def test_memory_history_sampler_appends_five_minute_snapshot_shape(
    tmp_path: Path, monkeypatch,
) -> None:
    sampler = MemoryHistorySampler(tmp_path, interval_seconds=300)
    monkeypatch.setattr(sampler, "_sample", lambda: {
        "recorded_at": "2026-11-11T12:00:00-03:00",
        "processes_top_private": [{
            "name": "learner.exe", "pid": 42, "private_bytes": 100, "working_set_bytes": 200,
        }],
        "memory": {"commit_used_bytes": 300, "commit_limit_bytes": 500,
                   "commit_available_bytes": 200, "potential_commit_limit_bytes": 700,
                   "potential_commit_available_bytes": 400},
        "sampler_error": None,
    })
    sampler.start()
    deadline = time.monotonic() + 2
    try:
        while not sampler.path.exists() and time.monotonic() < deadline:
            time.sleep(.01)
    finally:
        sampler.stop()
    rows = [json.loads(line) for line in sampler.path.read_text("utf-8").splitlines()]
    assert rows[0]["processes_top_private"][0]["pid"] == 42
    assert rows[0]["memory"]["potential_commit_available_bytes"] == 400
