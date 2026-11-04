from __future__ import annotations

import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
import torch

from voidgym.training import shared_learner as shared
from voidgym.training import shared_rollouts


def test_rollout_straggler_timeout_trains_on_live_actor_and_marks_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = tmp_path / "round-1-slot-0.npz"
    live.write_bytes(b"present")
    paths = {"actor-slot-0": live, "actor-slot-1": tmp_path / "round-1-slot-1.npz"}
    monkeypatch.setattr(shared, "rollout_path", lambda _root, _round, actor: paths[actor])
    monkeypatch.setattr(shared, "load_rollout", lambda *_args, **_kwargs: [
        SimpleNamespace(actions=torch.zeros((1, 4), dtype=torch.long))
    ])
    now = [10.0]
    clock = SimpleNamespace(
        monotonic=lambda: now[0],
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        time_ns=lambda: 0,
    )
    monkeypatch.setattr(shared, "time", clock)
    progress: dict[str, object] = {}
    lock = shared.threading.Lock()

    result = shared._wait_for_rollouts(
        tmp_path, 1, tuple(paths), 0, ("NOOP",), torch.device("cpu"),
        timeout=2.0, straggler_timeout=0.3, stop_file=None,
        progress=progress, progress_lock=lock,
    )

    assert result is not None
    assert result.missing_actors == ("actor-slot-1",)
    assert result.sequences[0].actions.numel() == 4
    assert result.wait_ms == pytest.approx(300.0)
    record = json.loads((tmp_path / "stragglers.jsonl").read_text("utf-8"))
    assert record["round"] == 1
    assert record["missing_actors"] == ["actor-slot-1"]
    assert record["present_actors"] == ["actor-slot-0"]
    assert record["wait_ms"] == pytest.approx(300.0)


def test_rollout_wait_fails_after_straggler_deadline_when_no_actor_is_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {actor: tmp_path / f"{actor}.npz" for actor in ("actor-0", "actor-1")}
    monkeypatch.setattr(shared, "rollout_path", lambda _root, _round, actor: paths[actor])
    now = [20.0]
    clock = SimpleNamespace(
        monotonic=lambda: now[0],
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        time_ns=lambda: 0,
    )
    monkeypatch.setattr(shared, "time", clock)

    with pytest.raises(TimeoutError, match="no actor published round 3 within 0.3s"):
        shared._wait_for_rollouts(
            tmp_path, 3, tuple(paths), 2, ("NOOP",), torch.device("cpu"),
            timeout=3600.0, straggler_timeout=0.3, stop_file=None,
            progress={}, progress_lock=shared.threading.Lock(),
        )
    assert now[0] < 21.0


def test_commit_guard_accepts_dynamic_pagefile_capacity_but_keeps_actual_floor() -> None:
    gib = 1024**3
    assert shared._commit_guard_failure({
        "commit_available_bytes": int(2.1 * gib),
        "potential_commit_available_bytes": int(18 * gib),
    }) is None
    assert "1.5 GiB current" in shared._commit_guard_failure({
        "commit_available_bytes": int(1.4 * gib),
        "potential_commit_available_bytes": int(18 * gib),
    })
    assert "3 GiB potential" in shared._commit_guard_failure({
        "commit_available_bytes": int(2 * gib),
        "potential_commit_available_bytes": int(2.9 * gib),
    })
    assert "3 GiB potential" in shared._commit_guard_failure({
        "commit_available_bytes": int(2 * gib),
    })


def test_publish_json_retries_transient_replace_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "learner.json"
    original = shared_rollouts.os.replace
    attempts = [0]

    def locked_twice(source, destination):
        attempts[0] += 1
        if attempts[0] <= 2:
            raise PermissionError("transient Windows file lock")
        return original(source, destination)

    monkeypatch.setattr(shared_rollouts.os, "replace", locked_twice)
    shared_rollouts.publish_json(target, {"status": "RUNNING"})

    assert attempts[0] == 3
    assert json.loads(target.read_text("utf-8")) == {"status": "RUNNING"}


def test_publish_json_uses_distinct_temporary_files_for_concurrent_writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "heartbeat.json"
    original = shared_rollouts.os.replace
    recorded_sources: list[str] = []
    source_lock = threading.Lock()
    errors: list[BaseException] = []

    def record_replace(source, destination):
        with source_lock:
            recorded_sources.append(str(source))
        return original(source, destination)

    monkeypatch.setattr(shared_rollouts.os, "replace", record_replace)

    def publish(value: int) -> None:
        try:
            shared_rollouts.publish_json(target, {"writer": value})
        except OSError as error:  # keep file publication failures visible to the test
            errors.append(error)

    threads = [threading.Thread(target=publish, args=(value,)) for value in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3.0)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert len(recorded_sources) == 2
    assert len(set(recorded_sources)) == 2
    assert json.loads(target.read_text("utf-8"))["writer"] in {1, 2}
    assert list(tmp_path.glob(".heartbeat.json.*.tmp")) == []
