from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import hk_agent.episode_worker as worker_module
from hk_agent.episode_worker import WorkerRequest, run_worker


class FakeAsyncProcess:
    pid = 456

    def __init__(self) -> None:
        self.returncode: int | None = None


def run_without_event_loop(coroutine: Any) -> int:
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return int(completed.value)
    raise AssertionError("test coroutine unexpectedly yielded")


def request(tmp_path: Path, *, maximum_seconds: float | None = 1.0) -> WorkerRequest:
    game = tmp_path / "game.exe"
    game.touch()
    return WorkerRequest(
        episode_id="episode-1",
        game_executable=game,
        game_arguments=(),
        personal_root=tmp_path / "personal",
        experimental_root=tmp_path / "experimental",
        stop_path=tmp_path / "stop.request",
        result_path=tmp_path / "result.json",
        close_timeout_seconds=0.1,
        maximum_episode_seconds=maximum_seconds,
        endpoint={
            "host": "127.0.0.1",
            "port": 12345,
            "run_id": "run-1",
            "token": "token",
            "game_profile": "test",
        },
    )


def test_invalid_isolation_prevents_process_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = False

    async def fake_create(*_: object, **__: object) -> FakeAsyncProcess:
        nonlocal created
        created = True
        return FakeAsyncProcess()

    monkeypatch.setattr(worker_module.asyncio, "create_subprocess_exec", fake_create)
    result = run_without_event_loop(run_worker(request(tmp_path)))
    assert result == 1
    assert created is False
    report = json.loads((tmp_path / "result.json").read_text("utf-8"))
    assert report["ok"] is False
    assert report["reason"].startswith("worker_failure:RuntimeError")


@pytest.mark.parametrize(
    ("stop_exists", "maximum_seconds", "expected_reason"),
    [
        (True, 1.0, "stop_requested"),
        (True, None, "stop_requested"),
        (False, 0.001, "episode_time_limit"),
    ],
)
def test_worker_closes_on_stop_or_episode_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop_exists: bool,
    maximum_seconds: float | None,
    expected_reason: str,
) -> None:
    fake_process = FakeAsyncProcess()
    close_calls: list[float] = []
    spawn_environment: dict[str, str] = {}

    async def fake_create(*_: object, **kwargs: object) -> FakeAsyncProcess:
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        spawn_environment.update(environment)
        return fake_process

    async def fake_close(
        process: Any, timeout: float
    ) -> tuple[int, bool, int]:
        assert process is fake_process
        close_calls.append(timeout)
        fake_process.returncode = 0
        return 0, False, 0

    monkeypatch.setattr(worker_module, "_assert_isolation", lambda _: None)
    monkeypatch.setattr(worker_module.asyncio, "create_subprocess_exec", fake_create)
    monkeypatch.setattr(worker_module, "_close_process", fake_close)
    async def immediate_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(worker_module.asyncio, "sleep", immediate_sleep)
    worker_request = request(tmp_path, maximum_seconds=maximum_seconds)
    if stop_exists:
        worker_request.stop_path.touch()

    result = run_without_event_loop(run_worker(worker_request))
    assert result == 0
    assert close_calls == [0.1]
    assert spawn_environment["HK_AGENT_SAVE_ROOT"] == str(
        worker_request.experimental_root.resolve(strict=False)
    )
    report = json.loads(worker_request.result_path.read_text("utf-8"))
    assert report == {
        "schema_version": 1,
        "episode_id": "episode-1",
        "ok": True,
        "reason": expected_reason,
        "game_exit_code": 0,
        "forced_termination": False,
    }
