from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import hk_agent.episode_worker as worker_module
from hk_agent.episode_worker import WorkerRequest, _game_environment, run_worker


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


def test_worker_instance_environment_comes_from_request_not_inherited_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "instance-data"
    worker_request = replace(
        request(tmp_path),
        redirected_save_mode=True,
        bridge_mode="instance",
        instance_product_name="Hollow Knight AI 1",
        instance_data_root=data_root,
    )
    # Stale values from both modes must neither leak nor steer the decision.
    monkeypatch.setenv("HK_AGENT_SAVE_ROOT", "stale-save")
    monkeypatch.setenv("HK_AGENT_INSTANCE_MODE", "stale-mode")
    monkeypatch.setenv("HK_AGENT_PRODUCT_NAME", "stale-product")
    monkeypatch.setenv("HK_AGENT_INSTANCE_DATA_ROOT", "stale-data")
    environment = _game_environment(worker_request)
    assert environment["HK_AGENT_INSTANCE_MODE"] == "1"
    assert environment["HK_AGENT_PRODUCT_NAME"] == "Hollow Knight AI 1"
    assert environment["HK_AGENT_INSTANCE_DATA_ROOT"] == str(data_root)
    assert "HK_AGENT_SAVE_ROOT" not in environment


def test_worker_legacy_request_ignores_inherited_instance_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker_request = request(tmp_path)
    monkeypatch.setenv("HK_AGENT_INSTANCE_MODE", "1")
    monkeypatch.setenv("HK_AGENT_PRODUCT_NAME", "stale-product")
    monkeypatch.setenv("HK_AGENT_INSTANCE_DATA_ROOT", "stale-data")
    environment = _game_environment(worker_request)
    assert environment["HK_AGENT_SAVE_ROOT"] == str(
        worker_request.experimental_root.resolve(strict=False)
    )
    assert not {
        "HK_AGENT_INSTANCE_MODE",
        "HK_AGENT_PRODUCT_NAME",
        "HK_AGENT_INSTANCE_DATA_ROOT",
    } & environment.keys()


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
        "game_process": {
            "pid": fake_process.pid,
            "creation_time": None,
            "executable": str(worker_request.game_executable.resolve(strict=False)),
            "launch_nonce": None,
        },
    }


def _no_real_process(_: int) -> dict[str, object]:
    raise OSError("fake pid is not a real process")


def test_stop_is_honoured_while_waiting_for_the_slot_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1: a window that never appears must not delay a cooperative stop."""
    fake_process = FakeAsyncProcess()
    lookups: list[int] = []
    sleeps = 0

    async def fake_create(*_: object, **__: object) -> FakeAsyncProcess:
        return fake_process

    async def fake_close(process: Any, timeout: float) -> tuple[int, bool, int]:
        fake_process.returncode = 0
        return 0, False, 0

    async def counting_sleep(_: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps == 3:
            worker_request.stop_path.touch()

    def no_window(pid: int) -> None:
        lookups.append(pid)

    clock = [1000.0]

    def fake_monotonic() -> float:
        clock[0] += 0.3
        return clock[0]

    monkeypatch.setenv("HK_AGENT_SLOT_ID", "1")
    monkeypatch.setattr(worker_module, "_assert_isolation", lambda _: None)
    monkeypatch.setattr(worker_module.asyncio, "create_subprocess_exec", fake_create)
    monkeypatch.setattr(worker_module, "_close_process", fake_close)
    monkeypatch.setattr(worker_module.asyncio, "sleep", counting_sleep)
    monkeypatch.setattr(worker_module, "process_identity", _no_real_process)
    monkeypatch.setattr(worker_module, "find_window_for_pid", no_window)
    monkeypatch.setattr(worker_module.time, "monotonic", fake_monotonic)
    worker_request = request(tmp_path, maximum_seconds=None)

    result = run_without_event_loop(run_worker(worker_request))
    report = json.loads(worker_request.result_path.read_text("utf-8"))
    assert result == 0
    assert report["reason"] == "stop_requested"
    assert lookups, "window discovery must run inside the supervision loop"
    assert sleeps == 3, "stop must be observed on the very next iteration"


def test_window_placement_retries_until_set_window_pos_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P2: a failed or recreated window is retried until the deadline."""
    windows = iter([None, 11, 12, 13])
    placed: list[tuple[int, int]] = []
    results = iter([False, False, True])

    def fake_find(_: int) -> int | None:
        return next(windows)

    def fake_place(hwnd: int, slot_id: int) -> bool:
        placed.append((hwnd, slot_id))
        return next(results)

    monkeypatch.setattr(worker_module, "find_window_for_pid", fake_find)
    monkeypatch.setattr(worker_module, "place_window", fake_place)
    state = worker_module.WindowPlacementState(pid=7, slot_id=0, give_up_at=100.0)
    now = 0.0
    for _ in range(10):
        state.attempt(now)
        now += worker_module.WINDOW_RETRY_SECONDS
    assert state.placed is True
    # The exact HWND that was found is the one moved (no second enumeration).
    assert placed == [(11, 0), (12, 0), (13, 0)]


def test_window_placement_stops_at_deadline_and_respects_retry_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []

    def fake_find(_: int) -> int | None:
        calls.append(0.0)
        return None

    monkeypatch.setattr(worker_module, "find_window_for_pid", fake_find)
    state = worker_module.WindowPlacementState(pid=7, slot_id=1, give_up_at=1.0)
    state.attempt(0.0)
    state.attempt(0.1)  # inside the retry interval: no new lookup
    state.attempt(0.3)
    state.attempt(5.0)  # past the deadline: gives up
    assert len(calls) == 2
    assert state.placed is False


def test_worker_writes_identity_sidecar_instead_of_launch_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P2: the worker must never rewrite the panel-owned launch.json."""
    fake_process = FakeAsyncProcess()
    launch_record = tmp_path / "launch.json"
    launch_record.write_text(json.dumps({"launch_id": "L1", "status": "STOP_REQUESTED"}), "utf-8")
    before = launch_record.read_bytes()
    sidecar = tmp_path / "game-process.json"

    async def fake_create(*_: object, **kwargs: object) -> FakeAsyncProcess:
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        assert worker_module.GAME_IDENTITY_ENV not in environment
        return fake_process

    async def fake_close(process: Any, timeout: float) -> tuple[int, bool, int]:
        fake_process.returncode = 0
        return 0, False, 0

    async def immediate_sleep(_: float) -> None:
        return None

    monkeypatch.setenv("HK_AGENT_LAUNCH_ID", "L1")
    monkeypatch.setenv(worker_module.GAME_IDENTITY_ENV, str(sidecar))
    monkeypatch.setattr(worker_module, "_assert_isolation", lambda _: None)
    monkeypatch.setattr(worker_module.asyncio, "create_subprocess_exec", fake_create)
    monkeypatch.setattr(worker_module, "_close_process", fake_close)
    monkeypatch.setattr(worker_module, "process_identity", _no_real_process)
    monkeypatch.setattr(worker_module.asyncio, "sleep", immediate_sleep)
    worker_request = request(tmp_path)
    worker_request.stop_path.touch()

    run_without_event_loop(run_worker(worker_request))
    assert launch_record.read_bytes() == before
    identity = json.loads(sidecar.read_text("utf-8"))
    assert identity["pid"] == fake_process.pid
    assert identity["launch_nonce"] == "L1"
