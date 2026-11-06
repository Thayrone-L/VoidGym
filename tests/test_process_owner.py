from __future__ import annotations

import base64
import json
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

import hk_agent.process_owner as process_owner_module
from hk_agent.bridge_environment import BRIDGE_VARIABLES
from hk_agent.process_owner import (
    EPISODE_REQUEST_ENV,
    IsolatedEpisodeProcessOwner,
    OwnedProcess,
    ProcessOwnerConfig,
)
from hk_agent.supervisor_backend import (
    ProcessLaunchCancellation,
    SupervisorEndpoint,
    SupervisorEnvironmentBackend,
)


class FakeProcess:
    def __init__(
        self,
        *,
        pid: int = 123,
        exit_code: int | None = None,
        worker_ok: bool = True,
    ) -> None:
        self._pid = pid
        self._exit_code = exit_code
        self.worker_ok = worker_ok
        self.worker_result_path: Path | None = None
        self.episode_id: str | None = None
        self.wait_timeout = False
        self.wait_calls = 0

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def returncode(self) -> int | None:
        return self._exit_code

    def poll(self) -> int | None:
        return self._exit_code

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self.wait_timeout:
            raise subprocess.TimeoutExpired("fake", timeout or 0.0)
        self._exit_code = 0 if self._exit_code is None else self._exit_code
        if self.worker_result_path is not None and self.episode_id is not None:
            self.worker_result_path.parent.mkdir(parents=True, exist_ok=True)
            self.worker_result_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "episode_id": self.episode_id,
                        "ok": self.worker_ok,
                        "reason": "stop_requested" if self.worker_ok else "worker_failure",
                        "forced_termination": False,
                    }
                ),
                encoding="utf-8",
            )
        return self._exit_code


class CaptureFactory:
    def __init__(self, process: FakeProcess, *additional: FakeProcess) -> None:
        self.processes = (process, *additional)
        self.calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def __call__(
        self, command: Sequence[str], environment: Mapping[str, str]
    ) -> OwnedProcess:
        self.calls.append((tuple(command), dict(environment)))
        process = self.processes[len(self.calls) - 1]
        request = decode_request(environment[EPISODE_REQUEST_ENV])
        process.worker_result_path = Path(str(request["result_path"]))
        process.episode_id = str(request["episode_id"])
        return process


def make_owner(
    tmp_path: Path,
    process: FakeProcess,
    *,
    enable_breakable_labels: bool = False,
) -> tuple[IsolatedEpisodeProcessOwner, CaptureFactory]:
    factory = CaptureFactory(process)
    owner = IsolatedEpisodeProcessOwner(
        ProcessOwnerConfig(
            game_executable=tmp_path / "Hollow Knight.exe",
            personal_root=tmp_path / "personal",
            experimental_root=tmp_path / "experimental",
            backup_root=tmp_path / "backups",
            transactional_launcher=tmp_path / "start-isolated.ps1",
            python_executable=tmp_path / "python.exe",
            state_root=tmp_path / "state",
            shutdown_timeout_seconds=0.1,
            enable_breakable_labels=enable_breakable_labels,
        ),
        process_factory=factory,
        powershell=tmp_path / "powershell.exe",
        validate_files=False,
    )
    return owner, factory


def test_owner_encodes_breakable_label_opt_in_for_isolated_worker(tmp_path: Path) -> None:
    owner, factory = make_owner(
        tmp_path, FakeProcess(), enable_breakable_labels=True
    )
    owner.start_episode(endpoint(), "episode-breakables")
    request = decode_request(factory.calls[0][1][EPISODE_REQUEST_ENV])
    assert request["enable_breakable_labels"] is True
    owner.shutdown()


def endpoint() -> SupervisorEndpoint:
    return SupervisorEndpoint("127.0.0.1", 43123, "run-1", "secret", "development")


def decode_request(encoded: str) -> dict[str, object]:
    value = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")))
    if not isinstance(value, dict):
        raise TypeError("episode request must be an object")
    return value


def test_owner_launches_transaction_worker_with_endpoint_environment(tmp_path: Path) -> None:
    process = FakeProcess()
    owner, factory = make_owner(tmp_path, process)
    owner.start_episode(endpoint(), "episode-1")

    assert owner.active_episode_id == "episode-1"
    command, environment = factory.calls[0]
    assert command[0].endswith("powershell.exe")
    assert "start-isolated-hollow-knight" not in " ".join(command)
    request = decode_request(environment[EPISODE_REQUEST_ENV])
    assert request["episode_id"] == "episode-1"
    assert request["endpoint"] == {
        "host": "127.0.0.1",
        "port": 43123,
        "run_id": "run-1",
        "token": "secret",
        "game_profile": "development",
    }
    assert environment["HK_OWNER_GAME"].endswith("Hollow Knight.exe")
    lifecycle = json.loads(
        (tmp_path / "state" / "episode-1" / "lifecycle.json").read_text("utf-8")
    )
    assert lifecycle["status"] == "STARTING"
    assert lifecycle["menu_bootstrap"] == "external"

    owner.shutdown()
    assert (tmp_path / "state" / "episode-1" / "stop.request").exists()
    completed = json.loads(
        (tmp_path / "state" / "episode-1" / "lifecycle.json").read_text("utf-8")
    )
    assert completed["status"] == "COMPLETED"
    assert owner.active_episode_id is None
    owner.shutdown()
    owner.close()


def test_instance_owner_uses_copy_and_player_log_without_save_redirect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executable = tmp_path / "hk-ai-1" / "Hollow Knight.exe"
    executable.parent.mkdir()
    executable.touch()
    data_root = tmp_path / "LocalLow" / "Hollow Knight AI 1"
    data_root.mkdir(parents=True)
    seed_root = tmp_path / "seed"
    seed_root.mkdir()
    process = FakeProcess()
    factory = CaptureFactory(process)
    owner = IsolatedEpisodeProcessOwner(
        ProcessOwnerConfig(
            game_executable=executable,
            personal_root=tmp_path / "personal",
            experimental_root=seed_root,
            backup_root=tmp_path / "backup",
            transactional_launcher=tmp_path / "unused.ps1",
            python_executable=tmp_path / "python.exe",
            redirected_save_mode=True,
            game_arguments=("-logFile", str(tmp_path / "runtime" / "launches" / "x" / "Player.log")),
            state_root=tmp_path / "episodes",
            instance_data_root=data_root,
            expected_product_name="Hollow Knight AI 1",
            other_instance_data_roots=(tmp_path / "LocalLow" / "Hollow Knight AI 2",),
        ),
        process_factory=factory,
        validate_files=False,
    )
    owner.start_episode(endpoint(), "episode-instance")

    command, environment = factory.calls[0]
    request = decode_request(environment[EPISODE_REQUEST_ENV])
    assert command[0].endswith("python.exe")
    assert request["game_executable"] == str(executable)
    assert request["game_arguments"][-2:] == ["-logFile", str(tmp_path / "runtime" / "launches" / "x" / "Player.log")]
    assert request["bridge_mode"] == "instance"
    assert request["instance_product_name"] == "Hollow Knight AI 1"
    assert request["instance_data_root"] == str(data_root.resolve(strict=False))
    # The owner hands the worker an explicit contract; no bridge variable is
    # forwarded through the inherited environment.
    assert not BRIDGE_VARIABLES & environment.keys()

    from hk_agent.episode_worker import WorkerRequest, _game_environment

    # Real handoff: the worker parses exactly what the owner emitted, while
    # stale values from either bridge mode exist in the worker's environment.
    monkeypatch.setenv(EPISODE_REQUEST_ENV, environment[EPISODE_REQUEST_ENV])
    monkeypatch.setenv("HK_AGENT_SAVE_ROOT", "stale-save")
    monkeypatch.setenv("HK_AGENT_PRODUCT_NAME", "stale-product")
    monkeypatch.delenv("HK_AGENT_INSTANCE_MODE", raising=False)
    worker_request = WorkerRequest.from_environment()
    game_environment = _game_environment(worker_request)
    assert game_environment["HK_AGENT_INSTANCE_MODE"] == "1"
    assert game_environment["HK_AGENT_PRODUCT_NAME"] == "Hollow Knight AI 1"
    assert game_environment["HK_AGENT_INSTANCE_DATA_ROOT"] == str(data_root.resolve(strict=False))
    assert "HK_AGENT_SAVE_ROOT" not in game_environment
    owner.shutdown()


def test_legacy_owner_handoff_ignores_inherited_instance_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "game" / "Hollow Knight.exe"
    executable.parent.mkdir()
    executable.touch()
    seed_root = tmp_path / "seed"
    seed_root.mkdir()
    monkeypatch.setenv("HK_AGENT_INSTANCE_MODE", "1")
    monkeypatch.setenv("HK_AGENT_PRODUCT_NAME", "stale-product")
    monkeypatch.setenv("HK_AGENT_INSTANCE_DATA_ROOT", "stale-data")
    factory = CaptureFactory(FakeProcess())
    owner = IsolatedEpisodeProcessOwner(
        ProcessOwnerConfig(
            game_executable=executable,
            personal_root=tmp_path / "personal",
            experimental_root=seed_root,
            backup_root=tmp_path / "backup",
            transactional_launcher=tmp_path / "unused.ps1",
            python_executable=tmp_path / "python.exe",
            redirected_save_mode=True,
            state_root=tmp_path / "episodes",
        ),
        process_factory=factory,
        validate_files=False,
    )
    owner.start_episode(endpoint(), "episode-legacy")
    _, environment = factory.calls[0]
    request = decode_request(environment[EPISODE_REQUEST_ENV])
    assert request["bridge_mode"] == "legacy"
    assert request["instance_product_name"] is None
    assert not BRIDGE_VARIABLES & environment.keys()

    from hk_agent.episode_worker import WorkerRequest, _game_environment

    monkeypatch.setenv(EPISODE_REQUEST_ENV, environment[EPISODE_REQUEST_ENV])
    game_environment = _game_environment(WorkerRequest.from_environment())
    assert game_environment["HK_AGENT_SAVE_ROOT"] == str(seed_root.resolve(strict=False))
    assert "HK_AGENT_INSTANCE_MODE" not in game_environment
    owner.shutdown()


def test_instance_owner_requires_redirected_save_mode(tmp_path: Path) -> None:
    config = ProcessOwnerConfig(
        game_executable=tmp_path / "Hollow Knight.exe",
        personal_root=tmp_path / "personal",
        experimental_root=tmp_path / "seed",
        backup_root=tmp_path / "backup",
        transactional_launcher=tmp_path / "launcher.ps1",
        instance_data_root=tmp_path / "LocalLow" / "Hollow Knight AI 1",
        expected_product_name="Hollow Knight AI 1",
    )
    with pytest.raises(ValueError, match="redirected save mode"):
        config.validate(require_files=False)


def test_redirected_owner_launches_worker_directly_with_isolated_root(
    tmp_path: Path,
) -> None:
    process = FakeProcess()
    experimental = tmp_path / "experimental"
    experimental.mkdir()
    factory = CaptureFactory(process)
    owner = IsolatedEpisodeProcessOwner(
        ProcessOwnerConfig(
            game_executable=tmp_path / "Hollow Knight.exe",
            personal_root=tmp_path / "personal",
            experimental_root=experimental,
            backup_root=tmp_path / "backups",
            transactional_launcher=tmp_path / "missing-launcher.ps1",
            python_executable=tmp_path / "python.exe",
            redirected_save_mode=True,
            state_root=tmp_path / "state",
        ),
        process_factory=factory,
        validate_files=False,
    )

    assert owner.expected_save_root == experimental.resolve(strict=False)
    owner.start_episode(endpoint(), "episode-direct")
    command, environment = factory.calls[0]
    assert command == (
        str(tmp_path / "python.exe"),
        "-m",
        "hk_agent.episode_worker",
    )
    assert "HK_OWNER_LAUNCHER" not in environment
    request = decode_request(environment[EPISODE_REQUEST_ENV])
    assert request["redirected_save_mode"] is True
    assert request["experimental_root"] == str(experimental.resolve(strict=False))
    owner.shutdown()


def test_redirected_mode_requires_existing_real_save_root(tmp_path: Path) -> None:
    config = ProcessOwnerConfig(
        game_executable=tmp_path / "game.exe",
        personal_root=tmp_path / "personal",
        experimental_root=tmp_path / "missing",
        backup_root=tmp_path / "backup",
        transactional_launcher=tmp_path / "unused.ps1",
        redirected_save_mode=True,
    )
    with pytest.raises(ValueError, match="must exist"):
        config.validate(require_files=False)


def test_consecutive_episode_reset_finishes_previous_transaction_first(
    tmp_path: Path,
) -> None:
    first = FakeProcess(pid=101)
    second = FakeProcess(pid=202)
    owner, _ = make_owner(tmp_path, first)
    factory = CaptureFactory(first, second)
    owner._factory = factory
    owner.start_episode(endpoint(), "episode-1")
    owner.start_episode(endpoint(), "episode-2")

    assert first.wait_calls == 1
    assert (tmp_path / "state" / "episode-1" / "stop.request").exists()
    lifecycle = json.loads(
        (tmp_path / "state" / "episode-1" / "lifecycle.json").read_text("utf-8")
    )
    assert lifecycle["status"] == "COMPLETED"
    assert owner.active_episode_id == "episode-2"
    assert len(factory.calls) == 2
    owner.shutdown()


def test_shutdown_timeout_is_bounded_and_preserves_transaction_owner(tmp_path: Path) -> None:
    process = FakeProcess()
    process.wait_timeout = True
    owner, _ = make_owner(tmp_path, process)
    owner.start_episode(endpoint(), "episode-1")
    with pytest.raises(TimeoutError, match="shutdown timed out"):
        owner.close()
    assert owner.active_episode_id == "episode-1"
    assert process.poll() is None

    process.wait_timeout = False
    owner.close()
    assert owner.active_episode_id is None


def test_nonzero_transaction_is_fail_closed(tmp_path: Path) -> None:
    process = FakeProcess(exit_code=7)
    owner, _ = make_owner(tmp_path, process)
    with pytest.raises(RuntimeError, match="exited with 7"):
        owner.start_episode(endpoint(), "episode-1")
    lifecycle = json.loads(
        (tmp_path / "state" / "episode-1" / "lifecycle.json").read_text("utf-8")
    )
    assert lifecycle["status"] == "FAILED"


def test_outer_success_does_not_hide_worker_failure(tmp_path: Path) -> None:
    process = FakeProcess(worker_ok=False)
    owner, _ = make_owner(tmp_path, process)
    owner.start_episode(endpoint(), "episode-1")
    with pytest.raises(RuntimeError, match="worker failed"):
        owner.shutdown()
    lifecycle = json.loads(
        (tmp_path / "state" / "episode-1" / "lifecycle.json").read_text("utf-8")
    )
    assert lifecycle["status"] == "FAILED"
    assert lifecycle["transaction_exit_code"] == 0
    assert lifecycle["worker_result_valid"] is False


def test_owner_rejects_unsafe_or_invalid_configuration(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    config = ProcessOwnerConfig(
        game_executable=tmp_path / "game.exe",
        personal_root=shared,
        experimental_root=shared,
        backup_root=tmp_path / "backup",
        transactional_launcher=tmp_path / "launcher.ps1",
    )
    with pytest.raises(ValueError, match="must differ"):
        config.validate(require_files=False)

    owner, _ = make_owner(tmp_path, FakeProcess())
    with pytest.raises(ValueError, match="invalid episode_id"):
        owner.start_episode(endpoint(), "../escape")


def test_timed_out_reset_cannot_launch_late_process(tmp_path: Path) -> None:
    process = FakeProcess()
    owner, factory = make_owner(tmp_path, process)
    release_callback = threading.Event()
    callback_finished = threading.Event()

    def delayed_reset(
        reset_endpoint: SupervisorEndpoint,
        episode_id: str,
        cancellation: ProcessLaunchCancellation,
    ) -> None:
        release_callback.wait()
        try:
            owner.start_episode(reset_endpoint, episode_id, cancellation)
        finally:
            callback_finished.set()

    backend = SupervisorEnvironmentBackend(
        reset_process=delayed_reset,
        bridge_timeout_seconds=0.1,
        command_timeout_seconds=0.1,
        process_timeout_seconds=0.05,
    )
    try:
        with pytest.raises(TimeoutError, match="reset_process timed out"):
            backend.reset(seed=None, options=None)
        release_callback.set()
        assert callback_finished.wait(timeout=1.0)
        time.sleep(0.02)
        assert factory.calls == []
        assert owner.active_episode_id is None
    finally:
        release_callback.set()
        backend.close()
        owner.close()


def test_factory_block_does_not_block_timeout_and_late_process_is_torn_down(
    tmp_path: Path,
) -> None:
    process = FakeProcess()
    factory_entered = threading.Event()
    release_factory = threading.Event()
    callback_finished = threading.Event()

    def blocking_factory(
        command: Sequence[str], environment: Mapping[str, str]
    ) -> OwnedProcess:
        del command
        request_value = decode_request(environment[EPISODE_REQUEST_ENV])
        process.worker_result_path = Path(str(request_value["result_path"]))
        process.episode_id = str(request_value["episode_id"])
        factory_entered.set()
        release_factory.wait()
        return process

    owner, _ = make_owner(tmp_path, process)
    owner._factory = blocking_factory

    def reset_callback(
        reset_endpoint: SupervisorEndpoint,
        episode_id: str,
        cancellation: ProcessLaunchCancellation,
    ) -> None:
        try:
            owner.start_episode(reset_endpoint, episode_id, cancellation)
        finally:
            callback_finished.set()

    backend = SupervisorEnvironmentBackend(
        reset_process=reset_callback,
        bridge_timeout_seconds=0.1,
        command_timeout_seconds=0.1,
        process_timeout_seconds=0.05,
    )
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="reset_process timed out"):
            backend.reset(seed=None, options=None)
        assert time.monotonic() - started < 0.5
        assert factory_entered.is_set()
        release_factory.set()
        assert callback_finished.wait(timeout=1.0)
        assert process.wait_calls == 1
        assert owner.active_episode_id is None
    finally:
        release_factory.set()
        backend.close()
        owner.close()


@pytest.mark.parametrize("writer_fails", [False, True])
def test_lifecycle_writer_block_cannot_strand_timed_out_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer_fails: bool,
) -> None:
    process = FakeProcess()
    owner, _ = make_owner(tmp_path, process)
    writer_entered = threading.Event()
    release_writer = threading.Event()
    callback_finished = threading.Event()
    real_writer = process_owner_module._write_json_atomic

    def controlled_writer(path: Path, value: Mapping[str, object]) -> None:
        if value.get("status") == "STARTING":
            writer_entered.set()
            release_writer.wait()
            if writer_fails:
                raise OSError("injected lifecycle failure")
        real_writer(path, value)

    monkeypatch.setattr(process_owner_module, "_write_json_atomic", controlled_writer)

    def reset_callback(
        reset_endpoint: SupervisorEndpoint,
        episode_id: str,
        cancellation: ProcessLaunchCancellation,
    ) -> None:
        try:
            owner.start_episode(reset_endpoint, episode_id, cancellation)
        finally:
            callback_finished.set()

    backend = SupervisorEnvironmentBackend(
        reset_process=reset_callback,
        bridge_timeout_seconds=0.1,
        command_timeout_seconds=0.1,
        process_timeout_seconds=0.05,
    )
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="reset_process timed out"):
            backend.reset(seed=None, options=None)
        assert time.monotonic() - started < 0.5
        assert writer_entered.wait(timeout=1.0)
        release_writer.set()
        assert callback_finished.wait(timeout=1.0)
        deadline = time.monotonic() + 1.0
        while owner.active_episode_id is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert owner.active_episode_id is None
        assert process.wait_calls == 1
    finally:
        release_writer.set()
        backend.close()
        owner.close()


def test_persistent_lifecycle_failure_clears_finished_process_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = FakeProcess()
    owner, _ = make_owner(tmp_path, process)

    def failed_writer(_: Path, __: Mapping[str, object]) -> None:
        raise OSError("persistent audit failure")

    monkeypatch.setattr(process_owner_module, "_write_json_atomic", failed_writer)
    with pytest.raises(OSError, match="persistent audit failure"):
        owner.start_episode(endpoint(), "episode-1")
    assert process.wait_calls == 1
    assert owner.active_episode_id is None
