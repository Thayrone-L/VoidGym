from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from hk_agent.bridge_environment import clean_bridge_environment
from hk_agent.supervisor_backend import (
    ProcessLaunchCancellation,
    ProcessLaunchCancelled,
    SupervisorEndpoint,
)
from hk_agent.win_job import WindowsJob

EPISODE_REQUEST_ENV = "HK_AGENT_EPISODE_REQUEST"
_EPISODE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class OwnedProcess(Protocol):
    @property
    def pid(self) -> int: ...

    @property
    def returncode(self) -> int | None: ...

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...


ProcessFactory = Callable[[Sequence[str], Mapping[str, str]], OwnedProcess]


@dataclass(frozen=True, slots=True)
class ProcessOwnerConfig:
    game_executable: Path
    personal_root: Path
    experimental_root: Path
    backup_root: Path
    transactional_launcher: Path
    python_executable: Path = Path(sys.executable)
    game_arguments: tuple[str, ...] = ()
    redirected_save_mode: bool = False
    shutdown_timeout_seconds: float = 20.0
    worker_close_timeout_seconds: float = 10.0
    maximum_episode_seconds: float | None = 3600.0
    state_root: Path = Path("artifacts/episodes")
    instance_data_root: Path | None = None
    expected_product_name: str | None = None
    other_instance_data_roots: tuple[Path, ...] = ()
    enable_breakable_labels: bool = False

    def validate(self, *, require_files: bool = True) -> None:
        if self.personal_root.resolve(strict=False) == self.experimental_root.resolve(
            strict=False
        ):
            raise ValueError("personal and experimental save roots must differ")
        if self.backup_root.resolve(strict=False) in {
            self.personal_root.resolve(strict=False),
            self.experimental_root.resolve(strict=False),
        }:
            raise ValueError("backup root must be separate from both save roots")
        if require_files:
            required = [
                ("game executable", self.game_executable),
                ("Python executable", self.python_executable),
            ]
            if not self.redirected_save_mode:
                required.append(("transactional launcher", self.transactional_launcher))
            for label, path in required:
                if not path.is_file():
                    raise ValueError(f"{label} not found: {path}")
        if self.redirected_save_mode and not self.experimental_root.is_dir():
            raise ValueError(
                "experimental save root must exist in redirected save mode"
            )
        if self.instance_data_root is not None:
            if self.instance_data_root.resolve(strict=False) == self.personal_root.resolve(strict=False):
                raise ValueError("instance data root cannot be the personal save root")
            if not self.expected_product_name:
                raise ValueError("instance mode requires the marker product name")
            if not self.redirected_save_mode:
                raise ValueError("instance mode requires redirected save mode")
        if type(self.enable_breakable_labels) is not bool:
            raise TypeError("enable_breakable_labels must be bool")

        for value, label in (
            (self.shutdown_timeout_seconds, "shutdown timeout"),
            (self.worker_close_timeout_seconds, "worker close timeout"),
        ):
            if value <= 0:
                raise ValueError(f"{label} must be positive")
        if self.maximum_episode_seconds is not None and self.maximum_episode_seconds <= 0:
            raise ValueError("maximum episode duration must be positive when enabled")


@dataclass(frozen=True, slots=True)
class EpisodeLaunch:
    episode_id: str
    owner_pid: int
    started_monotonic_ns: int
    stop_path: Path
    worker_result_path: Path
    lifecycle_path: Path


def _powershell_path() -> Path:
    return (
        Path(os.environ.get("SystemRoot", r"C:\Windows"))
        / "System32"
        / "WindowsPowerShell"
        / "v1.0"
        / "powershell.exe"
    )


def _encode_request(document: Mapping[str, object]) -> str:
    raw = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _default_process_factory(
    command: Sequence[str], environment: Mapping[str, str]
) -> OwnedProcess:
    return subprocess.Popen(
        list(command),
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class IsolatedEpisodeProcessOwner:
    """Own exactly one game process inside the transactional save launcher.

    ``start_episode`` and ``shutdown`` are suitable for the process callbacks of
    :class:`SupervisorEnvironmentBackend`.  This class contains no menu or
    gameplay actions; it only establishes and tears down an episode boundary.
    """

    def __init__(
        self,
        config: ProcessOwnerConfig,
        *,
        process_factory: ProcessFactory = _default_process_factory,
        powershell: Path | None = None,
        validate_files: bool = True,
    ) -> None:
        config.validate(require_files=validate_files)
        self._config = config
        self._factory = process_factory
        self._powershell = powershell or _powershell_path()
        if (
            validate_files
            and not config.redirected_save_mode
            and not self._powershell.is_file()
        ):
            raise ValueError(f"Windows PowerShell not found: {self._powershell}")
        self._lock = threading.RLock()
        self._process: OwnedProcess | None = None
        self._launch: EpisodeLaunch | None = None
        self._closed = False
        self._job = WindowsJob.create()
        if os.name == "nt":
            self._job.assign_pid(os.getpid())

    @property
    def active_episode_id(self) -> str | None:
        with self._lock:
            return self._launch.episode_id if self._launch is not None else None

    @property
    def expected_save_root(self) -> Path | None:
        if not self._config.redirected_save_mode or self._config.instance_data_root is not None:
            return None
        return self._config.experimental_root.resolve(strict=False)

    @property
    def expected_persistent_data_path(self) -> Path | None:
        root = self._config.instance_data_root
        return root.resolve(strict=False) if root is not None else None

    @property
    def forbidden_personal_root(self) -> Path | None:
        return self._config.personal_root.resolve(strict=False) if self._config.instance_data_root else None

    @property
    def expected_product_name(self) -> str | None:
        return self._config.expected_product_name

    @property
    def other_instance_data_roots(self) -> tuple[Path, ...]:
        return self._config.other_instance_data_roots

    def start_episode(
        self,
        endpoint: SupervisorEndpoint,
        episode_id: str,
        cancellation: ProcessLaunchCancellation | None = None,
    ) -> None:
        if not _EPISODE_ID.fullmatch(episode_id) or episode_id in {".", ".."}:
            raise ValueError("invalid episode_id")
        with self._lock:
            if self._closed:
                raise RuntimeError("process owner is closed")
            self._reap_finished_locked()
            if self._process is not None:
                # A Gym reset is also a process boundary. Finish restoration of
                # the previous transaction before creating the next one; never
                # overlap transactions and never terminate by process name.
                self._shutdown_locked()

            episode_root = self._config.state_root / episode_id
            stop_path = episode_root / "stop.request"
            worker_result = episode_root / "worker-result.json"
            lifecycle = episode_root / "lifecycle.json"
            if stop_path.exists():
                stop_path.unlink()
            request: dict[str, object] = {
                "schema_version": 1,
                "episode_id": episode_id,
                "game_executable": str(self._config.game_executable.resolve(strict=False)),
                "game_arguments": list(self._config.game_arguments),
                "personal_root": str(self._config.personal_root.resolve(strict=False)),
                "experimental_root": str(
                    self._config.experimental_root.resolve(strict=False)
                ),
                "redirected_save_mode": self._config.redirected_save_mode,
                "stop_path": str(stop_path.resolve(strict=False)),
                "result_path": str(worker_result.resolve(strict=False)),
                "close_timeout_seconds": self._config.worker_close_timeout_seconds,
                "maximum_episode_seconds": self._config.maximum_episode_seconds,
                "endpoint": asdict(endpoint),
                # The worker selects the bridge mode only from this explicit
                # contract, never from variables inherited by this process.
                "bridge_mode": "instance" if self._config.instance_data_root else "legacy",
                "instance_product_name": (
                    self._config.expected_product_name
                    if self._config.instance_data_root is not None
                    else None
                ),
                "instance_data_root": (
                    str(self._config.instance_data_root.resolve(strict=False))
                    if self._config.instance_data_root is not None
                    else None
                ),
                "enable_breakable_labels": self._config.enable_breakable_labels,
            }
            environment = clean_bridge_environment(os.environ)
            launch_nonce = os.environ.get("HK_AGENT_LAUNCH_ID") or uuid.uuid4().hex
            environment["HK_AGENT_LAUNCH_ID"] = launch_nonce
            environment[EPISODE_REQUEST_ENV] = _encode_request(request)
            if not self._config.redirected_save_mode:
                environment.update(
                    {
                        "HK_OWNER_LAUNCHER": str(self._config.transactional_launcher),
                        "HK_OWNER_PYTHON": str(self._config.python_executable),
                        "HK_OWNER_PERSONAL": str(self._config.personal_root),
                        "HK_OWNER_EXPERIMENTAL": str(self._config.experimental_root),
                        "HK_OWNER_BACKUPS": str(self._config.backup_root),
                        "HK_OWNER_GAME": str(self._config.game_executable),
                    }
                )
            command = self._command()
            if cancellation is not None:
                cancellation.raise_if_cancelled()
            process = self._factory(command, environment)
            if os.name == "nt" and self._factory is _default_process_factory:
                self._job.assign_pid(process.pid)
            launch = EpisodeLaunch(
                episode_id=episode_id,
                owner_pid=process.pid,
                started_monotonic_ns=time.monotonic_ns(),
                stop_path=stop_path,
                worker_result_path=worker_result,
                lifecycle_path=lifecycle,
            )
            self._process = process
            self._launch = launch
            try:
                if cancellation is not None:
                    # Register teardown immediately after assuming ownership. No
                    # filesystem I/O may occur in this cancellation window.
                    cancellation.register_cleanup(
                        lambda: self._cleanup_cancelled_episode(episode_id)
                    )
                _write_json_atomic(
                    lifecycle,
                    {
                        "schema_version": 1,
                        "episode_id": episode_id,
                        "status": "STARTING",
                        "transaction_owner_pid": process.pid,
                        "started_monotonic_ns": launch.started_monotonic_ns,
                        "menu_bootstrap": "external",
                    },
                )
                if cancellation is not None and cancellation.is_cancelled:
                    # The timeout raced with a factory that had already started.
                    # We now own its returned transaction process and must request
                    # full worker shutdown/save restoration before leaving.
                    self._shutdown_locked()
                    raise ProcessLaunchCancelled(
                        "process launch generation was cancelled during creation"
                    )
                self._reap_finished_locked(raise_on_failure=True)
            except Exception as error:
                # Once Popen returns, every later failure owns a live
                # transaction and must request teardown before propagating.
                if self._launch is launch:
                    try:
                        self._shutdown_locked()
                    except Exception as cleanup_error:  # noqa: BLE001
                        error.add_note(
                            "transaction cleanup also failed: "
                            f"{type(cleanup_error).__name__}: {cleanup_error}"
                        )
                raise

    def _cleanup_cancelled_episode(self, episode_id: str) -> None:
        with self._lock:
            if self._launch is None or self._launch.episode_id != episode_id:
                return
            self._shutdown_locked()

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown_locked()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._shutdown_locked()
            self._closed = True

    def _shutdown_locked(self) -> None:
        self._reap_finished_locked(raise_on_failure=True)
        if self._process is None or self._launch is None:
            return
        launch = self._launch
        launch.stop_path.parent.mkdir(parents=True, exist_ok=True)
        launch.stop_path.touch(exist_ok=True)
        try:
            exit_code = self._process.wait(
                timeout=self._config.shutdown_timeout_seconds
            )
        except subprocess.TimeoutExpired as error:
            # Do not kill the transaction owner: doing so could strand the save
            # junction and its game child.  Preserve state for a later retry/recovery.
            raise TimeoutError("isolated episode shutdown timed out") from error
        self._complete_locked(exit_code)

    def _reap_finished_locked(self, *, raise_on_failure: bool = False) -> None:
        if self._process is None:
            return
        exit_code = self._process.poll()
        if exit_code is None:
            return
        self._complete_locked(exit_code)
        if raise_on_failure and exit_code != 0:
            raise RuntimeError(f"isolated episode transaction exited with {exit_code}")

    def _complete_locked(self, exit_code: int) -> None:
        launch = self._launch
        if launch is None:
            raise RuntimeError("process owner state is inconsistent")
        worker_ok, worker_failure = self._validate_worker_result(launch)
        completed = exit_code == 0 and worker_ok
        try:
            _write_json_atomic(
                launch.lifecycle_path,
                {
                    "schema_version": 1,
                    "episode_id": launch.episode_id,
                    "status": "COMPLETED" if completed else "FAILED",
                    "transaction_owner_pid": launch.owner_pid,
                    "started_monotonic_ns": launch.started_monotonic_ns,
                    "finished_monotonic_ns": time.monotonic_ns(),
                    "transaction_exit_code": exit_code,
                    "worker_result": str(launch.worker_result_path),
                    "worker_result_valid": worker_ok,
                    "failure": worker_failure,
                    "menu_bootstrap": "external",
                },
            )
        finally:
            # The transaction process has ended. Audit persistence failure must
            # propagate, but must not leave a dead process marked as active.
            self._process = None
            self._launch = None
        if not completed:
            detail = (
                f"transaction exited with {exit_code}"
                if exit_code != 0
                else worker_failure or "worker result is invalid"
            )
            raise RuntimeError(f"isolated episode failed: {detail}")

    @staticmethod
    def _validate_worker_result(launch: EpisodeLaunch) -> tuple[bool, str | None]:
        try:
            value = json.loads(launch.worker_result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            return False, f"worker result unavailable: {type(error).__name__}"
        if not isinstance(value, dict):
            return False, "worker result is not an object"
        if value.get("schema_version") != 1:
            return False, "worker result schema mismatch"
        if value.get("episode_id") != launch.episode_id:
            return False, "worker result episode mismatch"
        if value.get("forced_termination") is not False:
            return False, "worker required forced termination"
        if value.get("ok") is not True:
            reason = value.get("reason")
            return False, f"worker failed: {reason}" if isinstance(reason, str) else "worker failed"
        return True, None

    def _command(self) -> tuple[str, ...]:
        if self._config.redirected_save_mode:
            return (
                str(self._config.python_executable),
                "-m",
                "hk_agent.episode_worker",
            )
        command_text = (
            "& $env:HK_OWNER_LAUNCHER "
            "-GameExecutable $env:HK_OWNER_PYTHON "
            "-PersonalRoot $env:HK_OWNER_PERSONAL "
            "-ExperimentalRoot $env:HK_OWNER_EXPERIMENTAL "
            "-BackupRoot $env:HK_OWNER_BACKUPS "
            "-GuardProcessExecutable $env:HK_OWNER_GAME "
            "-GameArguments @('-m','hk_agent.episode_worker')"
        )
        # Paths travel through environment variables, avoiding PowerShell quoting
        # and keeping secrets out of argv/process listings.
        return (
            str(self._powershell),
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command_text,
        )
