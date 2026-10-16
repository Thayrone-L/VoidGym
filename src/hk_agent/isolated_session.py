from __future__ import annotations

import argparse
import asyncio
import base64
import ctypes
import json
import math
import os
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hk_agent.bridge_environment import BRIDGE_VARIABLES, build_bridge_environment
from hk_agent.gameplay_control import GameplayControlServer
from hk_agent.protocol import ProtocolError
from hk_agent.supervisor import AcceptedBridge, SupervisorServer

REQUEST_ENVIRONMENT_VARIABLE = "HK_ISOLATED_SESSION_REQUEST"
BRIDGE_ENVIRONMENT_VARIABLES = BRIDGE_VARIABLES
MENU_BUTTONS = frozenset({"UP", "DOWN", "LEFT", "RIGHT", "SUBMIT", "CANCEL"})
MAXIMUM_MENU_LEASE_MILLISECONDS = 100


class SessionRequestError(ValueError):
    """Raised when an isolated-session request is malformed."""


@dataclass(frozen=True, slots=True)
class MenuStep:
    button: str
    lease_ms: int

    @classmethod
    def from_mapping(cls, value: Any) -> MenuStep:
        if not isinstance(value, dict):
            raise SessionRequestError("Each menu step must be a JSON object")
        button = value.get("button")
        lease_ms = value.get("lease_ms")
        if not isinstance(button, str) or button not in MENU_BUTTONS:
            raise SessionRequestError(
                "Menu button must be one of: UP, DOWN, LEFT, RIGHT, SUBMIT, CANCEL"
            )
        if (
            isinstance(lease_ms, bool)
            or not isinstance(lease_ms, int)
            or lease_ms < 1
            or lease_ms > MAXIMUM_MENU_LEASE_MILLISECONDS
        ):
            raise SessionRequestError("Menu lease_ms must be an integer between 1 and 100")
        return cls(button=button, lease_ms=lease_ms)


@dataclass(frozen=True, slots=True)
class SessionRequest:
    mode: str
    game_executable: Path
    game_arguments: tuple[str, ...]
    game_profile: str
    instance_id: str
    result_path: Path
    capture_output: Path | None
    personal_root: Path
    experimental_root: Path
    timeout_seconds: float
    close_timeout_seconds: float
    menu_steps: tuple[MenuStep, ...]
    menu_initial_wait_seconds: float
    menu_step_wait_seconds: float
    menu_final_wait_seconds: float
    menu_output_dir: Path | None
    menu_report_output: Path | None
    control_metadata_path: Path | None = None
    control_output_dir: Path | None = None
    control_idle_timeout_seconds: float = 300.0
    control_settle_seconds: float = 0.15
    instance_data_root: Path | None = None
    expected_product_name: str | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> SessionRequest:
        if not isinstance(value, dict):
            raise SessionRequestError("Session request must be a JSON object")

        required_strings = (
            "game_executable",
            "game_profile",
            "instance_id",
            "result_path",
            "personal_root",
            "experimental_root",
        )
        for name in required_strings:
            field = value.get(name)
            if not isinstance(field, str) or not field.strip():
                raise SessionRequestError(f"Session request field {name!r} must be a string")

        raw_arguments = value.get("game_arguments", [])
        if not isinstance(raw_arguments, list) or not all(
            isinstance(argument, str) for argument in raw_arguments
        ):
            raise SessionRequestError("game_arguments must be a list of strings")

        mode = value.get("mode", "smoke")
        if mode not in {"smoke", "menu", "interactive"}:
            raise SessionRequestError("Session mode must be 'smoke', 'menu', or 'interactive'")

        capture_value = value.get("capture_output")
        if capture_value is not None and (
            not isinstance(capture_value, str) or not capture_value.strip()
        ):
            raise SessionRequestError("capture_output must be null or a non-empty string")

        timeout = _positive_number(value.get("timeout_seconds", 30.0), "timeout_seconds")
        close_timeout = _positive_number(
            value.get("close_timeout_seconds", 10.0), "close_timeout_seconds"
        )
        raw_menu_steps = value.get("menu_steps", [])
        if not isinstance(raw_menu_steps, list):
            raise SessionRequestError("menu_steps must be a list")
        menu_steps = tuple(MenuStep.from_mapping(step) for step in raw_menu_steps)
        initial_wait = _bounded_nonnegative_number(
            value.get("menu_initial_wait_seconds", 1.0),
            "menu_initial_wait_seconds",
            30.0,
        )
        step_wait = _bounded_nonnegative_number(
            value.get("menu_step_wait_seconds", 0.08),
            "menu_step_wait_seconds",
            5.0,
        )
        final_wait = _bounded_nonnegative_number(
            value.get("menu_final_wait_seconds", 0.0),
            "menu_final_wait_seconds",
            180.0,
        )
        menu_output_value = value.get("menu_output_dir")
        menu_report_value = value.get("menu_report_output")
        if mode == "menu":
            for name, item in (
                ("menu_output_dir", menu_output_value),
                ("menu_report_output", menu_report_value),
            ):
                if not isinstance(item, str) or not item.strip():
                    raise SessionRequestError(f"{name} is required in menu mode")
        elif mode == "interactive":
            if not isinstance(menu_output_value, str) or not menu_output_value.strip():
                raise SessionRequestError("menu_output_dir is required in interactive mode")
            if menu_report_value is not None:
                raise SessionRequestError("menu_report_output is only valid in menu mode")
        elif raw_menu_steps or menu_output_value is not None or menu_report_value is not None:
            raise SessionRequestError("Menu fields are only valid in menu mode")
        control_metadata_value = value.get("control_metadata_path")
        control_output_value = value.get("control_output_dir")
        if mode == "interactive":
            for name, item in (
                ("control_metadata_path", control_metadata_value),
                ("control_output_dir", control_output_value),
            ):
                if not isinstance(item, str) or not item.strip():
                    raise SessionRequestError(f"{name} is required in interactive mode")
        elif control_metadata_value is not None or control_output_value is not None:
            raise SessionRequestError("Control fields are only valid in interactive mode")
        control_idle_timeout = _bounded_positive_number(
            value.get("control_idle_timeout_seconds", 300.0),
            "control_idle_timeout_seconds",
            3600.0,
        )
        control_settle = _bounded_nonnegative_number(
            value.get("control_settle_seconds", 0.15), "control_settle_seconds", 5.0
        )
        executable = Path(value["game_executable"]).resolve(strict=True)
        if not executable.is_file():
            raise SessionRequestError(f"Game executable is not a file: {executable}")
        data_value = value.get("instance_data_root")
        product_value = value.get("expected_product_name")
        if data_value is not None and (not isinstance(data_value, str) or not data_value.strip()):
            raise SessionRequestError("instance_data_root must be null or a non-empty path")
        if data_value is not None and (not isinstance(product_value, str) or not product_value.strip()):
            raise SessionRequestError("expected_product_name is required for instance mode")

        return cls(
            mode=mode,
            game_executable=executable,
            game_arguments=tuple(raw_arguments),
            game_profile=value["game_profile"].strip(),
            instance_id=value["instance_id"].strip(),
            result_path=Path(value["result_path"]).resolve(strict=False),
            capture_output=(
                Path(capture_value).resolve(strict=False) if capture_value is not None else None
            ),
            # resolve() follows the junction and would erase the evidence that the
            # personal path itself is a reparse point. abspath() canonicalizes the
            # spelling while preserving the path that must be verified.
            personal_root=Path(os.path.abspath(value["personal_root"])),
            experimental_root=Path(os.path.abspath(value["experimental_root"])),
            timeout_seconds=timeout,
            close_timeout_seconds=close_timeout,
            menu_steps=menu_steps,
            menu_initial_wait_seconds=initial_wait,
            menu_step_wait_seconds=step_wait,
            menu_final_wait_seconds=final_wait,
            menu_output_dir=(
                Path(menu_output_value).resolve(strict=False)
                if isinstance(menu_output_value, str)
                else None
            ),
            menu_report_output=(
                Path(menu_report_value).resolve(strict=False)
                if isinstance(menu_report_value, str)
                else None
            ),
            control_metadata_path=(
                Path(control_metadata_value).resolve(strict=False)
                if isinstance(control_metadata_value, str)
                else None
            ),
            control_output_dir=(
                Path(control_output_value).resolve(strict=False)
                if isinstance(control_output_value, str)
                else None
            ),
            control_idle_timeout_seconds=control_idle_timeout,
            control_settle_seconds=control_settle,
            instance_data_root=Path(os.path.abspath(data_value)) if isinstance(data_value, str) else None,
            expected_product_name=product_value if isinstance(product_value, str) else None,
        )


@dataclass(slots=True)
class SessionResult:
    mode: str = "smoke"
    ok: bool = False
    failure: str | None = None
    handshake_ok: bool = False
    heartbeat_ok: bool = False
    ping_ok: bool = False
    capture_ok: bool = False
    release_all_ok: bool = False
    run_id: str | None = None
    instance_id: str | None = None
    session_id: str | None = None
    game_profile: str | None = None
    game_version: str | None = None
    heartbeats_received: int = 0
    captured_frame_id: int | None = None
    captured_width: int | None = None
    captured_height: int | None = None
    captured_bytes: int = 0
    capture_output: str | None = None
    close_windows: int = 0
    forced_termination: bool = False
    game_exit_code: int | None = None
    menu_steps_completed: int = 0
    menu_captures: list[dict[str, object]] = field(default_factory=list)
    menu_report_output: str | None = None
    control_metadata_path: str | None = None
    control_output_dir: str | None = None
    control_close_reason: str | None = None
    control_commands_completed: int = 0
    control_captures: list[dict[str, object]] = field(default_factory=list)


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SessionRequestError(f"{name} must be a positive finite number")
    converted = float(value)
    maximum = 300.0 if name == "timeout_seconds" else 60.0
    if not math.isfinite(converted) or converted <= 0 or converted > maximum:
        raise SessionRequestError(f"{name} must be positive, finite, and at most {maximum:g}")
    return converted


def _bounded_nonnegative_number(value: Any, name: str, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SessionRequestError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0 or converted > maximum:
        raise SessionRequestError(
            f"{name} must be finite, non-negative, and at most {maximum:g}"
        )
    return converted


def _bounded_positive_number(value: Any, name: str, maximum: float) -> float:
    converted = _bounded_nonnegative_number(value, name, maximum)
    if converted == 0:
        raise SessionRequestError(f"{name} must be positive")
    return converted


def assert_save_isolation(request: SessionRequest) -> None:
    if request.instance_data_root is not None:
        if not request.instance_data_root.is_dir():
            raise SessionRequestError("Instance data root must exist before the game starts")
        try:
            if os.path.samefile(request.instance_data_root, request.personal_root):
                raise SessionRequestError("Instance data root resolves to personal saves")
        except FileNotFoundError:
            raise SessionRequestError("Instance and personal save roots must exist") from None
        return
    if not request.personal_root.is_dir() or not request.experimental_root.is_dir():
        raise SessionRequestError("Save-isolation roots must exist before the game starts")
    if not os.path.samefile(request.personal_root, request.experimental_root):
        raise SessionRequestError(
            "Personal save root is not redirected to the experimental save root"
        )
    stat_result = request.personal_root.lstat()
    reparse_attribute = getattr(stat_result, "st_file_attributes", 0) & 0x400
    if os.name == "nt" and not reparse_attribute:
        raise SessionRequestError("Personal save root is not a Windows reparse point")


def load_request_from_environment() -> SessionRequest:
    encoded = os.environ.pop(REQUEST_ENVIRONMENT_VARIABLE, None)
    if not encoded:
        raise SessionRequestError(f"{REQUEST_ENVIRONMENT_VARIABLE} is missing")
    try:
        document = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")))
    except (ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise SessionRequestError("The isolated-session request is invalid") from error
    return SessionRequest.from_mapping(document)


def _game_environment(supervisor: SupervisorServer, request: SessionRequest) -> dict[str, str]:
    if supervisor.port is None:
        raise RuntimeError("Supervisor did not bind a port")
    if request.instance_data_root is None:
        mode = "legacy"
        save_root = str(request.experimental_root.resolve(strict=False))
        product_name = instance_data_root = None
    else:
        mode = "instance"
        save_root = None
        product_name = request.expected_product_name
        instance_data_root = str(request.instance_data_root)
    environment = build_bridge_environment(
        os.environ,
        mode=mode,
        host="127.0.0.1",
        port=str(supervisor.port),
        run_id=supervisor.run_id,
        instance_id=request.instance_id,
        token=supervisor.token,
        game_profile=request.game_profile,
        save_root=save_root,
        product_name=product_name,
        instance_data_root=instance_data_root,
    )
    environment.pop(REQUEST_ENVIRONMENT_VARIABLE, None)
    return environment


def _request_close(process_id: int) -> int:
    if sys.platform != "win32":
        return 0

    user32 = ctypes.windll.user32
    window_count = 0
    enum_proc_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def enum_proc(window: int, _: int) -> bool:
        nonlocal window_count
        owner_process_id = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(window, ctypes.byref(owner_process_id))
        if owner_process_id.value == process_id:
            user32.PostMessageW(window, 0x0010, 0, 0)  # WM_CLOSE
            window_count += 1
        return True

    callback = enum_proc_type(enum_proc)
    user32.EnumWindows(callback, 0)
    return window_count


async def _close_process(
    process: asyncio.subprocess.Process, timeout: float
) -> tuple[int, bool, int]:
    if process.returncode is not None:
        return process.returncode, False, 0

    close_windows = _request_close(process.pid)
    if sys.platform != "win32":
        process.send_signal(signal.SIGTERM)
    try:
        return await asyncio.wait_for(process.wait(), timeout), False, close_windows
    except TimeoutError:
        process.terminate()
        try:
            return await asyncio.wait_for(process.wait(), timeout), True, close_windows
        except TimeoutError:
            process.kill()
            return await asyncio.wait_for(process.wait(), timeout), True, close_windows


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _write_result_atomic(path: Path, result: SessionResult) -> None:
    content = json.dumps(asdict(result), indent=2, ensure_ascii=False).encode("utf-8")
    _write_bytes_atomic(path, content)


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    content = json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8")
    _write_bytes_atomic(path, content)


async def _release_all(supervisor: SupervisorServer, bridge: AcceptedBridge) -> bool:
    try:
        response = await supervisor.send_command("RELEASE_ALL", {}, timeout=2.0, bridge=bridge)
        return response.message_type == "ACK"
    except (ConnectionError, OSError, ProtocolError, RuntimeError, TimeoutError):
        return False


async def _capture_menu_frame(
    request: SessionRequest,
    result: SessionResult,
    supervisor: SupervisorServer,
    bridge: AcceptedBridge,
    index: int,
    label: str,
) -> None:
    if request.menu_output_dir is None:
        raise RuntimeError("Menu output directory is unavailable")
    frame = await supervisor.capture_once(timeout=request.timeout_seconds, bridge=bridge)
    output = request.menu_output_dir / f"{index:03d}-{label}.png"
    _write_bytes_atomic(output, frame.data)
    result.menu_captures.append(
        {
            "index": index,
            "label": label,
            "path": str(output),
            "frame_id": frame.frame_id,
            "capture_monotonic_ns": frame.capture_monotonic_ns,
            "width": frame.width,
            "height": frame.height,
            "bytes": len(frame.data),
            "scene_transition": frame.scene_transition,
            "duplicate_of": frame.duplicate_of,
        }
    )


async def _run_menu_sequence(
    request: SessionRequest,
    result: SessionResult,
    supervisor: SupervisorServer,
    bridge: AcceptedBridge,
) -> None:
    await asyncio.sleep(request.menu_initial_wait_seconds)
    await _capture_menu_frame(request, result, supervisor, bridge, 0, "before")
    result.capture_ok = True

    for index, step in enumerate(request.menu_steps, start=1):
        # A prior step may have released successfully. Mark the new pulse as
        # unreleased before sending it so every error/timeout path triggers the
        # outer best-effort RELEASE_ALL during teardown.
        result.release_all_ok = False
        response = await supervisor.send_command(
            "APPLY_MENU_ACTION",
            {"buttons": [step.button], "lease_ms": step.lease_ms},
            timeout=request.timeout_seconds,
            bridge=bridge,
        )
        if response.message_type == "ERROR":
            error_code = response.payload.get("error_code", "unknown")
            raise ProtocolError(f"Menu action failed: {error_code}")
        if response.message_type != "ACK":
            raise ProtocolError(
                f"Expected ACK for APPLY_MENU_ACTION, received {response.message_type}"
            )

        released = False
        try:
            await asyncio.sleep(request.menu_step_wait_seconds)
        finally:
            released = await _release_all(supervisor, bridge)
            result.release_all_ok = released
        if not released:
            raise RuntimeError("RELEASE_ALL failed after acknowledged menu action")

        await _capture_menu_frame(
            request,
            result,
            supervisor,
            bridge,
            index,
            f"after-{step.button.lower()}",
        )
        result.menu_steps_completed = index

    result.release_all_ok = await _release_all(supervisor, bridge)
    if not result.release_all_ok:
        raise RuntimeError("Final RELEASE_ALL failed")
    if request.menu_final_wait_seconds > 0:
        await asyncio.sleep(request.menu_final_wait_seconds)
        await _capture_menu_frame(
            request,
            result,
            supervisor,
            bridge,
            len(request.menu_steps) + 1,
            "final-wait",
        )


async def run_session(request: SessionRequest) -> SessionResult:
    result = SessionResult(
        mode=request.mode,
        run_id=None,
        capture_output=str(request.capture_output) if request.capture_output is not None else None,
        menu_report_output=(
            str(request.menu_report_output) if request.menu_report_output is not None else None
        ),
        control_metadata_path=(
            str(request.control_metadata_path) if request.control_metadata_path is not None else None
        ),
        control_output_dir=(
            str(request.control_output_dir) if request.control_output_dir is not None else None
        ),
    )
    supervisor = SupervisorServer(
        game_profile=request.game_profile,
        expected_persistent_data_path=request.instance_data_root,
        forbidden_personal_root=request.personal_root if request.instance_data_root else None,
        expected_product_name=request.expected_product_name,
    )
    process: asyncio.subprocess.Process | None = None
    bridge: AcceptedBridge | None = None
    control: GameplayControlServer | None = None
    try:
        assert_save_isolation(request)
        await supervisor.start(host="127.0.0.1", port=0)
        result.run_id = supervisor.run_id
        process = await asyncio.create_subprocess_exec(
            str(request.game_executable),
            *request.game_arguments,
            cwd=request.game_executable.parent,
            env=_game_environment(supervisor, request),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        bridge = await supervisor.wait_for_bridge(request.timeout_seconds)
        result.handshake_ok = True
        result.instance_id = bridge.instance_id
        result.session_id = bridge.session_id
        result.game_profile = bridge.game_profile
        result.game_version = bridge.game_version

        bridge = await supervisor.wait_for_heartbeat(request.timeout_seconds)
        result.heartbeat_ok = True
        result.heartbeats_received = bridge.heartbeats_received

        if request.mode == "menu":
            await _run_menu_sequence(request, result, supervisor, bridge)
            result.ok = all(
                (
                    result.handshake_ok,
                    result.heartbeat_ok,
                    result.capture_ok,
                    result.release_all_ok,
                    result.menu_steps_completed == len(request.menu_steps),
                )
            )
        elif request.mode == "interactive":
            if request.control_metadata_path is None or request.control_output_dir is None:
                raise RuntimeError("Interactive control paths are unavailable")
            await _run_menu_sequence(request, result, supervisor, bridge)
            control = GameplayControlServer(
                supervisor,
                bridge,
                output_dir=request.control_output_dir,
                idle_timeout_seconds=request.control_idle_timeout_seconds,
                settle_seconds=request.control_settle_seconds,
                command_timeout_seconds=request.timeout_seconds,
            )
            await control.start()
            address = control.address
            _write_json_atomic(
                request.control_metadata_path,
                {
                    "schema_version": 1,
                    "status": "ready",
                    "host": address.host,
                    "port": address.port,
                    "token": address.token,
                    "run_id": supervisor.run_id,
                    "instance_id": bridge.instance_id,
                    "game_pid": process.pid,
                    "output_dir": str(request.control_output_dir),
                },
            )
            result.control_close_reason = await control.wait_until_closed()
            result.control_commands_completed = control.commands_completed
            result.control_captures = list(control.captures)
            result.release_all_ok = await _release_all(supervisor, bridge)
            result.capture_ok = True
            result.ok = all(
                (
                    result.handshake_ok,
                    result.heartbeat_ok,
                    result.release_all_ok,
                    result.control_close_reason == "client_close",
                )
            )
            if result.control_close_reason != "client_close":
                result.failure = f"Interactive session closed by {result.control_close_reason}"
        else:
            ping = await supervisor.send_command(
                "PING", {}, timeout=request.timeout_seconds, bridge=bridge
            )
            result.ping_ok = ping.message_type == "ACK"

            frame = await supervisor.capture_once(timeout=request.timeout_seconds, bridge=bridge)
            result.capture_ok = True
            result.captured_frame_id = frame.frame_id
            result.captured_width = frame.width
            result.captured_height = frame.height
            result.captured_bytes = len(frame.data)
            if request.capture_output is not None:
                _write_bytes_atomic(request.capture_output, frame.data)

            result.release_all_ok = await _release_all(supervisor, bridge)
            result.ok = all(
                (
                    result.handshake_ok,
                    result.heartbeat_ok,
                    result.ping_ok,
                    result.capture_ok,
                    result.release_all_ok,
                )
            )
    except (ConnectionError, OSError, RuntimeError, TimeoutError, ValueError) as error:
        result.failure = f"{type(error).__name__}: {error}"
    finally:
        if control is not None:
            try:
                await control.close()
            except (OSError, RuntimeError) as error:
                if result.failure is None:
                    result.failure = f"Control cleanup failed: {type(error).__name__}: {error}"
                result.ok = False
        if bridge is not None and not result.release_all_ok:
            result.release_all_ok = await _release_all(supervisor, bridge)
        # Keep the supervisor available while the game handles WM_CLOSE. The bridge
        # owns the peer socket and may keep it open until Unity actually exits; waiting
        # for writer.wait_closed() before closing the game creates a shutdown deadlock.
        if process is not None:
            try:
                exit_code, forced, close_windows = await _close_process(
                    process, request.close_timeout_seconds
                )
                result.game_exit_code = exit_code
                result.forced_termination = forced
                result.close_windows = close_windows
            except (OSError, RuntimeError, TimeoutError) as error:
                if result.failure is None:
                    result.failure = f"Game cleanup failed: {type(error).__name__}: {error}"
                result.ok = False
        try:
            await asyncio.wait_for(supervisor.close(), timeout=2.0)
        except (OSError, RuntimeError, TimeoutError) as error:
            if result.failure is None:
                result.failure = f"Supervisor cleanup failed: {type(error).__name__}: {error}"
            result.ok = False
        if result.forced_termination:
            result.ok = False
            if result.failure is None:
                result.failure = "Game required forced termination"
        if result.ok and result.game_exit_code not in (0, None):
            result.ok = False
            result.failure = f"Game exited with code {result.game_exit_code}"
        if request.menu_report_output is not None:
            try:
                _write_result_atomic(request.menu_report_output, result)
            except OSError as error:
                result.ok = False
                if result.failure is None:
                    result.failure = f"Menu report failed: {type(error).__name__}: {error}"
        if request.control_metadata_path is not None:
            try:
                _write_json_atomic(
                    request.control_metadata_path,
                    {
                        "schema_version": 1,
                        "status": "closed",
                        "run_id": result.run_id,
                        "reason": result.control_close_reason or result.failure or "startup_failed",
                        "commands_completed": result.control_commands_completed,
                    },
                )
            except OSError as error:
                result.ok = False
                if result.failure is None:
                    result.failure = f"Control metadata cleanup failed: {type(error).__name__}: {error}"
        _write_result_atomic(request.result_path, result)
    return result


async def run_smoke(request: SessionRequest) -> SessionResult:
    """Compatibility wrapper for callers that still use the original function name."""
    return await run_session(request)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bridge session inside save isolation")
    parser.add_argument(
        "--request-env",
        action="store_true",
        help=f"read the non-secret request from {REQUEST_ENVIRONMENT_VARIABLE}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.request_env:
        raise SystemExit("--request-env is required")
    request = load_request_from_environment()
    result = asyncio.run(run_session(request))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
