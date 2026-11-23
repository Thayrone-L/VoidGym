from __future__ import annotations

import argparse
import asyncio
import ctypes
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hk_agent.supervisor import CapturedFrame, SupervisorServer, require_png_frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_hashes(save_root: Path) -> dict[str, str]:
    if not save_root.is_dir():
        return {}
    return {
        str(path.resolve()): _sha256(path)
        for path in sorted(save_root.glob("user*.dat*"))
        if path.is_file()
    }


def _write_png_capture(output: Path, frame: CapturedFrame) -> None:
    require_png_frame(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(frame.data)


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


async def _wait_for_exit(process: asyncio.subprocess.Process, timeout: float) -> int:
    try:
        return await asyncio.wait_for(process.wait(), timeout)
    except TimeoutError:
        process.terminate()
        return await asyncio.wait_for(process.wait(), 10.0)


async def run(args: argparse.Namespace) -> int:
    executable = args.executable.resolve(strict=True)
    save_root = args.personal_save_root.resolve(strict=False)
    saves_before = _save_hashes(save_root)

    supervisor = SupervisorServer(game_profile=args.game_profile)
    await supervisor.start()
    if supervisor.port is None:
        raise RuntimeError("Supervisor did not bind a TCP port")

    environment = os.environ.copy()
    environment.update(
        {
            "HK_AGENT_HOST": "127.0.0.1",
            "HK_AGENT_PORT": str(supervisor.port),
            "HK_AGENT_RUN_ID": supervisor.run_id,
            "HK_AGENT_INSTANCE_ID": args.instance_id,
            "HK_AGENT_TOKEN": supervisor.token,
            "HK_AGENT_GAME_PROFILE": args.game_profile,
        }
    )

    process = await asyncio.create_subprocess_exec(
        str(executable),
        cwd=executable.parent,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    accepted = None
    ping_ok = False
    action_ok = False
    capture_ok = False
    captured_width: int | None = None
    captured_height: int | None = None
    captured_bytes = 0
    release_all_ok = False
    failure: str | None = None
    close_windows = 0
    exit_code: int | None = None
    try:
        accepted = await supervisor.wait_for_bridge(args.timeout)
        accepted = await supervisor.wait_for_heartbeat(args.timeout)
        ping = await supervisor.send_command("PING", {}, bridge=accepted)
        ping_ok = ping.message_type == "ACK"
        action = await supervisor.send_command(
            "APPLY_ACTION",
            {"action_id": 1, "buttons": ["LEFT"], "lease_ms": 100},
            bridge=accepted,
        )
        action_ok = action.message_type == "ACK"
        await asyncio.sleep(0.25)
        frame = await supervisor.capture_once(timeout=10.0, bridge=accepted)
        capture_ok = True
        captured_width = frame.width
        captured_height = frame.height
        captured_bytes = len(frame.data)
        if args.capture_output is not None:
            output = args.capture_output.resolve(strict=False)
            _write_png_capture(output, frame)
        released = await supervisor.send_command("RELEASE_ALL", {}, bridge=accepted)
        release_all_ok = released.message_type == "ACK"
        await asyncio.sleep(args.settle_seconds)
    except TimeoutError as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        close_windows = _request_close(process.pid)
        exit_code = await _wait_for_exit(process, 10.0)
        await supervisor.close()

    saves_after = _save_hashes(save_root)
    changed_saves = sorted(
        path
        for path in saves_before.keys() | saves_after.keys()
        if saves_before.get(path) != saves_after.get(path)
    )
    result = {
        "handshake_ok": accepted is not None,
        "failure": failure,
        "run_id": supervisor.run_id,
        "instance_id": accepted.instance_id if accepted else None,
        "session_id": accepted.session_id if accepted else None,
        "game_profile": accepted.game_profile if accepted else None,
        "game_version": accepted.game_version if accepted else None,
        "heartbeats_received": accepted.heartbeats_received if accepted else 0,
        "ping_ok": ping_ok,
        "action_ok": action_ok,
        "capture_ok": capture_ok,
        "captured_width": captured_width,
        "captured_height": captured_height,
        "captured_bytes": captured_bytes,
        "capture_output": str(args.capture_output.resolve()) if args.capture_output else None,
        "release_all_ok": release_all_ok,
        "game_exit_code": exit_code,
        "close_windows": close_windows,
        "changed_personal_saves": changed_saves,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return (
        0
        if accepted is not None
        and ping_ok
        and action_ok
        and capture_ok
        and release_all_ok
        and not changed_saves
        else 1
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the real Hollow Knight bridge")
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--personal-save-root", type=Path, required=True)
    parser.add_argument("--game-profile", default="gog-1.5.78.11833")
    parser.add_argument("--instance-id", default="smoke-01")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--capture-output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
