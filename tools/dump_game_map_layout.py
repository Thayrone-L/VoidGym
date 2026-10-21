from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hk_agent.instance_marker import load_instance_marker


def _request_window_close(pid: int) -> None:
    if os.name != "nt":
        return
    user32 = ctypes.windll.user32
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def close_for_pid(hwnd: int, _unused: int) -> bool:
        process_id = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        if process_id.value == pid and user32.IsWindowVisible(hwnd):
            user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
            return False
        return True

    user32.EnumWindows(callback_type(close_for_pid), 0)


def dump_layout(instance_root: Path, runtime_root: Path, *, timeout: float = 120.0) -> Path:
    root = instance_root.expanduser().resolve(strict=True)
    if root.is_symlink():
        raise ValueError("isolated game install must not be a symlink")
    marker = load_instance_marker(root, expected_index=2)
    executable = root / "Hollow Knight.exe"
    if not executable.is_file():
        raise FileNotFoundError(executable)
    target = runtime_root / "map-layout"
    target.mkdir(parents=True, exist_ok=True)
    request = target / "DUMP_MAP_LAYOUT.request"
    output = target / "game-map-layout.json"
    if request.exists():
        raise RuntimeError("a map layout dump request is already pending")
    if output.exists():
        backup = target / f"game-map-layout-{time.strftime('%Y%m%d-%H%M%S')}.json.bak"
        output.replace(backup)
    request.write_text(f"instance=hk-ai-2 marker={marker.bridge_dll_sha256}\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.pop("HK_AGENT_HOST", None)
    environment.pop("HK_AGENT_PORT", None)
    environment["HK_AGENT_MAP_LAYOUT_DUMP_REQUEST"] = str(request)
    process = subprocess.Popen([str(executable)], cwd=root, env=environment)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if output.is_file() and not request.exists():
                return output
            error = request.with_suffix(request.suffix + ".error.txt")
            if error.is_file():
                raise RuntimeError(error.read_text("utf-8", errors="replace"))
            if process.poll() is not None:
                raise RuntimeError(f"isolated Hollow Knight exited with status {process.returncode} before the dump")
            time.sleep(0.25)
        raise TimeoutError(f"map layout dump did not finish within {timeout:.0f} seconds")
    finally:
        if process.poll() is None:
            _request_window_close(process.pid)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                print(f"Isolated game {process.pid} did not close after WM_CLOSE; leaving it running for safe manual closure.", file=sys.stderr)
        if request.exists():
            request.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect GameMap layout from isolated instance 2.")
    parser.add_argument("--instance-root", type=Path,
                        default=Path.home() / "Games" / "HollowKnight-AI" / "instances" / "hk-ai-2")
    parser.add_argument("--runtime-root", type=Path,
                        default=Path.home() / "Games" / "HollowKnight-AI" / "runtime")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    try:
        path = dump_layout(args.instance_root, args.runtime_root, timeout=args.timeout)
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        print(f"GameMap layout dump failed: {error}", file=sys.stderr)
        return 1
    print(f"GameMap layout saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
