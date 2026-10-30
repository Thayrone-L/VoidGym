from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hk_agent.instance_install import (
    COMPANY_NAME,
    INSTANCES_ROOT,
    ORIGINAL_INSTALL,
    PERSONAL_ROOT,
    InstanceInstallError,
    installation_hashes,
)
from hk_agent.instance_marker import load_instance_marker

BACKUP_ROOT = Path(r"C:\Users\thayr\Games\HollowKnight-AI\save-backups")
RUNTIME_ROOT = Path(r"C:\Users\thayr\Games\HollowKnight-AI\runtime")
REGISTRY_ROOT = rf"HKCU\Software\{COMPANY_NAME}"
PERSONAL_REGISTRY_KEY = rf"{REGISTRY_ROOT}\Hollow Knight"
WAIT_TIMEOUT = 90
POST_CREATE_WAIT = 20


def tree_snapshot(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not root.exists():
        return result
    for path in root.rglob("*"):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            result[str(path.relative_to(root))] = digest
    return result


def snapshot_changes(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    before_keys, after_keys = set(before), set(after)
    return {
        "added": sorted(after_keys - before_keys),
        "removed": sorted(before_keys - after_keys),
        "changed": sorted(key for key in before_keys & after_keys if before[key] != after[key]),
    }


def pids_for_window_records(records: list[tuple[int, int]], target_pid: int) -> list[int]:
    """Return hwnds belonging only to the process launched by this verifier."""
    return [hwnd for hwnd, pid in records if pid == target_pid]


def _reg_export(key: str, output: Path) -> bool:
    result = subprocess.run(
        ["reg.exe", "export", key, str(output), "/y"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and output.is_file()


def _canonical_registry_key(key: str) -> str:
    normalized = key.strip().strip("[]").replace("/", "\\").lower()
    for alias, canonical in (("hkey_current_user", "hkcu"), ("hkey_local_machine", "hklm")):
        if normalized == alias or normalized.startswith(alias + "\\"):
            return canonical + normalized[len(alias):]
    return normalized


def _registry_section(export_path: Path, key: str) -> str:
    raw = export_path.read_bytes()
    text = raw.decode("utf-16") if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else raw.decode("utf-8-sig")
    wanted = _canonical_registry_key(key)
    current_key: str | None = None
    current_values: list[str] = []
    found: list[str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if current_key == wanted:
                found = sorted(current_values, key=str.casefold)
                break
            current_key = _canonical_registry_key(stripped)
            current_values = []
        elif current_key == wanted and stripped:
            current_values.append(stripped)
    if found is None and current_key == wanted:
        found = sorted(current_values, key=str.casefold)
    if found is None:
        raise InstanceInstallError(f"Registry export is missing required section: {key}")
    return wanted + "\n" + "\n".join(found)


def _running_game_pids() -> list[int]:
    command = (
        "Get-CimInstance Win32_Process -Filter \"Name='Hollow Knight.exe'\" "
        "| Select-Object -ExpandProperty ProcessId"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise InstanceInstallError(f"Could not inspect Hollow Knight processes: {result.stderr}")
    return [int(line) for line in result.stdout.splitlines() if line.strip().isdigit()]


def _windows_for_pid(pid: int) -> list[int]:
    user32 = ctypes.windll.user32
    enum_windows = user32.EnumWindows
    get_pid = user32.GetWindowThreadProcessId
    found: list[int] = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def callback(hwnd: int, _lparam: int) -> bool:
        process_id = ctypes.c_ulong()
        get_pid(hwnd, ctypes.byref(process_id))
        found.append((int(hwnd), int(process_id.value)))
        return True

    callback_ref = callback_type(callback)
    if not enum_windows(callback_ref, 0):
        raise OSError("EnumWindows failed")
    return pids_for_window_records(found, pid)


def _close_process_window(pid: int, process: subprocess.Popen[bytes]) -> None:
    user32 = ctypes.windll.user32
    for hwnd in _windows_for_pid(pid):
        user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        subprocess.run(
            ["taskkill.exe", "/PID", str(pid), "/F"],
            capture_output=True,
            check=False,
        )
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired as error:
            raise InstanceInstallError(f"Could not stop verifier PID {pid}") from error


def verify_instance(index: int) -> dict[str, object]:
    if os.name != "nt":
        raise InstanceInstallError("Isolation proof requires Windows")
    personal = PERSONAL_ROOT
    original_before = installation_hashes(ORIGINAL_INSTALL)
    install = INSTANCES_ROOT / f"hk-ai-{index}"
    try:
        marker_obj = load_instance_marker(install, expected_index=index)
    except ValueError as error:
        raise InstanceInstallError(str(error)) from error
    marker = marker_obj.__dict__
    if _running_game_pids():
        raise InstanceInstallError("Close every Hollow Knight.exe process before this proof")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = BACKUP_ROOT / f"pre-isolation-{timestamp}"
    backup.mkdir(parents=True, exist_ok=False)
    if personal.exists():
        import shutil

        shutil.copytree(personal, backup / personal.name)
    if not _reg_export(PERSONAL_REGISTRY_KEY, backup / "personal-playerprefs.reg"):
        raise InstanceInstallError("Could not export personal PlayerPrefs registry key")

    before_tree = tree_snapshot(personal)
    before_registry = backup / "team-cherry-before.reg"
    if not _reg_export(REGISTRY_ROOT, before_registry):
        raise InstanceInstallError("Could not snapshot Team Cherry registry tree")
    before_personal_reg = _registry_section(before_registry, PERSONAL_REGISTRY_KEY)

    log_dir = RUNTIME_ROOT / "isolation-tests" / f"instance-{index}-{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=False)
    log_path = log_dir / "Player.log"
    executable = install / "Hollow Knight.exe"
    env = os.environ.copy()
    for key in list(env):
        if key.upper().startswith("HK_AGENT_"):
            env.pop(key)
    command = [
        str(executable), "-screen-fullscreen", "0", "-screen-width", "960",
        "-screen-height", "540", "-logFile", str(log_path),
    ]
    started_utc = datetime.now(UTC).isoformat()
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=install, env=env)
    pid = process.pid
    data_dir = personal.parent / marker["product_name"]
    created = False
    try:
        while time.monotonic() - started < WAIT_TIMEOUT:
            if data_dir.is_dir():
                created = True
                break
            if process.poll() is not None:
                break
            time.sleep(1)
        if created:
            time.sleep(POST_CREATE_WAIT)
    finally:
        _close_process_window(pid, process)

    after_tree = tree_snapshot(personal)
    tree_delta = snapshot_changes(before_tree, after_tree)
    after_registry_path = log_dir / "team-cherry-after.reg"
    registry_exported = _reg_export(REGISTRY_ROOT, after_registry_path)
    after_personal_reg = (
        _registry_section(after_registry_path, PERSONAL_REGISTRY_KEY)
        if registry_exported else None
    )
    expected_reg = rf"{REGISTRY_ROOT}\{marker['product_name']}"
    expected_reg_file = log_dir / "expected-instance-playerprefs.reg"
    expected_reg_exists = _reg_export(expected_reg, expected_reg_file)
    original_after = installation_hashes(ORIGINAL_INSTALL)
    ended_utc = datetime.now(UTC).isoformat()
    checks = {
        "personal_tree_unchanged": not any(tree_delta.values()),
        "personal_registry_unchanged": before_personal_reg == after_personal_reg,
        "instance_data_directory_exists": data_dir.is_dir(),
        "instance_registry_key_exists": expected_reg_exists,
        "requested_log_created": log_path.is_file(),
        "original_install_unchanged": original_before == original_after,
    }
    manifest: dict[str, object] = {
        "index": index,
        "created_at": datetime.now(UTC).isoformat(),
        "started_at": started_utc,
        "ended_at": ended_utc,
        "installation": str(install),
        "instance_pid": pid,
        "started_elapsed_seconds": round(time.monotonic() - started, 3),
        "backup": str(backup),
        "runtime_log": str(log_path),
        "personal_data_path": str(personal),
        "instance_data_path": str(data_dir),
        "expected_registry_key": expected_reg,
        "original_hashes_before": {
            "executable": original_before.executable,
            "globalgamemanagers": original_before.globalgamemanagers,
        },
        "original_hashes_after": {
            "executable": original_after.executable,
            "globalgamemanagers": original_after.globalgamemanagers,
        },
        "personal_tree_changes": tree_delta,
        "personal_tree_hashes_before": before_tree,
        "personal_tree_hashes_after": after_tree,
        "registry_snapshots": {
            "personal_backup": {
                "path": str(backup / "personal-playerprefs.reg"),
                "sha256": hashlib.sha256(
                    (backup / "personal-playerprefs.reg").read_bytes()
                ).hexdigest(),
            },
            "team_cherry_before": {
                "path": str(before_registry),
                "sha256": hashlib.sha256(before_registry.read_bytes()).hexdigest(),
            },
            "team_cherry_after": {
                "path": str(after_registry_path) if registry_exported else None,
                "sha256": (
                    hashlib.sha256(after_registry_path.read_bytes()).hexdigest()
                    if registry_exported else None
                ),
            },
        },
        "checks": checks,
        "passed": all(checks.values()),
        "suggested_restore": (
            f"If personal files changed, close the game and manually restore the saved copy at "
            f"{backup / personal.name}; do not restore the registry unless its delta is reviewed."
        ),
    }
    manifest_path = log_dir / "isolation-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    if not all(checks.values()):
        changed = [*tree_delta["added"], *tree_delta["removed"], *tree_delta["changed"]]
        print("Personal tree differences: " + (", ".join(changed) if changed else "none"))
        raise InstanceInstallError(f"Isolation proof failed; manifest: {manifest_path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the authorized instance isolation proof")
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument(
        "--authorized", action="store_true",
        help="confirm the user explicitly authorized opening the game in this chat",
    )
    args = parser.parse_args()
    if not args.authorized:
        parser.error("Part B requires explicit user authorization; pass --authorized only after it")
    try:
        result = verify_instance(args.index)
    except (InstanceInstallError, OSError) as error:
        print(f"Isolation proof failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
