from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hk_agent.instance_install import (
    INSTANCES_ROOT,
    MARKER_NAME,
    InstanceInstallError,
    refresh_instance_bridge_marker,
    sha256_file,
)
from hk_agent.instance_marker import load_instance_marker

DEFAULT_BRIDGE = PROJECT_ROOT / "bridge" / "HkAgentBridge.Mod" / "bin" / "Release" / "HkAgentBridge.dll"


def deploy(bridge_dll: Path, *, instance_indices: list[int], instances_root: Path,
           backup_root: Path) -> list[dict[str, object]]:
    bridge = bridge_dll.expanduser().resolve(strict=True)
    if bridge.name.casefold() != "hkagentbridge.dll":
        raise InstanceInstallError(f"Expected HkAgentBridge.dll build artifact: {bridge}")
    build_hash = sha256_file(bridge)
    instance_base = instances_root.expanduser().resolve(strict=True)
    indices = list(instance_indices)
    if not indices or any(index < 1 for index in indices) or len(set(indices)) != len(indices):
        raise InstanceInstallError("Instance indices must be unique positive integers")

    # Validate the entire deployment set before creating backups or changing files.
    plan: list[dict[str, Any]] = []
    for index in indices:
        candidate = instance_base / f"hk-ai-{index}"
        if candidate.is_symlink():
            raise InstanceInstallError(f"Refusing symlinked instance root: {candidate}")
        root = candidate.resolve(strict=True)
        if root.parent != instance_base:
            raise InstanceInstallError(f"Refusing instance outside managed root: {root}")
        marker_path = root / MARKER_NAME
        target = root / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge" / "HkAgentBridge.dll"
        if marker_path.is_symlink() or target.is_symlink() or not marker_path.is_file() or not target.is_file():
            raise InstanceInstallError(f"Managed instance {index} is missing regular marker/DLL files")
        load_instance_marker(root, expected_index=index)
        plan.append({"index": index, "root": root, "target": target, "marker_path": marker_path,
                     "old_dll_hash": sha256_file(target), "old_marker_hash": sha256_file(marker_path)})

    try:
        import psutil
    except ImportError as error:
        raise InstanceInstallError("psutil is required to check whether managed games are running") from error
    for process in psutil.process_iter(["name", "exe"]):
        try:
            if str(process.info.get("name") or "").casefold() != "hollow knight.exe":
                continue
            executable = process.info.get("exe")
            if not executable:
                raise InstanceInstallError("Cannot identify a running Hollow Knight installation")
            running = os.path.normcase(str(Path(executable).resolve(strict=False)))
            for item in plan:
                if running == os.path.normcase(str((item["root"] / "Hollow Knight.exe").resolve(strict=False))):
                    raise InstanceInstallError(f"Instance {item['index']} game is running; deployment refused")
        except psutil.Error as error:
            raise InstanceInstallError("Cannot safely inspect running Hollow Knight processes") from error

    backup_dir = backup_root.expanduser().resolve(strict=False) / f"bridge-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    for item in plan:
        backup_instance = backup_dir / f"hk-ai-{item['index']}"
        backup_instance.mkdir(parents=True, exist_ok=False)
        item["backup_instance"] = backup_instance
        item["backup_dll"] = backup_instance / item["target"].name
        item["backup_marker"] = backup_instance / MARKER_NAME
        shutil.copy2(item["target"], item["backup_dll"])
        shutil.copy2(item["marker_path"], item["backup_marker"])
        if (sha256_file(item["backup_dll"]) != item["old_dll_hash"] or
                sha256_file(item["backup_marker"]) != item["old_marker_hash"]):
            raise InstanceInstallError(f"Backup validation failed for instance {item['index']}: {backup_instance}")

    touched: list[dict[str, Any]] = []
    try:
        for item in plan:
            touched.append(item)
            shutil.copy2(bridge, item["target"])
            if sha256_file(item["target"]) != build_hash:
                raise InstanceInstallError(f"Instance {item['index']} DLL differs from build after copy")
            refresh_instance_bridge_marker(item["root"], expected_index=item["index"], expected_bridge_hash=build_hash)
            validated = load_instance_marker(item["root"], expected_index=item["index"])
            if validated.bridge_dll_sha256 != build_hash:
                raise InstanceInstallError(f"Instance {item['index']} marker does not retain the build hash")
        return [{"index": item["index"], "root": str(item["root"]), "sha256": build_hash,
                 "backup": str(item["backup_instance"])} for item in plan]
    except Exception as error:
        rollback_errors: list[str] = []
        for item in reversed(touched):
            try:
                shutil.copy2(item["backup_dll"], item["target"])
                shutil.copy2(item["backup_marker"], item["marker_path"])
                if sha256_file(item["target"]) != item["old_dll_hash"]:
                    raise InstanceInstallError("restored DLL hash does not match backup")
                if sha256_file(item["marker_path"]) != item["old_marker_hash"]:
                    raise InstanceInstallError("restored marker hash does not match backup")
            except Exception as rollback_error:  # noqa: BLE001 - preserve original deployment error.
                rollback_errors.append(
                    f"instance {item['index']}: {rollback_error} (backup {item['backup_instance']})"
                )
        if rollback_errors:
            raise InstanceInstallError(
                f"Deployment failed: {error}; rollback also failed: {'; '.join(rollback_errors)}"
            ) from error
        raise InstanceInstallError(f"Deployment failed and all touched instances were restored: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description="Back up, deploy and validate the managed bridge DLL.")
    parser.add_argument("--bridge-dll", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--instances", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--instances-root", type=Path, default=INSTANCES_ROOT)
    parser.add_argument("--backup-root", type=Path, default=PROJECT_ROOT / "artifacts" / "bridge-deploy-backups")
    args = parser.parse_args()
    try:
        results = deploy(args.bridge_dll, instance_indices=args.instances,
                         instances_root=args.instances_root, backup_root=args.backup_root)
    except (OSError, ValueError, InstanceInstallError) as error:
        print(f"DLL deployment failed: {error}", file=sys.stderr)
        return 1
    for result in results:
        print(f"hk-ai-{result['index']}: deployed and validated sha256={result['sha256']} backup={result['backup']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
