from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

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
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup_dir = backup_root.expanduser().resolve(strict=False) / f"bridge-{stamp}"
    results: list[dict[str, object]] = []
    for index in instance_indices:
        if index < 1:
            raise InstanceInstallError("Instance indices must be positive")
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
        backup_instance = backup_dir / f"hk-ai-{index}"
        backup_instance.mkdir(parents=True, exist_ok=False)
        backup_dll = backup_instance / target.name
        backup_marker = backup_instance / MARKER_NAME
        shutil.copy2(target, backup_dll)
        shutil.copy2(marker_path, backup_marker)
        try:
            shutil.copy2(bridge, target)
            if sha256_file(target) != build_hash:
                raise InstanceInstallError(f"Instance {index} DLL differs from the selected build after copy")
            marker = refresh_instance_bridge_marker(
                root, expected_index=index, expected_bridge_hash=build_hash
            )
            validated = load_instance_marker(root, expected_index=index)
            if validated.bridge_dll_sha256 != build_hash:
                raise InstanceInstallError(f"Instance {index} marker validation did not retain the build hash")
            results.append({"index": index, "root": str(root), "sha256": build_hash,
                            "backup": str(backup_instance), "marker": marker})
        except Exception:
            shutil.copy2(backup_dll, target)
            shutil.copy2(backup_marker, marker_path)
            raise
    return results


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
