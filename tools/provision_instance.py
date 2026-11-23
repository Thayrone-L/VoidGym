from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hk_agent.instance_install import (
    INSTANCES_ROOT,
    ORIGINAL_INSTALL,
    InstanceInstallError,
    provision_instance,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Provision an isolated Hollow Knight install")
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--resync", action="store_true")
    parser.add_argument("--sync-mods", action="store_true")
    parser.add_argument("--bridge-dll", type=Path)
    parser.add_argument("--source", type=Path, default=ORIGINAL_INSTALL)
    parser.add_argument("--instances-root", type=Path, default=INSTANCES_ROOT)
    args = parser.parse_args()
    try:
        marker = provision_instance(
            args.index,
            source=args.source,
            instances_root=args.instances_root,
            resync=args.resync,
            sync_mods=args.sync_mods,
            bridge_dll=args.bridge_dll,
        )
    except (InstanceInstallError, OSError) as error:
        print(f"Provisioning failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(marker, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
