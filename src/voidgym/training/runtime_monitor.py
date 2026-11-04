"""Small, failure-tolerant resource snapshots for long-running v2 roles."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def resource_snapshot(device: str | None = None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"updated_unix_ns": time.time_ns()}
    try:
        import psutil  # type: ignore[import-untyped]

        memory = psutil.virtual_memory()
        snapshot["ram_available_bytes"] = int(memory.available)
        snapshot["ram_total_bytes"] = int(memory.total)
    except Exception:  # noqa: BLE001 - missing telemetry must not hide the primary failure
        snapshot["ram_available_bytes"] = None
        snapshot["ram_total_bytes"] = None
    try:
        from voidgym.core.hardware import memory_commit_stats

        snapshot.update(memory_commit_stats())
    except Exception:  # noqa: BLE001
        snapshot.update({"commit_available_bytes": None, "commit_limit_bytes": None})
    try:
        import torch

        if device and device.startswith("cuda") and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(device)
            snapshot["vram_free_bytes"] = int(free)
            snapshot["vram_total_bytes"] = int(total)
            snapshot["vram_allocated_bytes"] = int(torch.cuda.memory_allocated(device))
            snapshot["vram_reserved_bytes"] = int(torch.cuda.memory_reserved(device))
        else:
            snapshot.update({"vram_free_bytes": None, "vram_total_bytes": None,
                             "vram_allocated_bytes": None, "vram_reserved_bytes": None})
    except Exception:  # noqa: BLE001
        snapshot.update({"vram_free_bytes": None, "vram_total_bytes": None,
                         "vram_allocated_bytes": None, "vram_reserved_bytes": None})
    return snapshot


def publish_role_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
