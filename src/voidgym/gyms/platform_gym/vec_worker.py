"""Lightweight platform environment factory; deliberately does not import PyTorch."""

from __future__ import annotations

import faulthandler
import json
import os
import sys
from pathlib import Path
from typing import Any

from voidgym.envs.adapter_env import AdapterEnv
from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter
from voidgym.gyms.platform_gym.reward import CombinedPlatformReward

_FAULT_HANDLE: Any = None


def worker_memory() -> dict[str, int | None]:
    """Return process RSS and private resident bytes when psutil is available."""
    try:
        import psutil  # type: ignore[import-untyped]

        memory = psutil.Process().memory_info()
        full = psutil.Process().memory_full_info()
        return {"rss_bytes": int(memory.rss), "private_bytes": int(full.uss)}
    except (ImportError, OSError, AttributeError):
        return {"rss_bytes": None, "private_bytes": None}


def worker_startup_record(worker_index: int | str) -> dict[str, Any]:
    return {
        "worker_index": worker_index,
        "pid": os.getpid(),
        "executable": sys.executable,
        "torch_imported": "torch" in sys.modules,
        **worker_memory(),
    }


def make_platform_env(*, level: int, seed: int, worker_index: int | str = 0,
                      diagnostics_dir: Path | None = None) -> AdapterEnv:
    global _FAULT_HANDLE
    if diagnostics_dir is not None:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        _FAULT_HANDLE = (diagnostics_dir / f"faulthandler-worker-{worker_index}.log").open(
            "a", encoding="utf-8",
        )
        faulthandler.enable(file=_FAULT_HANDLE, all_threads=True)
    env = AdapterEnv(PlatformGymAdapter(level=level, seed=seed), CombinedPlatformReward())
    if diagnostics_dir is not None:
        record = worker_startup_record(worker_index)
        path = diagnostics_dir / f"worker-startup-{worker_index}.json"
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return env
