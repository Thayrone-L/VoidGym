"""Small, failure-tolerant resource snapshots for long-running v2 roles."""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any


def _process_role(pid: int, learner_pid: int, supervisor_pid: int | None,
                  executable: str, command: str) -> str | None:
    command_lower = command.casefold()
    executable_lower = executable.casefold().replace("/", "\\")
    if "run_actor.py" in command_lower:
        return "actor"
    if "run_learner.py" in command_lower:
        return "learner"
    if "supervise_hk_learner.py" in command_lower:
        return "supervisor"
    if pid == supervisor_pid:
        return "supervisor"
    if pid == learner_pid:
        return "learner"
    if executable_lower.endswith("\\hollow knight.exe"):
        return "game"
    return None


def _process_resource_record(process: Any, role: str, psutil: Any) -> dict[str, Any]:
    record: dict[str, Any] = {"pid": int(process.pid), "role": role}
    try:
        image_path = process.exe()
    except (psutil.Error, OSError):
        image_path = None
    record["image_path"] = image_path
    try:
        command = process.cmdline()
        record["executable"] = command[0] if command else image_path
    except (psutil.Error, OSError):
        record["executable"] = image_path
    try:
        memory = process.memory_info()
        full = process.memory_full_info()
        record["rss_bytes"] = int(memory.rss)
        record["private_bytes"] = int(getattr(full, "uss", memory.rss))
    except (psutil.Error, OSError):
        record["rss_bytes"] = None
        record["private_bytes"] = None
    try:
        module_paths = [mapping.path.casefold() for mapping in process.memory_maps()]
    except (psutil.Error, OSError):
        module_paths = None
    if process.pid == os.getpid():
        record["torch_loaded"] = any(name == "torch" or name.startswith("torch.")
                                      for name in sys.modules)
    else:
        record["torch_loaded"] = (
            any("torch" in path for path in module_paths) if module_paths is not None else None
        )
    cuda_markers = ("cublas", "cudnn", "cudart", "nvrtc", "nvcuda", "torch_cuda")
    record["cuda_loaded"] = (
        any(marker in path for marker in cuda_markers for path in module_paths)
        if module_paths is not None else None
    )
    return record


def _base_processes(psutil: Any) -> tuple[int, int | None, list[Any]]:
    try:
        learner = psutil.Process(os.getpid())
        supervisor = learner.parent()
        owner = supervisor.parent() if supervisor is not None else None
        supervisor_pid = supervisor.pid if supervisor is not None else None
    except (psutil.Error, OSError):
        learner = None
        supervisor = None
        owner = None
        supervisor_pid = None
    processes: list[Any] = []
    if learner is not None:
        try:
            processes = [learner]
            if supervisor is not None:
                processes.append(supervisor)
            if owner is not None:
                processes.extend(owner.children(recursive=True))
        except (psutil.Error, OSError):
            pass
    learner_pid = learner.pid if learner is not None else os.getpid()
    return learner_pid, supervisor_pid, processes


def _shared_processes(psutil: Any, shared_root: Path | None) -> list[Any]:
    if shared_root is None:
        return []
    # Actors are launched by the control plane as siblings of the learner
    # supervisor, so walking only the learner's ancestry misses them.
    marker = str(shared_root.resolve()).casefold().replace("/", "\\")
    try:
        candidates = psutil.process_iter()
    except (psutil.Error, OSError):
        return []
    processes: list[Any] = []
    for candidate in candidates:
        try:
            command = " ".join(candidate.cmdline()).casefold().replace("/", "\\")
            image = candidate.exe().casefold().replace("/", "\\")
            if marker in command:
                processes.append(candidate)
                processes.extend(candidate.children(recursive=True))
            elif (image.endswith("\\hollow knight.exe")
                  and "\\instances\\hk-ai-" in image):
                processes.append(candidate)
        except (psutil.Error, OSError):
            continue
    return processes


def campaign_process_snapshot(shared_root: Path | None = None) -> list[dict[str, Any]]:
    """Describe learner, supervisor, actors and game descendants for the HK heartbeat."""
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError:
        return []
    learner_pid, supervisor_pid, processes = _base_processes(psutil)
    processes.extend(_shared_processes(psutil, shared_root))
    records: dict[int, dict[str, Any]] = {}
    for process in processes:
        try:
            executable = process.exe()
            command = " ".join(process.cmdline())
            role = _process_role(process.pid, learner_pid, supervisor_pid,
                                 executable, command)
            if role is not None:
                records[process.pid] = _process_resource_record(process, role, psutil)
        except (psutil.Error, OSError):
            continue
    return sorted(records.values(), key=lambda item: (str(item["role"]), int(item["pid"])))


def resource_snapshot(device: str | None = None,
                      shared_root: Path | None = None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"updated_unix_ns": time.time_ns()}
    try:
        import psutil

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
    processes = campaign_process_snapshot(shared_root)
    snapshot["processes"] = processes
    snapshot["process_rss_total_bytes"] = sum(
        int(row["rss_bytes"]) for row in processes if isinstance(row.get("rss_bytes"), int)
    )
    snapshot["process_private_total_bytes"] = sum(
        int(row["private_bytes"]) for row in processes
        if isinstance(row.get("private_bytes"), int)
    )
    return snapshot


def publish_role_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        for attempt in range(100):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 99:
                    raise
                time.sleep(0.02)
    finally:
        temporary.unlink(missing_ok=True)
