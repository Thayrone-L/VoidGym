"""Collect read-only process, commit, and GPU memory diagnostics as JSONL."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from voidgym.core.hardware import memory_commit_stats  # type: ignore[import-untyped]


def summarize_process_memory(
    processes: list[dict[str, Any]], committed_bytes: int | None,
) -> dict[str, Any]:
    """Summarize a complete visible-process snapshot without labeling residuals."""
    private_total = sum(
        int(row["private_bytes"])
        for row in processes
        if isinstance(row.get("private_bytes"), (int, float))
    )
    return {
        "process_count": len(processes),
        "private_total_bytes": private_total,
        "committed_minus_process_private_bytes": (
            committed_bytes - private_total if committed_bytes is not None else None
        ),
        "processes_top_private": sorted(
            processes,
            key=lambda row: int(row.get("private_bytes") or 0),
            reverse=True,
        )[:10],
    }


def _windows_snapshot() -> tuple[list[dict[str, Any]], dict[str, Any] | None, str | None]:
    executable = shutil.which("powershell.exe") or shutil.which("pwsh")
    if executable is None:
        return [], None, "PowerShell unavailable"
    script = r"""
$ErrorActionPreference = 'Stop'
$processes = @(Get-Process -ErrorAction SilentlyContinue | ForEach-Object {
  [pscustomobject]@{
    name = $_.ProcessName
    pid = $_.Id
    private_bytes = $_.PrivateMemorySize64
    working_set_bytes = $_.WorkingSet64
  }
})
$memory = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory |
  Select-Object CommittedBytes, CommitLimit, PoolNonpagedBytes, PoolPagedBytes,
    CacheBytes, ModifiedPageListBytes
[pscustomobject]@{ processes = $processes; system = $memory } |
  ConvertTo-Json -Depth 4 -Compress
"""
    try:
        result = subprocess.run(
            [executable, "-NoProfile", "-NonInteractive", "-Command", script],
            check=False, capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], None, f"PowerShell snapshot failed: {exc}"
    if result.returncode != 0:
        detail = result.stderr.strip() or f"PowerShell exit {result.returncode}"
        return [], None, detail
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return [], None, f"PowerShell returned invalid JSON: {exc}"
    rows = payload.get("processes", [])
    if isinstance(rows, dict):
        rows = [rows]
    system = payload.get("system")
    return list(rows) if isinstance(rows, list) else [], system, None


def _portable_processes() -> tuple[list[dict[str, Any]], str | None]:
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError as exc:
        return [], f"psutil unavailable: {exc}"
    rows: list[dict[str, Any]] = []
    for process in psutil.process_iter(("pid", "name")):
        try:
            info = process.info
            rows.append({
                "name": info.get("name") or "",
                "pid": int(info["pid"]),
                "private_bytes": int(process.memory_full_info().uss),
                "working_set_bytes": int(process.memory_info().rss),
            })
        except (psutil.Error, OSError, KeyError, TypeError):
            continue
    return rows, None


def _gpu_snapshot() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi unavailable"}
    commands = {
        "memory": ["--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
        "compute_apps": ["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
    }
    results: dict[str, Any] = {"available": True}
    for name, arguments in commands.items():
        try:
            result = subprocess.run(
                [executable, *arguments], check=False, capture_output=True,
                text=True, timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            results[name] = {"error": str(exc)}
            continue
        results[name] = {
            "output": result.stdout.strip(),
            "error": result.stderr.strip() or None,
            "returncode": result.returncode,
        }
    return results


def _torch_process_memory(torch_module: Any = None) -> dict[str, Any]:
    """Read this process's CUDA allocator counters without importing PyTorch."""
    if torch_module is None:
        torch_module = sys.modules.get("torch")
    if torch_module is None:
        return {"available": False, "reason": "torch is not loaded in this process"}
    try:
        if not torch_module.cuda.is_available():
            return {"available": True, "cuda_available": False}
        device = torch_module.cuda.current_device()
        return {
            "available": True,
            "cuda_available": True,
            "device": int(device),
            "allocated_bytes": int(torch_module.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch_module.cuda.memory_reserved(device)),
        }
    except Exception as exc:  # noqa: BLE001 - telemetry should degrade gracefully
        return {"available": False, "error": str(exc)}


def collect_snapshot() -> dict[str, Any]:
    """Read host memory, visible processes, OS counters, and GPU usage."""
    import psutil

    process_error: str | None = None
    windows_system: dict[str, Any] | None = None
    if os.name == "nt":
        processes, windows_system, process_error = _windows_snapshot()
        source = "Get-Process"
        if not processes:
            processes, process_error = _portable_processes()
            source = "psutil fallback"
    else:
        processes, process_error = _portable_processes()
        source = "psutil"

    commits = memory_commit_stats()
    if windows_system is not None:
        # The project's helper can degrade to unknown on hosts where psapi
        # access is restricted; PerfOS counters remain a direct Windows source.
        perf_used = windows_system.get("CommittedBytes")
        perf_limit = windows_system.get("CommitLimit")
        if isinstance(perf_used, (int, float)):
            commits["commit_used_bytes"] = int(perf_used)
        if isinstance(perf_limit, (int, float)):
            commits["commit_limit_bytes"] = int(perf_limit)
        if isinstance(perf_used, (int, float)) and isinstance(perf_limit, (int, float)):
            commits["commit_available_bytes"] = int(perf_limit - perf_used)
    committed = commits.get("commit_used_bytes")
    summary = summarize_process_memory(
        processes, int(committed) if isinstance(committed, (int, float)) else None,
    )
    virtual_memory: dict[str, Any]
    try:
        counters = psutil.virtual_memory()
        virtual_memory = {
            "total_bytes": int(counters.total),
            "available_bytes": int(counters.available),
            "used_bytes": int(counters.used),
            "percent": float(counters.percent),
        }
    except (OSError, AttributeError) as exc:
        virtual_memory = {"error": str(exc)}
    return {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "process_source": source,
        "process_error": process_error,
        **summary,
        "system_memory": virtual_memory,
        "commit": commits,
        "windows_counters": windows_system,
        "gpu": _gpu_snapshot(),
        "torch_process_memory": _torch_process_memory(),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="append snapshots to this JSONL file")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--interval-seconds", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.samples < 1 or args.interval_seconds < 0:
        parser.error("samples must be positive and interval-seconds cannot be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output = args.output
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
    for index in range(args.samples):
        row = collect_snapshot()
        serialized = json.dumps(row, sort_keys=True)
        if output is None:
            print(serialized, flush=True)
        else:
            with output.open("a", encoding="utf-8") as stream:
                stream.write(serialized + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            print(serialized, flush=True)
        if index + 1 < args.samples:
            time.sleep(args.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
