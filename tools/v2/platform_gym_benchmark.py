"""Benchmark abaixo da prioridade normal para não interferir em treino externo."""

from concurrent.futures import ProcessPoolExecutor
import ctypes
import json
import multiprocessing
import os
import time


from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter


def _lower_priority() -> None:
    if os.name == "nt":
        from ctypes import wintypes
        ctypes.windll.kernel32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        ctypes.windll.kernel32.SetPriorityClass.restype = wintypes.BOOL
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)


def _worker(steps: int) -> int:
    _lower_priority()
    adapter = PlatformGymAdapter()
    right = adapter.action_table().index("RIGHT")
    for _ in range(steps):
        adapter.apply(right)
        adapter.frame_rgb()
        adapter.training_labels()
    return steps


def _process_rate(steps: int, workers: int) -> float:
    started = time.perf_counter()
    tasks = [steps // workers + int(index < steps % workers) for index in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        total = sum(pool.map(_worker, tasks))
    return total / (time.perf_counter() - started)


def main() -> int:
    _lower_priority()
    profile = multiprocessing.cpu_count()
    workers = max(1, profile - 2)
    steps = int(os.environ.get("VOIDGYM_BENCH_STEPS", "400"))
    started = time.perf_counter()
    _worker(steps)
    one_rate = steps / max(1e-9, time.perf_counter() - started)
    parallel_rate = _process_rate(steps, workers)
    print(json.dumps({"one_process_steps_per_second": one_rate, "workers": workers,
                      "parallel_steps_per_second": parallel_rate, "steps": steps,
                      "v1_cpu_percent": _v1_cpu_usage()}, indent=2))
    return 0


def _v1_cpu_usage() -> float | None:
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        total = 0.0
        found = False
        for process in psutil.process_iter(["name"]):
            name = str(process.info.get("name", "")).lower()
            if "hollow knight" in name or name in {"hollow_knight.exe", "hk.exe"}:
                found = True
                total += process.cpu_percent(interval=None)
        return total if found else None
    except (OSError, psutil.Error):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
