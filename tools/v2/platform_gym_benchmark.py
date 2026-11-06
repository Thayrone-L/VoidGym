"""Low-priority platform gym throughput benchmark, without starting game training."""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
from pathlib import Path
import sys
from time import perf_counter

import psutil  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from voidgym.gyms.platform_gym.generator import generate_room
from voidgym.gyms.platform_gym.labels import labels
from voidgym.gyms.platform_gym.physics import step_physics
from voidgym.gyms.platform_gym.render import render_room
from voidgym.gyms.platform_gym.world import Player


def lower_priority() -> None:
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)


def run_frames(frames: int) -> int:
    lower_priority()
    room = generate_room(3, 91)
    player = Player(*room.spawn, on_ground=True)
    for i in range(frames):
        step_physics(room, player, {"right": i % 100 < 70, "jump": i % 100 in range(30, 45)}, .02)
        render_room(room, player)
        labels(room, player)
    return frames


def measure_single(frames: int) -> float:
    lower_priority()
    t0 = perf_counter()
    run_frames(frames)
    return frames / (perf_counter() - t0)


def main() -> None:
    lower_priority()
    frames = int(os.environ.get("VOIDGYM_BENCH_FRAMES", "3000"))
    cpu_count = os.cpu_count() or 1
    workers = max(1, cpu_count - 2)
    v1 = []
    for proc in psutil.process_iter(("name", "cpu_percent")):
        if "hollow" in (proc.info.get("name") or "").lower():
            v1.append({"name": proc.info["name"], "cpu_percent": proc.cpu_percent(interval=0.1)})
    print(f"priority=BELOW_NORMAL frames={frames} workers={workers} v1_processes={v1 or 'not observed'}")
    single = measure_single(frames)
    print(f"single_fps={single:.1f}")
    per_worker = frames
    start = perf_counter()
    with mp.get_context("spawn").Pool(workers) as pool:
        done = sum(pool.map(run_frames, [per_worker] * workers))
    elapsed = perf_counter() - start
    print(f"multi_workers={workers} total_fps={done / elapsed:.1f} per_core_fps={done / elapsed / workers:.1f}")


if __name__ == "__main__":
    main()
