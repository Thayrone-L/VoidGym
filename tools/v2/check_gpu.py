"""Exibe hardware e compara convoluções curtas na GPU e na CPU."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from dataclasses import asdict

import torch

from voidgym.core.hardware import detect_hardware, plan_resources
from voidgym.core.paths import runtime_root


def _benchmark(device: torch.device, seconds: float) -> float:
    tensor = torch.randn(8, 16, 128, 128, device=device)
    kernel = torch.nn.Conv2d(16, 32, 3, padding=1).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    operations = 0
    while time.perf_counter() - started < seconds:
        kernel(tensor)
        operations += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return operations / max(time.perf_counter() - started, 1e-9)


def main() -> int:
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    profile = detect_hardware(runtime_root())
    plan = plan_resources(profile, "pretrain_platform")
    print(json.dumps({"profile": asdict(profile), "plan": asdict(plan)}, indent=2))
    cpu_rate = _benchmark(torch.device("cpu"), args.seconds)
    result = {"cpu_conv_per_second": cpu_rate, "cuda_conv_per_second": None, "speedup": None}
    if torch.cuda.is_available():
        cuda_rate = _benchmark(torch.device("cuda"), args.seconds)
        result.update(
            cuda_conv_per_second=cuda_rate,
            speedup=cuda_rate / max(cpu_rate, 1e-9),
            vram_peak_gb=torch.cuda.max_memory_allocated() / 2**30,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
