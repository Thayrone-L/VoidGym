"""Compara convolução CPU/CUDA e registra perfil/plano de recursos."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from voidgym.core.hardware import detect_hardware, plan_resources


def benchmark(device: str, seconds: float = 10.0) -> float:
    tensor = torch.randn(32, 3, 108, 192, device=device)
    kernel = torch.randn(32, 3, 11, 11, device=device)
    torch.nn.functional.conv2d(tensor, kernel, padding=5)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    operations = 0
    while time.perf_counter() - start < seconds:
        torch.nn.functional.conv2d(tensor, kernel, padding=5)
        operations += 1
    if device == "cuda":
        torch.cuda.synchronize()
    return operations / seconds


def main() -> int:
    profile = detect_hardware(Path("runtime"))
    plan = plan_resources(profile)
    print(json.dumps({"hardware": profile.__dict__ if hasattr(profile, "__dict__") else {
        field: getattr(profile, field) for field in profile.__dataclass_fields__},
        "plan": {field: getattr(plan, field) for field in plan.__dataclass_fields__}}, indent=2))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA não está disponível neste ambiente. Use tools/v2/setup_venv_cuda.ps1.")
    cpu = benchmark("cpu")
    cuda = benchmark("cuda")
    print(json.dumps({"cpu_conv_per_s": cpu, "cuda_conv_per_s": cuda, "speedup": cuda / cpu}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
