"""Detecção de hardware e plano reproduzível de recursos."""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HardwareProfile:
    physical_cores: int | None
    logical_cores: int | None
    ram_total_gb: float | None
    ram_available_gb: float | None
    gpu_name: str | None
    gpu_memory_total_gb: float | None
    gpu_memory_free_gb: float | None
    compute_capability: str | None
    disk_free_gb: float | None
    operating_system: str


@dataclass(frozen=True)
class ResourcePlan:
    train_device: str
    actor_device: str
    game_instances: int
    gym_envs: int
    torch_threads_learner: int
    torch_threads_actor: int
    batch_size: int
    amp: bool
    storage_budget_gb: float
    warnings: tuple[str, ...]


def _psutil_profile() -> tuple[int | None, float | None, float | None]:
    try:
        import psutil  # type: ignore[import-untyped]

        memory = psutil.virtual_memory()
        return psutil.cpu_count(logical=False), memory.total / 2**30, memory.available / 2**30
    except (ImportError, OSError, AttributeError):
        return None, None, None


def _gpu_profile() -> tuple[str | None, float | None, float | None, str | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None, None, None
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        capability = f"{props.major}.{props.minor}"
        return props.name, total_bytes / 2**30, free_bytes / 2**30, capability
    except (ImportError, RuntimeError, OSError, AttributeError):
        return None, None, None, None


def detect_hardware(runtime: Path | None = None) -> HardwareProfile:
    """Faz leituras best-effort; uma falha fica como desconhecido."""
    physical, ram_total, ram_free = _psutil_profile()
    gpu_name, gpu_total, gpu_free, capability = _gpu_profile()
    try:
        disk_free = shutil.disk_usage(runtime or Path.home()).free / 2**30
    except OSError:
        disk_free = None
    return HardwareProfile(
        physical_cores=physical,
        logical_cores=os.cpu_count(),
        ram_total_gb=ram_total,
        ram_available_gb=ram_free,
        gpu_name=gpu_name,
        gpu_memory_total_gb=gpu_total,
        gpu_memory_free_gb=gpu_free,
        compute_capability=capability,
        disk_free_gb=disk_free,
        operating_system=platform.system(),
    )


def plan_resources(
    profile: HardwareProfile,
    kind: str,
    overrides: Any = None,
    *,
    configured_storage_gb: float = 100.0,
) -> ResourcePlan:
    """Planeja recursos; sobrescritas validadas substituem os padrões."""
    physical = profile.physical_cores or profile.logical_cores or 2
    logical = profile.logical_cores or physical
    ram = profile.ram_total_gb or 8.0
    gpu = profile.gpu_name is not None and (profile.gpu_memory_free_gb or 0) >= 6
    instances = max(1, min(3, int((ram - 4) // 4), (physical - 2) // 2))
    envs = max(2, logical - 2 - 2 * instances)
    device = "cuda" if gpu else "cpu"
    threads = 2 if gpu else max(1, physical - 2 * instances - 1)
    if kind == "pretrain_platform":
        # This workload has no live game processes to reserve CPU for.
        instances = 0
        envs = max(2, logical - 2)
        threads = 2 if gpu else max(1, physical - 1)
    batch = 2048 if gpu and (profile.gpu_memory_total_gb or 0) >= 8 else 1024 if gpu else 512
    disk_budget = configured_storage_gb
    if profile.disk_free_gb is not None:
        disk_budget = min(configured_storage_gb, 0.8 * profile.disk_free_gb)
    warnings = _resource_warnings(profile, gpu)
    data: dict[str, Any] = {
        "game_instances": instances,
        "gym_envs": envs,
        "train_device": device,
        "torch_threads_learner": threads,
        "batch_size": batch,
        "storage_budget_gb": disk_budget,
    }
    data = _apply_overrides(data, overrides, instances, logical, disk_budget, gpu)
    return ResourcePlan(
        train_device=data["train_device"],
        actor_device="cpu",
        game_instances=int(data["game_instances"]),
        gym_envs=int(data["gym_envs"]),
        torch_threads_learner=int(data["torch_threads_learner"]),
        torch_threads_actor=1,
        batch_size=int(data["batch_size"]),
        amp=False,
        storage_budget_gb=float(data["storage_budget_gb"]),
        warnings=tuple(warnings),
    )


def _resource_warnings(profile: HardwareProfile, gpu: bool) -> list[str]:
    warnings = []
    if (
        profile.ram_total_gb
        and profile.ram_available_gb is not None
        and profile.ram_available_gb < 0.15 * profile.ram_total_gb
    ):
        warnings.append("Mais de 85% da RAM está em uso")
    if profile.disk_free_gb is not None and profile.disk_free_gb < 20:
        warnings.append("Menos de 20 GB livres no disco do runtime")
    if not gpu:
        warnings.append("Sem GPU com pelo menos 6 GB livres; treino usará CPU")
    return warnings


def _apply_overrides(
    values: dict[str, Any],
    overrides: Any,
    max_instances: int,
    logical_cores: int,
    disk_budget: float,
    gpu_available: bool,
) -> dict[str, Any]:
    for key in values:
        value = getattr(overrides, key, None) if overrides is not None else None
        if value is None:
            continue
        _validate_override(key, value, max_instances, logical_cores, disk_budget, gpu_available)
        values[key] = value
    return values


def _validate_override(
    key: str,
    value: Any,
    max_instances: int,
    logical_cores: int,
    disk_budget: float,
    gpu_available: bool,
) -> None:
    if key == "train_device":
        if value not in {"cpu", "cuda"} or (value == "cuda" and not gpu_available):
            raise ValueError("resources.train_device precisa de cpu ou de uma GPU disponível")
        return
    if value <= 0:
        raise ValueError(f"resources.{key} deve ser positivo")
    limits = {
        "game_instances": max_instances,
        "gym_envs": max(2, logical_cores - 2),
        "torch_threads_learner": logical_cores,
        "batch_size": 1_000_000,
        "storage_budget_gb": disk_budget,
    }
    if value > limits[key]:
        raise ValueError(f"resources.{key} excede o limite detectado ({limits[key]})")


def write_hardware_plan(profile: HardwareProfile, plan: ResourcePlan, root: Path) -> None:
    """Salva o perfil e o plano em JSON para auditoria de campanha."""
    import json

    root.mkdir(parents=True, exist_ok=True)
    (root / "hardware.json").write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")
    (root / "resource_plan.json").write_text(json.dumps(asdict(plan), indent=2), encoding="utf-8")
