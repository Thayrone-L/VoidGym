"""Detecção tolerante do hardware e planejamento inicial de recursos."""

import json
import os
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from voidgym.core.errors import VoidGymValueError as ValueError


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    physical_cores: int | None
    logical_cores: int | None
    memory_total_gb: float | None
    memory_free_gb: float | None
    gpu_name: str | None
    gpu_memory_total_gb: float | None
    gpu_memory_free_gb: float | None
    compute_capability: str | None
    disk_free_gb: float | None
    operating_system: str
    detection_warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
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


def _optional_hardware() -> tuple[int | None, float | None, float | None]:
    try:
        import psutil  # type: ignore[import-untyped]
        return psutil.cpu_count(logical=False), psutil.virtual_memory().total / 2**30, psutil.virtual_memory().available / 2**30
    except Exception:  # noqa: BLE001 - falha de hardware vira valor desconhecido
        return None, None, None


def detect_hardware(runtime_root: Path = Path("runtime")) -> HardwareProfile:
    physical, total_ram, free_ram = _optional_hardware()
    detection_warnings: list[str] = []
    gpu_name = None
    gpu_total = gpu_free = None
    capability = None
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            gpu_name = props.name
            gpu_total = props.total_memory / 2**30
            gpu_free = torch.cuda.mem_get_info(device)[0] / 2**30
            capability = f"{props.major}.{props.minor}"
    except Exception as error:  # noqa: BLE001 - hardware probing must degrade to unknown
        detection_warnings.append(f"GPU desconhecida: {type(error).__name__}.")
    try:
        disk = shutil.disk_usage(runtime_root.resolve()).free / 2**30
    except OSError:
        disk = None
    return HardwareProfile(physical, os.cpu_count(), total_ram, free_ram, gpu_name,
                           gpu_total, gpu_free, capability, disk, platform.platform(),
                           tuple(detection_warnings))


def plan_resources(profile: HardwareProfile, kind: str = "game_training",
                   overrides: dict[str, Any] | None = None) -> ResourcePlan:
    overrides = overrides or {}
    cores = profile.physical_cores or 1
    logical = profile.logical_cores or cores
    ram = profile.memory_total_gb or 4.0
    gpu = profile.gpu_memory_free_gb or 0.0
    device = "cuda" if gpu >= 6 else "cpu"
    v2_capacity = max(1, min(3, int((ram - 4) // 4), int((cores - 2) // 2)))
    v1_recommendation = max(1, cores // 4)
    instances = min(v2_capacity, v1_recommendation)
    gym_envs = max(2, logical - 2 - 2 * instances)
    threads = 2 if device == "cuda" else max(1, cores - 2 * instances - 1)
    batch = 2048 if profile.gpu_memory_total_gb and profile.gpu_memory_total_gb >= 8 else 1024 if gpu else 512
    warnings: list[str] = list(profile.detection_warnings)
    if profile.memory_total_gb and profile.memory_free_gb and profile.memory_free_gb < profile.memory_total_gb * 0.15:
        warnings.append("Mais de 85% da RAM está em uso.")
    if profile.disk_free_gb is not None and profile.disk_free_gb < 20:
        warnings.append("Menos de 20 GB livres no disco.")
    if not gpu:
        warnings.append("GPU CUDA indisponível; learner usará CPU.")
    max_instances = min(v2_capacity, v1_recommendation)
    if "game_instances" in overrides:
        requested = int(overrides["game_instances"])
        if requested < 1:
            raise ValueError("game_instances deve ser positivo")
        if requested > max_instances:
            warnings.append(f"Pedido de {requested} instâncias excede o plano recomendado ({max_instances}).")
        instances = requested
    gym_envs = int(overrides.get("gym_envs", gym_envs))
    if gym_envs < 1:
        raise ValueError("gym_envs deve ser positivo")
    device = str(overrides.get("train_device", device))
    if device not in {"cpu", "cuda"}:
        raise ValueError("train_device deve ser cpu ou cuda")
    budget = float(overrides.get("storage_budget_gb", 100.0))
    if budget <= 0:
        raise ValueError("storage_budget_gb deve ser positivo")
    if profile.disk_free_gb is not None:
        budget = min(budget, 0.8 * profile.disk_free_gb)
    return ResourcePlan(device, "cpu", instances, gym_envs, threads, 1, batch, False,
                        max(0.1, budget), tuple(warnings))


def save_resource_profile(profile: HardwareProfile, plan: ResourcePlan, campaign_root: Path) -> None:
    campaign_root.mkdir(parents=True, exist_ok=True)
    for name, item in (("hardware.json", profile), ("resource_plan.json", plan)):
        (campaign_root / name).write_text(json.dumps(asdict(item), indent=2), encoding="utf-8")
