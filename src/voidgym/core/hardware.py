"""Detecção tolerante do hardware e planejamento inicial de recursos."""

import ctypes
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
    commit_used_gb: float | None = None
    commit_limit_gb: float | None = None
    commit_available_gb: float | None = None
    paging_file_total_gb: float | None = None
    worker_private_gb: float | None = None


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
    commit_headroom_gb: float | None = None


def _optional_hardware() -> tuple[int | None, float | None, float | None]:
    try:
        import psutil  # type: ignore[import-untyped]
        return psutil.cpu_count(logical=False), psutil.virtual_memory().total / 2**30, psutil.virtual_memory().available / 2**30
    except Exception:  # noqa: BLE001 - falha de hardware vira valor desconhecido
        return None, None, None


def _memory_commit() -> tuple[float | None, float | None, float | None, float | None]:
    """Return committed, commit limit, available commit, and pagefile GiB."""
    try:
        import psutil
        pagefile_total = psutil.swap_memory().total / 2**30
        if os.name == "nt":
            class PerformanceInformation(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("CommitTotal", ctypes.c_size_t),
                    ("CommitLimit", ctypes.c_size_t),
                    ("CommitPeak", ctypes.c_size_t),
                    ("PhysicalTotal", ctypes.c_size_t),
                    ("PhysicalAvailable", ctypes.c_size_t),
                    ("SystemCache", ctypes.c_size_t),
                    ("KernelTotal", ctypes.c_size_t),
                    ("KernelPaged", ctypes.c_size_t),
                    ("KernelNonpaged", ctypes.c_size_t),
                    ("PageSize", ctypes.c_ulong),
                    ("HandleCount", ctypes.c_ulong),
                    ("ProcessCount", ctypes.c_ulong),
                    ("ThreadCount", ctypes.c_ulong),
                ]

            info = PerformanceInformation()
            info.cb = ctypes.sizeof(info)
            if ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
                used = info.CommitTotal * info.PageSize / 2**30
                limit = info.CommitLimit * info.PageSize / 2**30
                return used, limit, max(0.0, limit - used), pagefile_total

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError("GetPerformanceInfo and GlobalMemoryStatusEx failed")
            used = (status.ullTotalPageFile - status.ullAvailPageFile) / 2**30
            limit = status.ullTotalPageFile / 2**30
            return used, limit, status.ullAvailPageFile / 2**30, pagefile_total

        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        limit = (memory.total + swap.total) / 2**30
        used = (memory.total - memory.available + swap.used) / 2**30
        return used, limit, max(0.0, limit - used), pagefile_total
    except Exception:  # noqa: BLE001 - diagnostics must degrade gracefully
        return None, None, None, None


def memory_commit_stats() -> dict[str, int | None]:
    """Return Windows commit and pagefile counters as bytes for live telemetry."""
    used, limit, available, pagefile = _memory_commit()
    return {
        "commit_used_bytes": int(used * 2**30) if used is not None else None,
        "commit_limit_bytes": int(limit * 2**30) if limit is not None else None,
        "commit_available_bytes": int(available * 2**30) if available is not None else None,
        "paging_file_total_bytes": int(pagefile * 2**30) if pagefile is not None else None,
    }


def detect_hardware(runtime_root: Path = Path("runtime")) -> HardwareProfile:
    physical, total_ram, free_ram = _optional_hardware()
    commit_used, commit_limit, commit_available, pagefile_total = _memory_commit()
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
    return HardwareProfile(
        physical, os.cpu_count(), total_ram, free_ram, gpu_name, gpu_total, gpu_free,
        capability, disk, platform.platform(), tuple(detection_warnings),
        commit_used, commit_limit, commit_available, pagefile_total,
    )


def _plan_environment_count(profile: HardwareProfile, requested: int, *,
                            extra_learner_reserve_gb: float = 0.0,
                            worker_safety_factor: float = 1.25
                            ) -> tuple[int, float | None, tuple[str, ...]]:
    available = profile.commit_available_gb
    worker_memory = profile.worker_private_gb
    if available is None or worker_memory is None or worker_memory <= 0:
        return requested, available, ()
    # Keep 3 GiB available, reserve 1 GiB for learner growth, and account for
    # the pretrainer's measured learner footprint separately when requested.
    worker_budget = max(0.0, available - 4.0 - extra_learner_reserve_gb)
    worker_cost = worker_memory * worker_safety_factor
    safe_count = int(worker_budget // worker_cost)
    if safe_count < 1:
        raise ValueError("commit disponível insuficiente para manter 3 GiB de folga")
    chosen = min(requested, safe_count)
    warnings = ((f"Ambientes reduzidos de {requested} para {chosen} para manter folga de commit.",)
                if requested > safe_count else ())
    headroom = (available - chosen * worker_cost - extra_learner_reserve_gb)
    return chosen, headroom, warnings


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
    requested_envs = int(overrides.get("gym_envs", gym_envs))
    learner_reserve = 5.5 if kind == "pretrain_platform" else 0.0
    worker_factor = 3.0 if kind == "pretrain_platform" else 1.25
    gym_envs, commit_headroom, commit_warnings = _plan_environment_count(
        profile, requested_envs, extra_learner_reserve_gb=learner_reserve,
        worker_safety_factor=worker_factor,
    )
    threads = 2 if device == "cuda" else max(1, cores - 2 * instances - 1)
    batch = 2048 if profile.gpu_memory_total_gb and profile.gpu_memory_total_gb >= 8 else 1024 if gpu else 512
    warnings = list(profile.detection_warnings) + list(commit_warnings)
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
                        max(0.1, budget), tuple(warnings), commit_headroom)


def save_resource_profile(profile: HardwareProfile, plan: ResourcePlan, campaign_root: Path) -> None:
    campaign_root.mkdir(parents=True, exist_ok=True)
    for name, item in (("hardware.json", profile), ("resource_plan.json", plan)):
        (campaign_root / name).write_text(json.dumps(asdict(item), indent=2), encoding="utf-8")
