from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from voidgym.core.hardware import HardwareProfile, ResourcePlan
from voidgym.installer.manifest import ManifestError
from voidgym.installer.state import InstallState
from voidgym.installer.ui import ConsoleUI
from voidgym.installer.windows import WindowsSystem


@dataclass(slots=True)
class InstallContext:
    project_root: Path
    install_root: Path
    manifest: dict[str, Any]
    system: WindowsSystem
    ui: ConsoleUI
    state: InstallState
    logger: Any
    profile: HardwareProfile | None = None
    resource_plan: ResourcePlan | None = None
    game_path: Path | None = None
    instance_count: int = 0
    storage_budget_gb: float = 20.0


class Step(Protocol):
    step_id: str
    title: str

    def check(self, context: InstallContext) -> bool: ...
    def plan(self, context: InstallContext) -> str: ...
    def apply(self, context: InstallContext) -> str: ...
    def undo(self, context: InstallContext) -> None: ...


@dataclass(slots=True)
class ActionStep:
    step_id: str
    title: str
    check_action: Callable[[InstallContext], bool]
    plan_action: Callable[[InstallContext], str]
    apply_action: Callable[[InstallContext], str]
    undo_action: Callable[[InstallContext], None]

    def check(self, context: InstallContext) -> bool:
        return self.check_action(context)

    def plan(self, context: InstallContext) -> str:
        return self.plan_action(context)

    def apply(self, context: InstallContext) -> str:
        return self.apply_action(context)

    def undo(self, context: InstallContext) -> None:
        self.undo_action(context)


def execute_steps(context: InstallContext, steps: list[Step], *, dry_run: bool) -> list[str]:
    results: list[str] = []
    for number, step in enumerate(steps, start=1):
        prefix = f"[{number}/{len(steps)}] {step.title}"
        if dry_run:
            message = step.plan(context)
            context.ui.say(f"{prefix} — plano: {message}")
            context.logger.info("DRY-RUN %s: %s", step.step_id, message)
            results.append(f"plano: {message}")
            continue
        if step.check(context):
            context.ui.say(f"{prefix}… já estava pronto")
            context.logger.info("SKIP %s already ready", step.step_id)
            results.append("já estava pronto")
            context.state.complete_step(step.step_id)
            continue
        try:
            message = step.apply(context)
        except Exception as error:
            context.logger.exception("Step %s failed", step.step_id)
            context.ui.say(f"{prefix}… ERRO: {error}")
            raise RuntimeError(f"Etapa {number} ({step.title}): {error}") from error
        context.logger.info("OK %s: %s", step.step_id, message)
        context.ui.say(f"{prefix}… ok — {message}")
        results.append(message)
        context.state.complete_step(step.step_id)
    return results


def safe_download(url: str, expected_sha256: str, destination: Path, *, timeout: float = 45.0) -> Path:
    """Download to a sibling temp file, verify the pinned digest, then atomically publish."""
    if not url.startswith("https://"):
        raise ManifestError("Downloads exigem URL HTTPS fixa no manifesto.")
    if len(expected_sha256) != 64 or any(char not in "0123456789abcdefABCDEF" for char in expected_sha256):
        raise ManifestError("O download não tem SHA-256 fixo no manifesto.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".part", dir=destination.parent)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "VoidGym-Installer/0.1"})
        with os.fdopen(fd, "wb") as output, urllib.request.urlopen(request, timeout=timeout) as response:
            digest = hashlib.sha256()
            while block := response.read(1024 * 1024):
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest().casefold() != expected_sha256.casefold():
            raise ManifestError("SHA-256 divergente; download removido sem instalar.")
        os.replace(temporary, destination)
        return destination
    finally:
        Path(temporary).unlink(missing_ok=True)


def describe_resources(profile: HardwareProfile, plan: ResourcePlan) -> str:
    gpu = profile.gpu_name or "sem GPU CUDA detectada"
    ram = f"{profile.memory_total_gb:.1f} GB RAM" if profile.memory_total_gb is not None else "RAM desconhecida"
    return (f"{profile.logical_cores or 'núcleos?'} threads lógicos, {ram}, {gpu}; "
            f"recomendação de {plan.game_instances} instância(s), learner {plan.train_device}.")


def remove_owned_paths(context: InstallContext) -> list[str]:
    from voidgym.installer.state import safe_uninstall_path

    removed: list[str] = []
    app_root = context.install_root
    local_low = context.system.local_low
    for raw in reversed(context.state.data["created_paths"]):
        path = Path(raw)
        if not safe_uninstall_path(path, app_root=app_root, local_low=local_low):
            raise RuntimeError(f"Estado contém caminho fora das raízes permitidas; nada foi removido: {path}")
        if path.is_dir():
            import shutil
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
        removed.append(str(path))
    return removed
