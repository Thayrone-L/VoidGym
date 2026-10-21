"""Caminhos de dados e runtime fora do repositório."""

from pathlib import Path

from voidgym.core.config import runtime_root_from_environment


def runtime_root() -> Path:
    """Retorna o runtime configurado ou o padrão apropriado ao sistema."""
    configured = runtime_root_from_environment()
    if configured is not None:
        return configured
    local_app_data = Path.home() / "AppData" / "Local"
    if local_app_data.exists():
        return local_app_data / "VoidGym" / "runtime-v2"
    return Path.home() / ".local" / "share" / "voidgym" / "runtime-v2"
