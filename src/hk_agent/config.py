from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a local configuration violates a safety invariant."""


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    game_profile: str
    game_executable: Path
    personal_save_root: Path
    experimental_save_root: Path
    backup_root: Path
    training_enabled: bool
    parallel_enabled: bool


def _section(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing [{name}] section")
    return value


def _required_string(section: dict[str, Any], section_name: str, key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{section_name}.{key} must be a non-empty string")
    return value.strip()


def _required_bool(section: dict[str, Any], section_name: str, key: str) -> bool:
    value = section.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"{section_name}.{key} must be true or false")
    return value


def _normalized(path_value: str) -> Path:
    return Path(path_value).expanduser().resolve(strict=False)


def _is_nested(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return candidate != parent


def load_config(path: Path, *, allow_training: bool = False) -> ProjectConfig:
    if not path.is_file():
        raise ConfigError(f"Configuration not found: {path}")

    try:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"Invalid TOML in {path}: {error}") from error

    game = _section(document, "game")
    saves = _section(document, "saves")
    training = _section(document, "training")
    instances = _section(document, "instances")

    install_path = _normalized(_required_string(game, "game", "install_path"))
    executable_name = _required_string(game, "game", "executable")
    executable = install_path / executable_name
    if not executable.is_file():
        raise ConfigError(f"Game executable not found: {executable}")

    personal = _normalized(_required_string(saves, "saves", "personal_root"))
    experimental = _normalized(_required_string(saves, "saves", "experimental_root"))
    backup = _normalized(_required_string(saves, "saves", "backup_root"))

    if personal == experimental:
        raise ConfigError("Personal and experimental save roots must differ")
    if personal == backup:
        raise ConfigError("Personal and backup save roots must differ")
    if _is_nested(experimental, personal):
        raise ConfigError("Experimental saves must not be nested under personal saves")
    if _is_nested(backup, personal):
        raise ConfigError("Backups must not be nested under the active personal save root")

    training_enabled = _required_bool(training, "training", "enabled")
    parallel_enabled = _required_bool(instances, "instances", "parallel_enabled")
    if training_enabled and not allow_training:
        raise ConfigError("Training must remain disabled during environment preparation")
    if parallel_enabled:
        raise ConfigError("Parallel execution must remain disabled until isolation is proven")

    return ProjectConfig(
        game_profile=_required_string(game, "game", "profile"),
        game_executable=executable,
        personal_save_root=personal,
        experimental_save_root=experimental,
        backup_root=backup,
        training_enabled=training_enabled,
        parallel_enabled=parallel_enabled,
    )
