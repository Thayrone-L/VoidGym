from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hk_agent.instance_marker import load_instance_marker
from hk_agent.runtime_paths import validate_disjoint_root, validate_runtime_root


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
    instance_count: int
    runtime_root: Path
    training_device: str
    torch_threads: int
    torch_interop_threads: int
    install_roots: tuple[Path, ...]
    instance_data_roots: tuple[Path, ...]
    instance_product_names: tuple[str, ...]


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


def _optional_int(
    section: dict[str, Any], section_name: str, key: str, default: int, minimum: int, maximum: int
) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ConfigError(f"{section_name}.{key} must be an integer in [{minimum}, {maximum}]")
    return int(value)


def _normalized(path_value: str) -> Path:
    return Path(path_value).expanduser().resolve(strict=False)


def _is_nested(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return candidate != parent


def load_config(path: Path, *, allow_training: bool = False) -> ProjectConfig:
    path = path.expanduser().resolve(strict=False)
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
    instance_count = _optional_int(instances, "instances", "count", 1, 1, 4)
    runtime_value = instances.get("runtime_root", "artifacts/runtime")
    if not isinstance(runtime_value, str) or not runtime_value.strip():
        raise ConfigError("instances.runtime_root must be a non-empty string")
    runtime_path = Path(runtime_value).expanduser()
    if not runtime_path.is_absolute():
        runtime_path = path.parent / runtime_path
    try:
        runtime_root = validate_runtime_root(
            runtime_path,
            personal=personal,
            experimental=experimental,
            backup=backup,
            game_install=install_path,
        )
    except ValueError as error:
        raise ConfigError(str(error)) from error
    training_device = instances.get("training_device", "cpu")
    if not isinstance(training_device, str) or training_device not in {"cpu", "cuda", "auto"}:
        raise ConfigError("instances.training_device must be cpu, cuda, or auto")
    torch_threads = _optional_int(instances, "instances", "torch_threads", 3, 1, 8)
    torch_interop_threads = _optional_int(instances, "instances", "torch_interop_threads", 1, 1, 4)
    raw_roots = instances.get("install_roots", [])
    install_roots: list[Path] = []
    instance_data_roots: list[Path] = []
    instance_product_names: list[str] = []
    if not isinstance(raw_roots, list):
        raise ConfigError("instances.install_roots must be an array of paths")
    product_names: set[str] = set()
    managed_roots: list[Path] = []
    if raw_roots and len(raw_roots) != instance_count:
        raise ConfigError("instances.install_roots must contain exactly instances.count paths")
    protected = (install_path, runtime_root, personal, experimental, backup)
    for index, raw_root in enumerate(raw_roots, start=1):
        if not isinstance(raw_root, str) or not raw_root.strip():
            raise ConfigError("instances.install_roots entries must be non-empty paths")
        root = _normalized(raw_root)
        try:
            marker = load_instance_marker(root, expected_index=index)
        except (OSError, ValueError, TypeError) as error:
            marker_path = root / ".hk-agent-instance.json"
            raise ConfigError(f"managed instance marker is missing or invalid: {marker_path}") from error
        product = marker.product_name
        data_value = marker.persistent_data_path_expected
        if product.casefold() in product_names:
            raise ConfigError("managed instance productName values must be unique")
        product_names.add(product.casefold())
        data_root = _normalized(data_value)
        for protected_root in protected:
            try:
                root = validate_disjoint_root(root, protected_root)
                data_root = validate_disjoint_root(data_root, protected_root)
            except ValueError as error:
                raise ConfigError(str(error)) from error
        if any(root == previous for previous in install_roots):
            raise ConfigError("managed instance install roots must be unique")
        if any(data_root == previous for previous in instance_data_roots):
            raise ConfigError("managed instance data roots must be unique")
        for existing in managed_roots:
            try:
                validate_disjoint_root(root, existing)
                validate_disjoint_root(data_root, existing)
            except ValueError as error:
                raise ConfigError(str(error)) from error
        managed_roots.extend((root, data_root))
        install_roots.append(root)
        instance_data_roots.append(data_root)
        instance_product_names.append(product)
    if training_enabled and not allow_training:
        raise ConfigError("Training must remain disabled during environment preparation")
    if parallel_enabled != (instance_count > 1):
        raise ConfigError(
            "instances.parallel_enabled must be true exactly when instances.count is greater than 1"
        )
    if parallel_enabled and training_device != "cpu":
        raise ConfigError("GPU reservada para as instâncias do jogo; use cpu com count > 1")
    if parallel_enabled and not raw_roots:
        raise ConfigError("instances.install_roots is required for parallel execution")

    return ProjectConfig(
        game_profile=_required_string(game, "game", "profile"),
        game_executable=executable,
        personal_save_root=personal,
        experimental_save_root=experimental,
        backup_root=backup,
        training_enabled=training_enabled,
        parallel_enabled=parallel_enabled,
        instance_count=instance_count,
        runtime_root=runtime_root,
        training_device=training_device,
        torch_threads=torch_threads,
        torch_interop_threads=torch_interop_threads,
        install_roots=tuple(install_roots),
        instance_data_roots=tuple(instance_data_roots),
        instance_product_names=tuple(instance_product_names),
    )
