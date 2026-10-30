from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class StateError(ValueError):
    """Installer state failed validation."""


class InstallState:
    SCHEMA = 1

    def __init__(self, path: Path) -> None:
        self.path = path.resolve(strict=False)
        self.data: dict[str, Any] = self._read()

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": self.SCHEMA, "created_paths": [], "registry_keys": [], "shortcuts": [], "completed_steps": []}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StateError(f"Estado do instalador inválido: {error}") from error
        if not isinstance(value, dict) or value.get("schema_version") != self.SCHEMA:
            raise StateError("Versão ou estrutura do estado do instalador inválida.")
        for key in ("created_paths", "registry_keys", "shortcuts", "completed_steps"):
            if not isinstance(value.get(key), list) or not all(isinstance(item, str) for item in value[key]):
                raise StateError(f"Campo inválido no estado: {key}")
        return value

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".install-state-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(self.data, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.path)
        finally:
            Path(temp_name).unlink(missing_ok=True)

    def record_path(self, path: Path) -> None:
        value = str(path.resolve(strict=False))
        if value not in self.data["created_paths"]:
            self.data["created_paths"].append(value)
            self.save()

    def record_registry_key(self, key: str) -> None:
        if key not in self.data["registry_keys"]:
            self.data["registry_keys"].append(key)
            self.save()

    def record_shortcut(self, path: Path) -> None:
        value = str(path.resolve(strict=False))
        if value not in self.data["shortcuts"]:
            self.data["shortcuts"].append(value)
            self.save()

    def complete_step(self, step_id: str) -> None:
        if step_id not in self.data["completed_steps"]:
            self.data["completed_steps"].append(step_id)
            self.save()


def safe_uninstall_path(path: Path, *, app_root: Path, local_low: Path) -> bool:
    target = path.resolve(strict=False)
    roots = (app_root.resolve(strict=False), local_low.resolve(strict=False))
    if not any(_is_relative_to(target, root) for root in roots):
        return False
    if _is_relative_to(target, roots[1]) and not target.name.casefold().startswith("hollow knight ai "):
        return False
    return target != roots[0] and target != roots[1]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def safe_uninstall_registry_key(key: str) -> bool:
    normalized = key.replace("/", "\\").strip("\\")
    if normalized.casefold().startswith("hkcu\\"):
        normalized = normalized[5:]
    prefix = "Software\\Team Cherry\\Hollow Knight AI "
    if not normalized.casefold().startswith(prefix.casefold()):
        return False
    return all(part not in {"", ".", ".."} for part in normalized.split("\\"))
