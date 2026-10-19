from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from hk_agent.runtime_paths import (
    _ensure_no_reparse_points,
    ensure_inside_runtime,
    validate_disjoint_root,
)

SAVE_FILE_PATTERN = re.compile(
    r"^user\d+(?:_[0-9.]+)?\.(?:dat|modded\.json)(?:\.bak\d*)?$", re.IGNORECASE
)
BACKUP_LIMIT = 12
BACKUP_INTERVAL_SECONDS = 300
_last_progress_write: dict[str, float] = {}


def _save_files(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_file() and SAVE_FILE_PATTERN.fullmatch(path.name)) if root.is_dir() else []


def _hashes(root: Path) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in _save_files(root)}


def save_signature(root: Path) -> dict[str, str]:
    return _hashes(root)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def playthrough_root(runtime_root: Path, campaign_id: str, slot: int) -> Path:
    if not campaign_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in campaign_id):
        raise ValueError("campaign_id must be a safe path component")
    if isinstance(slot, bool) or not isinstance(slot, int) or not 1 <= slot <= 4:
        raise ValueError("slot must be in [1, 4]")
    return ensure_inside_runtime(runtime_root / "playthroughs" / campaign_id / f"slot-{slot}", runtime_root)


def initialize_playthrough(
    root: Path,
    baseline: Path,
    *,
    runtime_root: Path,
    personal_root: Path,
    backup_root: Path,
    game_install: Path,
    source: str,
    save_slot: int,
) -> tuple[Path, bool]:
    if source not in {"existing_save", "new_game", "continuous_playthrough"}:
        raise ValueError("unsupported playthrough source")
    root = validate_disjoint_root(root, personal_root, baseline, backup_root, game_install)
    root = ensure_inside_runtime(root, runtime_root)
    baseline = baseline.resolve(strict=True)
    _ensure_no_reparse_points(baseline)
    manifest = root / "playthrough.json"
    if manifest.exists():
        _read_manifest(root)
        return root, False
    if root.exists():
        if any(root.iterdir()):
            raise RuntimeError("uninitialized playthrough directory is not empty")
    else:
        root.mkdir(parents=True)
    seed_source = "existing_save" if source == "continuous_playthrough" else source
    selected = baseline / f"user{save_slot}.dat"
    if seed_source == "existing_save" and not selected.is_file():
        raise RuntimeError(f"selected existing save slot is empty: {selected.name}")
    if seed_source == "new_game" and selected.exists():
        raise RuntimeError(f"new-game slot is occupied and will not be overwritten: {selected.name}")
    staging = root / ".seed-staging"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir()
    try:
        for path in _save_files(baseline):
            shutil.copy2(path, staging / path.name)
        if seed_source == "new_game":
            for path in _save_files(staging):
                if path.name.casefold().startswith(f"user{save_slot}".casefold()):
                    path.unlink()
        for path in _save_files(staging):
            os.replace(path, root / path.name)
        _atomic_json(manifest, {
            "schema_version": 1,
            "campaign_id": root.parent.name,
            "slot": int(root.name.removeprefix("slot-")),
            "save_slot": save_slot,
            "seed_source": seed_source,
            "created_unix_ns": time.time_ns(),
            "last_scene": None,
            "progress": {},
            "save_sha256": _hashes(root),
        })
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return root, True


def _read_manifest(root: Path) -> dict[str, Any]:
    try:
        value = json.loads((root / "playthrough.json").read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("playthrough manifest is missing or corrupt") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RuntimeError("unsupported playthrough manifest")
    return value


def backup_playthrough(root: Path, *, reason: str, keep: int = BACKUP_LIMIT) -> Path | None:
    saves = _save_files(root)
    if not saves:
        return None
    backups = root / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    name = f"{time.time_ns()}-{reason[:32]}"
    target = backups / name
    staging = backups / f".{name}.staging"
    staging.mkdir()
    try:
        for path in saves:
            shutil.copy2(path, staging / path.name)
        _atomic_json(staging / "backup.json", {
            "schema_version": 1,
            "created_unix_ns": time.time_ns(),
            "reason": reason,
            "sha256": {path.name: hashlib.sha256((staging / path.name).read_bytes()).hexdigest() for path in saves},
        })
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    entries = sorted(path for path in backups.iterdir() if path.is_dir() and not path.name.startswith("."))
    for old in entries[:-keep]:
        shutil.rmtree(old)
    return target


def persist_playthrough(root: Path, instance_root: Path, *, force_backup: bool = False) -> dict[str, str]:
    previous = _hashes(root)
    current = _hashes(instance_root)
    if force_backup or (previous and previous != current):
        backup_playthrough(root, reason="save-change" if previous != current else "checkpoint")
    staging = root / ".save-staging"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir()
    try:
        for path in _save_files(instance_root):
            shutil.copy2(path, staging / path.name)
        # Replace only managed save files, keeping manifests and backup history.
        for path in _save_files(root):
            path.unlink()
        for path in _save_files(staging):
            os.replace(path, root / path.name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    hashes = _hashes(root)
    manifest = _read_manifest(root)
    manifest["save_sha256"] = hashes
    manifest["saved_unix_ns"] = time.time_ns()
    _atomic_json(root / "playthrough.json", manifest)
    return hashes


def restore_last_backup(root: Path, *, instance_root: Path | None = None) -> Path:
    root = root.resolve(strict=True)
    _ensure_no_reparse_points(root)
    backups = root / "backups"
    candidates = sorted(path for path in backups.iterdir() if path.is_dir()) if backups.is_dir() else []
    for candidate in reversed(candidates):
        try:
            value = json.loads((candidate / "backup.json").read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        hashes = value.get("sha256") if isinstance(value, dict) else None
        if not isinstance(hashes, dict) or not hashes:
            continue
        valid = all(isinstance(name, str) and isinstance(digest, str) and (candidate / name).is_file()
                    and hashlib.sha256((candidate / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
        if not valid:
            continue
        target = root / ".restore-staging"
        if target.exists():
            raise FileExistsError(target)
        target.mkdir()
        try:
            for name in hashes:
                shutil.copy2(candidate / name, target / name)
            for path in _save_files(root):
                path.unlink()
            for path in _save_files(target):
                os.replace(path, root / path.name)
            if instance_root is not None:
                _ensure_no_reparse_points(instance_root)
                instance_root.mkdir(parents=True, exist_ok=True)
                temp = instance_root / ".restore-staging"
                if temp.exists():
                    raise FileExistsError(temp)
                temp.mkdir()
                try:
                    for path in _save_files(root):
                        shutil.copy2(path, temp / path.name)
                    for path in _save_files(instance_root):
                        path.unlink()
                    for path in _save_files(temp):
                        os.replace(path, instance_root / path.name)
                finally:
                    shutil.rmtree(temp, ignore_errors=True)
            manifest = _read_manifest(root)
            manifest["save_sha256"] = _hashes(root)
            manifest["saved_unix_ns"] = time.time_ns()
            _atomic_json(root / "playthrough.json", manifest)
            return candidate
        finally:
            shutil.rmtree(target, ignore_errors=True)
    raise RuntimeError("no valid playthrough backup is available")


def update_playthrough_progress(root: Path, *, scene: str | None, progress: dict[str, Any]) -> None:
    manifest = _read_manifest(root)
    current_progress = manifest.get("progress", {})
    if not isinstance(current_progress, dict):
        current_progress = {}
    key = str(root.resolve(strict=False))
    now = time.monotonic()
    meaningful_change = False
    if scene and scene != manifest.get("last_scene"):
        manifest["last_scene"] = scene
        meaningful_change = True
    for name, value in progress.items():
        if value is None:
            continue
        if name == "step" and now - _last_progress_write.get(key, 0.0) < 30.0:
            continue
        if current_progress.get(name) != value:
            current_progress[name] = value
            meaningful_change = True
    if not meaningful_change:
        return
    manifest["progress"] = current_progress
    manifest["updated_unix_ns"] = time.time_ns()
    _atomic_json(root / "playthrough.json", manifest)
    _last_progress_write[key] = now


def playthrough_status(root: Path) -> dict[str, Any] | None:
    if not (root / "playthrough.json").is_file():
        return None
    manifest = _read_manifest(root)
    return {
        "path": str(root),
        "last_scene": manifest.get("last_scene"),
        "progress": manifest.get("progress", {}),
        "last_saved_unix_ns": manifest.get("saved_unix_ns"),
        "backup_count": len(list((root / "backups").glob("*/backup.json"))) if (root / "backups").is_dir() else 0,
        "seed_source": manifest.get("seed_source"),
    }
