from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import uuid
from collections.abc import Callable
from pathlib import Path

# Every file Hollow Knight / the Modding API keeps per save slot:
#   user1.dat, user1_1.5.78.11833.dat (versioned copy), user1.dat.bak1 (rotating
#   backup the game creates on save), user1.modded.json and their .bak variants.
SAVE_FILE_PATTERN = re.compile(
    r"^user\d+(?:_[0-9.]+)?\.(?:dat|modded\.json)(?:\.bak\d*)?$", re.IGNORECASE
)


def is_save_file(name: str) -> bool:
    """True for any per-slot save artifact, including the game's rotating backups."""
    return SAVE_FILE_PATTERN.fullmatch(name) is not None


def _save_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.iterdir() if path.is_file() and is_save_file(path.name)]


def _check_fence(
    token: str | None, is_fence_current: Callable[[str], bool] | None
) -> None:
    if token is not None and (is_fence_current is None or not is_fence_current(token)):
        raise InterruptedError("instance save slot fence is no longer current")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_save_atomically(source: Path, destination: Path) -> None:
    """Copy across volumes safely, then atomically replace on the destination volume."""
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        if _file_sha256(source) != _file_sha256(temporary):
            raise OSError(f"save copy failed SHA-256 verification: {source}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_instance_saves(
    instance_root: Path,
    baseline_root: Path,
    launch_root: Path,
    *,
    fence_token: str | None = None,
    is_fence_current: Callable[[str], bool] | None = None,
) -> tuple[Path, Path]:
    """Preserve the instance's prior saves, then seed it from this launch baseline."""
    runtime_root = launch_root.parents[1]
    launch_root = ensure_inside_runtime(launch_root, runtime_root)
    instance_root = instance_root.expanduser().resolve(strict=False)
    baseline_root = baseline_root.expanduser().resolve(strict=False)
    _ensure_no_reparse_points(instance_root)
    _ensure_no_reparse_points(baseline_root)
    if _paths_overlap(instance_root, baseline_root):
        raise ValueError("Instance data root and save baseline must be separate")
    previous = launch_root / "previous-instance-state"
    if previous.exists():
        raise FileExistsError(previous)
    staging = ensure_inside_runtime(launch_root / ".instance-state-staging", runtime_root)
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    previous_stage = staging / "previous"
    seed_stage = staging / "seed"
    previous_stage.mkdir()
    seed_stage.mkdir()
    try:
        for path in _save_files(instance_root):
            _check_fence(fence_token, is_fence_current)
            shutil.copy2(path, previous_stage / path.name)
        for path in _save_files(baseline_root):
            _check_fence(fence_token, is_fence_current)
            shutil.copy2(path, seed_stage / path.name)
        _check_fence(fence_token, is_fence_current)
        if previous_stage.exists():
            previous_stage.rename(previous)
        instance_root.mkdir(parents=True, exist_ok=True)
        backups = staging / "replaced"
        backups.mkdir()
        existing_saves = _save_files(instance_root)
        for path in existing_saves:
            _check_fence(fence_token, is_fence_current)
            backup = backups / path.name
            shutil.copy2(path, backup)
            if _file_sha256(path) != _file_sha256(backup):
                raise OSError(f"save backup failed SHA-256 verification: {path}")
        for path in existing_saves:
            _check_fence(fence_token, is_fence_current)
            path.unlink()
        try:
            for path in _save_files(seed_stage):
                _check_fence(fence_token, is_fence_current)
                _copy_save_atomically(path, instance_root / path.name)
        except BaseException:
            for path in _save_files(instance_root):
                _check_fence(fence_token, is_fence_current)
                path.unlink(missing_ok=True)
            for path in backups.iterdir():
                _check_fence(fence_token, is_fence_current)
                _copy_save_atomically(path, instance_root / path.name)
            raise
    except BaseException:
        if previous.exists() and (fence_token is None or is_fence_current is not None and is_fence_current(fence_token)):
            for path in previous.iterdir():
                _check_fence(fence_token, is_fence_current)
                if not (instance_root / path.name).exists():
                    _copy_save_atomically(path, instance_root / path.name)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return previous, instance_root


def capture_instance_saves(
    instance_root: Path,
    launch_root: Path,
    *,
    fence_token: str | None = None,
    is_fence_current: Callable[[str], bool] | None = None,
) -> Path:
    runtime_root = launch_root.parents[1]
    launch_root = ensure_inside_runtime(launch_root, runtime_root)
    instance_root = instance_root.expanduser().resolve(strict=False)
    _ensure_no_reparse_points(instance_root)
    evidence = launch_root / "final-save"
    if evidence.is_dir():
        return evidence
    staging = ensure_inside_runtime(launch_root / ".final-save-staging", runtime_root)
    if evidence.exists() or staging.exists():
        raise FileExistsError(evidence)
    staging.mkdir(parents=True)
    try:
        for path in _save_files(instance_root):
            _check_fence(fence_token, is_fence_current)
            shutil.copy2(path, staging / path.name)
        _check_fence(fence_token, is_fence_current)
        staging.rename(evidence)
    except BaseException:
        # Keep the partial evidence under a recoverable name for diagnosis.
        if staging.exists():
            staging.rename(launch_root / "final-save-incomplete")
        raise
    return evidence


def _absolute_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return Path.cwd() / expanded


def _ensure_no_reparse_points(path: Path) -> None:
    absolute = _absolute_path(path)
    current = Path(absolute.anchor)
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for part in parts:
        if part == "..":
            current = current.parent
            continue
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            continue
        attributes = getattr(metadata, "st_file_attributes", 0)
        is_reparse_point = bool(
            attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
        if (os.name == "nt" and is_reparse_point) or (
            os.name != "nt" and current.is_symlink()
        ):
            raise ValueError(f"Runtime path must not contain symlinks or reparse points: {current}")


def _path_key(path: Path) -> str:
    value = os.fspath(path.resolve(strict=False))
    return os.path.normcase(value) if os.name == "nt" else value


def _paths_overlap(first: Path, second: Path) -> bool:
    first_key = _path_key(first)
    second_key = _path_key(second)
    try:
        common = os.path.commonpath((first_key, second_key))
    except ValueError:
        return False
    return common in {first_key, second_key}


def validate_disjoint_root(candidate: Path, *protected_roots: Path) -> Path:
    """Resolve a managed root and reject reparse points or overlap in either direction."""
    _ensure_no_reparse_points(candidate)
    resolved = candidate.expanduser().resolve(strict=False)
    for protected in protected_roots:
        _ensure_no_reparse_points(protected)
        if _paths_overlap(resolved, protected):
            raise ValueError(f"managed root must not overlap protected root: {protected}")
    return resolved


def validate_runtime_root(
    runtime_root: Path,
    *,
    personal: Path,
    experimental: Path,
    backup: Path,
    game_install: Path,
) -> Path:
    """Resolve and validate a runtime root against protected filesystem roots."""
    _ensure_no_reparse_points(runtime_root)
    resolved = runtime_root.expanduser().resolve(strict=False)
    protected_roots = {
        "personal saves": personal,
        "experimental saves": experimental,
        "save backups": backup,
        "game installation": game_install,
    }
    for label, protected in protected_roots.items():
        if _paths_overlap(resolved, protected):
            raise ValueError(f"Runtime root must not overlap {label}: {protected}")
    return resolved


def ensure_inside_runtime(candidate: Path, runtime_root: Path) -> Path:
    """Resolve a launch path and require it to remain inside a safe runtime root."""
    _ensure_no_reparse_points(runtime_root)
    _ensure_no_reparse_points(candidate)
    resolved_root = runtime_root.expanduser().resolve(strict=False)
    resolved_candidate = candidate.expanduser().resolve(strict=False)
    root_key = _path_key(resolved_root)
    candidate_key = _path_key(resolved_candidate)
    try:
        common = os.path.commonpath((candidate_key, root_key))
    except ValueError as error:
        raise ValueError(f"Launch path must remain inside runtime root: {candidate}") from error
    if common != root_key:
        raise ValueError(f"Launch path must remain inside runtime root: {candidate}")
    return resolved_candidate
