from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_directory(
    root: Path, *, maximum_files: int = 1_000, maximum_bytes: int = 64 * 1024 * 1024
) -> dict[str, Any]:
    """Freeze a bounded, content-addressed inventory without copying its contents."""

    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir() or resolved_root.is_symlink():
        raise ValueError("fingerprint root must be a real directory")
    records: list[dict[str, object]] = []
    total_bytes = 0
    tree_digest = hashlib.sha256()
    for candidate in sorted(resolved_root.rglob("*"), key=lambda path: path.as_posix()):
        if candidate.is_dir():
            continue
        if candidate.is_symlink():
            raise ValueError("fingerprint tree cannot contain symbolic links")
        resolved = candidate.resolve(strict=True)
        try:
            relative = resolved.relative_to(resolved_root).as_posix()
        except ValueError as error:
            raise ValueError("fingerprint entry escaped its root") from error
        size = resolved.stat().st_size
        total_bytes += size
        if len(records) >= maximum_files or total_bytes > maximum_bytes:
            raise ValueError("fingerprint tree exceeds its bounded inventory")
        digest = sha256_file(resolved)
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(str(size).encode("ascii"))
        tree_digest.update(b"\0")
        tree_digest.update(digest.encode("ascii"))
        tree_digest.update(b"\n")
        records.append({"path": relative, "size": size, "sha256": digest})
    return {
        "path": str(resolved_root),
        "file_count": len(records),
        "total_bytes": total_bytes,
        "tree_sha256": tree_digest.hexdigest(),
        "files": records,
    }
