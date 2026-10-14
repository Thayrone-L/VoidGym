from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from hk_agent.distributed_knowledge import (
    MAX_DOCUMENT_BYTES,
    KnowledgeValidationError,
    validate_champion_file,
    validate_knowledge_tree,
)

MAX_CHANGED_CONTRIBUTIONS = 50
MAX_CHANGED_BYTES = 10 * 1024 * 1024
_CONTRIBUTION_PATH_RE = re.compile(r"^knowledge/contributions/[0-9a-f]{64}\.json$")
_ALWAYS_ALLOWED_PATHS = frozenset({"knowledge/index.json", "knowledge/champion.json"})


@dataclass(frozen=True)
class GitChange:
    status: str
    path: str


def _git_executable() -> str:
    override = os.environ.get("VOIDGYM_GIT_EXE")
    if override:
        candidate = Path(override).resolve(strict=False)
        if not candidate.is_file():
            raise KnowledgeValidationError("VOIDGYM_GIT_EXE is not a regular file")
        return str(candidate)
    if os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/cmd/git.exe"
        if candidate.is_file():
            return str(candidate)
    return "git"


def _run_git(repository: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            [_git_executable(), "-C", str(repository.resolve()), *arguments],
            check=False,
            capture_output=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as error:
        raise KnowledgeValidationError("Git inspection timed out") from error
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise KnowledgeValidationError(f"Git inspection failed: {detail}")
    return result.stdout


def _decode_records(raw: bytes, *, label: str) -> list[str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise KnowledgeValidationError(f"{label} contains a non-UTF-8 path") from error
    if text and not text.endswith("\0"):
        raise KnowledgeValidationError(f"{label} has a malformed Git response")
    return text.split("\0")[:-1] if text else []


def read_changes(repository: Path, base_sha: str, candidate_sha: str) -> list[GitChange]:
    records = _decode_records(
        _run_git(
            repository,
            "diff",
            "--no-ext-diff",
            "--no-renames",
            "--name-status",
            "-z",
            base_sha,
            candidate_sha,
            "--",
        ),
        label="Git diff",
    )
    if len(records) % 2:
        raise KnowledgeValidationError("Git diff has an incomplete status/path pair")
    return [GitChange(records[index], records[index + 1]) for index in range(0, len(records), 2)]


def read_tracked_modes(repository: Path) -> dict[str, str]:
    records = _decode_records(
        _run_git(repository, "ls-files", "-s", "-z", "--", "knowledge"),
        label="Git index",
    )
    modes: dict[str, str] = {}
    for record in records:
        metadata, separator, path = record.partition("\t")
        fields = metadata.split()
        if not separator or len(fields) != 3:
            raise KnowledgeValidationError("Git index entry is malformed")
        mode = fields[0]
        if path in modes:
            raise KnowledgeValidationError(f"duplicate Git index entry: {path}")
        modes[path] = mode
    return modes


def _is_allowed_path(path: str) -> bool:
    return path in _ALWAYS_ALLOWED_PATHS or _CONTRIBUTION_PATH_RE.fullmatch(path) is not None


def validate_diff_policy(changes: list[GitChange]) -> None:
    added_contributions = 0
    for change in changes:
        if not _is_allowed_path(change.path):
            raise KnowledgeValidationError(f"knowledge PR changes a forbidden path: {change.path}")
        if change.status not in {"A", "M"}:
            raise KnowledgeValidationError(
                f"knowledge PR contains a forbidden {change.status} change: {change.path}"
            )
        if _CONTRIBUTION_PATH_RE.fullmatch(change.path):
            if change.status != "A":
                raise KnowledgeValidationError(
                    f"published contributions are immutable: {change.path}"
                )
            added_contributions += 1
    if added_contributions > MAX_CHANGED_CONTRIBUTIONS:
        raise KnowledgeValidationError(
            f"knowledge PR adds {added_contributions} contributions; limit is "
            f"{MAX_CHANGED_CONTRIBUTIONS}"
        )


def validate_inventory(repository: Path, knowledge_root: Path) -> None:
    modes = read_tracked_modes(repository)
    if set(_ALWAYS_ALLOWED_PATHS) - set(modes):
        raise KnowledgeValidationError("knowledge index.json and champion.json must be tracked")
    for path, mode in modes.items():
        if not _is_allowed_path(path):
            raise KnowledgeValidationError(f"knowledge contains an unexpected file: {path}")
        if mode != "100644":
            raise KnowledgeValidationError(f"knowledge entry must be a regular file: {path}")

    indexed = validate_knowledge_tree(knowledge_root)
    expected = {entry["path"] for entry in indexed["contributions"]}
    actual = {path for path in modes if _CONTRIBUTION_PATH_RE.fullmatch(path)}
    if actual != expected:
        missing = sorted(expected - actual)
        orphaned = sorted(actual - expected)
        raise KnowledgeValidationError(
            f"knowledge contribution inventory differs from index; missing={missing}, "
            f"orphaned={orphaned}"
        )


def validate_changed_sizes(candidate_root: Path, changes: list[GitChange]) -> None:
    total = 0
    for change in changes:
        path = candidate_root / Path(*change.path.split("/"))
        if path.is_symlink() or not path.is_file():
            raise KnowledgeValidationError(f"changed knowledge path is not a regular file: {change.path}")
        size = path.stat().st_size
        if size > MAX_DOCUMENT_BYTES:
            raise KnowledgeValidationError(f"knowledge document exceeds size limit: {change.path}")
        total += size
    if total > MAX_CHANGED_BYTES:
        raise KnowledgeValidationError(
            f"knowledge PR changes {total} bytes; limit is {MAX_CHANGED_BYTES}"
        )


def validate_champion_transition(
    base_root: Path,
    candidate_root: Path,
    *,
    changed: bool,
    allow_update: bool,
) -> None:
    if not changed:
        return
    if not allow_update:
        raise KnowledgeValidationError("external pull requests cannot update the champion")
    base = validate_champion_file(base_root / "champion.json")
    candidate = validate_champion_file(candidate_root / "champion.json")
    if candidate["generation"] != base["generation"] + 1:
        raise KnowledgeValidationError("champion update must advance exactly one generation")

    import json

    from hk_agent.distributed_knowledge import validate_champion

    try:
        document = validate_champion(
            json.loads((candidate_root / "champion.json").read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise KnowledgeValidationError("candidate champion is not valid UTF-8 JSON") from error
    if document["parent_champion_sha256"] != base["sha256"]:
        raise KnowledgeValidationError("champion update does not descend from the base champion")


def validate_knowledge_pull_request(
    repository: Path,
    trusted_knowledge_root: Path,
    *,
    base_sha: str,
    candidate_sha: str,
    allow_champion_update: bool,
) -> None:
    candidate_root = repository.resolve()
    knowledge_root = candidate_root / "knowledge"
    changes = read_changes(candidate_root, base_sha, candidate_sha)
    validate_diff_policy(changes)
    validate_inventory(candidate_root, knowledge_root)
    validate_changed_sizes(candidate_root, changes)
    validate_champion_transition(
        trusted_knowledge_root.resolve(),
        knowledge_root,
        changed=any(change.path == "knowledge/champion.json" for change in changes),
        allow_update=allow_champion_update,
    )
