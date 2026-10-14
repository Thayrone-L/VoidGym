from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hk_agent.distributed_knowledge import (
    KnowledgeValidationError,
    canonical_document_sha256,
    contribution_from_manifest,
    sync_from_git_ref,
    validate_champion,
    validate_contribution,
    validate_index,
    validate_knowledge_tree,
)
from hk_agent.first_training import sha256_file

_WORKER_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_POLICY_BYTES = 512 * 1024 * 1024


class WorkerCycleError(ValueError):
    """Raised when a distributed worker input fails closed validation."""


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    worker_id: str
    trusted_ref: str
    knowledge_root: Path
    cache_root: Path
    outbox_root: Path
    stage_checkpoint: bool = True


@dataclass(frozen=True, slots=True)
class PublicationReceipt:
    bundle_id: str
    branch: str
    commit: str
    contribution_id: str
    source_commit: str
    worktree: str


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise WorkerCycleError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise WorkerCycleError(f"{label} has unexpected or missing fields")


def _resolve_config_path(config_path: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise WorkerCycleError(f"{label} must be a non-empty path")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return candidate.resolve(strict=False)


def load_worker_config(path: Path) -> WorkerConfig:
    config_path = path.resolve(strict=True)
    try:
        raw = json.loads(config_path.read_text("utf-8"))
    except json.JSONDecodeError as error:
        raise WorkerCycleError("worker config is invalid JSON") from error
    document = _object(raw, "worker config")
    _exact_keys(
        document,
        {
            "schema_version",
            "worker_id",
            "trusted_ref",
            "knowledge_root",
            "cache_root",
            "outbox_root",
            "stage_checkpoint",
        },
        "worker config",
    )
    if document["schema_version"] != 1:
        raise WorkerCycleError("unsupported worker config schema_version")
    worker_id = document["worker_id"]
    if not isinstance(worker_id, str) or not _WORKER_ID.fullmatch(worker_id):
        raise WorkerCycleError("worker_id must be a lowercase slug")
    stage_checkpoint = document["stage_checkpoint"]
    if not isinstance(stage_checkpoint, bool):
        raise WorkerCycleError("stage_checkpoint must be a boolean")
    trusted_ref = document["trusted_ref"]
    if trusted_ref != "origin/knowledge":
        raise WorkerCycleError("trusted_ref must be exactly origin/knowledge")
    return WorkerConfig(
        worker_id=worker_id,
        trusted_ref=trusted_ref,
        knowledge_root=_resolve_config_path(
            config_path, document["knowledge_root"], "knowledge_root"
        ),
        cache_root=_resolve_config_path(config_path, document["cache_root"], "cache_root"),
        outbox_root=_resolve_config_path(config_path, document["outbox_root"], "outbox_root"),
        stage_checkpoint=stage_checkpoint,
    )


def sync_worker_knowledge(
    config: WorkerConfig, *, repo: Path, ref: str | None = None
) -> dict[str, Any]:
    """Import an already-fetched Git ref; this never fetches, pushes, or uses credentials."""

    selected_ref = ref or config.trusted_ref
    if selected_ref != config.trusted_ref:
        raise WorkerCycleError("sync ref differs from the configured trusted maintainer ref")
    repository = repo.resolve(strict=True)
    source_commit = _git(
        repository,
        "rev-parse",
        f"{selected_ref}^{{commit}}",
        safe_error="cannot resolve the trusted ref to an immutable commit",
    )
    if not re.fullmatch(r"[0-9a-f]{40,64}", source_commit):
        raise WorkerCycleError("trusted ref did not resolve to a full commit hash")
    report = sync_from_git_ref(repository, source_commit, config.knowledge_root)
    champion, champion_hash = _load_champion(config.knowledge_root)
    if (
        report.get("champion_generation") != champion["generation"]
        or report.get("champion_sha256") != champion_hash
    ):
        raise WorkerCycleError("synced champion differs from the validated sync report")
    _atomic_json(
        config.cache_root / "trusted-source.json",
        {
            "schema_version": 1,
            "trust_tier": "maintainer-promoted",
            "ref": selected_ref,
            "source_commit": source_commit,
            "generation": champion["generation"],
            "champion_sha256": champion_hash,
        },
    )
    return {
        "worker_id": config.worker_id,
        **report,
        "ref": selected_ref,
        "source_commit": source_commit,
    }


def _load_champion(knowledge_root: Path) -> tuple[dict[str, Any], str]:
    validate_knowledge_tree(knowledge_root)
    champion_path = knowledge_root / "champion.json"
    try:
        raw = json.loads(champion_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerCycleError("validated knowledge has no readable champion.json") from error
    try:
        champion = validate_champion(raw)
    except KnowledgeValidationError as error:
        raise WorkerCycleError(str(error)) from error
    return champion, canonical_document_sha256(champion)


def _is_link_or_junction(path: Path) -> bool:
    return path.is_symlink() or (
        hasattr(os.path, "isjunction") and os.path.isjunction(path)
    )


def _cached_policy_path(config: WorkerConfig, digest: str) -> Path:
    policy = config.cache_root / "champions" / digest / "policies" / "policy.safetensors"
    for candidate in (
        config.cache_root,
        policy.parent.parent,
        policy.parent,
        policy,
    ):
        if _is_link_or_junction(candidate):
            raise WorkerCycleError("champion cache path cannot contain links or junctions")
    return policy


def _policy_from_cache(config: WorkerConfig, champion: Mapping[str, Any]) -> Path:
    artifact = champion.get("artifact")
    if not isinstance(artifact, Mapping):
        raise WorkerCycleError("non-canonical champion has no artifact")
    digest = artifact.get("sha256")
    size = artifact.get("size_bytes")
    artifact_format = artifact.get("format")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise WorkerCycleError("champion artifact has an invalid sha256")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or not 0 < size <= _MAX_POLICY_BYTES
    ):
        raise WorkerCycleError("champion artifact has an invalid size")
    if artifact_format != "safetensors":
        raise WorkerCycleError("distributed champions must use safetensors")
    bundle = config.cache_root / "champions" / digest
    policy = _cached_policy_path(config, digest)
    manifest_path = bundle / "manifest.json"
    try:
        if policy.is_symlink() or manifest_path.is_symlink():
            raise WorkerCycleError("champion cache bundle cannot contain symlinks")
        if policy.stat().st_size != size:
            raise WorkerCycleError("cached policy size differs from champion.json")
        if sha256_file(policy) != digest:
            raise WorkerCycleError("cached champion hash differs from champion.json")
        manifest = _object(json.loads(manifest_path.read_text("utf-8")), "champion manifest")
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerCycleError("champion cache bundle is incomplete") from error
    compatibility = champion.get("compatibility_fingerprint")
    if manifest.get("compatibility_fingerprint") != compatibility:
        raise WorkerCycleError("cached champion manifest compatibility differs")
    inventory = manifest.get("portable_policies")
    if not isinstance(inventory, list) or not any(
        isinstance(entry, Mapping)
        and entry.get("sha256") == digest
        and entry.get("format") == "safetensors"
        for entry in inventory
    ):
        raise WorkerCycleError("cached policy hash is absent from its manifest")
    return policy


def preflight_worker(config: WorkerConfig, session_id: str) -> dict[str, Any]:
    """Freeze the validated champion snapshot to use for exactly one new session."""

    if not _WORKER_ID.fullmatch(session_id):
        raise WorkerCycleError("session_id must be a lowercase slug")
    champion, champion_hash = _load_champion(config.knowledge_root)
    generation = champion["generation"]
    trusted_path = config.cache_root / "trusted-source.json"
    try:
        trusted = _object(json.loads(trusted_path.read_text("utf-8")), "trusted source")
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerCycleError("sync the trusted maintainer ref before preflight") from error
    _exact_keys(
        trusted,
        {
            "schema_version",
            "trust_tier",
            "ref",
            "source_commit",
            "generation",
            "champion_sha256",
        },
        "trusted source",
    )
    if (
        trusted["schema_version"] != 1
        or trusted["trust_tier"] != "maintainer-promoted"
        or trusted["ref"] != config.trusted_ref
        or not isinstance(trusted["source_commit"], str)
        or not re.fullmatch(r"[0-9a-f]{40,64}", trusted["source_commit"])
        or trusted["generation"] != generation
        or trusted["champion_sha256"] != champion_hash
    ):
        raise WorkerCycleError("champion is not the exact trusted maintainer-promoted snapshot")
    policy: Path | None = None
    if generation != 0:
        policy = _policy_from_cache(config, champion)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "worker_id": config.worker_id,
        "session_id": session_id,
        "parent_champion": {"generation": generation, "sha256": champion_hash},
        "source_commit": trusted["source_commit"],
        "trust_tier": "maintainer-promoted",
        "compatibility_fingerprint": champion.get("compatibility_fingerprint"),
        "resume_checkpoint": None,
        "resume_checkpoint_sha256": None,
        "warm_start_policy": str(policy) if policy is not None else None,
        "warm_start_policy_sha256": (
            champion["artifact"]["sha256"] if generation != 0 else None
        ),
        "warm_start_policy_format": "safetensors" if generation != 0 else None,
    }
    receipt_path = config.cache_root / "preflights" / f"{session_id}.json"
    if receipt_path.exists():
        try:
            existing = _object(json.loads(receipt_path.read_text("utf-8")), "preflight receipt")
        except (OSError, json.JSONDecodeError) as error:
            raise WorkerCycleError("existing preflight receipt is unreadable") from error
        if existing != receipt:
            raise WorkerCycleError("session_id already froze a different champion")
    else:
        _atomic_json(receipt_path, receipt)
    return receipt


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    pending.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(pending, path)


def _git_executable() -> str:
    override = os.environ.get("VOIDGYM_GIT_EXE")
    if override:
        candidate = Path(override).resolve(strict=False)
        if not candidate.is_file():
            raise WorkerCycleError("VOIDGYM_GIT_EXE is not a regular file")
        return str(candidate)
    if os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/cmd/git.exe"
        if candidate.is_file():
            return str(candidate)
    return "git"


def _run(
    arguments: list[str],
    *,
    cwd: Path,
    safe_error: str,
    timeout: int = 60,
) -> str:
    environment = dict(os.environ)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        result = subprocess.run(
            arguments,
            cwd=cwd,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise WorkerCycleError(safe_error) from error
    if result.returncode != 0:
        raise WorkerCycleError(safe_error)
    return result.stdout.strip()


def _git(repo: Path, *arguments: str, safe_error: str) -> str:
    return _run(
        [_git_executable(), "-C", str(repo), *arguments],
        cwd=repo,
        safe_error=safe_error,
    )


def _latest_checkpoint(manifest_path: Path, manifest: Mapping[str, Any]) -> tuple[Path, str, int]:
    if manifest.get("status") not in {"COMPLETED", "CANCELLED"}:
        raise WorkerCycleError("only a finished training manifest can become a candidate")
    if manifest.get("resumable_checkpoint") is not True:
        raise WorkerCycleError("training manifest has no resumable checkpoint")
    inventory = manifest.get("checkpoints")
    if not isinstance(inventory, list) or not inventory:
        raise WorkerCycleError("training manifest has no checkpoint inventory")
    entries = [entry for entry in inventory if isinstance(entry, Mapping)]
    if not entries:
        raise WorkerCycleError("training checkpoint inventory is invalid")
    latest = max(entries, key=lambda entry: int(entry.get("step", -1)))
    path_value = latest.get("path")
    digest = latest.get("sha256")
    step = latest.get("step")
    if (
        not isinstance(path_value, str)
        or not isinstance(digest, str)
        or not _SHA256.fullmatch(digest)
    ):
        raise WorkerCycleError("latest checkpoint metadata is invalid")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise WorkerCycleError("latest checkpoint step is invalid")
    unresolved = Path(path_value)
    if not unresolved.is_absolute():
        unresolved = manifest_path.parent / unresolved
    if unresolved.is_symlink():
        raise WorkerCycleError("checkpoint cannot be a symlink")
    checkpoint = unresolved.resolve(strict=True)
    expected_parent = (manifest_path.parent / "checkpoints").resolve(strict=True)
    try:
        checkpoint.relative_to(expected_parent)
    except ValueError as error:
        raise WorkerCycleError("checkpoint escaped the run checkpoint directory") from error
    if checkpoint.parent != expected_parent:
        raise WorkerCycleError("checkpoint must be directly inside the run checkpoint directory")
    if sha256_file(checkpoint) != digest:
        raise WorkerCycleError("latest checkpoint hash differs from the manifest")
    return checkpoint, digest, step


def _load_preflight(config: WorkerConfig, session_id: str) -> dict[str, Any]:
    if not _WORKER_ID.fullmatch(session_id):
        raise WorkerCycleError("session_id must be a lowercase slug")
    path = config.cache_root / "preflights" / f"{session_id}.json"
    try:
        receipt = _object(json.loads(path.read_text("utf-8")), "preflight receipt")
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerCycleError("run preflight before finalizing a candidate") from error
    _exact_keys(
        receipt,
        {
            "schema_version",
            "worker_id",
            "session_id",
            "parent_champion",
            "source_commit",
            "trust_tier",
            "compatibility_fingerprint",
            "resume_checkpoint",
            "resume_checkpoint_sha256",
            "warm_start_policy",
            "warm_start_policy_sha256",
            "warm_start_policy_format",
        },
        "preflight receipt",
    )
    if (
        receipt["schema_version"] != 1
        or receipt["worker_id"] != config.worker_id
        or receipt["session_id"] != session_id
        or receipt["trust_tier"] != "maintainer-promoted"
        or not isinstance(receipt["source_commit"], str)
        or not re.fullmatch(r"[0-9a-f]{40,64}", receipt["source_commit"])
    ):
        raise WorkerCycleError("preflight receipt belongs to another worker or schema")
    if receipt["resume_checkpoint"] is not None or receipt["resume_checkpoint_sha256"] is not None:
        raise WorkerCycleError("distributed preflight cannot authorize an SB3 resume checkpoint")
    parent = _object(receipt["parent_champion"], "preflight parent champion")
    _exact_keys(parent, {"generation", "sha256"}, "preflight parent champion")
    parent_generation = parent["generation"]
    parent_sha = parent["sha256"]
    if (
        not isinstance(parent_generation, int)
        or isinstance(parent_generation, bool)
        or parent_generation < 0
        or not isinstance(parent_sha, str)
        or not _SHA256.fullmatch(parent_sha)
    ):
        raise WorkerCycleError("preflight parent champion is invalid")
    warm_path = receipt["warm_start_policy"]
    warm_digest = receipt["warm_start_policy_sha256"]
    warm_format = receipt["warm_start_policy_format"]
    if warm_path is None or warm_digest is None or warm_format is None:
        if (warm_path, warm_digest, warm_format) != (None, None, None):
            raise WorkerCycleError("preflight warm-start policy fields are incomplete")
        if parent_generation != 0:
            raise WorkerCycleError("promoted champion preflight has no safe warm-start policy")
    elif (
        not isinstance(warm_path, str)
        or not isinstance(warm_digest, str)
        or not _SHA256.fullmatch(warm_digest)
        or warm_format != "safetensors"
    ):
        raise WorkerCycleError("preflight warm-start policy is invalid")
    elif parent_generation == 0:
        raise WorkerCycleError("generation-zero preflight cannot contain a warm-start policy")
    else:
        expected_policy = _cached_policy_path(config, warm_digest)
        supplied_policy = Path(warm_path)
        if supplied_policy != expected_policy:
            raise WorkerCycleError("preflight warm-start policy escaped the canonical cache path")
        try:
            size = expected_policy.stat().st_size
        except OSError as error:
            raise WorkerCycleError("preflight warm-start policy is unavailable") from error
        if not 0 < size <= _MAX_POLICY_BYTES or sha256_file(expected_policy) != warm_digest:
            raise WorkerCycleError("preflight warm-start policy failed integrity revalidation")
    return receipt


def finalize_candidate(
    config: WorkerConfig, manifest_path: Path, session_id: str
) -> dict[str, Any]:
    """Create an idempotent, sanitized outbox bundle after a session has ended."""

    resolved_manifest = manifest_path.resolve(strict=True)
    try:
        manifest = _object(json.loads(resolved_manifest.read_text("utf-8")), "training manifest")
    except json.JSONDecodeError as error:
        raise WorkerCycleError("training manifest is invalid JSON") from error
    receipt = _load_preflight(config, session_id)
    compatibility = manifest.get("compatibility_fingerprint")
    if not isinstance(compatibility, str) or not _SHA256.fullmatch(compatibility):
        raise WorkerCycleError("training manifest has no compatibility fingerprint")
    expected = receipt["compatibility_fingerprint"]
    if expected is not None and compatibility != expected:
        raise WorkerCycleError("training session is incompatible with its frozen champion")
    resume_digest = receipt["resume_checkpoint_sha256"]
    resumed_from = manifest.get("resumed_from")
    if resume_digest is None:
        if resumed_from is not None:
            raise WorkerCycleError("generation-zero session unexpectedly resumed a checkpoint")
    elif not isinstance(resumed_from, Mapping) or resumed_from.get("sha256") != resume_digest:
        raise WorkerCycleError("training session did not resume its frozen champion")
    warm_start_digest = receipt["warm_start_policy_sha256"]
    warm_started_from = manifest.get("warm_started_from")
    if warm_start_digest is None:
        if warm_started_from is not None:
            raise WorkerCycleError("generation-zero session unexpectedly used a warm start")
    elif (
        not isinstance(warm_started_from, Mapping)
        or warm_started_from.get("sha256") != warm_start_digest
        or warm_started_from.get("format") != "safetensors"
    ):
        raise WorkerCycleError("training session did not warm-start its frozen champion policy")
    checkpoint, checkpoint_hash, checkpoint_step = _latest_checkpoint(resolved_manifest, manifest)
    contribution = contribution_from_manifest(manifest)
    if contribution["compatibility_fingerprint"] != compatibility:
        raise WorkerCycleError("training manifest compatibility fingerprint is not canonical")
    contribution_summary = {
        "contribution_id": contribution["contribution_id"],
        "sha256": canonical_document_sha256(contribution),
        "compatibility_fingerprint": contribution["compatibility_fingerprint"],
        "filename": "contribution.json",
    }
    parent = receipt["parent_champion"]
    if not isinstance(parent, Mapping):
        raise WorkerCycleError("preflight parent champion is invalid")
    body: dict[str, Any] = {
        "schema_version": 1,
        "worker_id": config.worker_id,
        "session_id": session_id,
        "parent_champion": dict(parent),
        "source_commit": receipt["source_commit"],
        "compatibility_fingerprint": compatibility,
        "contribution": contribution_summary,
        "artifact": {
            "sha256": checkpoint_hash,
            "size_bytes": checkpoint.stat().st_size,
            "step": checkpoint_step,
            "filename": "checkpoint.zip" if config.stage_checkpoint else None,
        },
        "finished_unix_ns": manifest.get("finished_unix_ns"),
    }
    bundle_id = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    # This is deliberately a private staging receipt, not the public candidate
    # manifest.  Publication adds the immutable HTTPS artifact URI and runs the
    # closed public candidate validator in distributed_knowledge.
    staging = {"bundle_id": bundle_id, **body}
    candidate_root = config.outbox_root / "candidates" / bundle_id
    candidate_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(candidate_root / "contribution.json", contribution)
    if config.stage_checkpoint:
        destination = candidate_root / "checkpoint.zip"
        if destination.exists():
            if sha256_file(destination) != checkpoint_hash:
                raise WorkerCycleError("existing staged candidate checkpoint has wrong hash")
        else:
            pending = candidate_root / f".checkpoint.{os.getpid()}.pending"
            try:
                shutil.copyfile(checkpoint, pending)
                if sha256_file(pending) != checkpoint_hash:
                    raise WorkerCycleError("staged candidate checkpoint failed hash verification")
                os.replace(pending, destination)
            finally:
                if pending.exists():
                    pending.unlink()
    _atomic_json(candidate_root / "staging.json", staging)
    return staging


def _load_staged_contribution(
    config: WorkerConfig, bundle_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _SHA256.fullmatch(bundle_id):
        raise WorkerCycleError("bundle_id must be a lowercase SHA-256")
    bundle = config.outbox_root / "candidates" / bundle_id
    staging_path = bundle / "staging.json"
    contribution_path = bundle / "contribution.json"
    if bundle.is_symlink() or staging_path.is_symlink() or contribution_path.is_symlink():
        raise WorkerCycleError("publication bundle cannot contain symlinks")
    try:
        staging = _object(json.loads(staging_path.read_text("utf-8")), "staging receipt")
        contribution = validate_contribution(json.loads(contribution_path.read_text("utf-8")))
    except (OSError, json.JSONDecodeError, KnowledgeValidationError) as error:
        raise WorkerCycleError("publication bundle is incomplete or invalid") from error
    if staging.get("bundle_id") != bundle_id:
        raise WorkerCycleError("staging receipt belongs to another bundle")
    summary = staging.get("contribution")
    expected = {
        "contribution_id": contribution["contribution_id"],
        "sha256": canonical_document_sha256(contribution),
        "compatibility_fingerprint": contribution["compatibility_fingerprint"],
        "filename": "contribution.json",
    }
    if summary != expected:
        raise WorkerCycleError("staged contribution differs from its receipt")
    return staging, contribution


def _publication_receipt(config: WorkerConfig, bundle_id: str) -> Path:
    return config.outbox_root / "publications" / f"{bundle_id}.json"


def _read_publication_receipt(config: WorkerConfig, bundle_id: str) -> PublicationReceipt:
    path = _publication_receipt(config, bundle_id)
    try:
        document = _object(json.loads(path.read_text("utf-8")), "publication receipt")
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerCycleError("prepare this bundle before publishing it") from error
    _exact_keys(
        document,
        {
            "schema_version",
            "bundle_id",
            "branch",
            "commit",
            "contribution_id",
            "source_commit",
            "worktree",
        },
        "publication receipt",
    )
    if document["schema_version"] != 1 or document["bundle_id"] != bundle_id:
        raise WorkerCycleError("publication receipt belongs to another bundle or schema")
    for key in ("branch", "commit", "contribution_id", "source_commit", "worktree"):
        if not isinstance(document[key], str):
            raise WorkerCycleError(f"publication receipt {key} must be a string")
    if not re.fullmatch(r"[0-9a-f]{40,64}", document["commit"]):
        raise WorkerCycleError("publication receipt has an invalid commit")
    if not re.fullmatch(r"[0-9a-f]{40,64}", document["source_commit"]):
        raise WorkerCycleError("publication receipt has an invalid source_commit")
    if not _SHA256.fullmatch(document["contribution_id"]):
        raise WorkerCycleError("publication receipt has an invalid contribution_id")
    return PublicationReceipt(
        bundle_id=bundle_id,
        branch=document["branch"],
        commit=document["commit"],
        contribution_id=document["contribution_id"],
        source_commit=document["source_commit"],
        worktree=document["worktree"],
    )


def _verify_prepared_publication(config: WorkerConfig, receipt: PublicationReceipt) -> Path:
    expected_branch = f"codex/knowledge-{config.worker_id}-{receipt.contribution_id[:12]}"
    if receipt.branch != expected_branch:
        raise WorkerCycleError("publication branch does not match the immutable bundle")
    worktree = Path(receipt.worktree).resolve(strict=True)
    expected_worktree = (config.outbox_root / "wt" / receipt.bundle_id[:16]).resolve(strict=False)
    if worktree != expected_worktree or worktree.is_symlink():
        raise WorkerCycleError("publication worktree escaped the configured outbox")
    branch = _git(
        worktree,
        "branch",
        "--show-current",
        safe_error="cannot inspect the prepared publication branch",
    )
    commit = _git(
        worktree,
        "rev-parse",
        "HEAD",
        safe_error="cannot inspect the prepared publication commit",
    )
    status = _git(
        worktree,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        safe_error="cannot inspect the prepared publication worktree",
    )
    if branch != receipt.branch or commit != receipt.commit or status:
        raise WorkerCycleError("prepared publication changed after its receipt was created")
    parents = _git(
        worktree,
        "rev-list",
        "--parents",
        "-n",
        "1",
        receipt.commit,
        safe_error="cannot inspect the prepared publication lineage",
    ).split()
    if parents != [receipt.commit, receipt.source_commit]:
        raise WorkerCycleError("prepared publication is not one commit above its trusted snapshot")
    changed = set(
        _git(
            worktree,
            "diff",
            "--name-only",
            f"{receipt.source_commit}..{receipt.commit}",
            safe_error="cannot inspect prepared publication paths",
        ).splitlines()
    )
    expected_paths = {
        "knowledge/index.json",
        f"knowledge/contributions/{receipt.contribution_id}.json",
    }
    if changed != expected_paths:
        raise WorkerCycleError("prepared publication contains unexpected paths")
    validate_knowledge_tree(worktree / "knowledge")
    return worktree


def prepare_publication(config: WorkerConfig, *, repo: Path, bundle_id: str) -> dict[str, Any]:
    """Create an immutable contribution commit in an isolated Git worktree.

    This does not fetch, push, open a pull request, upload the checkpoint, or modify
    the caller's checkout. Only the sanitized contribution and index are committed.
    """

    staging, contribution = _load_staged_contribution(config, bundle_id)
    receipt_path = _publication_receipt(config, bundle_id)
    if receipt_path.exists():
        receipt = _read_publication_receipt(config, bundle_id)
        _verify_prepared_publication(config, receipt)
        return {
            "schema_version": 1,
            "bundle_id": receipt.bundle_id,
            "branch": receipt.branch,
            "commit": receipt.commit,
            "contribution_id": receipt.contribution_id,
            "source_commit": receipt.source_commit,
            "worktree": receipt.worktree,
        }

    repository = repo.resolve(strict=True)
    contribution_id = contribution["contribution_id"]
    branch = f"codex/knowledge-{config.worker_id}-{contribution_id[:12]}"
    worktree = (config.outbox_root / "wt" / bundle_id[:16]).resolve(strict=False)
    if worktree.exists():
        raise WorkerCycleError("publication worktree already exists without a receipt")
    worktree.parent.mkdir(parents=True, exist_ok=True)
    hooks = (config.outbox_root / "empty-git-hooks").resolve(strict=False)
    hooks.mkdir(parents=True, exist_ok=True)
    source_commit = _git(
        repository,
        "rev-parse",
        f"{config.trusted_ref}^{{commit}}",
        safe_error="cannot resolve the trusted ref to an immutable commit",
    )
    if not re.fullmatch(r"[0-9a-f]{40,64}", source_commit):
        raise WorkerCycleError("trusted ref did not resolve to a full commit hash")
    if staging.get("source_commit") != source_commit:
        raise WorkerCycleError(
            "trusted ref no longer matches the commit frozen for this training session"
        )

    # `show-ref` exit 1 means absent. Any other output/error is intentionally hidden.
    try:
        branch_probe = subprocess.run(
            [
                _git_executable(),
                "-C",
                str(repository),
                "show-ref",
                "--verify",
                "--quiet",
                f"refs/heads/{branch}",
            ],
            cwd=repository,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise WorkerCycleError("cannot inspect local publication branches") from error
    if branch_probe.returncode == 0:
        raise WorkerCycleError("immutable publication branch already exists")
    if branch_probe.returncode != 1:
        raise WorkerCycleError("cannot inspect local publication branches")

    worktree_created = False
    branch_created = False
    receipt_written = False
    try:
        _git(
            repository,
            "-c",
            f"core.hooksPath={hooks}",
            "worktree",
            "add",
            "--detach",
            str(worktree),
            source_commit,
            safe_error="cannot create the isolated publication worktree",
        )
        worktree_created = True
        _git(
            worktree,
            "-c",
            f"core.hooksPath={hooks}",
            "checkout",
            "-b",
            branch,
            safe_error="cannot create the immutable publication branch",
        )
        branch_created = True

        # The checked-out tree must be exactly the validated maintainer snapshot used by
        # the worker. This catches a fetch without a matching worker sync.
        checked_index = validate_knowledge_tree(worktree / "knowledge")
        trusted_index = validate_knowledge_tree(config.knowledge_root)
        checked_champion, checked_hash = _load_champion(worktree / "knowledge")
        trusted_champion, trusted_hash = _load_champion(config.knowledge_root)
        if (
            checked_index != trusted_index
            or checked_champion != trusted_champion
            or checked_hash != trusted_hash
        ):
            raise WorkerCycleError("trusted ref changed; sync the worker again before preparing")

        destination = worktree / "knowledge" / "contributions" / f"{contribution_id}.json"
        if destination.exists():
            raise WorkerCycleError("contribution is already present in the trusted branch")
        _atomic_json(destination, contribution)
        index_path = worktree / "knowledge" / "index.json"
        try:
            index = validate_index(json.loads(index_path.read_text("utf-8")))
        except (OSError, json.JSONDecodeError, KnowledgeValidationError) as error:
            raise WorkerCycleError("trusted publication index is invalid") from error
        entry = {
            "contribution_id": contribution_id,
            "path": f"knowledge/contributions/{contribution_id}.json",
            "sha256": canonical_document_sha256(contribution),
            "compatibility_fingerprint": contribution["compatibility_fingerprint"],
        }
        index["contributions"] = sorted(
            [*index["contributions"], entry], key=lambda item: item["contribution_id"]
        )
        _atomic_json(index_path, index)
        validate_knowledge_tree(worktree / "knowledge")

        expected_paths = {
            "knowledge/index.json",
            f"knowledge/contributions/{contribution_id}.json",
        }
        _git(
            worktree,
            "add",
            "--",
            *sorted(expected_paths),
            safe_error="cannot stage the sanitized knowledge documents",
        )
        staged = _git(
            worktree,
            "diff",
            "--cached",
            "--name-only",
            safe_error="cannot inspect staged publication paths",
        )
        if set(staged.splitlines()) != expected_paths:
            raise WorkerCycleError("publication attempted to stage unexpected paths")
        _git(
            worktree,
            "-c",
            f"core.hooksPath={hooks}",
            "-c",
            "user.name=VoidGym Worker",
            "-c",
            "user.email=voidgym-worker@users.noreply.github.com",
            "commit",
            "-m",
            f"Add knowledge contribution {contribution_id[:12]}",
            safe_error="cannot commit the sanitized publication",
        )
        commit = _git(
            worktree,
            "rev-parse",
            "HEAD",
            safe_error="cannot resolve the publication commit",
        )
        receipt_document: dict[str, object] = {
            "schema_version": 1,
            "bundle_id": bundle_id,
            "branch": branch,
            "commit": commit,
            "contribution_id": contribution_id,
            "source_commit": source_commit,
            "worktree": str(worktree),
        }
        _atomic_json(receipt_path, receipt_document)
        receipt_written = True
        receipt = _read_publication_receipt(config, bundle_id)
        _verify_prepared_publication(config, receipt)
        return dict(receipt_document)
    except Exception as error:
        if worktree_created or worktree.exists():
            try:
                _git(
                    repository,
                    "-c",
                    f"core.hooksPath={hooks}",
                    "worktree",
                    "remove",
                    "--force",
                    str(worktree),
                    safe_error="cannot remove failed publication worktree",
                )
            except WorkerCycleError:
                pass
        if branch_created:
            try:
                _git(
                    repository,
                    "branch",
                    "-D",
                    branch,
                    safe_error="cannot remove failed publication branch",
                )
            except WorkerCycleError:
                pass
        if receipt_written:
            try:
                receipt_path.unlink()
            except OSError:
                pass
        if isinstance(error, KnowledgeValidationError):
            raise WorkerCycleError("trusted publication snapshot is invalid") from error
        raise
