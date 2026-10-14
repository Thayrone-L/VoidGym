from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
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
    validate_knowledge_tree,
)
from hk_agent.first_training import sha256_file

_WORKER_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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
    if not isinstance(trusted_ref, str) or not trusted_ref.startswith("origin/"):
        raise WorkerCycleError("trusted_ref must name an origin/* maintainer ref")
    return WorkerConfig(
        worker_id=worker_id,
        trusted_ref=trusted_ref,
        knowledge_root=_resolve_config_path(
            config_path, document["knowledge_root"], "knowledge_root"
        ),
        cache_root=_resolve_config_path(config_path, document["cache_root"], "cache_root"),
        outbox_root=_resolve_config_path(
            config_path, document["outbox_root"], "outbox_root"
        ),
        stage_checkpoint=stage_checkpoint,
    )


def sync_worker_knowledge(
    config: WorkerConfig, *, repo: Path, ref: str | None = None
) -> dict[str, Any]:
    """Import an already-fetched Git ref; this never fetches, pushes, or uses credentials."""

    selected_ref = ref or config.trusted_ref
    if selected_ref != config.trusted_ref:
        raise WorkerCycleError("sync ref differs from the configured trusted maintainer ref")
    report = sync_from_git_ref(
        repo.resolve(strict=True), selected_ref, config.knowledge_root
    )
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
            "generation": champion["generation"],
            "champion_sha256": champion_hash,
        },
    )
    return {"worker_id": config.worker_id, **report}


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


def _checkpoint_from_cache(config: WorkerConfig, champion: Mapping[str, Any]) -> Path:
    artifact = champion.get("artifact")
    if not isinstance(artifact, Mapping):
        raise WorkerCycleError("non-canonical champion has no artifact")
    digest = artifact.get("sha256")
    size = artifact.get("size_bytes")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise WorkerCycleError("champion artifact has an invalid sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise WorkerCycleError("champion artifact has an invalid size")
    bundle = config.cache_root / "champions" / digest
    checkpoint = bundle / "checkpoints" / "checkpoint.zip"
    manifest_path = bundle / "manifest.json"
    try:
        if checkpoint.is_symlink() or manifest_path.is_symlink():
            raise WorkerCycleError("champion cache bundle cannot contain symlinks")
        if checkpoint.stat().st_size != size:
            raise WorkerCycleError("cached champion size differs from champion.json")
        if sha256_file(checkpoint) != digest:
            raise WorkerCycleError("cached champion hash differs from champion.json")
        manifest = _object(json.loads(manifest_path.read_text("utf-8")), "champion manifest")
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerCycleError("champion cache bundle is incomplete") from error
    compatibility = champion.get("compatibility_fingerprint")
    if manifest.get("compatibility_fingerprint") != compatibility:
        raise WorkerCycleError("cached champion manifest compatibility differs")
    inventory = manifest.get("checkpoints")
    if not isinstance(inventory, list) or not any(
        isinstance(entry, Mapping) and entry.get("sha256") == digest for entry in inventory
    ):
        raise WorkerCycleError("cached champion hash is absent from its manifest")
    return checkpoint


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
            "generation",
            "champion_sha256",
        },
        "trusted source",
    )
    if (
        trusted["schema_version"] != 1
        or trusted["trust_tier"] != "maintainer-promoted"
        or trusted["ref"] != config.trusted_ref
        or trusted["generation"] != generation
        or trusted["champion_sha256"] != champion_hash
    ):
        raise WorkerCycleError("champion is not the exact trusted maintainer-promoted snapshot")
    checkpoint: Path | None = None
    if generation != 0:
        checkpoint = _checkpoint_from_cache(config, champion)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "worker_id": config.worker_id,
        "session_id": session_id,
        "parent_champion": {"generation": generation, "sha256": champion_hash},
        "trust_tier": "maintainer-promoted",
        "compatibility_fingerprint": champion.get("compatibility_fingerprint"),
        "resume_checkpoint": str(checkpoint) if checkpoint is not None else None,
        "resume_checkpoint_sha256": (
            champion["artifact"]["sha256"] if generation != 0 else None
        ),
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
    if not isinstance(path_value, str) or not isinstance(digest, str) or not _SHA256.fullmatch(
        digest
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
            "trust_tier",
            "compatibility_fingerprint",
            "resume_checkpoint",
            "resume_checkpoint_sha256",
        },
        "preflight receipt",
    )
    if (
        receipt["schema_version"] != 1
        or receipt["worker_id"] != config.worker_id
        or receipt["session_id"] != session_id
        or receipt["trust_tier"] != "maintainer-promoted"
    ):
        raise WorkerCycleError("preflight receipt belongs to another worker or schema")
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
    checkpoint, checkpoint_hash, checkpoint_step = _latest_checkpoint(
        resolved_manifest, manifest
    )
    contribution = contribution_from_manifest(manifest)
    if contribution["compatibility_fingerprint"] != compatibility:
        raise WorkerCycleError(
            "training manifest compatibility fingerprint is not canonical"
        )
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
