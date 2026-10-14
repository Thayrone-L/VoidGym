import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

KNOWLEDGE_SCHEMA_VERSION = 1
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_CONTRIBUTION_PATH_RE = re.compile(r"^knowledge/contributions/[0-9a-f]{64}\.json$")
_WORKER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SEMANTIC_FILE_HASHES = frozenset(
    {
        "game_executable",
        "mod_dll",
        "bootstrap_plan",
        "readiness_reference",
        "curriculum_source",
        "temporal_actions_source",
        "environment_source",
    }
)
_TRAINING_KEYS = {
    "algorithm",
    "seed",
    "reward_version",
    "status",
    "completed_timesteps",
    "episode_count",
    "gameplay_run_index",
    "gameplay_run_failures",
}


class KnowledgeValidationError(ValueError):
    """Raised when public knowledge does not satisfy the closed schema."""


def canonical_document_sha256(value: object) -> str:
    """Return the stable digest used for registry compare-and-swap operations."""

    return _sha256_json(value)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _expect_dict(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise KnowledgeValidationError(f"{label} must be a JSON object with string keys")
    return value


def _expect_exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise KnowledgeValidationError(
            f"{label} keys must be exactly {sorted(keys)}; got {sorted(value)}"
        )


def _portable_origin(provenance: Mapping[str, Any]) -> dict[str, Any]:
    raw = provenance.get("campaign_origin", {})
    if not isinstance(raw, dict):
        return {}
    result: dict[str, Any] = {}
    for key in ("schema_version", "save_source", "resolved_plan_id", "resolved_plan_sha256"):
        value = raw.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            result[key] = value
    steps = raw.get("resolved_steps")
    if isinstance(steps, list):
        clean_steps: list[dict[str, Any]] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            clean = {
                key: step[key]
                for key in ("button", "lease_ms", "settle_seconds", "label")
                if key in step and isinstance(step[key], (str, int, float))
            }
            clean_steps.append(clean)
        result["resolved_steps"] = clean_steps
    subsequent = raw.get("subsequent_episode_bootstrap")
    if isinstance(subsequent, dict):
        result["subsequent_episode_bootstrap"] = _portable_origin({"campaign_origin": subsequent})
    return result


def _portable_source_hashes(provenance: Mapping[str, Any]) -> dict[str, str]:
    files = provenance.get("files", {})
    source_hashes: dict[str, str] = {}
    if not isinstance(files, dict):
        return source_hashes
    for name, record in sorted(files.items()):
        if name not in _SEMANTIC_FILE_HASHES or not isinstance(record, dict):
            continue
        digest = record.get("sha256")
        if isinstance(digest, str) and _SHA256_RE.fullmatch(digest):
            source_hashes[name] = digest
    return source_hashes


def _fingerprint_from_portable_parts(
    task_contract: Mapping[str, Any],
    *,
    game_profile: str | None,
    campaign_origin: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    observation_dimensions: Mapping[str, Any],
) -> str:
    return _sha256_json(
        {
            "task_contract": dict(task_contract),
            "game_profile": game_profile,
            "campaign_origin": dict(campaign_origin),
            "source_hashes": dict(source_hashes),
            "observation_dimensions": dict(observation_dimensions),
        }
    )


def portable_compatibility_fingerprint(
    task_contract: Mapping[str, Any], provenance: Mapping[str, Any]
) -> str:
    """Hash only cross-machine semantics, never absolute paths or save/screen data."""

    dimensions = provenance.get("observation_dimensions", {})
    if not isinstance(dimensions, dict):
        dimensions = {}
    game_profile = provenance.get("game_profile")
    return _fingerprint_from_portable_parts(
        task_contract,
        game_profile=game_profile if isinstance(game_profile, str) else None,
        campaign_origin=_portable_origin(provenance),
        source_hashes=_portable_source_hashes(provenance),
        observation_dimensions=dimensions,
    )


def contribution_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Create a public, allowlisted summary from a private training manifest."""

    task_contract = _expect_dict(manifest.get("task_contract"), "manifest.task_contract")
    provenance = _expect_dict(manifest.get("provenance"), "manifest.provenance")
    checkpoints: list[dict[str, Any]] = []
    for checkpoint in manifest.get("checkpoints", []):
        if not isinstance(checkpoint, dict):
            continue
        step, digest = checkpoint.get("step"), checkpoint.get("sha256")
        if (
            isinstance(step, int)
            and not isinstance(step, bool)
            and isinstance(digest, str)
            and _SHA256_RE.fullmatch(digest)
        ):
            checkpoints.append({"step": step, "sha256": digest})

    versions = manifest.get("versions", {})
    clean_versions = (
        {str(key): value for key, value in versions.items() if isinstance(value, str)}
        if isinstance(versions, dict)
        else {}
    )
    body: dict[str, Any] = {
        "schema_version": KNOWLEDGE_SCHEMA_VERSION,
        "compatibility_fingerprint": portable_compatibility_fingerprint(task_contract, provenance),
        "training": {
            "algorithm": manifest.get("algorithm"),
            "seed": manifest.get("seed"),
            "reward_version": manifest.get("reward_version"),
            "status": manifest.get("status"),
            "completed_timesteps": manifest.get("completed_timesteps", 0),
            "episode_count": manifest.get("episode_count", 0),
            "gameplay_run_index": manifest.get("gameplay_run_index", 0),
            "gameplay_run_failures": manifest.get("gameplay_run_failures", 0),
        },
        "task_contract": task_contract,
        "portable_provenance": {
            "game_profile": provenance.get("game_profile"),
            "campaign_origin": _portable_origin(provenance),
            "observation_dimensions": provenance.get("observation_dimensions", {}),
            "source_hashes": _portable_source_hashes(provenance),
        },
        "versions": clean_versions,
        "checkpoint_fingerprints": sorted(checkpoints, key=lambda item: item["step"]),
    }
    contribution_id = _sha256_json(body)
    contribution = {"contribution_id": contribution_id, **body}
    validate_contribution(contribution)
    return contribution


def validate_contribution(value: object) -> dict[str, Any]:
    contribution = _expect_dict(value, "contribution")
    _expect_exact_keys(
        contribution,
        {
            "contribution_id",
            "schema_version",
            "compatibility_fingerprint",
            "training",
            "task_contract",
            "portable_provenance",
            "versions",
            "checkpoint_fingerprints",
        },
        "contribution",
    )
    if contribution["schema_version"] != KNOWLEDGE_SCHEMA_VERSION:
        raise KnowledgeValidationError("unsupported contribution schema_version")
    for key in ("contribution_id", "compatibility_fingerprint"):
        if not isinstance(contribution[key], str) or not _SHA256_RE.fullmatch(contribution[key]):
            raise KnowledgeValidationError(f"contribution.{key} must be lowercase SHA-256")
    training = _expect_dict(contribution["training"], "contribution.training")
    _expect_exact_keys(training, _TRAINING_KEYS, "contribution.training")
    for key in ("algorithm", "reward_version", "status"):
        if not isinstance(training[key], str):
            raise KnowledgeValidationError(f"training.{key} must be a string")
    for key in (
        "seed",
        "completed_timesteps",
        "episode_count",
        "gameplay_run_index",
        "gameplay_run_failures",
    ):
        if not isinstance(training[key], int) or isinstance(training[key], bool):
            raise KnowledgeValidationError(f"training.{key} must be an integer")
    task_contract = _expect_dict(contribution["task_contract"], "contribution.task_contract")
    portable = _expect_dict(contribution["portable_provenance"], "portable_provenance")
    _expect_exact_keys(
        portable,
        {"game_profile", "campaign_origin", "observation_dimensions", "source_hashes"},
        "portable_provenance",
    )
    game_profile = portable["game_profile"]
    if game_profile is not None and not isinstance(game_profile, str):
        raise KnowledgeValidationError("portable_provenance.game_profile must be a string or null")
    origin = _expect_dict(portable["campaign_origin"], "portable_provenance.campaign_origin")
    dimensions = _expect_dict(
        portable["observation_dimensions"], "portable_provenance.observation_dimensions"
    )
    source_hashes = _expect_dict(portable["source_hashes"], "portable_provenance.source_hashes")
    for name, digest in source_hashes.items():
        if name not in _SEMANTIC_FILE_HASHES:
            raise KnowledgeValidationError("portable_provenance contains a non-semantic file hash")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise KnowledgeValidationError("portable source hash is invalid")
    versions = _expect_dict(contribution["versions"], "contribution.versions")
    if not all(isinstance(value, str) for value in versions.values()):
        raise KnowledgeValidationError("contribution versions must be strings")
    expected_fingerprint = _fingerprint_from_portable_parts(
        task_contract,
        game_profile=game_profile,
        campaign_origin=origin,
        source_hashes=source_hashes,
        observation_dimensions=dimensions,
    )
    if contribution["compatibility_fingerprint"] != expected_fingerprint:
        raise KnowledgeValidationError(
            "compatibility_fingerprint does not match portable semantics"
        )
    fingerprints = contribution["checkpoint_fingerprints"]
    if not isinstance(fingerprints, list):
        raise KnowledgeValidationError("checkpoint_fingerprints must be a list")
    for item in fingerprints:
        record = _expect_dict(item, "checkpoint fingerprint")
        _expect_exact_keys(record, {"step", "sha256"}, "checkpoint fingerprint")
        if not isinstance(record["step"], int) or isinstance(record["step"], bool):
            raise KnowledgeValidationError("checkpoint step must be an integer")
        if not isinstance(record["sha256"], str) or not _SHA256_RE.fullmatch(record["sha256"]):
            raise KnowledgeValidationError("checkpoint sha256 is invalid")
    body = dict(contribution)
    claimed_id = body.pop("contribution_id")
    if _sha256_json(body) != claimed_id:
        raise KnowledgeValidationError("contribution_id does not match canonical content")
    forbidden = re.compile(r'(?i)([a-z]:\\|/users/|\\users\\|\.png|\.dat(?:\b|\.)|"path"\s*:)')
    if forbidden.search(_canonical_bytes(contribution).decode("utf-8")):
        raise KnowledgeValidationError(
            "contribution contains a local path, save, or screen reference"
        )
    return contribution


def validate_index(value: object) -> dict[str, Any]:
    index = _expect_dict(value, "knowledge index")
    _expect_exact_keys(index, {"schema_version", "contributions"}, "knowledge index")
    if index["schema_version"] != KNOWLEDGE_SCHEMA_VERSION:
        raise KnowledgeValidationError("unsupported knowledge index schema_version")
    entries = index["contributions"]
    if not isinstance(entries, list):
        raise KnowledgeValidationError("knowledge index contributions must be a list")
    if len(entries) > 1_000:
        raise KnowledgeValidationError("knowledge index exceeds 1000 contributions")
    seen: set[str] = set()
    for raw in entries:
        entry = _expect_dict(raw, "knowledge index entry")
        _expect_exact_keys(
            entry,
            {"contribution_id", "path", "sha256", "compatibility_fingerprint"},
            "knowledge index entry",
        )
        contribution_id = entry["contribution_id"]
        if not isinstance(contribution_id, str) or not _SHA256_RE.fullmatch(contribution_id):
            raise KnowledgeValidationError("index contribution_id is invalid")
        expected_path = f"knowledge/contributions/{contribution_id}.json"
        if entry["path"] != expected_path or not _CONTRIBUTION_PATH_RE.fullmatch(entry["path"]):
            raise KnowledgeValidationError("index contribution path is not canonical")
        for key in ("sha256", "compatibility_fingerprint"):
            if not isinstance(entry[key], str) or not _SHA256_RE.fullmatch(entry[key]):
                raise KnowledgeValidationError(f"index {key} is invalid")
        if contribution_id in seen:
            raise KnowledgeValidationError("knowledge index contains duplicate contributions")
        seen.add(contribution_id)
    return index


def _validate_artifact(value: object, label: str = "artifact") -> dict[str, Any]:
    artifact = _expect_dict(value, label)
    _expect_exact_keys(artifact, {"uri", "sha256", "size_bytes", "format"}, label)
    uri = artifact["uri"]
    if not isinstance(uri, str) or len(uri) > 2_048:
        raise KnowledgeValidationError(f"{label}.uri must be an HTTPS URL")
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise KnowledgeValidationError(
            f"{label}.uri must be HTTPS without credentials, query, or fragment"
        )
    digest = artifact["sha256"]
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise KnowledgeValidationError(f"{label}.sha256 must be lowercase SHA-256")
    size = artifact["size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= 2 * 1024**3:
        raise KnowledgeValidationError(f"{label}.size_bytes must be between 1 and 2 GiB")
    if artifact["format"] not in {"sb3-zip", "safetensors"}:
        raise KnowledgeValidationError(f"{label}.format is not supported")
    return artifact


def _validate_parent_champion(value: object) -> dict[str, Any]:
    parent = _expect_dict(value, "candidate.parent_champion")
    _expect_exact_keys(parent, {"generation", "sha256"}, "candidate.parent_champion")
    generation = parent["generation"]
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
        raise KnowledgeValidationError("candidate parent generation must be non-negative")
    digest = parent["sha256"]
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise KnowledgeValidationError("candidate parent sha256 must be lowercase SHA-256")
    return parent


def candidate_from_contribution(
    contribution: Mapping[str, Any],
    *,
    worker_id: str,
    artifact_uri: str,
    artifact_sha256: str,
    artifact_size_bytes: int,
    artifact_format: str,
    trust_tier: str,
    parent_generation: int,
    parent_sha256: str,
) -> dict[str, Any]:
    """Build a content-addressed candidate without reading its model artifact."""

    clean_contribution = validate_contribution(dict(contribution))
    body: dict[str, Any] = {
        "schema_version": KNOWLEDGE_SCHEMA_VERSION,
        "worker_id": worker_id,
        "trust_tier": trust_tier,
        "contribution_id": clean_contribution["contribution_id"],
        "compatibility_fingerprint": clean_contribution["compatibility_fingerprint"],
        "parent_champion": {"generation": parent_generation, "sha256": parent_sha256},
        "artifact": {
            "uri": artifact_uri,
            "sha256": artifact_sha256,
            "size_bytes": artifact_size_bytes,
            "format": artifact_format,
        },
    }
    candidate = {"candidate_id": _sha256_json(body), **body}
    return validate_candidate(candidate)


def validate_candidate(value: object) -> dict[str, Any]:
    candidate = _expect_dict(value, "candidate")
    _expect_exact_keys(
        candidate,
        {
            "candidate_id",
            "schema_version",
            "worker_id",
            "trust_tier",
            "contribution_id",
            "compatibility_fingerprint",
            "parent_champion",
            "artifact",
        },
        "candidate",
    )
    if candidate["schema_version"] != KNOWLEDGE_SCHEMA_VERSION:
        raise KnowledgeValidationError("unsupported candidate schema_version")
    worker_id = candidate["worker_id"]
    if not isinstance(worker_id, str) or not _WORKER_ID_RE.fullmatch(worker_id):
        raise KnowledgeValidationError("candidate.worker_id is not a safe public identifier")
    if candidate["trust_tier"] not in {"owned", "quarantined", "external-untrusted"}:
        raise KnowledgeValidationError("candidate.trust_tier is not supported")
    for key in ("candidate_id", "contribution_id", "compatibility_fingerprint"):
        if not isinstance(candidate[key], str) or not _SHA256_RE.fullmatch(candidate[key]):
            raise KnowledgeValidationError(f"candidate.{key} must be lowercase SHA-256")
    _validate_parent_champion(candidate["parent_champion"])
    _validate_artifact(candidate["artifact"], "candidate.artifact")
    body = dict(candidate)
    claimed_id = body.pop("candidate_id")
    if _sha256_json(body) != claimed_id:
        raise KnowledgeValidationError("candidate_id does not match canonical content")
    return candidate


def initial_champion() -> dict[str, Any]:
    return {
        "schema_version": KNOWLEDGE_SCHEMA_VERSION,
        "generation": 0,
        "candidate_id": None,
        "contribution_id": None,
        "compatibility_fingerprint": None,
        "artifact": None,
        "parent_champion_sha256": None,
        "evaluation": None,
    }


def validate_champion(value: object) -> dict[str, Any]:
    champion = _expect_dict(value, "champion")
    keys = {
        "schema_version",
        "generation",
        "candidate_id",
        "contribution_id",
        "compatibility_fingerprint",
        "artifact",
        "parent_champion_sha256",
        "evaluation",
    }
    _expect_exact_keys(champion, keys, "champion")
    if champion["schema_version"] != KNOWLEDGE_SCHEMA_VERSION:
        raise KnowledgeValidationError("unsupported champion schema_version")
    generation = champion["generation"]
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
        raise KnowledgeValidationError("champion.generation must be non-negative")
    nullable_hashes = (
        "candidate_id",
        "contribution_id",
        "compatibility_fingerprint",
        "parent_champion_sha256",
    )
    for key in nullable_hashes:
        item = champion[key]
        if item is not None and (not isinstance(item, str) or not _SHA256_RE.fullmatch(item)):
            raise KnowledgeValidationError(f"champion.{key} must be null or lowercase SHA-256")
    if generation == 0:
        if champion != initial_champion():
            raise KnowledgeValidationError("generation-zero champion must be canonical and empty")
    else:
        if any(champion[key] is None for key in nullable_hashes) or champion["artifact"] is None:
            raise KnowledgeValidationError("promoted champion fields cannot be null")
        _validate_artifact(champion["artifact"], "champion.artifact")
        evaluation = _expect_dict(champion["evaluation"], "champion.evaluation")
        _expect_exact_keys(
            evaluation,
            {
                "report_sha256",
                "suite_sha256",
                "policy_sha256",
                "candidate_id",
                "parent_champion_sha256",
                "evaluator_commit",
            },
            "champion.evaluation",
        )
        for key in (
            "report_sha256",
            "suite_sha256",
            "policy_sha256",
            "candidate_id",
            "parent_champion_sha256",
        ):
            if not isinstance(evaluation[key], str) or not _SHA256_RE.fullmatch(evaluation[key]):
                raise KnowledgeValidationError(f"champion evaluation {key} is invalid")
        commit = evaluation["evaluator_commit"]
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise KnowledgeValidationError("champion evaluator_commit must be a full Git SHA-1")
        if evaluation["policy_sha256"] != champion["artifact"]["sha256"]:
            raise KnowledgeValidationError(
                "champion evaluation policy hash does not match artifact"
            )
        if evaluation["candidate_id"] != champion["candidate_id"]:
            raise KnowledgeValidationError("champion evaluation candidate binding differs")
        if evaluation["parent_champion_sha256"] != champion["parent_champion_sha256"]:
            raise KnowledgeValidationError("champion evaluation parent binding differs")
    return champion


def promote_candidate(
    champion: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    expected_generation: int,
    expected_champion_sha256: str,
    evaluation_report_sha256: str,
    evaluation_suite_sha256: str,
    evaluator_commit: str,
) -> dict[str, Any]:
    """Promote locally using generation+digest CAS; never read candidate model bytes."""

    current = validate_champion(dict(champion))
    clean_candidate = validate_candidate(dict(candidate))
    current_sha256 = canonical_document_sha256(current)
    parent = clean_candidate["parent_champion"]
    if (
        current["candidate_id"] == clean_candidate["candidate_id"]
        and current["generation"] == parent["generation"] + 1
        and current["parent_champion_sha256"] == parent["sha256"]
    ):
        evaluation = current.get("evaluation")
        if not isinstance(evaluation, Mapping) or (
            evaluation.get("report_sha256") != evaluation_report_sha256
            or evaluation.get("suite_sha256") != evaluation_suite_sha256
            or evaluation.get("evaluator_commit") != evaluator_commit
        ):
            raise KnowledgeValidationError(
                "idempotent promotion evidence differs from the promoted champion"
            )
        return current
    if expected_generation != current["generation"] or expected_champion_sha256 != current_sha256:
        raise KnowledgeValidationError("champion compare-and-swap precondition failed")
    if parent["generation"] != current["generation"] or parent["sha256"] != current_sha256:
        raise KnowledgeValidationError("candidate was not trained from the current champion")
    current_fingerprint = current["compatibility_fingerprint"]
    if current_fingerprint is not None and (
        clean_candidate["compatibility_fingerprint"] != current_fingerprint
    ):
        raise KnowledgeValidationError("candidate is incompatible with the current champion")
    if clean_candidate["trust_tier"] == "external-untrusted":
        raise KnowledgeValidationError("external-untrusted candidate must be quarantined first")
    promoted = {
        "schema_version": KNOWLEDGE_SCHEMA_VERSION,
        "generation": current["generation"] + 1,
        "candidate_id": clean_candidate["candidate_id"],
        "contribution_id": clean_candidate["contribution_id"],
        "compatibility_fingerprint": clean_candidate["compatibility_fingerprint"],
        "artifact": dict(clean_candidate["artifact"]),
        "parent_champion_sha256": current_sha256,
        "evaluation": {
            "report_sha256": evaluation_report_sha256,
            "suite_sha256": evaluation_suite_sha256,
            "policy_sha256": clean_candidate["artifact"]["sha256"],
            "candidate_id": clean_candidate["candidate_id"],
            "parent_champion_sha256": current_sha256,
            "evaluator_commit": evaluator_commit,
        },
    }
    return validate_champion(promoted)


def _load_json(path: Path) -> object:
    if not path.is_file() or path.is_symlink():
        raise KnowledgeValidationError(f"expected a regular JSON file: {path}")
    if path.stat().st_size > MAX_DOCUMENT_BYTES:
        raise KnowledgeValidationError(f"JSON document exceeds size limit: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise KnowledgeValidationError(f"cannot read JSON document {path}: {error}") from error


def validate_knowledge_tree(root: Path) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    index = validate_index(_load_json(resolved / "index.json"))
    for entry in index["contributions"]:
        relative = PurePosixPath(entry["path"])
        path = resolved / Path(*relative.parts[1:])
        contribution = validate_contribution(_load_json(path))
        if contribution["contribution_id"] != entry["contribution_id"]:
            raise KnowledgeValidationError("index and contribution IDs differ")
        if contribution["compatibility_fingerprint"] != entry["compatibility_fingerprint"]:
            raise KnowledgeValidationError("index and contribution fingerprints differ")
        if hashlib.sha256(_canonical_bytes(contribution)).hexdigest() != entry["sha256"]:
            raise KnowledgeValidationError("indexed contribution content hash differs")
    champion_path = resolved / "champion.json"
    if champion_path.exists():
        validate_champion(_load_json(champion_path))
    return index


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_bytes(value) + b"\n")
    temporary.replace(path)


def export_contribution(manifest_path: Path, knowledge_root: Path) -> dict[str, Any]:
    manifest = _expect_dict(_load_json(manifest_path), "training manifest")
    contribution = contribution_from_manifest(manifest)
    root = knowledge_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    index: dict[str, Any]
    if index_path.exists():
        index = validate_index(_load_json(index_path))
    else:
        index = {"schema_version": KNOWLEDGE_SCHEMA_VERSION, "contributions": []}
    contribution_id = contribution["contribution_id"]
    path = root / "contributions" / f"{contribution_id}.json"
    _atomic_json(path, contribution)
    entry = {
        "contribution_id": contribution_id,
        "path": f"knowledge/contributions/{contribution_id}.json",
        "sha256": hashlib.sha256(_canonical_bytes(contribution)).hexdigest(),
        "compatibility_fingerprint": contribution["compatibility_fingerprint"],
    }
    retained = [
        item for item in index["contributions"] if item["contribution_id"] != contribution_id
    ]
    index["contributions"] = sorted(retained + [entry], key=lambda item: item["contribution_id"])
    _atomic_json(index_path, index)
    validate_knowledge_tree(root)
    return entry


def export_candidate(
    contribution_path: Path,
    output_root: Path,
    *,
    worker_id: str,
    artifact_uri: str,
    artifact_sha256: str,
    artifact_size_bytes: int,
    artifact_format: str,
    trust_tier: str,
    parent_generation: int,
    parent_sha256: str,
) -> dict[str, Any]:
    contribution = _expect_dict(_load_json(contribution_path), "contribution")
    candidate = candidate_from_contribution(
        contribution,
        worker_id=worker_id,
        artifact_uri=artifact_uri,
        artifact_sha256=artifact_sha256,
        artifact_size_bytes=artifact_size_bytes,
        artifact_format=artifact_format,
        trust_tier=trust_tier,
        parent_generation=parent_generation,
        parent_sha256=parent_sha256,
    )
    output = output_root.resolve() / "outbox" / "candidates" / f"{candidate['candidate_id']}.json"
    _atomic_json(output, candidate)
    return {"candidate_id": candidate["candidate_id"], "path": str(output)}


def validate_champion_file(path: Path) -> dict[str, Any]:
    champion = validate_champion(_load_json(path))
    return {
        "generation": champion["generation"],
        "sha256": canonical_document_sha256(champion),
        "candidate_id": champion["candidate_id"],
    }


def promote_candidate_file(
    champion_path: Path,
    candidate_path: Path,
    *,
    expected_generation: int,
    expected_champion_sha256: str,
    evaluation_report_sha256: str,
    evaluation_suite_sha256: str,
    evaluator_commit: str,
) -> dict[str, Any]:
    current = _expect_dict(_load_json(champion_path), "champion")
    candidate = _expect_dict(_load_json(candidate_path), "candidate")
    promoted = promote_candidate(
        current,
        candidate,
        expected_generation=expected_generation,
        expected_champion_sha256=expected_champion_sha256,
        evaluation_report_sha256=evaluation_report_sha256,
        evaluation_suite_sha256=evaluation_suite_sha256,
        evaluator_commit=evaluator_commit,
    )
    _atomic_json(champion_path, promoted)
    return {
        "generation": promoted["generation"],
        "sha256": canonical_document_sha256(promoted),
        "candidate_id": promoted["candidate_id"],
    }


def _validate_git_ref(ref: str) -> None:
    if not _REF_RE.fullmatch(ref) or ".." in ref or "//" in ref or ref.endswith("/") or "@{" in ref:
        raise KnowledgeValidationError("unsafe Git ref")


def _git_show(repo: Path, ref: str, path: str) -> bytes:
    if path not in {"knowledge/index.json", "knowledge/champion.json"} and not (
        _CONTRIBUTION_PATH_RE.fullmatch(path)
    ):
        raise KnowledgeValidationError("unsafe knowledge path")
    try:
        result = subprocess.run(
            ["git", "-C", str(repo.resolve()), "show", f"{ref}:{path}"],
            check=False,
            capture_output=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as error:
        raise KnowledgeValidationError(f"git show timed out for {path}") from error
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise KnowledgeValidationError(f"git show failed for {path}: {message}")
    if len(result.stdout) > MAX_DOCUMENT_BYTES:
        raise KnowledgeValidationError(f"Git knowledge document exceeds size limit: {path}")
    return result.stdout


def sync_from_git_ref(repo: Path, ref: str, knowledge_root: Path) -> dict[str, Any]:
    """Import a validated branch snapshot without checkout, merge, hooks, or execution."""

    _validate_git_ref(ref)
    try:
        remote_index_raw = _git_show(repo, ref, "knowledge/index.json")
        remote_index = validate_index(json.loads(remote_index_raw.decode("utf-8")))
        remote_champion_raw = _git_show(repo, ref, "knowledge/champion.json")
        remote_champion = validate_champion(json.loads(remote_champion_raw.decode("utf-8")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise KnowledgeValidationError("remote knowledge index is invalid JSON") from error

    staged: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for entry in remote_index["contributions"]:
        raw = _git_show(repo, ref, entry["path"])
        try:
            contribution = validate_contribution(json.loads(raw.decode("utf-8")))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise KnowledgeValidationError("remote contribution is invalid JSON") from error
        canonical_hash = hashlib.sha256(_canonical_bytes(contribution)).hexdigest()
        if canonical_hash != entry["sha256"]:
            raise KnowledgeValidationError("remote contribution does not match its index hash")
        if contribution["contribution_id"] != entry["contribution_id"]:
            raise KnowledgeValidationError("remote contribution does not match its index ID")
        if contribution["compatibility_fingerprint"] != entry["compatibility_fingerprint"]:
            raise KnowledgeValidationError("remote contribution has a mismatched fingerprint")
        staged.append((entry, contribution))

    root = knowledge_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    local_index_path = root / "index.json"
    local_champion_path = root / "champion.json"
    if local_champion_path.exists():
        local_champion = validate_champion(_load_json(local_champion_path))
        local_generation = local_champion["generation"]
        remote_generation = remote_champion["generation"]
        if remote_generation < local_generation:
            raise KnowledgeValidationError("remote champion generation would roll back local state")
        if remote_generation == local_generation and remote_champion != local_champion:
            raise KnowledgeValidationError("champion fork detected at the same generation")
        if remote_generation == local_generation + 1 and (
            remote_champion["parent_champion_sha256"] != canonical_document_sha256(local_champion)
        ):
            raise KnowledgeValidationError("remote champion does not descend from local champion")
        if remote_generation > local_generation + 1:
            raise KnowledgeValidationError(
                "remote champion skipped generations; verify the promotion chain explicitly"
            )
    local_index: dict[str, Any]
    if local_index_path.exists():
        local_index = validate_index(_load_json(local_index_path))
    else:
        local_index = {"schema_version": KNOWLEDGE_SCHEMA_VERSION, "contributions": []}
    entries = {item["contribution_id"]: item for item in local_index["contributions"]}
    for entry, contribution in staged:
        existing = entries.get(entry["contribution_id"])
        if existing is not None and existing != entry:
            raise KnowledgeValidationError("immutable contribution ID has conflicting content")
        _atomic_json(root / "contributions" / f"{entry['contribution_id']}.json", contribution)
        entries[entry["contribution_id"]] = entry
    merged = {
        "schema_version": KNOWLEDGE_SCHEMA_VERSION,
        "contributions": sorted(entries.values(), key=lambda item: item["contribution_id"]),
    }
    _atomic_json(local_champion_path, remote_champion)
    _atomic_json(local_index_path, merged)
    validate_knowledge_tree(root)
    return {
        "imported": len(staged),
        "total": len(entries),
        "ref": ref,
        "champion_generation": remote_champion["generation"],
        "champion_sha256": canonical_document_sha256(remote_champion),
    }
