from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from hk_agent.distributed_knowledge import (
    canonical_document_sha256,
    initial_champion,
    portable_compatibility_fingerprint,
)
from hk_agent.worker_cycle import (
    WorkerCycleError,
    finalize_candidate,
    load_worker_config,
    preflight_worker,
    sync_worker_knowledge,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _config(tmp_path: Path, *, stage: bool = True) -> Path:
    path = tmp_path / "config.worker.local.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "worker_id": "pc-um",
            "trusted_ref": "origin/knowledge",
            "knowledge_root": "knowledge",
            "cache_root": "cache",
            "outbox_root": "outbox",
            "stage_checkpoint": stage,
        },
    )
    return path


def _knowledge(tmp_path: Path, champion: dict[str, Any] | None = None) -> None:
    root = tmp_path / "knowledge"
    _write_json(root / "index.json", {"schema_version": 1, "contributions": []})
    _write_json(root / "champion.json", champion or initial_champion())


def _mark_trusted(tmp_path: Path, champion: dict[str, Any] | None = None) -> None:
    selected = champion or initial_champion()
    _write_json(
        tmp_path / "cache" / "trusted-source.json",
        {
            "schema_version": 1,
            "trust_tier": "maintainer-promoted",
            "ref": "origin/knowledge",
            "generation": selected["generation"],
            "champion_sha256": canonical_document_sha256(selected),
        },
    )


def _manifest(tmp_path: Path, checkpoint_bytes: bytes = b"owned checkpoint") -> Path:
    training = tmp_path / "run" / "training"
    checkpoint = training / "checkpoints" / "step-128.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(checkpoint_bytes)
    task_contract = {
        "reward_version": "visual_motion_v1",
        "action_names": ["NOOP", "ATTACK"],
        "config": {"enemy_kill_reward": 2.0},
    }
    provenance = {
        "game_profile": "gog-1.5.78.11833",
        "campaign_origin": {},
        "files": {},
        "observation_dimensions": {"policy": {"height": 90, "width": 160}},
    }
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    manifest = {
        "schema_version": 3,
        "status": "COMPLETED",
        "algorithm": "PPO",
        "seed": 7,
        "reward_version": "visual_motion_v1",
        "completed_timesteps": 128,
        "episode_count": 1,
        "gameplay_run_index": 1,
        "gameplay_run_failures": 0,
        "resumable_checkpoint": True,
        "task_contract": task_contract,
        "provenance": provenance,
        "compatibility_fingerprint": portable_compatibility_fingerprint(
            task_contract, provenance
        ),
        "versions": {"python": "3.12"},
        "checkpoints": [{"step": 128, "path": str(checkpoint), "sha256": digest}],
        "finished_unix_ns": 123456,
    }
    path = training / "manifest.json"
    _write_json(path, manifest)
    return path


def test_initial_preflight_freezes_canonical_parent_without_checkpoint(
    tmp_path: Path,
) -> None:
    _knowledge(tmp_path)
    _mark_trusted(tmp_path)
    config = load_worker_config(_config(tmp_path))
    receipt = preflight_worker(config, "session-1")

    assert receipt["worker_id"] == "pc-um"
    assert receipt["resume_checkpoint"] is None
    assert receipt["parent_champion"] == {
        "generation": 0,
        "sha256": canonical_document_sha256(initial_champion()),
    }
    assert (
        json.loads(
            (tmp_path / "cache" / "preflights" / "session-1.json").read_text("utf-8")
        )
        == receipt
    )


def test_preflight_rejects_locally_copied_champion_without_trusted_sync(
    tmp_path: Path,
) -> None:
    _knowledge(tmp_path)
    config = load_worker_config(_config(tmp_path))

    with pytest.raises(WorkerCycleError, match="sync the trusted maintainer ref"):
        preflight_worker(config, "session-1")


def test_sync_marks_only_the_configured_maintainer_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _knowledge(tmp_path)
    config = load_worker_config(_config(tmp_path))

    def fake_sync(repo: Path, ref: str, knowledge_root: Path) -> dict[str, Any]:
        assert repo == tmp_path.resolve()
        assert ref == "origin/knowledge"
        assert knowledge_root == (tmp_path / "knowledge").resolve()
        return {
            "imported": 0,
            "total": 0,
            "ref": ref,
            "champion_generation": 0,
            "champion_sha256": canonical_document_sha256(initial_champion()),
        }

    monkeypatch.setattr("hk_agent.worker_cycle.sync_from_git_ref", fake_sync)
    report = sync_worker_knowledge(config, repo=tmp_path)
    assert report["champion_generation"] == 0
    trusted = json.loads((tmp_path / "cache" / "trusted-source.json").read_text("utf-8"))
    assert trusted["trust_tier"] == "maintainer-promoted"

    with pytest.raises(WorkerCycleError, match="configured trusted maintainer ref"):
        sync_worker_knowledge(config, repo=tmp_path, ref="origin/friend-candidate")


def test_promoted_champion_is_selected_only_from_verified_cache(tmp_path: Path) -> None:
    checkpoint_bytes = b"promoted owned checkpoint"
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    compatibility = "a" * 64
    initial_hash = canonical_document_sha256(initial_champion())
    champion = {
        "schema_version": 1,
        "generation": 1,
        "candidate_id": "b" * 64,
        "contribution_id": "c" * 64,
        "compatibility_fingerprint": compatibility,
        "artifact": {
            "uri": "https://github.com/example/releases/download/v1/checkpoint.zip",
            "sha256": digest,
            "size_bytes": len(checkpoint_bytes),
            "format": "sb3-zip",
        },
        "parent_champion_sha256": initial_hash,
        "evaluation": {
            "report_sha256": "d" * 64,
            "suite_sha256": "f" * 64,
            "policy_sha256": digest,
            "candidate_id": "b" * 64,
            "parent_champion_sha256": initial_hash,
            "evaluator_commit": "e" * 40,
        },
    }
    _knowledge(tmp_path, champion)
    _mark_trusted(tmp_path, champion)
    bundle = tmp_path / "cache" / "champions" / digest
    checkpoint = bundle / "checkpoints" / "checkpoint.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(checkpoint_bytes)
    _write_json(
        bundle / "manifest.json",
        {
            "compatibility_fingerprint": compatibility,
            "checkpoints": [{"step": 10, "sha256": digest}],
        },
    )

    receipt = preflight_worker(load_worker_config(_config(tmp_path)), "session-1")
    assert receipt["resume_checkpoint"] == str(checkpoint)

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(WorkerCycleError, match="size differs"):
        preflight_worker(load_worker_config(_config(tmp_path)), "session-2")


def test_finalize_is_sanitized_staged_and_idempotent(tmp_path: Path) -> None:
    _knowledge(tmp_path)
    _mark_trusted(tmp_path)
    config = load_worker_config(_config(tmp_path))
    preflight_worker(config, "session-1")
    manifest = _manifest(tmp_path)

    first = finalize_candidate(config, manifest, "session-1")
    second = finalize_candidate(config, manifest, "session-1")

    assert first == second
    bundle = tmp_path / "outbox" / "candidates" / str(first["bundle_id"])
    assert (bundle / "checkpoint.zip").read_bytes() == b"owned checkpoint"
    text = (bundle / "staging.json").read_text("utf-8")
    assert str(tmp_path) not in text
    assert "manifest.json" not in text
    assert "checkpoint_fingerprints" not in text
    contribution = json.loads((bundle / "contribution.json").read_text("utf-8"))
    assert contribution["contribution_id"] == first["contribution"]["contribution_id"]


def test_finalize_rejects_checkpoint_outside_run_directory(tmp_path: Path) -> None:
    _knowledge(tmp_path)
    _mark_trusted(tmp_path)
    config = load_worker_config(_config(tmp_path))
    preflight_worker(config, "session-1")
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text("utf-8"))
    outside = tmp_path / "outside.zip"
    outside.write_bytes(b"owned checkpoint")
    manifest["checkpoints"][0]["path"] = str(outside)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError):
        finalize_candidate(config, manifest_path, "session-1")


def test_finalize_requires_the_exact_frozen_promoted_checkpoint(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text("utf-8"))
    parent_bytes = b"promoted parent"
    parent_digest = hashlib.sha256(parent_bytes).hexdigest()
    initial_hash = canonical_document_sha256(initial_champion())
    champion = {
        "schema_version": 1,
        "generation": 1,
        "candidate_id": "b" * 64,
        "contribution_id": "c" * 64,
        "compatibility_fingerprint": manifest["compatibility_fingerprint"],
        "artifact": {
            "uri": "https://github.com/example/releases/download/v1/checkpoint.zip",
            "sha256": parent_digest,
            "size_bytes": len(parent_bytes),
            "format": "sb3-zip",
        },
        "parent_champion_sha256": initial_hash,
        "evaluation": {
            "report_sha256": "d" * 64,
            "suite_sha256": "f" * 64,
            "policy_sha256": parent_digest,
            "candidate_id": "b" * 64,
            "parent_champion_sha256": initial_hash,
            "evaluator_commit": "e" * 40,
        },
    }
    _knowledge(tmp_path, champion)
    _mark_trusted(tmp_path, champion)
    bundle = tmp_path / "cache" / "champions" / parent_digest
    checkpoint = bundle / "checkpoints" / "checkpoint.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(parent_bytes)
    _write_json(
        bundle / "manifest.json",
        {
            "compatibility_fingerprint": manifest["compatibility_fingerprint"],
            "checkpoints": [{"step": 10, "sha256": parent_digest}],
        },
    )
    config = load_worker_config(_config(tmp_path))
    preflight_worker(config, "session-promoted")

    with pytest.raises(WorkerCycleError, match="did not resume"):
        finalize_candidate(config, manifest_path, "session-promoted")

    manifest["resumed_from"] = {"sha256": parent_digest}
    _write_json(manifest_path, manifest)
    result = finalize_candidate(config, manifest_path, "session-promoted")
    assert result["parent_champion"]["generation"] == 1
