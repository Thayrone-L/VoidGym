from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
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
    prepare_publication,
    sync_worker_knowledge,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _git(repo: Path, *arguments: str) -> str:
    executable = "git"
    if os.name == "nt":
        installed = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git/cmd/git.exe"
        if installed.is_file():
            executable = str(installed)
    result = subprocess.run(
        [executable, "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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


def _mark_trusted(
    tmp_path: Path,
    champion: dict[str, Any] | None = None,
    *,
    source_commit: str = "a" * 40,
) -> None:
    selected = champion or initial_champion()
    _write_json(
        tmp_path / "cache" / "trusted-source.json",
        {
            "schema_version": 1,
            "trust_tier": "maintainer-promoted",
            "ref": "origin/knowledge",
            "source_commit": source_commit,
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
        "compatibility_fingerprint": portable_compatibility_fingerprint(task_contract, provenance),
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
    assert receipt["warm_start_policy"] is None
    assert receipt["parent_champion"] == {
        "generation": 0,
        "sha256": canonical_document_sha256(initial_champion()),
    }
    assert (
        json.loads((tmp_path / "cache" / "preflights" / "session-1.json").read_text("utf-8"))
        == receipt
    )


def test_preflight_rejects_locally_copied_champion_without_trusted_sync(
    tmp_path: Path,
) -> None:
    _knowledge(tmp_path)
    config = load_worker_config(_config(tmp_path))

    with pytest.raises(WorkerCycleError, match="sync the trusted maintainer ref"):
        preflight_worker(config, "session-1")


def test_worker_config_accepts_only_the_maintainer_knowledge_ref(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    document = json.loads(config_path.read_text("utf-8"))
    document["trusted_ref"] = "origin/friend-candidate"
    _write_json(config_path, document)

    with pytest.raises(WorkerCycleError, match="exactly origin/knowledge"):
        load_worker_config(config_path)


def test_sync_marks_only_the_configured_maintainer_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _knowledge(tmp_path)
    config = load_worker_config(_config(tmp_path))

    def fake_sync(repo: Path, ref: str, knowledge_root: Path) -> dict[str, Any]:
        assert repo == tmp_path.resolve()
        assert ref == "1" * 40
        assert knowledge_root == (tmp_path / "knowledge").resolve()
        return {
            "imported": 0,
            "total": 0,
            "ref": ref,
            "champion_generation": 0,
            "champion_sha256": canonical_document_sha256(initial_champion()),
        }

    monkeypatch.setattr("hk_agent.worker_cycle.sync_from_git_ref", fake_sync)
    monkeypatch.setattr("hk_agent.worker_cycle._git", lambda *args, **kwargs: "1" * 40)
    report = sync_worker_knowledge(config, repo=tmp_path)
    assert report["champion_generation"] == 0
    trusted = json.loads((tmp_path / "cache" / "trusted-source.json").read_text("utf-8"))
    assert trusted["trust_tier"] == "maintainer-promoted"

    with pytest.raises(WorkerCycleError, match="configured trusted maintainer ref"):
        sync_worker_knowledge(config, repo=tmp_path, ref="origin/friend-candidate")


def test_promoted_champion_is_selected_only_as_safe_warm_start(tmp_path: Path) -> None:
    policy_bytes = b"safe tensor fixture"
    digest = hashlib.sha256(policy_bytes).hexdigest()
    compatibility = "a" * 64
    initial_hash = canonical_document_sha256(initial_champion())
    champion = {
        "schema_version": 1,
        "generation": 1,
        "candidate_id": "b" * 64,
        "contribution_id": "c" * 64,
        "compatibility_fingerprint": compatibility,
        "artifact": {
            "uri": "https://github.com/example/releases/download/v1/policy.safetensors",
            "sha256": digest,
            "size_bytes": len(policy_bytes),
            "format": "safetensors",
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
    policy = bundle / "policies" / "policy.safetensors"
    policy.parent.mkdir(parents=True)
    policy.write_bytes(policy_bytes)
    _write_json(
        bundle / "manifest.json",
        {
            "compatibility_fingerprint": compatibility,
            "portable_policies": [
                {"step": 10, "sha256": digest, "format": "safetensors"}
            ],
        },
    )

    receipt = preflight_worker(load_worker_config(_config(tmp_path)), "session-1")
    assert receipt["resume_checkpoint"] is None
    assert receipt["warm_start_policy"] == str(policy)
    assert receipt["warm_start_policy_sha256"] == digest

    policy.write_bytes(b"tampered")
    with pytest.raises(WorkerCycleError, match="policy size differs"):
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


def test_finalize_rejects_tampered_distributed_resume_receipt(tmp_path: Path) -> None:
    _knowledge(tmp_path)
    _mark_trusted(tmp_path)
    config = load_worker_config(_config(tmp_path))
    preflight_worker(config, "session-1")
    receipt_path = tmp_path / "cache" / "preflights" / "session-1.json"
    receipt = json.loads(receipt_path.read_text("utf-8"))
    receipt["resume_checkpoint"] = str(tmp_path / "attacker.zip")
    receipt["resume_checkpoint_sha256"] = "f" * 64
    _write_json(receipt_path, receipt)

    with pytest.raises(WorkerCycleError, match="cannot authorize an SB3 resume"):
        finalize_candidate(config, _manifest(tmp_path), "session-1")


def test_finalize_revalidates_warm_start_policy_after_preflight(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text("utf-8"))
    policy_bytes = b"safe tensor fixture"
    digest = hashlib.sha256(policy_bytes).hexdigest()
    initial_hash = canonical_document_sha256(initial_champion())
    champion = {
        "schema_version": 1,
        "generation": 1,
        "candidate_id": "b" * 64,
        "contribution_id": "c" * 64,
        "compatibility_fingerprint": manifest["compatibility_fingerprint"],
        "artifact": {
            "uri": "https://github.com/example/releases/download/v1/policy.safetensors",
            "sha256": digest,
            "size_bytes": len(policy_bytes),
            "format": "safetensors",
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
    policy = bundle / "policies" / "policy.safetensors"
    policy.parent.mkdir(parents=True)
    policy.write_bytes(policy_bytes)
    _write_json(
        bundle / "manifest.json",
        {
            "compatibility_fingerprint": manifest["compatibility_fingerprint"],
            "portable_policies": [
                {"step": 10, "sha256": digest, "format": "safetensors"}
            ],
        },
    )
    config = load_worker_config(_config(tmp_path))
    preflight_worker(config, "session-promoted")
    policy.write_bytes(b"tampered after preflight")

    with pytest.raises(WorkerCycleError, match="integrity revalidation"):
        finalize_candidate(config, manifest_path, "session-promoted")


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
            "uri": "https://github.com/example/releases/download/v1/policy.safetensors",
            "sha256": parent_digest,
            "size_bytes": len(parent_bytes),
            "format": "safetensors",
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
    policy = bundle / "policies" / "policy.safetensors"
    policy.parent.mkdir(parents=True)
    policy.write_bytes(parent_bytes)
    _write_json(
        bundle / "manifest.json",
        {
            "compatibility_fingerprint": manifest["compatibility_fingerprint"],
            "portable_policies": [
                {"step": 10, "sha256": parent_digest, "format": "safetensors"}
            ],
        },
    )
    config = load_worker_config(_config(tmp_path))
    preflight_worker(config, "session-promoted")

    with pytest.raises(WorkerCycleError, match="did not warm-start"):
        finalize_candidate(config, manifest_path, "session-promoted")

    manifest["warm_started_from"] = {
        "sha256": parent_digest,
        "format": "safetensors",
    }
    _write_json(manifest_path, manifest)
    result = finalize_candidate(config, manifest_path, "session-promoted")
    assert result["parent_champion"]["generation"] == 1


def test_prepare_publication_uses_isolated_immutable_branch(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _knowledge(repository)
    _git(repository, "add", "knowledge")
    _git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "trusted knowledge",
    )
    trusted_commit = _git(repository, "rev-parse", "HEAD")
    _git(repository, "update-ref", "refs/remotes/origin/knowledge", trusted_commit)

    config_path = tmp_path / "worker.json"
    _write_json(
        config_path,
        {
            "schema_version": 1,
            "worker_id": "pc-um",
            "trusted_ref": "origin/knowledge",
            "knowledge_root": str(repository / "knowledge"),
            "cache_root": str(tmp_path / "cache"),
            "outbox_root": str(tmp_path / "outbox"),
            "stage_checkpoint": False,
        },
    )
    config = load_worker_config(config_path)
    _mark_trusted(tmp_path, source_commit=trusted_commit)
    preflight_worker(config, "session-1")
    staging = finalize_candidate(config, _manifest(tmp_path), "session-1")
    main_before = _git(repository, "rev-parse", "HEAD")

    prepared = prepare_publication(config, repo=repository, bundle_id=str(staging["bundle_id"]))
    repeated = prepare_publication(config, repo=repository, bundle_id=str(staging["bundle_id"]))

    assert prepared == repeated
    assert prepared["source_commit"] == trusted_commit
    assert prepared["branch"].startswith("codex/knowledge-pc-um-")
    assert _git(repository, "rev-parse", "HEAD") == main_before
    assert _git(repository, "branch", "--show-current") == "main"
    worktree = Path(str(prepared["worktree"]))
    assert set(_git(worktree, "diff", "--name-only", f"{trusted_commit}..HEAD").splitlines()) == {
        "knowledge/index.json",
        f"knowledge/contributions/{prepared['contribution_id']}.json",
    }
    assert not (worktree / "checkpoint.zip").exists()
    assert not (worktree / "staging.json").exists()


def test_prepare_publication_rolls_back_invalid_trusted_snapshot(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _knowledge(repository)
    trusted_copy = tmp_path / "trusted-knowledge"
    shutil.copytree(repository / "knowledge", trusted_copy)
    _write_json(repository / "knowledge" / "champion.json", {"schema_version": 1})
    _git(repository, "add", "knowledge")
    _git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "invalid remote snapshot",
    )
    source_commit = _git(repository, "rev-parse", "HEAD")
    _git(repository, "update-ref", "refs/remotes/origin/knowledge", source_commit)
    config_path = tmp_path / "worker.json"
    _write_json(
        config_path,
        {
            "schema_version": 1,
            "worker_id": "pc-um",
            "trusted_ref": "origin/knowledge",
            "knowledge_root": str(trusted_copy),
            "cache_root": str(tmp_path / "cache"),
            "outbox_root": str(tmp_path / "outbox"),
            "stage_checkpoint": False,
        },
    )
    config = load_worker_config(config_path)
    _mark_trusted(tmp_path, source_commit=source_commit)
    preflight_worker(config, "session-1")
    staging = finalize_candidate(config, _manifest(tmp_path), "session-1")
    contribution_id = str(staging["contribution"]["contribution_id"])
    branch = f"codex/knowledge-pc-um-{contribution_id[:12]}"

    with pytest.raises(WorkerCycleError):
        prepare_publication(config, repo=repository, bundle_id=str(staging["bundle_id"]))

    assert _git(repository, "branch", "--list", branch) == ""
    assert not (tmp_path / "outbox" / "wt" / str(staging["bundle_id"])[:16]).exists()


def test_prepare_publication_rejects_a_moved_trusted_ref(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _knowledge(repository)
    _git(repository, "add", "knowledge")
    _git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "trusted knowledge",
    )
    trusted_commit = _git(repository, "rev-parse", "HEAD")
    _git(repository, "update-ref", "refs/remotes/origin/knowledge", trusted_commit)
    config_path = tmp_path / "worker.json"
    _write_json(
        config_path,
        {
            "schema_version": 1,
            "worker_id": "pc-um",
            "trusted_ref": "origin/knowledge",
            "knowledge_root": str(repository / "knowledge"),
            "cache_root": str(tmp_path / "cache"),
            "outbox_root": str(tmp_path / "outbox"),
            "stage_checkpoint": False,
        },
    )
    config = load_worker_config(config_path)
    _mark_trusted(tmp_path, source_commit=trusted_commit)
    preflight_worker(config, "session-1")
    staging = finalize_candidate(config, _manifest(tmp_path), "session-1")

    (repository / "README.md").write_text("new trusted head", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "advance trusted ref",
    )
    advanced_commit = _git(repository, "rev-parse", "HEAD")
    _git(repository, "update-ref", "refs/remotes/origin/knowledge", advanced_commit)

    with pytest.raises(WorkerCycleError, match="no longer matches"):
        prepare_publication(config, repo=repository, bundle_id=str(staging["bundle_id"]))

    assert not (tmp_path / "outbox" / "publications").exists()
