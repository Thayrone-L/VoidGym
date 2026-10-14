from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hk_agent.distributed_knowledge import (
    KnowledgeValidationError,
    candidate_from_contribution,
    canonical_document_sha256,
    contribution_from_manifest,
    export_contribution,
    initial_champion,
    portable_compatibility_fingerprint,
    promote_candidate,
    sync_from_git_ref,
    validate_candidate,
    validate_champion,
    validate_knowledge_tree,
)


def _manifest(root: str = r"C:\Users\alice\project") -> dict[str, Any]:
    return {
        "algorithm": "PPO",
        "seed": 37,
        "reward_version": "visual_motion_v1",
        "status": "COMPLETED",
        "completed_timesteps": 1024,
        "episode_count": 2,
        "gameplay_run_index": 3,
        "gameplay_run_failures": 1,
        "task_contract": {
            "reward_version": "visual_motion_v1",
            "action_names": ["NOOP", "ATTACK"],
            "config": {"enemy_kill_reward": 2.0},
        },
        "provenance": {
            "game_profile": "gog-1.5.78.11833",
            "campaign_origin": {
                "schema_version": 1,
                "save_source": "existing_save",
                "save_slot": 1,
                "save_file_name": "user1.dat",
                "base_plan_path": f"{root}\\configs\\plan.json",
                "resolved_plan_id": "continue-v1",
                "resolved_plan_sha256": "a" * 64,
                "resolved_steps": [{"button": "SUBMIT", "lease_ms": 50, "settle_seconds": 2.0}],
                "subsequent_episode_bootstrap": {
                    "schema_version": 1,
                    "save_source": "existing_save",
                    "save_slot": 1,
                    "base_plan_path": f"{root}\\configs\\resume.json",
                    "resolved_plan_id": "resume-v1",
                    "resolved_plan_sha256": "e" * 64,
                    "resolved_steps": [{"button": "SUBMIT", "lease_ms": 50}],
                },
            },
            "files": {
                "config": {"path": f"{root}\\config.local.toml", "sha256": "b" * 64},
                "game_executable": {"path": r"C:\Games\HK.exe", "sha256": "c" * 64},
            },
            "observation_dimensions": {"policy": {"height": 90, "width": 160}},
        },
        "versions": {"python": "3.12.10", "torch": "2.9.0"},
        "checkpoints": [
            {"step": 1024, "path": f"{root}\\checkpoints\\step-1024.zip", "sha256": "d" * 64}
        ],
    }


def _write_manifest(path: Path, manifest: dict[str, Any] | None = None) -> None:
    path.write_text(json.dumps(manifest or _manifest()), encoding="utf-8")


def test_portable_fingerprint_ignores_machine_paths_and_save_details() -> None:
    first = _manifest(r"C:\Users\alice\project")
    second = _manifest(r"D:\work\hk")
    second["provenance"]["campaign_origin"]["save_slot"] = 3
    second["provenance"]["files"]["config"]["sha256"] = "f" * 64
    assert portable_compatibility_fingerprint(
        first["task_contract"], first["provenance"]
    ) == portable_compatibility_fingerprint(second["task_contract"], second["provenance"])


def test_portable_fingerprint_changes_for_semantic_binary_hash() -> None:
    first = _manifest()
    second = _manifest()
    second["provenance"]["files"]["game_executable"]["sha256"] = "f" * 64
    assert portable_compatibility_fingerprint(
        first["task_contract"], first["provenance"]
    ) != portable_compatibility_fingerprint(second["task_contract"], second["provenance"])


def test_export_is_sanitized_content_addressed_and_valid(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    root = tmp_path / "knowledge"
    entry = export_contribution(manifest_path, root)
    index = validate_knowledge_tree(root)

    assert index["contributions"] == [entry]
    contribution_text = (root / "contributions" / f"{entry['contribution_id']}.json").read_text()
    assert "alice" not in contribution_text
    assert "user1.dat" not in contribution_text
    assert ".png" not in contribution_text
    assert "step-1024.zip" not in contribution_text
    assert '"sha256":"dddd' in contribution_text


def test_export_is_idempotent(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    root = tmp_path / "knowledge"
    first = export_contribution(manifest_path, root)
    second = export_contribution(manifest_path, root)
    assert first == second
    assert len(validate_knowledge_tree(root)["contributions"]) == 1


def test_contribution_rejects_tampering() -> None:
    contribution = contribution_from_manifest(_manifest())
    contribution["training"]["completed_timesteps"] = 9999
    from hk_agent.distributed_knowledge import validate_contribution

    with pytest.raises(KnowledgeValidationError, match="contribution_id"):
        validate_contribution(contribution)


def _candidate(champion: dict[str, Any] | None = None) -> dict[str, Any]:
    parent = champion or initial_champion()
    return candidate_from_contribution(
        contribution_from_manifest(_manifest()),
        worker_id="machine-1",
        artifact_uri="https://github.com/example/VoidGym/releases/download/run-1/model.zip",
        artifact_sha256="9" * 64,
        artifact_size_bytes=123_456,
        artifact_format="sb3-zip",
        trust_tier="owned",
        parent_generation=parent["generation"],
        parent_sha256=canonical_document_sha256(parent),
    )


def test_candidate_is_closed_content_addressed_and_https_only() -> None:
    candidate = _candidate()
    assert validate_candidate(candidate) == candidate

    tampered = dict(candidate)
    tampered["worker_id"] = "other-machine"
    with pytest.raises(KnowledgeValidationError, match="candidate_id"):
        validate_candidate(tampered)

    with pytest.raises(KnowledgeValidationError, match="HTTPS"):
        candidate_from_contribution(
            contribution_from_manifest(_manifest()),
            worker_id="machine-1",
            artifact_uri="https://example.test/model.zip?token=secret",
            artifact_sha256="9" * 64,
            artifact_size_bytes=123,
            artifact_format="sb3-zip",
            trust_tier="owned",
            parent_generation=0,
            parent_sha256=canonical_document_sha256(initial_champion()),
        )


def test_champion_promotion_uses_cas_lineage_and_is_idempotent() -> None:
    champion = initial_champion()
    candidate = _candidate(champion)
    champion_sha = canonical_document_sha256(champion)
    promoted = promote_candidate(
        champion,
        candidate,
        expected_generation=0,
        expected_champion_sha256=champion_sha,
        evaluation_report_sha256="8" * 64,
        evaluation_suite_sha256="6" * 64,
        evaluator_commit="7" * 40,
    )
    assert validate_champion(promoted)["generation"] == 1
    assert promoted["parent_champion_sha256"] == champion_sha
    assert (
        promote_candidate(
            promoted,
            candidate,
            expected_generation=0,
            expected_champion_sha256=champion_sha,
            evaluation_report_sha256="8" * 64,
            evaluation_suite_sha256="6" * 64,
            evaluator_commit="7" * 40,
        )
        == promoted
    )
    with pytest.raises(KnowledgeValidationError, match="evidence differs"):
        promote_candidate(
            promoted,
            candidate,
            expected_generation=0,
            expected_champion_sha256=champion_sha,
            evaluation_report_sha256="5" * 64,
            evaluation_suite_sha256="6" * 64,
            evaluator_commit="7" * 40,
        )


def test_champion_promotion_rejects_stale_or_incompatible_candidate() -> None:
    champion = initial_champion()
    candidate = _candidate(champion)
    with pytest.raises(KnowledgeValidationError, match="compare-and-swap"):
        promote_candidate(
            champion,
            candidate,
            expected_generation=1,
            expected_champion_sha256=canonical_document_sha256(champion),
            evaluation_report_sha256="8" * 64,
            evaluation_suite_sha256="6" * 64,
            evaluator_commit="7" * 40,
        )


def test_sync_reads_validated_git_ref_without_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    source = tmp_path / "source" / "knowledge"
    export_contribution(manifest_path, source)
    (source / "champion.json").write_text(json.dumps(initial_champion()), encoding="utf-8")
    calls: list[str] = []

    def fake_git_show(repo: Path, ref: str, path: str) -> bytes:
        del repo
        assert ref == "knowledge"
        calls.append(path)
        relative = Path(*Path(path).parts[1:])
        return (source / relative).read_bytes()

    monkeypatch.setattr("hk_agent.distributed_knowledge._git_show", fake_git_show)

    destination = tmp_path / "consumer" / "knowledge"
    report = sync_from_git_ref(tmp_path, "knowledge", destination)

    assert report == {
        "imported": 1,
        "total": 1,
        "ref": "knowledge",
        "champion_generation": 0,
        "champion_sha256": canonical_document_sha256(initial_champion()),
    }
    assert calls[0] == "knowledge/index.json"
    assert all(
        path in {"knowledge/index.json", "knowledge/champion.json"}
        or path.startswith("knowledge/contributions/")
        for path in calls
    )
    validate_knowledge_tree(destination)


@pytest.mark.parametrize("ref", ["--help", "knowledge^{tree}", "a..b", "a@{0}", "a//b"])
def test_sync_rejects_unsafe_ref(tmp_path: Path, ref: str) -> None:
    with pytest.raises(KnowledgeValidationError, match="unsafe Git ref"):
        sync_from_git_ref(tmp_path, ref, tmp_path / "knowledge")
