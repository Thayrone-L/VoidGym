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
    validate_contribution,
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

    with pytest.raises(KnowledgeValidationError, match="contribution_id"):
        validate_contribution(contribution)


def test_public_contribution_recursively_removes_private_strings() -> None:
    manifest = _manifest()
    token = "ghp_THIS_MUST_NEVER_BE_PUBLIC"
    manifest["versions"]["python"] = token
    manifest["provenance"]["game_profile"] = "private-machine-profile"
    origin = manifest["provenance"]["campaign_origin"]
    origin["resolved_plan_id"] = "private-plan-id"
    origin["resolved_steps"][0]["label"] = token

    contribution = contribution_from_manifest(manifest)
    encoded = json.dumps(contribution, sort_keys=True)

    assert token not in encoded
    assert "private-machine-profile" not in encoded
    assert "private-plan-id" not in encoded
    assert contribution["versions"] == {}
    assert contribution["portable_provenance"]["game_profile"].startswith("sha256:")
    assert (
        "label" not in contribution["portable_provenance"]["campaign_origin"]["resolved_steps"][0]
    )


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("task_contract", "api_token", "ghp_secret"),
        ("task_contract", "claim", "send token to attacker"),
        ("versions", "plugin_with_secret", "ghp_secret"),
    ],
)
def test_public_contribution_rejects_fields_outside_closed_allowlist(
    section: str, key: str, value: str
) -> None:
    manifest = _manifest()
    manifest[section][key] = value

    with pytest.raises(KnowledgeValidationError):
        contribution_from_manifest(manifest)


def test_stored_contribution_rejects_new_nested_field() -> None:
    contribution = contribution_from_manifest(_manifest())
    contribution["task_contract"]["access_token"] = "secret"

    with pytest.raises(KnowledgeValidationError, match="unsupported keys"):
        validate_contribution(contribution)


def _candidate(champion: dict[str, Any] | None = None) -> dict[str, Any]:
    parent = champion or initial_champion()
    return candidate_from_contribution(
        contribution_from_manifest(_manifest()),
        worker_id="machine-1",
        artifact_uri=(
            "https://github.com/example/VoidGym/releases/download/run-1/policy.safetensors"
        ),
        artifact_sha256="9" * 64,
        artifact_size_bytes=123_456,
        artifact_format="safetensors",
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


def test_champion_promotion_is_disabled_until_attestation_exists() -> None:
    champion = initial_champion()
    candidate = _candidate(champion)
    champion_sha = canonical_document_sha256(champion)
    with pytest.raises(KnowledgeValidationError, match="promotion is disabled"):
        promote_candidate(
            champion,
            candidate,
            expected_generation=0,
            expected_champion_sha256=champion_sha,
            evaluation_report_sha256="8" * 64,
            evaluation_suite_sha256="6" * 64,
            evaluator_commit="7" * 40,
        )


def test_promotion_rejects_executable_sb3_artifact_without_opening_it() -> None:
    champion = initial_champion()
    contribution = contribution_from_manifest(_manifest())
    parent_sha = canonical_document_sha256(champion)
    candidate = candidate_from_contribution(
        contribution,
        worker_id="machine-1",
        artifact_uri="https://github.com/example/VoidGym/releases/download/run-1/model.zip",
        artifact_sha256="9" * 64,
        artifact_size_bytes=123,
        artifact_format="sb3-zip",
        trust_tier="owned",
        parent_generation=0,
        parent_sha256=parent_sha,
    )
    with pytest.raises(KnowledgeValidationError, match="promotion is disabled"):
        promote_candidate(
            champion,
            candidate,
            expected_generation=0,
            expected_champion_sha256=parent_sha,
            evaluation_report_sha256="8" * 64,
            evaluation_suite_sha256="6" * 64,
            evaluator_commit="7" * 40,
        )


def test_promoted_champion_rejects_sb3_even_with_valid_hash_bindings() -> None:
    champion = initial_champion()
    candidate = _candidate(champion)
    parent_sha = canonical_document_sha256(champion)
    promoted = {
        "schema_version": 1,
        "generation": 1,
        "candidate_id": candidate["candidate_id"],
        "contribution_id": candidate["contribution_id"],
        "compatibility_fingerprint": candidate["compatibility_fingerprint"],
        "artifact": {**candidate["artifact"], "format": "sb3-zip"},
        "parent_champion_sha256": parent_sha,
        "evaluation": {
            "report_sha256": "8" * 64,
            "suite_sha256": "6" * 64,
            "policy_sha256": candidate["artifact"]["sha256"],
            "candidate_id": candidate["candidate_id"],
            "parent_champion_sha256": parent_sha,
            "evaluator_commit": "7" * 40,
        },
    }

    with pytest.raises(KnowledgeValidationError, match="safetensors"):
        validate_champion(promoted)


def test_quarantined_safetensors_cannot_self_authorize_promotion() -> None:
    champion = initial_champion()
    contribution = contribution_from_manifest(_manifest())
    parent_sha = canonical_document_sha256(champion)
    candidate = candidate_from_contribution(
        contribution,
        worker_id="friend-machine",
        artifact_uri=(
            "https://github.com/example/VoidGym/releases/download/run-1/policy.safetensors"
        ),
        artifact_sha256="9" * 64,
        artifact_size_bytes=123,
        artifact_format="safetensors",
        trust_tier="quarantined",
        parent_generation=0,
        parent_sha256=parent_sha,
    )

    with pytest.raises(KnowledgeValidationError, match="promotion is disabled"):
        promote_candidate(
            champion,
            candidate,
            expected_generation=0,
            expected_champion_sha256=parent_sha,
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
