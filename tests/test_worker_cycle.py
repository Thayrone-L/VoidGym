from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import save_file

from hk_agent import worker_cycle as worker_cycle_module
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
    publish_fork,
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
            "schema_version": 2,
            "worker_id": "pc-um",
            "upstream_repository": "Thayrone-L/VoidGym",
            "upstream_remote": "upstream",
            "fork_repository": "voidgym-worker/VoidGym",
            "fork_remote": "fork",
            "branch_prefix": "voidgym/knowledge",
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
            "ref": "upstream/knowledge",
            "source_commit": source_commit,
            "generation": selected["generation"],
            "champion_sha256": canonical_document_sha256(selected),
        },
    )


def _write_portable_policy(
    policy: Path, compatibility: str
) -> tuple[dict[str, Any], Path]:
    policy.parent.mkdir(parents=True, exist_ok=True)
    inventory = [{"name": "weight", "dtype": "F32", "shape": [1], "numel": 1}]
    canonical_inventory = json.dumps(
        inventory, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    schema_hash = hashlib.sha256(canonical_inventory).hexdigest()
    save_file(
        {"weight": np.asarray([0.25], dtype=np.float32)},
        str(policy),
        metadata={
            "format": "voidgym-policy-v1",
            "compatibility_fingerprint": compatibility,
            "tensor_schema_sha256": schema_hash,
        },
    )
    artifact = {
        "sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
        "size_bytes": policy.stat().st_size,
    }
    body: dict[str, Any] = {
        "schema_version": 1,
        "format": "safetensors",
        "compatibility_fingerprint": compatibility,
        "policy_class": "stable_baselines3.common.policies.MultiInputActorCriticPolicy",
        "tensor_schema_sha256": schema_hash,
        "tensors": inventory,
        "artifact": artifact,
    }
    manifest_id = hashlib.sha256(
        json.dumps(
            body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    manifest = {"manifest_id": manifest_id, **body}
    sidecar = policy.with_suffix(".manifest.json")
    _write_json(sidecar, manifest)
    return manifest, sidecar


def _install_promoted_policy(
    tmp_path: Path, compatibility: str
) -> tuple[dict[str, Any], Path, Path]:
    provisional = tmp_path / "cache" / "provisional" / "policy.safetensors"
    policy_manifest, provisional_sidecar = _write_portable_policy(
        provisional, compatibility
    )
    digest = str(policy_manifest["artifact"]["sha256"])
    bundle = tmp_path / "cache" / "champions" / digest
    policy = bundle / "policies" / "policy.safetensors"
    sidecar = bundle / "policies" / "policy.manifest.json"
    policy.parent.mkdir(parents=True, exist_ok=True)
    provisional.replace(policy)
    provisional_sidecar.replace(sidecar)
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
            "size_bytes": policy.stat().st_size,
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
    _write_json(
        bundle / "manifest.json",
        {
            "compatibility_fingerprint": compatibility,
            "portable_policies": [
                {
                    "step": 10,
                    "sha256": digest,
                    "format": "safetensors",
                    "manifest_id": policy_manifest["manifest_id"],
                    "manifest_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    return champion, policy, sidecar


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
    compatibility = portable_compatibility_fingerprint(task_contract, provenance)
    policy = training / "policies" / "step-128.safetensors"
    policy_manifest, sidecar = _write_portable_policy(policy, compatibility)
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
        "compatibility_fingerprint": compatibility,
        "versions": {"python": "3.12"},
        "checkpoints": [{"step": 128, "path": str(checkpoint), "sha256": digest}],
        "portable_policies": [
            {
                "step": 128,
                "path": str(policy),
                "manifest_path": str(sidecar),
                "sha256": policy_manifest["artifact"]["sha256"],
                "size_bytes": policy_manifest["artifact"]["size_bytes"],
                "format": "safetensors",
                "manifest_id": policy_manifest["manifest_id"],
                "manifest_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            }
        ],
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
    assert receipt["warm_start_policy_manifest"] is None
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


def test_worker_config_requires_distinct_upstream_and_fork(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    document = json.loads(config_path.read_text("utf-8"))
    document["fork_repository"] = document["upstream_repository"]
    _write_json(config_path, document)

    with pytest.raises(WorkerCycleError, match="different GitHub owner"):
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
    monkeypatch.setattr("hk_agent.worker_cycle._verify_remote", lambda *args, **kwargs: None)
    monkeypatch.setattr("hk_agent.worker_cycle._git", lambda *args, **kwargs: "1" * 40)
    report = sync_worker_knowledge(config, repo=tmp_path)
    assert report["champion_generation"] == 0
    trusted = json.loads((tmp_path / "cache" / "trusted-source.json").read_text("utf-8"))
    assert trusted["trust_tier"] == "maintainer-promoted"

    with pytest.raises(WorkerCycleError, match="configured trusted maintainer ref"):
        sync_worker_knowledge(config, repo=tmp_path, ref="origin/friend-candidate")


def test_promoted_champion_is_selected_only_as_safe_warm_start(tmp_path: Path) -> None:
    compatibility = "a" * 64
    champion, policy, sidecar = _install_promoted_policy(tmp_path, compatibility)
    digest = champion["artifact"]["sha256"]

    receipt = preflight_worker(load_worker_config(_config(tmp_path)), "session-1")
    assert receipt["resume_checkpoint"] is None
    assert receipt["warm_start_policy"] == str(policy)
    assert receipt["warm_start_policy_sha256"] == digest
    assert receipt["warm_start_policy_manifest"] == str(sidecar)
    assert receipt["warm_start_policy_manifest_id"]

    policy.write_bytes(b"tampered")
    with pytest.raises(WorkerCycleError, match="policy size differs|canonical sidecar"):
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
    assert (bundle / "policy.safetensors").is_file()
    assert (bundle / "policy.manifest.json").is_file()
    assert not (bundle / "checkpoint.zip").exists()
    assert first["artifact"]["format"] == "safetensors"
    text = (bundle / "staging.json").read_text("utf-8")
    assert str(tmp_path) not in text
    assert "training/manifest.json" not in text
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
    _, policy, _ = _install_promoted_policy(
        tmp_path, manifest["compatibility_fingerprint"]
    )
    config = load_worker_config(_config(tmp_path))
    preflight_worker(config, "session-promoted")
    policy.write_bytes(b"tampered after preflight")

    with pytest.raises(WorkerCycleError, match="canonical sidecar|integrity revalidation"):
        finalize_candidate(config, manifest_path, "session-promoted")


def test_finalize_requires_the_exact_frozen_promoted_checkpoint(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text("utf-8"))
    champion, _, _ = _install_promoted_policy(
        tmp_path, manifest["compatibility_fingerprint"]
    )
    parent_digest = champion["artifact"]["sha256"]
    config = load_worker_config(_config(tmp_path))
    preflight_worker(config, "session-promoted")

    with pytest.raises(WorkerCycleError, match="did not warm-start"):
        finalize_candidate(config, manifest_path, "session-promoted")

    frozen = preflight_worker(config, "session-promoted")
    manifest["warm_started_from"] = {
        "sha256": parent_digest,
        "format": "safetensors",
        "manifest_id": frozen["warm_start_policy_manifest_id"],
        "manifest_sha256": frozen["warm_start_policy_manifest_sha256"],
    }
    _write_json(manifest_path, manifest)
    result = finalize_candidate(config, manifest_path, "session-promoted")
    assert result["parent_champion"]["generation"] == 1


def _publication_fixture(tmp_path: Path) -> tuple[Path, Path, object, dict[str, Any], str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "remote", "add", "upstream", "https://github.com/Thayrone-L/VoidGym.git")
    _git(
        repository,
        "remote",
        "set-url",
        "--push",
        "upstream",
        "disabled://voidgym-upstream-read-only",
    )
    _git(repository, "config", "--unset-all", "remote.upstream.fetch")
    _git(
        repository,
        "config",
        "--add",
        "remote.upstream.fetch",
        "+refs/heads/knowledge:refs/remotes/upstream/knowledge",
    )
    _git(repository, "remote", "add", "fork", "https://github.com/voidgym-worker/VoidGym.git")
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
    _git(repository, "update-ref", "refs/remotes/upstream/knowledge", trusted_commit)

    config_path = tmp_path / "worker.json"
    _write_json(
        config_path,
        {
            "schema_version": 2,
            "worker_id": "pc-um",
            "upstream_repository": "Thayrone-L/VoidGym",
            "upstream_remote": "upstream",
            "fork_repository": "voidgym-worker/VoidGym",
            "fork_remote": "fork",
            "branch_prefix": "voidgym/knowledge",
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
    return repository, config_path, config, staging, trusted_commit


def test_prepare_publication_uses_unique_isolated_attempt(tmp_path: Path) -> None:
    repository, _, config, staging, trusted_commit = _publication_fixture(tmp_path)
    main_before = _git(repository, "rev-parse", "HEAD")
    attempt = "1" * 32

    prepared = prepare_publication(
        config, repo=repository, bundle_id=str(staging["bundle_id"]), attempt_id=attempt
    )
    repeated = prepare_publication(
        config, repo=repository, bundle_id=str(staging["bundle_id"]), attempt_id=attempt
    )

    assert prepared == repeated
    assert prepared["trained_source_commit"] == trusted_commit
    assert prepared["publication_base_commit"] == trusted_commit
    assert prepared["remote_branch"] == f"voidgym/knowledge/pc-um/{attempt}"
    assert _git(repository, "rev-parse", "HEAD") == main_before
    assert _git(repository, "branch", "--show-current") == "main"
    worktree = Path(str(prepared["worktree"]))
    assert set(
        _git(worktree, "diff", "--name-only", f"{trusted_commit}..HEAD").splitlines()
    ) == {
        "knowledge/index.json",
        f"knowledge/contributions/{prepared['contribution_id']}.json",
    }
    assert not (worktree / "checkpoint.zip").exists()
    assert not (worktree / "staging.json").exists()


def test_prepare_publication_rebuilds_on_latest_upstream_snapshot(tmp_path: Path) -> None:
    repository, _, config, staging, trained_commit = _publication_fixture(tmp_path)
    (repository / "README.md").write_text("new upstream head", encoding="utf-8")
    _git(repository, "add", "README.md")
    _git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "advance upstream knowledge",
    )
    latest_commit = _git(repository, "rev-parse", "HEAD")
    _git(repository, "update-ref", "refs/remotes/upstream/knowledge", latest_commit)

    prepared = prepare_publication(
        config,
        repo=repository,
        bundle_id=str(staging["bundle_id"]),
        attempt_id="2" * 32,
    )

    assert prepared["trained_source_commit"] == trained_commit
    assert prepared["publication_base_commit"] == latest_commit
    assert _git(Path(prepared["worktree"]), "show", "HEAD^:README.md") == "new upstream head"


def test_prepare_publication_rejects_mismatched_remote(tmp_path: Path) -> None:
    repository, _, config, staging, _ = _publication_fixture(tmp_path)
    _git(
        repository,
        "remote",
        "set-url",
        "--push",
        "fork",
        "https://github.com/attacker/VoidGym.git",
    )

    with pytest.raises(WorkerCycleError, match="does not match"):
        prepare_publication(
            config,
            repo=repository,
            bundle_id=str(staging["bundle_id"]),
            attempt_id="3" * 32,
        )


def test_prepare_publication_rejects_noncanonical_upstream_refspec(tmp_path: Path) -> None:
    repository, _, config, staging, _ = _publication_fixture(tmp_path)
    _git(
        repository,
        "config",
        "--replace-all",
        "remote.upstream.fetch",
        "+refs/pull/1/head:refs/remotes/upstream/knowledge",
    )

    with pytest.raises(WorkerCycleError, match="fetch only refs/heads/knowledge"):
        prepare_publication(
            config,
            repo=repository,
            bundle_id=str(staging["bundle_id"]),
            attempt_id="5" * 32,
        )


def test_prepare_publication_rejects_divergent_upstream_history(tmp_path: Path) -> None:
    repository, _, config, staging, trained_commit = _publication_fixture(tmp_path)
    tree = _git(repository, "rev-parse", f"{trained_commit}^{{tree}}")
    divergent = _git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit-tree",
        tree,
        "-m",
        "divergent knowledge",
    )
    _git(repository, "update-ref", "refs/remotes/upstream/knowledge", divergent)

    with pytest.raises(WorkerCycleError, match="not a descendant"):
        prepare_publication(
            config,
            repo=repository,
            bundle_id=str(staging["bundle_id"]),
            attempt_id="6" * 32,
        )


def test_prepare_publication_claim_prevents_attempt_collision(tmp_path: Path) -> None:
    repository, _, config, staging, _ = _publication_fixture(tmp_path)
    attempt = "7" * 32
    claim = (
        tmp_path
        / "outbox"
        / "publications"
        / str(staging["bundle_id"])
        / attempt
        / "preparing.lock"
    )
    claim.parent.mkdir(parents=True)
    claim.write_text("owned by another process", encoding="ascii")

    with pytest.raises(WorkerCycleError, match="already being prepared"):
        prepare_publication(
            config,
            repo=repository,
            bundle_id=str(staging["bundle_id"]),
            attempt_id=attempt,
        )
    assert claim.read_text("ascii") == "owned by another process"
    assert not (tmp_path / "outbox" / "wt" / attempt).exists()


def test_publish_fork_pushes_exact_commit_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _, config, staging, _ = _publication_fixture(tmp_path)
    prepared = prepare_publication(
        config,
        repo=repository,
        bundle_id=str(staging["bundle_id"]),
        attempt_id="4" * 32,
    )
    calls: list[list[str]] = []

    def fake_run(arguments: list[str], **_: object) -> str:
        calls.append(arguments)
        if "ls-remote" in arguments and len([call for call in calls if "ls-remote" in call]) == 1:
            return ""
        if "ls-remote" in arguments:
            return f"{prepared['publication_commit']}\trefs/heads/{prepared['remote_branch']}"
        return "ok"

    monkeypatch.setattr("hk_agent.worker_cycle._run", fake_run)
    monkeypatch.setattr("hk_agent.worker_cycle._verify_remote", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "hk_agent.worker_cycle._verify_prepared_publication",
        lambda *_: Path(str(prepared["worktree"])),
    )
    result = publish_fork(
        config,
        repo=repository,
        bundle_id=str(staging["bundle_id"]),
        attempt_id="4" * 32,
    )

    push = next(arguments for arguments in calls if "push" in arguments)
    assert not any(argument.startswith(("--force", "+")) for argument in push)
    assert "https://github.com/voidgym-worker/VoidGym.git" in push
    assert all("Thayrone-L/VoidGym.git" not in argument for argument in push)
    assert result["compare_url"].startswith(
        "https://github.com/Thayrone-L/VoidGym/compare/knowledge...voidgym-worker:"
    )


def test_publish_fork_adopts_matching_remote_branch_after_lost_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, _, config, staging, _ = _publication_fixture(tmp_path)
    attempt = "8" * 32
    prepared = prepare_publication(
        config,
        repo=repository,
        bundle_id=str(staging["bundle_id"]),
        attempt_id=attempt,
    )
    calls: list[list[str]] = []

    def fake_run(arguments: list[str], **_: object) -> str:
        calls.append(arguments)
        if "ls-remote" in arguments:
            return f"{prepared['publication_commit']}\trefs/heads/{prepared['remote_branch']}"
        return "ok"

    monkeypatch.setattr(worker_cycle_module, "_run", fake_run)
    monkeypatch.setattr(worker_cycle_module, "_verify_remote", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        worker_cycle_module,
        "_verify_prepared_publication",
        lambda *_: Path(str(prepared["worktree"])),
    )

    result = publish_fork(
        config,
        repo=repository,
        bundle_id=str(staging["bundle_id"]),
        attempt_id=attempt,
    )

    assert result["publication_commit"] == prepared["publication_commit"]
    assert not any("push" in arguments for arguments in calls)


@pytest.mark.skipif(os.name != "nt", reason="Win32 extended UNC path behavior")
def test_native_io_path_uses_valid_extended_unc_prefix() -> None:
    converted = worker_cycle_module._native_io_path(
        Path(r"\\server\share\voidgym\receipt.json")
    )
    assert str(converted).startswith("\\\\?\\UNC\\server\\share\\")


def test_subprocess_environment_drops_ambient_github_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, str] = {}

    def fake_subprocess_run(*_: object, **kwargs: object) -> object:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        captured.update(environment)

        class Result:
            returncode = 0
            stdout = "ok"

        return Result()

    monkeypatch.setenv("GH_TOKEN", "must-not-leak")
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-leak")
    monkeypatch.setattr(worker_cycle_module.subprocess, "run", fake_subprocess_run)

    assert worker_cycle_module._run(
        ["git", "version"], cwd=tmp_path, safe_error="failed"
    ) == "ok"
    assert "GH_TOKEN" not in captured
    assert "GITHUB_TOKEN" not in captured
    assert captured["GIT_TERMINAL_PROMPT"] == "0"
