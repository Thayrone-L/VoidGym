from __future__ import annotations

import json
from pathlib import Path

import pytest

from hk_agent.distributed_knowledge import KnowledgeValidationError, initial_champion
from hk_agent.knowledge_ci import (
    GitChange,
    validate_champion_transition,
    validate_diff_policy,
    validate_inventory,
)


def _write_initial_knowledge(root: Path) -> Path:
    knowledge = root / "knowledge"
    knowledge.mkdir(parents=True)
    (knowledge / "index.json").write_text(
        json.dumps({"schema_version": 1, "contributions": []}), encoding="utf-8"
    )
    (knowledge / "champion.json").write_text(
        json.dumps(initial_champion()), encoding="utf-8"
    )
    return knowledge


def test_diff_policy_accepts_new_immutable_contribution() -> None:
    digest = "a" * 64
    validate_diff_policy(
        [
            GitChange("M", "knowledge/index.json"),
            GitChange("A", f"knowledge/contributions/{digest}.json"),
        ]
    )


@pytest.mark.parametrize(
    ("status", "path", "message"),
    [
        ("M", "src/hk_agent/policy.py", "forbidden path"),
        ("D", "knowledge/index.json", "forbidden D change"),
        ("M", f"knowledge/contributions/{'a' * 64}.json", "immutable"),
    ],
)
def test_diff_policy_rejects_non_data_and_mutation(
    status: str, path: str, message: str
) -> None:
    with pytest.raises(KnowledgeValidationError, match=message):
        validate_diff_policy([GitChange(status, path)])


def test_inventory_rejects_orphaned_contribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    knowledge = _write_initial_knowledge(tmp_path)
    orphan = f"knowledge/contributions/{'b' * 64}.json"
    monkeypatch.setattr(
        "hk_agent.knowledge_ci.read_tracked_modes",
        lambda _repository: {
            "knowledge/index.json": "100644",
            "knowledge/champion.json": "100644",
            orphan: "100644",
        },
    )
    with pytest.raises(KnowledgeValidationError, match="orphaned"):
        validate_inventory(tmp_path, knowledge)


def test_inventory_rejects_symlink_or_submodule_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    knowledge = _write_initial_knowledge(tmp_path)
    monkeypatch.setattr(
        "hk_agent.knowledge_ci.read_tracked_modes",
        lambda _repository: {
            "knowledge/index.json": "100644",
            "knowledge/champion.json": "120000",
        },
    )
    with pytest.raises(KnowledgeValidationError, match="regular file"):
        validate_inventory(tmp_path, knowledge)


def test_external_pr_cannot_update_champion(tmp_path: Path) -> None:
    base = _write_initial_knowledge(tmp_path / "base")
    candidate = _write_initial_knowledge(tmp_path / "candidate")
    with pytest.raises(KnowledgeValidationError, match="external"):
        validate_champion_transition(base, candidate, changed=True, allow_update=False)
