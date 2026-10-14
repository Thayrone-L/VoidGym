from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hk_agent.campaign_source import (
    CampaignBootstrapSequence,
    CampaignOrigin,
    ResolvedCampaignBootstrap,
    resolve_campaign_bootstrap,
    validate_save_precondition,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTINUE = PROJECT_ROOT / "configs" / "menu-bootstrap" / "continue-kings-pass-v1.json"
NEW_GAME = PROJECT_ROOT / "configs" / "menu-bootstrap" / "new-game-kings-pass-v1.json"


def _resolve(source: str, slot: int) -> ResolvedCampaignBootstrap:
    return resolve_campaign_bootstrap(
        continue_plan_path=CONTINUE,
        new_game_plan_path=NEW_GAME,
        origin=CampaignOrigin(source, slot),
    )


def test_existing_slot_one_preserves_the_reviewed_base_plan() -> None:
    resolved = _resolve("existing_save", 1)
    assert [step.button for step in resolved.plan.steps] == [
        "SUBMIT",
        "CANCEL",
        "SUBMIT",
        "SUBMIT",
        "SUBMIT",
    ]
    assert resolved.plan.plan_id == "continue-kings-pass-v1"
    assert resolved.provenance()["resolved_plan_sha256"] == resolved.canonical_sha256


def test_existing_slot_three_inserts_deterministic_navigation() -> None:
    resolved = _resolve("existing_save", 3)
    assert [step.button for step in resolved.plan.steps] == [
        "SUBMIT",
        "CANCEL",
        "SUBMIT",
        "DOWN",
        "DOWN",
        "SUBMIT",
        "SUBMIT",
    ]
    assert resolved.plan.steps[5].label == "select-slot-3"


def test_new_game_slot_four_uses_new_game_plan_and_has_readiness() -> None:
    resolved = _resolve("new_game", 4)
    assert resolved.base_path == NEW_GAME.resolve()
    assert [step.button for step in resolved.plan.steps] == [
        "SUBMIT",
        "CANCEL",
        "SUBMIT",
        "DOWN",
        "DOWN",
        "DOWN",
        "SUBMIT",
        "SUBMIT",
    ]
    assert resolved.plan.readiness is not None
    assert resolved.plan.steps[6].label == "select-new-game-slot-4"


def test_new_game_sequence_resumes_the_same_slot_after_first_bootstrap() -> None:
    initial = _resolve("new_game", 3)
    subsequent = _resolve("existing_save", 3)
    sequence = CampaignBootstrapSequence(initial.plan, subsequent.plan)
    assert sequence().plan_id == "new-game-kings-pass-v1-slot-3"
    assert sequence().plan_id == "continue-kings-pass-v1-slot-3"
    assert sequence().plan_id == "continue-kings-pass-v1-slot-3"


@pytest.mark.parametrize(("source", "slot"), [("other", 1), ("existing_save", 0), ("new_game", 5)])
def test_campaign_origin_rejects_invalid_values(source: str, slot: int) -> None:
    with pytest.raises(ValueError):
        CampaignOrigin(source, slot)


def test_save_preconditions_never_overwrite_an_occupied_slot(tmp_path: Path) -> None:
    (tmp_path / "user2.dat").write_bytes(b"save")
    assert validate_save_precondition(tmp_path, CampaignOrigin("existing_save", 2)).name == "user2.dat"
    with pytest.raises(RuntimeError, match="will not be overwritten"):
        validate_save_precondition(tmp_path, CampaignOrigin("new_game", 2))
    with pytest.raises(RuntimeError, match="slot is empty"):
        validate_save_precondition(tmp_path, CampaignOrigin("existing_save", 3))
    assert validate_save_precondition(tmp_path, CampaignOrigin("new_game", 3)).name == "user3.dat"


@pytest.mark.parametrize(
    "runner",
    ["run_first_training.py", "run_scene_change_probe.py", "run_scene_change_training.py"],
)
def test_runners_report_resolved_origin_in_dry_run(runner: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / runner),
            "--save-source",
            "new_game",
            "--save-slot",
            "4",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["campaign_origin"]["save_source"] == "new_game"
    assert plan["campaign_origin"]["save_slot"] == 4
    buttons = [step["button"] for step in plan["campaign_origin"]["resolved_steps"]]
    assert buttons == [
        "SUBMIT",
        "CANCEL",
        "SUBMIT",
        "DOWN",
        "DOWN",
        "DOWN",
        "SUBMIT",
        "SUBMIT",
    ]
