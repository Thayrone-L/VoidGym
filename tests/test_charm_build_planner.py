from __future__ import annotations

import json
from pathlib import Path

import pytest

from hk_agent.charm_build_planner import CharmBuildPlanner, PlannerScope


def _scope(campaign: str = "campaign-a") -> PlannerScope:
    return PlannerScope("sha256:game", "save-1", campaign)


def _planner(path: Path, **kwargs: object) -> CharmBuildPlanner:
    return CharmBuildPlanner(path, _scope(), **kwargs)  # type: ignore[arg-type]


def test_visually_recognises_candidates_without_semantic_names(tmp_path: Path) -> None:
    planner = _planner(tmp_path / "charms.json")
    first = planner.observe_candidate([0.1, 0.2, 0.3], notch_cost=2)
    same = planner.observe_candidate([0.11, 0.19, 0.31], notch_cost=2)
    second = planner.observe_candidate([0.8, 0.7, 0.9], notch_cost=1)

    assert first == same == "visual-charm-0001"
    assert second == "visual-charm-0002"
    assert set(planner.snapshot()["candidates"]) == {first, second}


def test_conflicting_cost_for_same_visual_candidate_is_rejected(tmp_path: Path) -> None:
    planner = _planner(tmp_path / "charms.json")
    planner.observe_candidate([0.1, 0.2], notch_cost=1)
    with pytest.raises(ValueError, match="conflicting notch cost"):
        planner.observe_candidate([0.1, 0.2], notch_cost=2)


def test_build_search_respects_capacity_and_is_bounded(tmp_path: Path) -> None:
    planner = _planner(tmp_path / "charms.json", beam_width=4, max_candidates=5)
    ids = [
        planner.observe_candidate([index / 5, 0.0], notch_cost=2)
        for index in range(5)
    ]
    build = planner.suggest_build(
        notch_capacity=4,
        available_candidate_ids=ids,
        exploration_weight=1.0,
    )

    assert build.notch_cost <= 4
    assert len(build.candidate_ids) <= 2
    assert planner.snapshot()["beam_width"] == 4
    with pytest.raises(ValueError, match="candidate limit"):
        planner.observe_candidate([1.0, 1.0], notch_cost=1)


def test_learns_individual_effect_and_pair_synergy_from_run_outcomes(tmp_path: Path) -> None:
    planner = _planner(tmp_path / "charms.json")
    first = planner.observe_candidate([0.1, 0.2], notch_cost=1)
    second = planner.observe_candidate([0.8, 0.9], notch_cost=1)

    # A complete 2x2 empirical contrast: the pair is better than the sum of
    # either individual effect.  These are observations, not encoded charm lore.
    for build, outcome in (((), 0.0), ((first,), 2.0), ((second,), 1.0), ((first, second), 6.0)):
        planner.record_run(build, outcome=outcome, context={"rooms": outcome})

    assert float(planner.estimate_effect(first)["effect"]) > 0
    synergy = planner.estimate_synergy(first, second)
    assert synergy["complete_contrast"] == 1
    assert synergy["synergy"] == pytest.approx(3.0)
    chosen = planner.suggest_build(notch_capacity=2, exploration_weight=0)
    assert chosen.candidate_ids == (first, second)
    assert chosen.notch_cost == 2


def test_uncertainty_encourages_untested_candidate_and_shrinks_with_evidence(
    tmp_path: Path,
) -> None:
    planner = _planner(tmp_path / "charms.json")
    candidate = planner.observe_candidate([0.2, 0.4], notch_cost=1)
    before = float(planner.estimate_effect(candidate)["uncertainty"])
    for _ in range(8):
        planner.record_run([], outcome=0)
        planner.record_run([candidate], outcome=0)
    after = float(planner.estimate_effect(candidate)["uncertainty"])

    assert before > after
    assert planner.suggest_build(notch_capacity=1, exploration_weight=1).notch_cost <= 1


def test_persistence_is_scoped_and_rejects_cross_save_evidence(tmp_path: Path) -> None:
    path = tmp_path / "charms.json"
    planner = _planner(path)
    candidate = planner.observe_candidate([0.2, 0.4], notch_cost=1)
    planner.record_run([candidate], outcome=4.5)

    restored = _planner(path)
    assert restored.snapshot()["runs"][0]["outcome"] == 4.5
    with pytest.raises(ValueError, match="different scope"):
        CharmBuildPlanner(path, _scope("campaign-b"))


def test_contract_is_pixel_only_and_declares_non_exhaustive_search() -> None:
    contract = CharmBuildPlanner.contract()
    assert contract["information_boundary"] == "pixel_only"
    assert contract["exhaustive_power_set_enumeration"] is False
    assert contract["search"] == "capacity_constrained_bounded_beam"
    assert contract["menu_integration"] == "planner_only_not_implemented"
    assert "PlayerData" in contract["forbidden_inputs"]
    assert "wiki_or_external_charm_database" in contract["forbidden_inputs"]


def test_corrupt_persistence_is_rejected_without_overwriting(tmp_path: Path) -> None:
    path = tmp_path / "charms.json"
    original = {"schema_version": 999}
    path.write_text(json.dumps(original), "utf-8")
    with pytest.raises(ValueError, match="schema version"):
        _planner(path)
    assert json.loads(path.read_text("utf-8")) == original
