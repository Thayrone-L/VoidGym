from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hk_agent.route_memory import ROUTE_MEMORY_SCHEMA_VERSION, TopologicalRouteMemory

SCOPE = {
    "game_profile": "test-profile",
    "save_source": "existing_save",
    "save_slot": 1,
    "campaign_id": "test-campaign",
    "detector_signature": "test-detector-v1",
}


def _descriptor(value: float) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    return np.full((2, 3), value, dtype=np.float32)


def test_records_directed_routes_and_finds_shortest_path(tmp_path: Path) -> None:
    memory = TopologicalRouteMemory(tmp_path / "routes.json", scope=SCOPE)
    start = memory.observe_start(_descriptor(0.1))
    middle = memory.observe_transition(_descriptor(0.4), action_trace=(2, 2, 4))
    end = memory.observe_transition(_descriptor(0.8), action_trace=(3, 5))

    assert start["new_landmark"] is True
    assert middle["new_edge"] is True
    assert memory.shortest_path(start["destination"], end["destination"]) == (
        start["destination"],
        middle["destination"],
        end["destination"],
    )
    assert memory.shortest_path(end["destination"], start["destination"]) is None
    assert memory.route_plan(start["destination"], end["destination"]) == [
        {
            "source": start["destination"],
            "destination": middle["destination"],
            "temporal_action_trace": [2, 2, 4],
            "observed_traversals": 1,
            "confidence": pytest.approx(1 / 3),
        },
        {
            "source": middle["destination"],
            "destination": end["destination"],
            "temporal_action_trace": [3, 5],
            "observed_traversals": 1,
            "confidence": pytest.approx(1 / 3),
        },
    ]


def test_revisits_are_retained_and_never_blocked(tmp_path: Path) -> None:
    path = tmp_path / "routes.json"
    memory = TopologicalRouteMemory(path, scope=SCOPE)
    first = memory.observe_start(_descriptor(0.2))
    second = memory.observe_transition(_descriptor(0.7), action_trace=(2, 2))
    revisit = memory.observe_transition(_descriptor(0.201), action_trace=(1,))

    assert revisit["destination"] == first["destination"]
    assert revisit["revisit"] is True
    assert revisit["new_landmark"] is False
    assert revisit["edge_traversals"] == 1
    assert memory.summary() == {
        "schema_version": ROUTE_MEMORY_SCHEMA_VERSION,
        "current_node": first["destination"],
        "known_nodes": 2,
        "known_edges": 2,
        "observed_traversals": 2,
    }
    persisted = json.loads(path.read_text("utf-8"))
    visits = {item["id"]: item["visits"] for item in persisted["nodes"]}
    assert visits[first["destination"]] == 2
    assert visits[second["destination"]] == 1


def test_memory_reloads_and_accumulates_traversal_evidence(tmp_path: Path) -> None:
    path = tmp_path / "routes.json"
    memory = TopologicalRouteMemory(path, scope=SCOPE)
    source = memory.observe_start(_descriptor(0.1))["destination"]
    destination = memory.observe_transition(_descriptor(0.9), action_trace=(2, 3))[
        "destination"
    ]

    del memory
    restored = TopologicalRouteMemory(path, scope=SCOPE)
    assert restored.shortest_path(source, destination) == (source, destination)
    restored.reset_episode()
    assert restored.observe_start(_descriptor(0.1))["revisit"] is True
    event = restored.observe_transition(_descriptor(0.9), action_trace=(2,))
    assert event["edge_traversals"] == 2
    assert event["new_edge"] is False


def test_corrupt_or_incompatible_memory_is_rejected_without_overwrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "routes.json"
    original = '{"schema_version":99,"nodes":[],"edges":[]}'
    path.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="schema version"):
        TopologicalRouteMemory(path, scope=SCOPE)
    assert path.read_text("utf-8") == original


def test_scope_mismatch_and_concurrent_writer_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "routes.json"
    first = TopologicalRouteMemory(path, scope=SCOPE)
    first.observe_start(_descriptor(0.1))
    second = TopologicalRouteMemory(path, scope=SCOPE)
    first.observe_transition(_descriptor(0.5), action_trace=(2,))

    second.reset_episode()
    with pytest.raises(RuntimeError, match="another writer"):
        second.observe_start(_descriptor(0.1))

    wrong_scope = {**SCOPE, "save_slot": 2}
    with pytest.raises(ValueError, match="scope differs"):
        TopologicalRouteMemory(path, scope=wrong_scope)


def test_contract_explicitly_excludes_policy_and_reward_feedback() -> None:
    contract = TopologicalRouteMemory.contract()
    assert contract["policy_integration"] == "external_planner_only"
    assert contract["reward_integration"] == "none"
    assert contract["privileged_state"] == []
