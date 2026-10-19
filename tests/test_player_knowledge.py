from __future__ import annotations

import numpy as np

from hk_agent.player_knowledge import (
    ABILITY_FLAGS,
    GRID_HEIGHT,
    GRID_WIDTH,
    ITEM_FLAGS,
    KEY_ITEM_FLAGS,
    PlayerKnowledge,
)


def test_item_descriptors_are_zero_until_acquisition_and_anonymous() -> None:
    knowledge = PlayerKnowledge()
    vector = knowledge.vector()
    assert np.all(vector[: len(ITEM_FLAGS) * 5] == 0)
    knowledge.update({"operator_telemetry": {
        "scene_name": "RoomA", "scene_bounds": [0, 0, 16, 9], "x": 8, "y": 4.5,
        "progress_flags": {"hasDash": True},
    }})
    vector = knowledge.vector()
    assert vector[0] == 1
    assert np.any(vector[1:5] > 0)
    assert np.all(vector[5: len(ITEM_FLAGS) * 5] == 0)


def test_charm_effect_descriptor_and_cost_are_visible_only_after_acquisition() -> None:
    knowledge = PlayerKnowledge()
    charm_start = (len(ABILITY_FLAGS) + len(KEY_ITEM_FLAGS)) * 5
    assert np.all(knowledge.vector()[charm_start : charm_start + 5] == 0)
    knowledge.update({"operator_telemetry": {
        "scene_name": "RoomA", "scene_bounds": [0, 0, 16, 9], "x": 8, "y": 4,
        "progress_flags": {"gotCharm_1": True}, "charm_costs": {"gotCharm_1": 1},
    }})
    block = knowledge.vector()[charm_start : charm_start + 5]
    assert block[0] == 1
    assert block[1] > 0 and block[2] > 0 and block[3] == 1 / 6
    assert np.all(np.isfinite(block))


def test_map_only_contains_cells_lived_and_exits_seen() -> None:
    knowledge = PlayerKnowledge()
    knowledge.update({"operator_telemetry": {
        "scene_name": "RoomA", "scene_bounds": [0, 0, 16, 9], "x": 1, "y": 1,
    }})
    knowledge.update({"operator_telemetry": {
        "scene_name": "RoomA", "scene_bounds": [0, 0, 16, 9], "x": 2, "y": 1,
    }})
    map_start = len(ITEM_FLAGS) * 5
    visited = knowledge.vector()[map_start : map_start + GRID_WIDTH * GRID_HEIGHT]
    assert int(visited.sum()) == 2
    knowledge.update({"operator_telemetry": {
        "scene_name": "RoomB", "scene_bounds": [0, 0, 16, 9], "x": 2, "y": 1,
    }})
    assert "RoomB" in knowledge.visited
    assert "RoomA" not in knowledge.visited or knowledge.visited["RoomA"]
    assert knowledge.exits["RoomA"]
    assert visited.sum() == 2


def test_missing_position_or_bounds_never_fills_map() -> None:
    knowledge = PlayerKnowledge()
    knowledge.update({"operator_telemetry": {"scene_name": "RoomA", "x": 3, "y": 4}})
    vector = knowledge.vector()
    map_start = len(ITEM_FLAGS) * 5
    assert np.all(vector[map_start : map_start + GRID_WIDTH * GRID_HEIGHT] == 0)
    assert GRID_HEIGHT == 9


def test_exit_records_pois_and_memory_round_trip_are_bounded() -> None:
    memory = PlayerKnowledge()
    memory.update({"operator_telemetry": {
        "scene_name": "RoomA", "scene_bounds": [0, 0, 16, 9], "x": 1, "y": 1,
        "milestone_events_since_last_frame": [{"category": "death"}],
    }})
    memory.update({"operator_telemetry": {
        "scene_name": "RoomB", "scene_bounds": [0, 0, 16, 9], "x": 1, "y": 1,
    }})
    assert memory.exit_records["RoomA"][0][3] == 0
    assert memory.points["RoomA"]
    restored = PlayerKnowledge.from_payload(memory.to_payload())
    assert restored.visited == memory.visited
    assert restored.exit_records == memory.exit_records
    assert restored.points == memory.points
