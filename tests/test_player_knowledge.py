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


def test_special_places_are_zero_until_seen_and_remember_only_visible_positions() -> None:
    memory = PlayerKnowledge()
    place_block_start = len(ITEM_FLAGS) * 5 + GRID_WIDTH * GRID_HEIGHT * 2 + 8 * 6 + 8 * 5
    assert np.all(memory.vector()[place_block_start:] == 0)
    memory.update({"operator_telemetry": {
        "scene_name": "RoomA", "scene_bounds": [0, 0, 16, 9], "x": 8, "y": 4,
        "visible_places": [{"kind": "bench", "world_x": 4.0, "world_y": 3.0}],
    }})
    assert memory.seen_places == {"bench"}
    assert memory.special_places["RoomA"]["bench"] == {(4, 3)}
    block = memory.vector()[place_block_start : place_block_start + 4]
    assert block[0] == 1
    place_map_start = place_block_start + 4 * 4
    assert memory.vector()[place_map_start + 3 * GRID_WIDTH + 4] == 1
    assert "RoomB" not in memory.special_places
    restored = PlayerKnowledge.from_payload(memory.to_payload())
    assert restored.special_places == memory.special_places
    assert restored.seen_places == memory.seen_places


def test_story_objectives_wait_for_game_flags_and_pay_once() -> None:
    memory = PlayerKnowledge()
    base = {"scene_bounds": [0, 0, 16, 9], "x": 4, "y": 4}
    memory.update({"operator_telemetry": {**base, "scene_name": "Tutorial_01", "progress_flags": {}}})
    assert memory.current_objective is None
    memory.update({"operator_telemetry": {**base, "scene_name": "Tutorial_01", "progress_flags": {"enteredTutorialFirstTime": True}}})
    assert memory.current_objective == "reach_crossroads"
    memory.update({"operator_telemetry": {**base, "scene_name": "Crossroads_01", "progress_flags": {"enteredTutorialFirstTime": True}}})
    assert "reach_crossroads" in memory.completed_objectives
    assert "objective_completed" in {event["type"] for event in memory.new_events}
    assert "first_target_arrival" in {event["type"] for event in memory.new_events}
    memory.update({"operator_telemetry": {**base, "scene_name": "Crossroads_01", "progress_flags": {"metElderbug": True}}})
    assert memory.current_objective == "defeat_false_knight"
    memory.update({"operator_telemetry": {**base, "scene_name": "Crossroads_10", "progress_flags": {"metElderbug": True, "falseKnightDefeated": True}}})
    assert "defeat_false_knight" in memory.completed_objectives
    memory.update({"operator_telemetry": {**base, "scene_name": "Crossroads_10", "progress_flags": {"metElderbug": True, "falseKnightDefeated": True}}})
    assert memory.current_objective == "reach_greenpath"
    memory.update({"operator_telemetry": {**base, "scene_name": "Fungus1_01", "progress_flags": {"falseKnightDefeated": True}}})
    assert "reach_greenpath" in memory.completed_objectives
    restored = PlayerKnowledge.from_payload(memory.to_payload())
    assert restored.completed_objectives == memory.completed_objectives
    assert len(restored.claimed_tutorial_rewards) == len(memory.claimed_tutorial_rewards)


def test_tutorial_rewards_are_once_per_observed_bench_station_and_travel() -> None:
    memory = PlayerKnowledge()
    telemetry = {
        "scene_name": "Crossroads_01", "scene_bounds": [0, 0, 16, 9], "x": 4, "y": 4,
        "at_bench": True,
        "milestone_events_since_last_frame": [
            {"category": "flag", "key": "openedCrossroads"},
            {"category": "scene_transition", "transition_kind": "stag"},
        ],
    }
    memory.update({"operator_telemetry": telemetry})
    first = [event["type"] for event in memory.new_events]
    assert first.count("first_bench_rest") == 1
    assert first.count("station_opened") == 1
    assert first.count("first_fast_travel") == 1
    memory.update({"operator_telemetry": telemetry})
    assert memory.new_events == []


def test_bestiary_learns_only_from_visible_enemy_experience_and_round_trips() -> None:
    memory = PlayerKnowledge()
    enemy_id = "e_0123456789"
    telemetry = {
        "scene_name": "RoomA", "scene_bounds": [0, 0, 16, 9], "x": 4, "y": 4,
        "visible_enemies": [{
            "enemy_type_id": enemy_id, "instance_id": 41, "screen_x": 0.6,
            "screen_y": 0.5, "struck_this_encounter": 1,
        }],
    }
    memory.update({"operator_telemetry": telemetry, "hp_lost_since_last_frame": 1})
    experience = memory.bestiary.experience[enemy_id]
    assert experience.encounters == 1
    assert experience.damage_taken == 1
    assert experience.defeated == 0
    assert memory.bestiary.slots[0][0] == enemy_id
    memory.update({
        "operator_telemetry": {**telemetry, "visible_enemies": []},
        "bestiary_events_since_last_frame": [{
            "enemy_type_id": enemy_id, "hits_dealt": 3, "geo_awarded": 5,
        }],
    })
    assert experience.defeated == 1
    assert experience.total_hits_to_defeat == 3
    assert experience.total_geo == 5
    restored = PlayerKnowledge.from_payload(memory.to_payload())
    assert restored.bestiary.to_payload() == memory.bestiary.to_payload()
    assert len(restored.bestiary.vector()) == 6 * 13
    assert np.all(np.isfinite(restored.vector()))


def test_bestiary_rejects_unseen_or_malformed_enemy_slots() -> None:
    memory = PlayerKnowledge()
    memory.update({"operator_telemetry": {
        "scene_name": "RoomA", "visible_enemies": [
            {"enemy_type_id": "Crawler", "instance_id": 1, "screen_x": 0.5,
             "screen_y": 0.5, "struck_this_encounter": 0},
            {"enemy_type_id": "e_0123456789", "instance_id": 2, "screen_x": 1.5,
             "screen_y": 0.5, "struck_this_encounter": 0},
        ],
    }})
    assert memory.bestiary.slots == []
    assert memory.bestiary.experience == {}


def test_bestiary_expires_fought_instance_after_ten_missing_frames() -> None:
    memory = PlayerKnowledge()
    enemy_id = "e_0123456789"
    visible = {
        "visible_enemies": [{
            "enemy_type_id": enemy_id, "instance_id": 41, "screen_x": 0.6,
            "screen_y": 0.5, "struck_this_encounter": 1,
        }],
    }
    memory.update({"operator_telemetry": visible})
    for _ in range(10):
        memory.update({"operator_telemetry": {"visible_enemies": []}})
    experience = memory.bestiary.experience[enemy_id]
    assert experience.encounters == 1
    assert experience.fought_encounters == 1
    assert memory.bestiary.slots == []
