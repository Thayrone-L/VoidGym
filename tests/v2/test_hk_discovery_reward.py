from __future__ import annotations

import pytest

from voidgym.adapters.hollow_knight.discovery_reward import (
    DiscoveryReward,
    DiscoveryRewardConfig,
)


def _event(kind: str, object_id: str, scene: str = "room-a") -> dict[str, object]:
    return {"kind": kind, "scene_name": scene, "object_id": object_id, "amount": 0}


def test_geo_rock_hit_and_break_pay_once_across_episode_slice_resets() -> None:
    reward = DiscoveryReward()
    assert reward({}, {"combat_events": [_event("hit_geo_rock", "rock-1")]}) == .05
    assert reward({}, {"combat_events": [_event("hit_geo_rock", "rock-1")]}) == 0
    assert reward({}, {"combat_events": [_event("geo_rock_broken", "rock-1")]}) == .1
    reward.reset_slice()
    assert reward({}, {"combat_events": [_event("hit_geo_rock", "rock-1")]}) == 0
    assert reward({}, {"combat_events": [_event("hit_geo_rock", "rock-2", "room-b")]}) == .05


def test_enemy_kill_discovery_is_once_per_scene_instance() -> None:
    reward = DiscoveryReward()
    event = _event("kill_enemy", "enemy-1")
    assert reward({}, {"combat_events": [event]}) == .15
    reward.reset_slice()
    assert reward({}, {"combat_events": [event]}) == 0
    assert reward({}, {"combat_events": [_event("kill_enemy", "enemy-1", "room-b")]}) == .15


def test_room_and_episode_slice_budgets_are_enforced() -> None:
    reward = DiscoveryReward(DiscoveryRewardConfig(
        first_geo_rock_hit=.2, geo_rock_broken=.2, first_enemy_kill=.2,
        room_budget=.25, episode_slice_budget=.3,
    ))
    assert reward({}, {"combat_events": [
        _event("hit_geo_rock", "rock-1"), _event("geo_rock_broken", "rock-1"),
    ]}) == pytest.approx(.25)
    assert reward({}, {"combat_events": [_event("kill_enemy", "enemy-1", "room-b")]}) == pytest.approx(.05)
    assert reward({}, {"combat_events": [_event("hit_geo_rock", "rock-2", "room-c")]}) == 0


def test_reward_config_rejects_negative_and_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        DiscoveryReward(DiscoveryRewardConfig(first_enemy_kill=-.1))
    with pytest.raises(ValueError):
        DiscoveryReward(DiscoveryRewardConfig(room_budget=float("inf")))
