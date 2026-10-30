from __future__ import annotations

import pytest

from voidgym.core.config import RewardConfig
from voidgym.gyms.platform_gym.physics import Event
from voidgym.gyms.platform_gym.reward import (
    CombinedPlatformReward,
    GameReward,
    GameRewardConfig,
)


def _state(x: float, *, room: int = 0) -> dict[str, object]:
    return {"room_id": room, "player_pos_room": (x / 40, .25), "player_pos_world": (x, 1.0),
            "hp": 3, "step_game_ms": 80, "events": (), "progress": x / 40}


def test_world_novelty_uses_two_unit_cells_and_caps_per_room() -> None:
    config = RewardConfig(novelty=.02, progress=0, damage=0, death=0, time_cost=0, room_change=0)
    reward = CombinedPlatformReward(config)
    assert reward(_state(1), _state(1)) == pytest.approx(.02)
    assert reward(_state(1), _state(1)) == pytest.approx(0)
    assert reward(_state(2), _state(2)) == pytest.approx(.02)
    total = .04
    for index in range(2, 80):
        x = float(index * 2 + .1)
        total += reward(_state(x), _state(x))
    assert total == pytest.approx(1.0)
    assert reward(_state(200), _state(200)) == pytest.approx(0)
    assert reward(_state(1, room=1), _state(1, room=1)) == pytest.approx(.02)


def test_game_reward_values_and_damage_cost() -> None:
    reward = CombinedPlatformReward()
    before = _state(1)
    after = _state(1)
    after.update(hp=2, events=(Event("exit"), Event("pickup", "coin"), Event("ability")))
    assert reward(before, after) == pytest.approx(.02 - .3 + 1.0 + .2 + 1.0 - .001)


def test_first_enemy_and_treasure_rock_hits_pay_once_and_enemy_kill_is_increased() -> None:
    game = GameReward()
    before = _state(1)
    after = _state(1)
    after["events"] = (Event("hit_enemy", "enemy-1"), Event("kill", "enemy-1"))
    assert game(before, after) == pytest.approx(.35)
    after["events"] = (Event("hit_enemy", "enemy-1"),)
    assert game(before, after) == 0
    after["events"] = (Event("hit_treasure_rock", "rock-1"),)
    assert game(before, after) == pytest.approx(.05)
    assert game(before, after) == 0
    assert game(before, {**after, "events": (Event("kill", "enemy-2"),)}) == pytest.approx(.3)
    assert game(before, {**after, "events": (Event("kill", "enemy-3"),)}) == pytest.approx(.3)
    assert game(before, {**after, "events": (Event("kill", "enemy-4"),)}) == 0


def test_game_reward_values_are_configurable_and_validated() -> None:
    game = GameReward(GameRewardConfig(first_hit=.08, enemy_kill=.4,
                                       kill_limit_per_room=2, coin=.25))
    before = _state(1)
    assert game(before, {**before, "events": (Event("hit_enemy", "e1"),
                                               Event("kill", "e1"),
                                               Event("pickup", "coin"))}) == pytest.approx(.73)
    with pytest.raises(ValueError):
        GameReward(GameRewardConfig(first_hit=-.1))
