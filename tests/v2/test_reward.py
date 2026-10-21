"""Retorno universal tem custo temporal contínuo e partes separadas."""

import pytest

from voidgym.core.config import RewardConfig
from voidgym.core.reward import UniversalReward, reward_zero_alert


def test_twenty_thousand_quiet_steps_still_have_time_cost() -> None:
    reward = UniversalReward(RewardConfig(time=-0.001))
    before = {"room_id": "room", "player_pos_room": [0.1, 0.1], "hp": 5}
    after = dict(before, progress=0.0)
    reward.compute(before, after)
    total = sum(reward.compute(before, after).universal for _ in range(20_000))
    assert total < -19.0
    assert total != 0


def test_game_reward_is_recorded_separately_and_zero_alert() -> None:
    result = UniversalReward().step({}, {"room_id": "a", "player_pos_room": [0.2, 0.4]}, 0.25)
    assert result.game == 0.25
    assert result.universal > result.game
    assert result.total == result.universal + result.game
    assert reward_zero_alert([0] * 10 + [1])
    assert not reward_zero_alert([1, 1, 1, 1])


def test_positive_rewards_are_capped_by_episode_slice_and_rolling_window() -> None:
    rewarder = UniversalReward(
        RewardConfig(
            novelty=0.0,
            progress=0.0,
            time=-0.01,
            episode_slice_steps=2,
            episode_slice_budget=1.0,
            rolling_window_steps=3,
            rolling_window_budget=1.5,
        )
    )
    before = {"hp": 1}
    after = {"hp": 1, "room_id": "r"}
    values = [rewarder.step(before, after, game_reward=10.0).game for _ in range(4)]
    assert values[0] == pytest.approx(0.01)
    assert values[1:] == [0.0, 1.0, 0.0]
    assert rewarder.step(before, after).universal < 0
