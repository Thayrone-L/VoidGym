"""Regression tests for adapter-constrained platform-room search."""

from voidgym.gyms.platform_gym.action_solver import solve_with_actions
from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter


def test_action_solver_returns_a_replayable_adapter_route() -> None:
    adapter = PlatformGymAdapter(level=0, seed=8600, split="test")
    result = solve_with_actions(adapter, max_expansions=5_000)

    assert result.success
    assert result.actions
    replay = PlatformGymAdapter(level=0, seed=8600, split="test")
    action_table = replay.action_table()
    for name in result.actions:
        replay.apply(action_table.index(name))
        signals = replay.signals()
        if signals["room_changed"]:
            break

    assert replay.signals()["room_changed"]


def test_action_solver_reports_exhausted_search_budget() -> None:
    adapter = PlatformGymAdapter(level=0, seed=8600, split="test")
    result = solve_with_actions(adapter, max_expansions=1)

    assert not result.success
    assert result.reason == "node_limit"
