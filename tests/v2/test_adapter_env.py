from time import perf_counter

from tests.v2.fake_adapter import FakeAdapter
from voidgym.envs.adapter_env import AdapterEnv


def test_fake_adapter_action_mapping_round_trips() -> None:
    adapter = FakeAdapter()
    for index in range(len(adapter.action_table())):
        assert adapter.from_universal(adapter.to_universal(index)) == index


def test_reset_step_keep_signals_out_of_observation() -> None:
    env = AdapterEnv(FakeAdapter(), lambda before, after: after["position"] - before["position"])
    observation, info = env.reset(seed=1)
    assert env.observation_space.contains(observation)
    assert "signals" in info
    assert "position" not in observation
    next_observation, reward, terminated, truncated, _ = env.step(2)
    assert env.observation_space.contains(next_observation)
    assert reward == 1
    assert not terminated and not truncated
    assert env.action_masks().tolist() == [True, True, False, True, True]


def test_thousand_steps_under_two_seconds() -> None:
    env = AdapterEnv(FakeAdapter(), lambda _before, _after: 0.0)
    env.reset(seed=1)
    start = perf_counter()
    for _ in range(1000):
        env.step(0)
    assert perf_counter() - start < 2.0
