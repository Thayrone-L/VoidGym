"""Testes da composição Gymnasium e do contrato de observação universal."""

import time

import numpy as np

from tests.v2.fake_adapter import FakeAdapter
from voidgym.core.observation import fit_to_universal, split_observation
from voidgym.envs.adapter_env import AdapterEnv


def test_fit_to_universal_handles_common_aspect_ratios() -> None:
    for height, width in ((108, 192), (144, 192), (81, 192)):
        fitted = fit_to_universal(np.full((height, width, 3), 100, dtype=np.uint8))
        assert fitted.shape == (108, 192, 3)
        assert fitted.dtype == np.uint8


def test_reset_step_spaces_signals_and_masks() -> None:
    adapter = FakeAdapter()
    env = AdapterEnv(
        adapter, lambda before, after: float(after["position"] - before.get("position", 0))
    )
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert info["signals"]["position"] == adapter.position
    assert env.action_masks().tolist() == [True] * 5
    obs, _reward, _terminated, _truncated, info = env.step(2)
    assert env.observation_space.contains(obs)
    assert set(obs) == {"pixels", "action_history"}
    assert info["signals"]["position"] == adapter.position
    universal, extras = split_observation({**obs, "game_extra": np.asarray([1])})
    assert set(universal) == {"pixels", "action_history"}
    assert set(extras) == {"game_extra"}


def test_info_labels_never_enter_observation_and_speed() -> None:
    adapter = FakeAdapter()
    adapter._labels = {"sem": np.zeros((27, 48), dtype=np.uint8)}
    env = AdapterEnv(adapter, lambda _before, _after: 0.0)
    env.reset()
    started = time.perf_counter()
    for _ in range(1000):
        obs, *_rest, info = env.step(0)
        assert env.observation_space.contains(obs)
        assert "sem" not in obs
        assert "training_labels" in info
    assert time.perf_counter() - started < 2.0


def test_invalid_action_mask_shape_is_rejected() -> None:
    class BadMask(FakeAdapter):
        def action_mask(self) -> np.ndarray:
            return np.ones(2, dtype=np.bool_)

    env = AdapterEnv(BadMask(), lambda _before, _after: 0.0)
    try:
        env.action_masks()
    except ValueError as error:
        assert "máscara" in str(error)
    else:
        raise AssertionError("a forma inválida da máscara deve ser rejeitada")
