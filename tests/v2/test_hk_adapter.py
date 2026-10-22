import pytest

from tests.test_gym_env import FakeBackend, make_env
from voidgym.adapters.hollow_knight.adapter import HollowKnightAdapter
from voidgym.core.observation import split_observation


class EmptyEnv:
    def __init__(self) -> None:
        self._frames = []
        self.latest_action_availability = None


def test_action_order_matches_current_temporal_table() -> None:
    adapter = HollowKnightAdapter(EmptyEnv())
    assert adapter.action_table() == (
        "NOOP", "LEFT", "RIGHT", "JUMP_TAP", "JUMP_HOLD", "LEFT+JUMP_TAP",
        "LEFT+JUMP_HOLD", "RIGHT+JUMP_TAP", "RIGHT+JUMP_HOLD", "JUMP_MAX",
        "LEFT+JUMP_MAX", "RIGHT+JUMP_MAX", "ATTACK", "LEFT+ATTACK", "RIGHT+ATTACK",
        "UP+ATTACK", "DOWN+ATTACK", "JUMP+ATTACK", "LEFT+JUMP+ATTACK",
        "RIGHT+JUMP+ATTACK", "CAST", "UP+CAST", "DOWN+CAST", "FOCUS_HOLD",
        "NAIL_ART_CHARGE", "LEFT+NAIL_ART_CHARGE", "RIGHT+NAIL_ART_CHARGE",
        "UP+NAIL_ART_CHARGE", "DOWN+NAIL_ART_CHARGE", "DASH", "LEFT+DASH",
        "RIGHT+DASH", "DOWN+DASH", "SUPER_DASH_HOLD", "DREAM_NAIL", "INTERACT",
        "MENU_UP", "MENU_DOWN", "MENU_LEFT", "MENU_RIGHT", "MENU_SUBMIT",
        "MENU_CANCEL", "OPEN_CHARMS",
    )


def test_every_hk_action_round_trips() -> None:
    adapter = HollowKnightAdapter(EmptyEnv())
    for index in range(len(adapter.action_table())):
        assert adapter.from_universal(adapter.to_universal(index)) == index


def test_hk_observation_split_preserves_keys_shapes_and_dtypes(tmp_path) -> None:
    env = make_env(FakeBackend(), tmp_path)
    observation, _info = env.reset(seed=12)
    universal, specific = split_observation(observation)
    assert set(universal) == {"pixels", "action_history"}
    assert specific == {}
    assert all(observation[key].shape == universal[key].shape for key in observation)
    assert all(observation[key].dtype == universal[key].dtype for key in observation)
    env.close()


@pytest.mark.skip(reason="checkpoint da campanha não fica versionado no checkout")
def test_latest_campaign_checkpoint_contract() -> None:
    """Validar espaço e ações quando existir runtime local disponível."""
