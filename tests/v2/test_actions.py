import pytest

from voidgym.core.actions import Duration, UNIVERSAL_ACTIONS, UniversalAction
from voidgym.core.errors import VoidGymError


def test_universal_actions_use_only_contract_values_and_unique_names() -> None:
    assert len(UNIVERSAL_ACTIONS) >= 43
    assert len({action.name for action in UNIVERSAL_ACTIONS}) == len(UNIVERSAL_ACTIONS)
    assert all(action.special in {None, "cast", "focus", "super_dash", "dream"}
               for action in UNIVERSAL_ACTIONS)
    assert all(action.menu in {None, "up", "down", "left", "right", "submit", "cancel"}
               for action in UNIVERSAL_ACTIONS)
    assert all(action.jump is None or isinstance(action.jump, Duration)
               for action in UNIVERSAL_ACTIONS)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"special": "charge"},
        {"menu": "open"},
        {"game_time_ms": 1.5},
        {"attack": 1},
    ],
)
def test_universal_action_rejects_values_outside_contract(kwargs) -> None:
    with pytest.raises(VoidGymError):
        UniversalAction("INVALID", **kwargs)
