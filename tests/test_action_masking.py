from __future__ import annotations

import pytest

from hk_agent.action_masking import (
    ActionAvailability,
    build_action_mask,
    parse_action_availability,
)
from hk_agent.protocol import ProtocolError
from hk_agent.training_task import TASK_ACTION_NAMES


def test_mask_covers_spell_focus_and_nail_art_availability() -> None:
    unavailable = ActionAvailability(0, 33, 33, 0, 0, 0, False, False, False, False)
    mask = build_action_mask(TASK_ACTION_NAMES, unavailable)
    assert mask[TASK_ACTION_NAMES.index("NOOP")]
    assert not mask[TASK_ACTION_NAMES.index("FOCUS_HOLD")]
    assert not mask[TASK_ACTION_NAMES.index("CAST")]
    assert not mask[TASK_ACTION_NAMES.index("UP+CAST")]
    assert not mask[TASK_ACTION_NAMES.index("DOWN+CAST")]
    assert not mask[TASK_ACTION_NAMES.index("NAIL_ART_CHARGE")]


def test_mask_allows_available_actions_and_unknown_bridge() -> None:
    available = ActionAvailability(33, 33, 33, 1, 1, 1, True, True, True, True)
    mask = build_action_mask(TASK_ACTION_NAMES, available)
    assert mask.all()
    assert build_action_mask(TASK_ACTION_NAMES, None).all()


@pytest.mark.parametrize(
    "value",
    [
        {"mp_charge": -1},
        {"mp_charge": True},
        {"mp_charge": 0.5},
        {"mp_charge": 0, "spell_cost": 0},
    ],
)
def test_malformed_availability_is_protocol_error(value: object) -> None:
    with pytest.raises(ProtocolError):
        parse_action_availability(value)
