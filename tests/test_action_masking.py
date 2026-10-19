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
    available = ActionAvailability(33, 33, 33, 1, 1, 1, True, True, True, True,
                                   has_dash=True, has_super_dash=True, has_dream_nail=True,
                                   has_dashmaster=True)
    mask = build_action_mask(TASK_ACTION_NAMES, available)
    assert mask.all()
    unknown_mask = build_action_mask(TASK_ACTION_NAMES, None)
    assert unknown_mask[TASK_ACTION_NAMES.index("LEFT")]
    assert not unknown_mask[TASK_ACTION_NAMES.index("DASH")]


def test_new_ability_actions_remain_masked_until_owned() -> None:
    unavailable = ActionAvailability(33, 33, 33, 0, 0, 0, False, False, False, False)
    mask = build_action_mask(TASK_ACTION_NAMES, unavailable)
    assert not mask[TASK_ACTION_NAMES.index("DASH")]
    assert not mask[TASK_ACTION_NAMES.index("SUPER_DASH_HOLD")]
    assert not mask[TASK_ACTION_NAMES.index("DREAM_NAIL")]
    assert not mask[TASK_ACTION_NAMES.index("LEFT+DASH")]
    owned = ActionAvailability(33, 33, 33, 0, 0, 0, False, False, False, False,
                               has_dash=True, has_super_dash=True, has_dream_nail=True,
                               has_dashmaster=True)
    owned_mask = build_action_mask(TASK_ACTION_NAMES, owned)
    assert owned_mask[TASK_ACTION_NAMES.index("DASH")]
    assert owned_mask[TASK_ACTION_NAMES.index("SUPER_DASH_HOLD")]
    assert owned_mask[TASK_ACTION_NAMES.index("DREAM_NAIL")]


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
