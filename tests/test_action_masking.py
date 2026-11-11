from __future__ import annotations

import pytest

from hk_agent.action_masking import (
    ActionAvailability,
    build_action_mask,
    parse_action_availability,
)
from hk_agent.protocol import ProtocolError
from hk_agent.scene_change_task import SCENE_CHANGE_INTERACTION_ACTION_NAMES
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


def test_jump_remains_available_in_traversal_without_special_abilities() -> None:
    unavailable_abilities = ActionAvailability(0, 33, 33, 0, 0, 0, False, False, False, False)
    mask = build_action_mask(TASK_ACTION_NAMES, unavailable_abilities)
    assert mask[TASK_ACTION_NAMES.index("JUMP_TAP")]
    assert mask[TASK_ACTION_NAMES.index("JUMP_HOLD")]
    assert mask[TASK_ACTION_NAMES.index("LEFT+JUMP_HOLD")]
    assert mask[TASK_ACTION_NAMES.index("RIGHT+JUMP_HOLD")]


def test_mask_allows_available_actions_and_unknown_bridge() -> None:
    available = ActionAvailability(33, 33, 33, 1, 1, 1, True, True, True, True,
                                   has_dash=True, has_super_dash=True, has_dream_nail=True,
                                   has_dashmaster=True)
    mask = build_action_mask(TASK_ACTION_NAMES, available)
    assert all(mask[index] for index, name in enumerate(TASK_ACTION_NAMES) if not name.startswith("MENU_"))
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


def test_interaction_is_always_available_and_menu_actions_are_contextual() -> None:
    names = SCENE_CHANGE_INTERACTION_ACTION_NAMES
    closed = ActionAvailability(33, 33, 33, 1, 1, 1, True, True, True, True)
    closed_mask = build_action_mask(names, closed)
    assert closed_mask[names.index("INTERACT")]
    for name in ("MENU_UP", "MENU_DOWN", "MENU_LEFT", "MENU_RIGHT", "MENU_SUBMIT", "MENU_CANCEL"):
        assert not closed_mask[names.index(name)]
    open_menu = ActionAvailability(33, 33, 33, 1, 1, 1, True, True, True, True, menu_open=True)
    open_mask = build_action_mask(names, open_menu)
    assert open_mask[names.index("MENU_UP")]
    assert open_mask[names.index("MENU_SUBMIT")]
    assert open_mask[names.index("MENU_CANCEL")]
    assert not open_mask[names.index("LEFT")]
    assert not open_mask[names.index("ATTACK")]
    assert open_mask[names.index("NOOP")]
    assert not closed_mask[names.index("OPEN_CHARMS")]


def test_open_charms_requires_a_seated_bench_and_menu_actions_require_menu_state() -> None:
    names = SCENE_CHANGE_INTERACTION_ACTION_NAMES
    away = ActionAvailability(33, 33, 33, 1, 1, 1, True, True, True, True)
    seated = ActionAvailability(33, 33, 33, 1, 1, 1, True, True, True, True, at_bench=True)
    assert not build_action_mask(names, away)[names.index("OPEN_CHARMS")]
    assert build_action_mask(names, seated)[names.index("OPEN_CHARMS")]
    on_menu = ActionAvailability(33, 33, 33, 1, 1, 1, True, True, True, True,
                                 at_bench=True, menu_open=True, charm_menu_open=True,
                                 charm_cursor=3, equipped_charm_mask=5,
                                 charm_notches_used=2, charm_notches_total=3)
    mask = build_action_mask(names, on_menu)
    assert not mask[names.index("OPEN_CHARMS")]
    assert mask[names.index("MENU_UP")]
    assert not mask[names.index("ATTACK")]


def test_legacy_action_availability_defaults_menu_flags_closed() -> None:
    values = {
        "mp_charge": 0, "spell_cost": 33, "focus_cost": 33,
        "fireball_level": 0, "quake_level": 0, "scream_level": 0,
        "has_nail_art": False, "has_cyclone": False,
        "has_dash_slash": False, "has_upward_slash": False,
    }
    parsed = parse_action_availability(values)
    assert parsed is not None and not parsed.dialog_open and not parsed.menu_open
    assert not parsed.at_bench and parsed.equipped_charm_mask == 0


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
