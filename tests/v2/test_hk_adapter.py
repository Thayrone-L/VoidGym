"""Contrato e compatibilidade do adaptador HK; sem teste real do jogo neste PC."""

import pytest

from voidgym.adapters.hollow_knight.adapter import HollowKnightAdapter
from voidgym.core.actions import UNIVERSAL_ACTIONS

EXPECTED_HK_ACTIONS = (
    "NOOP",
    "LEFT",
    "RIGHT",
    "JUMP_TAP",
    "JUMP_HOLD",
    "LEFT+JUMP_TAP",
    "LEFT+JUMP_HOLD",
    "RIGHT+JUMP_TAP",
    "RIGHT+JUMP_HOLD",
    "JUMP_MAX",
    "LEFT+JUMP_MAX",
    "RIGHT+JUMP_MAX",
    "ATTACK",
    "LEFT+ATTACK",
    "RIGHT+ATTACK",
    "UP+ATTACK",
    "DOWN+ATTACK",
    "JUMP+ATTACK",
    "LEFT+JUMP+ATTACK",
    "RIGHT+JUMP+ATTACK",
    "CAST",
    "UP+CAST",
    "DOWN+CAST",
    "FOCUS_HOLD",
    "NAIL_ART_CHARGE",
    "LEFT+NAIL_ART_CHARGE",
    "RIGHT+NAIL_ART_CHARGE",
    "UP+NAIL_ART_CHARGE",
    "DOWN+NAIL_ART_CHARGE",
    "DASH",
    "LEFT+DASH",
    "RIGHT+DASH",
    "DOWN+DASH",
    "SUPER_DASH_HOLD",
    "DREAM_NAIL",
    "INTERACT",
    "MENU_UP",
    "MENU_DOWN",
    "MENU_LEFT",
    "MENU_RIGHT",
    "MENU_SUBMIT",
    "MENU_CANCEL",
    "OPEN_CHARMS",
)


def test_action_table_and_round_trip() -> None:
    class DummyTask:
        pass

    adapter = HollowKnightAdapter(DummyTask())
    assert tuple(adapter.action_table()) == EXPECTED_HK_ACTIONS
    for index, name in enumerate(adapter.action_table()):
        action = adapter.to_universal(index)
        assert action.name == EXPECTED_HK_ACTIONS[index] == name
        assert adapter.from_universal(action) == index
    assert tuple(item.name for item in UNIVERSAL_ACTIONS) == EXPECTED_HK_ACTIONS


@pytest.mark.skip(reason="precisa do jogo e do bridge — validar no PC principal")
def test_hollow_knight_observation_matches_live_environment() -> None:
    raise NotImplementedError


@pytest.mark.skip(reason="checkpoint não existe nesta máquina — validar no PC principal")
def test_existing_checkpoint_spaces_match_live_environment() -> None:
    raise NotImplementedError
