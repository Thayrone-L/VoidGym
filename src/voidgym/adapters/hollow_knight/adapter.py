"""Shim de leitura da v1; migração da bridge e do treino fica para o prompt 86."""

from dataclasses import replace
from typing import Any, cast

import numpy as np

from hk_agent.action_masking import build_action_mask  # V1-SHIM: portar no prompt 86
from hk_agent.temporal_actions import (  # V1-SHIM: portar no prompt 86
    PLAYER_INTERACTION_ACTIONS,
    TEMPORAL_COMBAT_ACTIONS,
)
from voidgym.core.actions import UNIVERSAL_ACTIONS, UniversalAction

_TEMPLATES = TEMPORAL_COMBAT_ACTIONS + PLAYER_INTERACTION_ACTIONS
_NAMES = tuple(template.name for template in _TEMPLATES)
_HK_TO_UNIVERSAL = {
    "NOOP": "NOOP", "LEFT": "MOVE_LEFT", "RIGHT": "MOVE_RIGHT",
    "JUMP_TAP": "JUMP_SHORT", "JUMP_HOLD": "JUMP_HOLD",
    "LEFT+JUMP_TAP": "MOVE_LEFT+JUMP_SHORT", "LEFT+JUMP_HOLD": "MOVE_LEFT+JUMP_HOLD",
    "RIGHT+JUMP_TAP": "MOVE_RIGHT+JUMP_SHORT", "RIGHT+JUMP_HOLD": "MOVE_RIGHT+JUMP_HOLD",
    "JUMP_MAX": "JUMP_MAX", "LEFT+JUMP_MAX": "MOVE_LEFT+JUMP_MAX",
    "RIGHT+JUMP_MAX": "MOVE_RIGHT+JUMP_MAX", "ATTACK": "ATTACK",
    "LEFT+ATTACK": "MOVE_LEFT+ATTACK", "RIGHT+ATTACK": "MOVE_RIGHT+ATTACK",
    "UP+ATTACK": "AIM_UP+ATTACK", "DOWN+ATTACK": "AIM_DOWN+ATTACK",
    "JUMP+ATTACK": "JUMP+ATTACK", "LEFT+JUMP+ATTACK": "MOVE_LEFT+JUMP+ATTACK",
    "RIGHT+JUMP+ATTACK": "MOVE_RIGHT+JUMP+ATTACK", "CAST": "SPECIAL_CAST",
    "UP+CAST": "AIM_UP+CAST", "DOWN+CAST": "AIM_DOWN+CAST", "FOCUS_HOLD": "FOCUS",
    "NAIL_ART_CHARGE": "NAIL_ART", "LEFT+NAIL_ART_CHARGE": "MOVE_LEFT+NAIL_ART",
    "RIGHT+NAIL_ART_CHARGE": "MOVE_RIGHT+NAIL_ART", "UP+NAIL_ART_CHARGE": "AIM_UP+NAIL_ART",
    "DOWN+NAIL_ART_CHARGE": "AIM_DOWN+NAIL_ART", "DASH": "DASH",
    "LEFT+DASH": "MOVE_LEFT+DASH", "RIGHT+DASH": "MOVE_RIGHT+DASH",
    "DOWN+DASH": "AIM_DOWN+DASH", "SUPER_DASH_HOLD": "SUPER_DASH",
    "DREAM_NAIL": "DREAM", "INTERACT": "INTERACT", "MENU_UP": "MENU_UP",
    "MENU_DOWN": "MENU_DOWN", "MENU_LEFT": "MENU_LEFT", "MENU_RIGHT": "MENU_RIGHT",
    "MENU_SUBMIT": "MENU_SUBMIT", "MENU_CANCEL": "MENU_CANCEL", "OPEN_CHARMS": "OPEN_MENU",
}
_UNIVERSAL = {action.name: action for action in UNIVERSAL_ACTIONS}


class HollowKnightAdapter:
    """Adapta um HollowKnightEnv existente sem alterar nem mover código da v1."""

    game_id = "hollow_knight"

    def __init__(self, env: Any) -> None:
        self.env = env
        if set(_HK_TO_UNIVERSAL) != set(_NAMES):
            raise RuntimeError("Mapeamento universal de ações do HK incompleto")

    def action_table(self) -> tuple[str, ...]:
        return _NAMES

    def to_universal(self, game_action_index: int) -> UniversalAction:
        template = _TEMPLATES[game_action_index]
        action = _UNIVERSAL[_HK_TO_UNIVERSAL[template.name]]
        return replace(action, game_time_ms=template.resolve_lease_ms(80))

    def from_universal(self, action: UniversalAction) -> int | None:
        for index in range(len(_TEMPLATES)):
            candidate = self.to_universal(index)
            if candidate == action:
                return index
        return None

    def frame_rgb(self) -> np.ndarray:
        if not self.env._frames:
            raise RuntimeError("Hollow Knight ainda não forneceu um quadro")
        return cast(np.ndarray, self.env._frames[-1].pixels)

    def apply(self, game_action_index: int) -> None:
        template = _TEMPLATES[game_action_index]
        self.env.step(template.raw_action_index, lease_ms=template.resolve_lease_ms(80))

    def action_mask(self) -> np.ndarray:
        availability = self.env.latest_action_availability
        return build_action_mask(_NAMES, availability)

    def signals(self) -> dict[str, Any]:
        if not self.env._frames:
            return {}
        frame = self.env._frames[-1]
        return {
            "scene": frame.scene_name,
            "position": frame.operator_telemetry,
            "hp_lost": frame.hp_lost_since_last_frame,
            "died": frame.player_dead,
            "enemy_kills": frame.enemy_kills_since_last_frame,
            "geo_gained": frame.geo_gained_since_last_frame,
        }

    def training_labels(self) -> None:
        return None

    def game_time_scale(self) -> float:
        return 1.0
