"""Shim de leitura da v1; migração da bridge e do treino fica para o prompt 86."""

from dataclasses import replace
from typing import Any, cast

import numpy as np

from hk_agent.action_masking import (  # V1-SHIM: portar no prompt 86
    availability_dict,
    build_action_mask,
)
from hk_agent.temporal_actions import (  # V1-SHIM: portar no prompt 86
    PLAYER_INTERACTION_ACTIONS,
    TEMPORAL_COMBAT_ACTIONS,
)
from voidgym.core.actions import UNIVERSAL_ACTIONS, UniversalAction
from voidgym.core.errors import VoidGymRuntimeError as RuntimeError

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
        self._last_info: dict[str, Any] = {}
        self._last_reward = 0.0
        self._task_mode = hasattr(env, "environment") and hasattr(env, "config") and hasattr(env, "action_masks")
        if self._task_mode and tuple(getattr(env, "_action_names", ())) != _NAMES:
            raise RuntimeError("explore_v4 action table differs from the universal HK adapter")
        if set(_HK_TO_UNIVERSAL) != set(_NAMES):
            raise RuntimeError("Mapeamento universal de ações do HK incompleto")

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        _observation, info = self.env.reset(seed=seed, options=options)
        self._last_info = dict(info)
        self._last_reward = 0.0

    def _game_env(self) -> Any:
        return self.env.environment if self._task_mode else self.env

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
        game_env = self._game_env()
        if not game_env._frames:
            raise RuntimeError("Hollow Knight ainda não forneceu um quadro")
        return cast(np.ndarray, game_env._frames[-1].pixels)

    def apply(self, game_action_index: int) -> None:
        template = _TEMPLATES[game_action_index]
        if self._task_mode:
            _observation, reward, terminated, truncated, info = self.env.step(
                game_action_index
            )
            info = {**info, "terminated": bool(terminated), "truncated": bool(truncated)}
        else:
            _observation, reward, _terminated, _truncated, info = self.env.step(
                template.raw_action_index, lease_ms=template.resolve_lease_ms(80)
            )
        self._last_reward = float(reward)
        self._last_info = dict(info)

    def action_mask(self) -> np.ndarray:
        if self._task_mode:
            return np.asarray(self.env.action_masks(), dtype=np.bool_)
        availability = self.env.latest_action_availability
        return build_action_mask(_NAMES, availability)

    def signals(self) -> dict[str, Any]:
        game_env = self._game_env()
        if not game_env._frames:
            return {}
        if self._task_mode:
            frame = game_env._frames[-1]
            result = {"scene_name": frame.scene_name,
                      "terminated": bool(self._last_info.get("terminated", False)),
                      "truncated": bool(self._last_info.get("truncated", False)),
                      "room_changed": bool(self._last_info.get("scene_change_event", False)),
                      "died": self._last_info.get("termination_reason") == "player_dead",
                      "progress": max((float(value) for value in self._last_info.get("max_progress_by_scene", {}).values()), default=0.0)
                      if isinstance(self._last_info.get("max_progress_by_scene"), dict) else 0.0}
            telemetry = frame.operator_telemetry
            if telemetry:
                result["operator_telemetry"] = dict(telemetry)
                bounds = telemetry.get("scene_bounds")
                x, y = telemetry.get("x"), telemetry.get("y")
                if (isinstance(bounds, (list, tuple)) and len(bounds) == 4
                        and all(isinstance(item, (int, float)) for item in bounds)
                        and isinstance(x, (int, float)) and isinstance(y, (int, float))):
                    left, bottom, right, top = (float(item) for item in bounds)
                    result["player_pos_room"] = (
                        min(1.0, max(0.0, (float(x) - left) / max(1e-6, right - left))),
                        min(1.0, max(0.0, (float(y) - bottom) / max(1e-6, top - bottom))),
                    )
                if "hp" in telemetry:
                    result["hp"] = telemetry["hp"]
            return result
        frame = game_env._frames[-1]
        signal_names = {
            "enemy_kills_since_last_frame", "hp_lost_since_last_frame",
            "geo_gained_since_last_frame", "geo_spent_since_last_frame",
            "hp_restored_since_last_frame", "shade_hits_since_last_frame",
            "shade_distance", "scene_name", "shade_pending", "enemy_events_since_last_frame",
            "bestiary_events_since_last_frame", "run_failed", "supervisor",
            "operator_telemetry", "map_zone", "x", "y", "camera_x", "camera_y",
            "scene_bounds", "milestone_events_since_last_frame",
        }
        result = {key: value for key, value in self._last_info.items() if key in signal_names}
        result.update({
            "scene_name": frame.scene_name,
            "scene_transition": frame.scene_transition,
            "shade_pending": frame.shade_pending,
            "shade_recovered": frame.shade_recovered,
            "run_failed": frame.run_failed,
            "enemy_kills_since_last_frame": frame.enemy_kills_since_last_frame,
            "hp_lost_since_last_frame": frame.hp_lost_since_last_frame,
            "geo_gained_since_last_frame": frame.geo_gained_since_last_frame,
            "geo_spent_since_last_frame": frame.geo_spent_since_last_frame,
            "hp_restored_since_last_frame": frame.hp_restored_since_last_frame,
            "shade_hits_since_last_frame": frame.shade_hits_since_last_frame,
            "shade_distance": frame.shade_distance,
            "enemy_events_since_last_frame": frame.enemy_events_since_last_frame,
            "bestiary_events_since_last_frame": frame.bestiary_events,
            "action_availability": availability_dict(frame.action_availability),
        })
        if frame.operator_telemetry is not None:
            telemetry = dict(frame.operator_telemetry)
            result["operator_telemetry"] = telemetry
            result["position"] = telemetry
            for key in ("map_zone", "x", "y", "camera_x", "camera_y", "scene_bounds",
                        "milestone_events_since_last_frame"):
                if key in telemetry:
                    result[key] = telemetry[key]
        return result

    def training_labels(self) -> dict[str, Any] | None:
        """Return optional bridge oracle labels; these never enter observations."""
        game_env = self._game_env()
        if not game_env._frames:
            return None
        frame = game_env._frames[-1]
        sem = getattr(frame, "semantic_labels", None)
        if sem is None:
            return None
        labels: dict[str, Any] = {
            "sem": sem,
            "interact_prompt": int(bool(getattr(frame, "interact_prompt", False))),
            "interact_prompt_valid": bool(getattr(frame, "interact_prompt_valid", False)),
        }
        motion = getattr(frame, "hero_motion", None)
        if motion is not None:
            labels["hero_velocity"] = np.asarray(motion[:2], dtype=np.float32)
            labels["on_ground"] = int(motion[2])
        return labels

    def game_extras(self) -> np.ndarray:
        """GameExtrasAdapter[hollow_knight]: private player/bestiary knowledge."""
        if not self._task_mode:
            return np.zeros(1, dtype=np.float32)
        knowledge = getattr(self.env, "_player_knowledge", None)
        if knowledge is None:
            raise RuntimeError("explore_v4 has not initialized player knowledge")
        return np.asarray(knowledge.vector(hint_level=self.env.config.tutorial_hint_level),
                          dtype=np.float32)

    @staticmethod
    def game_extras_size() -> int:
        from hk_agent.player_knowledge import PlayerKnowledge

        return int(PlayerKnowledge.space().shape[0])

    def game_time_scale(self) -> float:
        return 1.0

    @property
    def last_reward(self) -> float:
        return self._last_reward

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
