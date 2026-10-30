"""GameAdapter implementation that exposes the same ordered actions as HK."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from voidgym.core.actions import UNIVERSAL_ACTIONS, UniversalAction
from voidgym.gyms.platform_gym.generator import generate_room, generate_room_set
from voidgym.gyms.platform_gym.labels import labels
from voidgym.gyms.platform_gym.params import PHYSICS_DT_SECONDS, Split
from voidgym.gyms.platform_gym.physics import Event, interaction_prompt, step_physics
from voidgym.gyms.platform_gym.render import render_room
from voidgym.gyms.platform_gym.world import Player

_NAMES = (
    "NOOP", "LEFT", "RIGHT", "JUMP_TAP", "JUMP_HOLD", "LEFT+JUMP_TAP", "LEFT+JUMP_HOLD",
    "RIGHT+JUMP_TAP", "RIGHT+JUMP_HOLD", "JUMP_MAX", "LEFT+JUMP_MAX", "RIGHT+JUMP_MAX",
    "ATTACK", "LEFT+ATTACK", "RIGHT+ATTACK", "UP+ATTACK", "DOWN+ATTACK", "JUMP+ATTACK",
    "LEFT+JUMP+ATTACK", "RIGHT+JUMP+ATTACK", "CAST", "UP+CAST", "DOWN+CAST", "FOCUS_HOLD",
    "NAIL_ART_CHARGE", "LEFT+NAIL_ART_CHARGE", "RIGHT+NAIL_ART_CHARGE", "UP+NAIL_ART_CHARGE",
    "DOWN+NAIL_ART_CHARGE", "DASH", "LEFT+DASH", "RIGHT+DASH", "DOWN+DASH",
    "SUPER_DASH_HOLD", "DREAM_NAIL", "INTERACT", "MENU_UP", "MENU_DOWN", "MENU_LEFT",
    "MENU_RIGHT", "MENU_SUBMIT", "MENU_CANCEL", "OPEN_CHARMS",
)
_NAME_TO_UNIVERSAL = {
    "LEFT": "MOVE_LEFT", "RIGHT": "MOVE_RIGHT", "JUMP_TAP": "JUMP_SHORT",
    "LEFT+JUMP_TAP": "MOVE_LEFT+JUMP_SHORT", "LEFT+JUMP_HOLD": "MOVE_LEFT+JUMP_HOLD",
    "RIGHT+JUMP_TAP": "MOVE_RIGHT+JUMP_SHORT", "RIGHT+JUMP_HOLD": "MOVE_RIGHT+JUMP_HOLD",
    "LEFT+JUMP+ATTACK": "MOVE_LEFT+JUMP+ATTACK", "RIGHT+JUMP+ATTACK": "MOVE_RIGHT+JUMP+ATTACK",
    "LEFT+JUMP_MAX": "MOVE_LEFT+JUMP_MAX", "RIGHT+JUMP_MAX": "MOVE_RIGHT+JUMP_MAX",
    "LEFT+ATTACK": "MOVE_LEFT+ATTACK", "RIGHT+ATTACK": "MOVE_RIGHT+ATTACK",
    "UP+ATTACK": "AIM_UP+ATTACK", "DOWN+ATTACK": "AIM_DOWN+ATTACK",
    "CAST": "SPECIAL_CAST", "UP+CAST": "AIM_UP+CAST", "DOWN+CAST": "AIM_DOWN+CAST",
    "FOCUS_HOLD": "FOCUS", "NAIL_ART_CHARGE": "NAIL_ART",
    "LEFT+NAIL_ART_CHARGE": "MOVE_LEFT+NAIL_ART", "RIGHT+NAIL_ART_CHARGE": "MOVE_RIGHT+NAIL_ART",
    "UP+NAIL_ART_CHARGE": "AIM_UP+NAIL_ART", "DOWN+NAIL_ART_CHARGE": "AIM_DOWN+NAIL_ART",
    "LEFT+DASH": "MOVE_LEFT+DASH", "RIGHT+DASH": "MOVE_RIGHT+DASH", "DOWN+DASH": "AIM_DOWN+DASH",
    "SUPER_DASH_HOLD": "SUPER_DASH", "DREAM_NAIL": "DREAM", "OPEN_CHARMS": "OPEN_MENU",
}
_ACTIONS = {a.name: a for a in UNIVERSAL_ACTIONS}
_HK_ACTIONS = tuple(_ACTIONS[_NAME_TO_UNIVERSAL.get(name, name)] for name in _NAMES)


class PlatformGymAdapter:
    game_id = "platform_gym"

    def __init__(self, *, level: int = 0, seed: int = 0, split: Split = "train") -> None:
        self.level, self.seed, self.split = level, seed, split
        self._rooms = generate_room_set(seed) if level == 8 else [generate_room(level, seed)]
        self._room_index = 0
        self.room = self._rooms[0]
        if level != 8:
            self.room.id = 0
        self.player = Player(*self.room.spawn, on_ground=True)
        self._events: list[Event] = []
        self._episode_steps = 0
        self._room_counter = 0
        self._last_step_game_ms = 80
        self._finished = False
        self._steps_since_new_cell = 0
        self._visited: set[tuple[int, int, int]] = set()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        del options
        if seed is not None:
            self.seed = seed
        self._room_counter = 0
        self._rooms = generate_room_set(self.seed) if self.level == 8 else [generate_room(self.level, self.seed + self._room_counter)]
        self._room_index = 0
        self.room = self._rooms[0]
        if self.level != 8:
            self.room.id = 0
        self.player = Player(*self.room.spawn, on_ground=True)
        self._events.clear()
        self._episode_steps = 0
        self._last_step_game_ms = 80
        self._finished = False
        self._steps_since_new_cell = 0
        self._visited.clear()

    def action_table(self) -> Sequence[str]:
        return _NAMES

    def to_universal(self, game_action_index: int) -> UniversalAction:
        template = _HK_ACTIONS[game_action_index]
        duration = 50 if "TAP" in _NAMES[game_action_index] or _NAMES[game_action_index] in {"ATTACK", "CAST", "INTERACT"} else template.game_time_ms
        return replace(template, game_time_ms=duration)

    def from_universal(self, action: UniversalAction) -> int | None:
        return next((i for i in range(len(_NAMES)) if self.to_universal(i) == action), None)

    def frame_rgb(self) -> np.ndarray:
        return render_room(self.room, self.player, split=self.split)

    def apply(self, game_action_index: int) -> None:
        action = self.to_universal(game_action_index)
        self._last_step_game_ms = action.game_time_ms
        action_name = _NAMES[game_action_index]
        # Held jump retains the button across repeated policy actions; all other short
        # presses release on the next internal tick. JUMP_MAX spans its declared 450 ms.
        jump = action.jump is not None
        btn = {
            "left": action.move < 0, "right": action.move > 0, "up": action.aim == "up",
            "down": action.aim == "down", "jump": jump, "attack": action.attack,
            "dash": action.dash and "ability_dash" in self.player.abilities,
            "interact": action.interact and interaction_prompt(self.room, self.player),
        }
        if action.special or action.menu or action_name in {"CAST", "NAIL_ART_CHARGE"}:
            btn["jump"] = False
        count = max(1, round(action.game_time_ms / (PHYSICS_DT_SECONDS * 1000)))
        self._events = []
        for tick in range(count):
            frame_btn = dict(btn)
            if action.jump is not None and action_name.endswith("TAP") and tick > 0:
                frame_btn["jump"] = False
            if action.interact and tick > 0:
                frame_btn["interact"] = False
            self._events.extend(step_physics(self.room, self.player, frame_btn, PHYSICS_DT_SECONDS))
            if any(event.kind == "exit" for event in self._events):
                self._room_counter += 1
                if self._room_index + 1 < len(self._rooms):
                    abilities = set(self.player.abilities)
                    self._room_index += 1
                    self.room = self._rooms[self._room_index]
                    self.player = Player(*self.room.spawn, on_ground=True, abilities=abilities)
                elif self.level == 8:
                    self._finished = True
                else:
                    self.room = generate_room(self.level, self.seed + self._room_counter)
                    self.room.id = self._room_counter
                    self.player = Player(*self.room.spawn, on_ground=True)
                break
        self._episode_steps += 1
        cell = (self.room.id, int(self.player.x / 2), int(self.player.y / 2))
        if cell not in self._visited:
            self._visited.add(cell)
            self._steps_since_new_cell = 0
        else:
            self._steps_since_new_cell += 1

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(len(_NAMES), dtype=np.bool_)
        for i, name in enumerate(_NAMES):
            action = _HK_ACTIONS[i]
            unavailable_special = bool(action.special or action.menu or name in {"CAST", "UP+CAST", "DOWN+CAST", "FOCUS_HOLD", "NAIL_ART_CHARGE", "LEFT+NAIL_ART_CHARGE", "RIGHT+NAIL_ART_CHARGE", "UP+NAIL_ART_CHARGE", "DOWN+NAIL_ART_CHARGE", "SUPER_DASH_HOLD", "DREAM_NAIL", "MENU_UP", "MENU_DOWN", "MENU_LEFT", "MENU_RIGHT", "MENU_SUBMIT", "MENU_CANCEL", "OPEN_CHARMS"})
            mask[i] = not unavailable_special
            if action.dash and "ability_dash" not in self.player.abilities:
                mask[i] = False
            if action.interact and not interaction_prompt(self.room, self.player):
                mask[i] = False
        mask[0] = True
        return mask

    def signals(self) -> Mapping[str, Any]:
        return {"room_id": self.room.id, "player_pos_room": (self.player.x / self.room.width, self.player.y / self.room.height),
                "player_pos_world": (self.player.x, self.player.y),
                "hp": self.player.hp, "events": tuple(self._events), "step_game_ms": self._last_step_game_ms,
                "progress": self.player.x / self.room.width, "position": self.player.x,
                "died": any(e.kind == "death" for e in self._events),
                "damage_taken": sum(e.kind == "damage" for e in self._events),
                "room_changed": any(e.kind == "exit" for e in self._events),
                "terminated": self._finished,
                "truncated": self._episode_steps >= 1500 or self._steps_since_new_cell >= 300,
                "stuck": self._steps_since_new_cell >= 300}

    def training_labels(self) -> Mapping[str, Any]:
        return labels(self.room, self.player, split=self.split)

    def game_time_scale(self) -> float:
        return 1.0
