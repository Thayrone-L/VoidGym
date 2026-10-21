"""Sinais específicos do ginásio; ações e observações permanecem universais."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from voidgym.core.actions import UNIVERSAL_ACTION_BY_NAME, UNIVERSAL_ACTIONS, UniversalAction
from voidgym.gyms.platform_gym.params import SUBSTEPS
from voidgym.gyms.platform_gym.physics import Event, step_physics
from voidgym.gyms.platform_gym.render import render
from voidgym.gyms.platform_gym.styles import RenderStyle, make_style
from voidgym.gyms.platform_gym.world import Player, Room


class PlatformGymAdapter:
    game_id = "platform_gym"

    def __init__(self, room: Room | None = None, *, split: str = "train") -> None:
        from voidgym.gyms.platform_gym.generator import generate_room

        self.room = room or generate_room(0, 1)
        self.player = Player(*self.room.spawn)
        self.player.last_safe_pos = self.room.spawn
        self.style: RenderStyle = make_style(self.room.style_seed, split)
        self.events: list[Event] = []
        self.steps = 0
        self._last_cell = (int(self.player.x), int(self.player.y))
        self._stuck_steps = 0
        self.terminated = False
        self.truncated = False
        self._last_action = 0
        self._names = tuple(action.name for action in UNIVERSAL_ACTIONS)

    def action_table(self) -> Sequence[str]:
        return self._names

    def to_universal(self, game_action_index: int) -> UniversalAction:
        try:
            return UNIVERSAL_ACTION_BY_NAME[self._names[game_action_index]]
        except (IndexError, KeyError) as error:
            raise ValueError("Índice de ação do ginásio inválido") from error

    def from_universal(self, action: UniversalAction) -> int | None:
        return self._names.index(action.name) if action.name in self._names else None

    def frame_rgb(self) -> NDArray[np.uint8]:
        return render(self.room, self.player, self.style)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        from voidgym.gyms.platform_gym.generator import generate_room

        level = int((options or {}).get("level", 0))
        chosen_seed = seed if seed is not None else self.room.style_seed
        self.room = generate_room(level, chosen_seed)
        self.player = Player(*self.room.spawn)
        self.player.last_safe_pos = self.room.spawn
        self.style = make_style(chosen_seed, self.style.style_params.split)
        self.events = []
        self.steps = 0
        self._last_cell = (int(self.player.x), int(self.player.y))
        self._stuck_steps = 0
        self.terminated = False
        self.truncated = False

    def apply(self, game_action_index: int) -> None:
        action = self.to_universal(game_action_index)
        self._last_action = game_action_index
        self.events = []
        for substep in range(SUBSTEPS):
            elapsed_ms = substep * 20
            buttons = {
                "left": action.move < 0,
                "right": action.move > 0,
                "up": action.aim == "up",
                "down": action.aim == "down",
                "jump": action.jump is not None and (action.jump.value == "max" or elapsed_ms < action.game_time_ms),
                "attack": action.attack and elapsed_ms < action.game_time_ms,
                "dash": action.dash and elapsed_ms < action.game_time_ms,
                "interact": action.interact and bool(self.training_labels()["interact_prompt"]),
            }
            self.events.extend(step_physics(self.room, self.player, buttons))
        self.steps += 1
        cell = (int(self.player.x), int(self.player.y))
        self._stuck_steps = 0 if cell != self._last_cell else self._stuck_steps + 1
        self._last_cell = cell
        self.terminated = any(event.kind == "exit" for event in self.events)
        self.truncated = self.steps >= 1500 or self._stuck_steps >= 300

    def action_mask(self) -> NDArray[np.bool_]:
        mask = np.ones(len(self._names), dtype=np.bool_)
        for index, action in enumerate(UNIVERSAL_ACTIONS):
            if action.special or action.menu:
                mask[index] = False
            if action.dash and "ability_dash" not in self.player.abilities:
                mask[index] = False
            if action.interact and not self.training_labels()["interact_prompt"]:
                mask[index] = False
        return mask

    def signals(self) -> Mapping[str, Any]:
        return {"room_id": str(self.room.id), "player_pos_room": [self.player.x / self.room.width, self.player.y / self.room.height],
                "hp": self.player.hp, "max_hp": self.player.max_hp, "died": any(e.kind == "death" for e in self.events),
                "room_changed": any(e.kind == "exit" for e in self.events), "progress": self.player.x / self.room.width,
                "player_pos_world": [self.player.x, self.player.y],
                "events": [event.kind for event in self.events], "terminated": self.terminated, "truncated": self.truncated}

    def training_labels(self) -> Mapping[str, Any]:
        from voidgym.gyms.platform_gym.labels import labels

        camera_x = min(max(self.player.x - self.style.style_params.camera_width * .42, 0), max(0, self.room.width - self.style.style_params.camera_width))
        return labels(self.room, self.player, camera_x, max(0, self.player.y - 3), self.style)

    def game_time_scale(self) -> float:
        return 1.0

    def game_reward(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
        del before
        reward = sum({"exit": 1.0, "pickup": .2, "ability": 1.0, "bench": .5, "break": .3, "kill": .1, "damage": -.3, "death": -1.0}.get(kind, 0.0) for kind in after.get("events", []))
        return reward - .001
