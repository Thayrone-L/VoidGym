"""Game reward com memória de progresso por sala."""

from collections.abc import Mapping
from typing import Any


class GymReward:
    def __init__(self) -> None:
        self.visited: dict[str, set[tuple[int, int]]] = {}
        self.events: set[tuple[str, str]] = set()
        self.kills: dict[str, int] = {}

    def reset(self) -> None:
        self.visited.clear()
        self.events.clear()
        self.kills.clear()

    def __call__(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
        del before
        room = str(after.get("room_id", "unknown"))
        position = after.get("player_pos_world", after.get("player_pos_room", (0, 0)))
        divisor = 2.0 if "player_pos_world" in after else .125
        cell = (int(float(position[0]) / divisor), int(float(position[1]) / divisor))
        cells = self.visited.setdefault(room, set())
        reward = .02 if cell not in cells and len(cells) < 50 else 0.0
        cells.add(cell)
        for event in after.get("events", []):
            key = (room, str(event))
            if event == "kill" and self.kills.get(room, 0) < 3:
                self.kills[room] = self.kills.get(room, 0) + 1
                reward += .1
            elif key not in self.events:
                self.events.add(key)
                reward += {"exit": 1., "pickup": .2, "ability": 1., "bench": .5, "break": .3}.get(event, 0.)
        return reward - .001
