"""Game reward com memória de progresso por sala."""

from collections.abc import Mapping
from typing import Any


class GymReward:
    def __init__(self) -> None:
        self.events: set[tuple[str, str, str]] = set()
        self.kills: dict[str, int] = {}

    def reset(self) -> None:
        self.events.clear()
        self.kills.clear()

    def __call__(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
        del before
        room = str(after.get("room_id", "unknown"))
        reward = 0.0
        details = after.get("event_details", [])
        if not details:
            details = [(str(event), "") for event in after.get("events", [])]
        for event, detail in details:
            event, detail = str(event), str(detail)
            key = (room, event, detail)
            if event == "damage":
                reward -= .3
            elif event == "death":
                reward -= 1.
            elif event == "kill" and self.kills.get(room, 0) < 3:
                self.kills[room] = self.kills.get(room, 0) + 1
                reward += .1
            elif key not in self.events:
                self.events.add(key)
                reward += {
                    "exit": 1., "pickup": .2 if detail.startswith("coin") else 0.,
                    "ability": 1., "bench": .5, "break": .3,
                }.get(event, 0.)
        return reward
