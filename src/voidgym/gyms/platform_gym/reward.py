"""Game-specific shaping; UniversalReward remains responsible for universal signals."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from voidgym.core.config import RewardConfig
from voidgym.core.observation import STEP_GAME_MS
from voidgym.core.reward import UniversalReward


class GameReward:
    """One-time platform-gym achievements, added to UniversalReward."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.seen: set[tuple[int, str, str]] = set()
        self.kills: dict[int, int] = defaultdict(int)

    def __call__(self, before: dict[str, Any], after: dict[str, Any]) -> float:
        value = 0.0
        room = int(after.get("room_id", 0))
        for event in after.get("events", ()):
            kind = getattr(event, "kind", None)
            target = str(getattr(event, "target", ""))
            key = (room, str(kind), target)
            if kind == "exit":
                value += 1.0
            elif kind == "pickup" and target == "coin":
                value += 0.2
            elif kind == "ability":
                value += 1.0
            elif kind == "bench" and key not in self.seen:
                value += 0.5
                self.seen.add(key)
            elif kind == "break" and key not in self.seen:
                value += 0.3
                self.seen.add(key)
            elif kind == "kill" and self.kills[room] < 3:
                self.kills[room] += 1
                value += 0.1
        return value


class CombinedPlatformReward:
    """Compose universal progress/damage/time shaping with gym achievements."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig(novelty=.02, progress=0, damage=.3, death=1,
                                             time_cost=.001 / STEP_GAME_MS, room_change=0)
        self.reset()

    def reset(self) -> None:
        self.universal = UniversalReward(self.config, world_cell_size=2.0, novelty_cap_per_room=1.0)
        self.game = GameReward()

    def __call__(self, before: dict[str, Any], after: dict[str, Any]) -> float:
        return float(self.universal(before, after) + self.game(before, after))
