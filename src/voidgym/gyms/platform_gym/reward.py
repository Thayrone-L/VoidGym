"""Game-specific shaping; UniversalReward remains responsible for universal signals."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from voidgym.core.config import RewardConfig
from voidgym.core.observation import STEP_GAME_MS
from voidgym.core.reward import UniversalReward


@dataclass(frozen=True, slots=True)
class GameRewardConfig:
    first_hit: float = 0.05
    enemy_kill: float = 0.3
    kill_limit_per_room: int = 3
    coin: float = 0.2

    def validate(self) -> None:
        if any(not math.isfinite(value) or value < 0
               for value in (self.first_hit, self.enemy_kill, self.coin)):
            raise ValueError("game rewards must be finite and non-negative")
        if self.kill_limit_per_room < 0:
            raise ValueError("kill_limit_per_room must be non-negative")


class GameReward:
    """One-time platform-gym achievements, added to UniversalReward."""

    def __init__(self, config: GameRewardConfig | None = None) -> None:
        self.config = config or GameRewardConfig()
        self.config.validate()
        self.reset()

    def reset(self) -> None:
        self.seen: set[tuple[int, str, str]] = set()
        self.kills: dict[int, int] = defaultdict(int)
        self.hit_rewards: set[tuple[int, str, str]] = set()

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
                value += self.config.coin
            elif kind in {"hit_enemy", "hit_treasure_rock"}:
                hit_key = (room, str(kind), target)
                if hit_key not in self.hit_rewards:
                    self.hit_rewards.add(hit_key)
                    value += self.config.first_hit
            elif kind == "ability":
                value += 1.0
            elif kind == "bench" and key not in self.seen:
                value += 0.5
                self.seen.add(key)
            elif kind == "break" and key not in self.seen:
                value += 0.3
                self.seen.add(key)
            elif (kind == "kill"
                  and self.kills[room] < self.config.kill_limit_per_room):
                self.kills[room] += 1
                value += self.config.enemy_kill
        return value


class CombinedPlatformReward:
    """Compose universal progress/damage/time shaping with gym achievements."""

    def __init__(self, config: RewardConfig | None = None,
                 game_config: GameRewardConfig | None = None) -> None:
        self.config = config or RewardConfig(novelty=.02, progress=0, damage=.3, death=1,
                                             time_cost=.001 / STEP_GAME_MS, room_change=0)
        self.game_config = game_config or GameRewardConfig()
        self.reset()

    def reset(self) -> None:
        self.universal = UniversalReward(self.config, world_cell_size=2.0, novelty_cap_per_room=1.0)
        self.game = GameReward(self.game_config)

    def __call__(self, before: dict[str, Any], after: dict[str, Any]) -> float:
        return float(self.universal(before, after) + self.game(before, after))
