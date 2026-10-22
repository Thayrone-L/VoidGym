"""Componentes de recompensa genéricos, independentes do jogo concreto."""

from collections import Counter
from typing import Any, Protocol

from voidgym.core.config import RewardConfig


class GameReward(Protocol):
    def __call__(self, before: dict[str, Any], after: dict[str, Any]) -> float: ...


class UniversalReward:
    """Novidade, progresso e custo temporal sem tetos arbitrários por episódio."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self.visits: Counter[tuple[str, int, int]] = Counter()
        self.best: dict[str, float] = {}
        self.stuck: Counter[tuple[str, int, int]] = Counter()
        self.steps = 0
        self.zero_steps = 0

    def __call__(self, before: dict[str, Any], after: dict[str, Any]) -> float:
        self.steps += 1
        room = str(after.get("room_id", "unknown"))
        position = after.get("player_pos_room", (0.5, 0.5))
        cell = (int(float(position[0]) * 20), int(float(position[1]) * 12))
        key = (room, *cell)
        value = 0.0
        if not self.visits[key]:
            value += self.config.novelty
        self.visits[key] += 1
        progress = float(after.get("progress", position[0]))
        if progress > self.best.get(room, -1.0):
            value += self.config.progress * (progress - self.best.get(room, 0.0))
            self.best[room] = progress
        hp_before, hp_after = before.get("hp"), after.get("hp")
        if isinstance(hp_before, (int, float)) and isinstance(hp_after, (int, float)):
            damage = max(0.0, float(hp_before) - float(hp_after))
        else:
            damage = max(0.0, float(after.get("damage_taken", 0.0)))
        value -= self.config.damage * damage
        if after.get("died"):
            value -= self.config.death
        value -= self.config.time_cost * max(1.0, float(after.get("step_game_ms", 80)))
        if after.get("room_changed"):
            value += self.config.room_change
        if before.get("player_pos_room") == after.get("player_pos_room"):
            self.stuck[key] += 1
            if self.stuck[key] > 10:
                value -= 0.001
        else:
            self.stuck[key] = 0
        if value == 0:
            self.zero_steps += 1
        return value

    def zero_reward_fraction(self) -> float:
        return self.zero_steps / self.steps if self.steps else 0.0
