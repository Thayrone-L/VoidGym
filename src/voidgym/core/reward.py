"""Recompensa compartilhada entre jogos e alerta contra retorno zerado."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from voidgym.core.config import RewardConfig


class GameReward(Protocol):
    """Parte de recompensa específica do adaptador de jogo."""

    def __call__(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> float: ...


@dataclass(frozen=True)
class RewardBreakdown:
    universal: float
    game: float
    components: dict[str, float]

    @property
    def total(self) -> float:
        return self.universal + self.game


class UniversalReward:
    """Recompensa por novidade, progresso e sobrevivência, sem consultar regras do jogo."""

    def __init__(
        self,
        config: RewardConfig | None = None,
        *,
        grid_width: int = 16,
        grid_height: int = 9,
        stuck_steps: int = 300,
    ) -> None:
        self.config = config or RewardConfig()
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.stuck_steps = stuck_steps
        self._visited: dict[str, set[tuple[int, int]]] = {}
        self._best: dict[str, float] = {}
        self._steps_by_place: dict[tuple[str, int, int], int] = {}
        self._paid_stuck: set[tuple[str, int, int]] = set()
        self._rooms: set[str] = set()
        self._slice_rewards: deque[float] = deque()
        self._rolling_rewards: deque[float] = deque()
        self._episode_steps = 0

    def reset(self) -> None:
        self._visited.clear()
        self._best.clear()
        self._steps_by_place.clear()
        self._paid_stuck.clear()
        self._rooms.clear()
        self._slice_rewards.clear()
        self._rolling_rewards.clear()
        self._episode_steps = 0

    def __call__(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
        return self.compute(before, after).universal

    def compute(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> RewardBreakdown:
        components: dict[str, float] = {"time": self.config.time}
        room = str(after.get("room_id", "unknown"))
        cell = _cell(after.get("player_pos_room"), self.grid_width, self.grid_height)
        visited = self._visited.setdefault(room, set())
        if cell is not None and cell not in visited:
            visited.add(cell)
            components["novelty"] = self.config.novelty
        progress = float(after.get("progress", 0.0))
        if progress > self._best.get(room, float("-inf")):
            self._best[room] = progress
            components["progress"] = self.config.progress
        hp_before = float(before.get("hp", after.get("hp", 0)))
        hp_after = float(after.get("hp", hp_before))
        if hp_after < hp_before:
            components["damage"] = self.config.damage * (hp_before - hp_after)
        if after.get("died") is True:
            components["death"] = self.config.death
        if after.get("room_changed") is True or room not in self._rooms:
            self._rooms.add(room)
            components["new_room"] = 1.0
        self._stuck_component(room, cell, components)
        total = self._apply_budgets(sum(components.values()))
        return RewardBreakdown(total, 0.0, components)

    def step(
        self,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        game_reward: float = 0.0,
    ) -> RewardBreakdown:
        shared = self.compute(before, after)
        game = self._apply_budgets(game_reward, advance=False)
        return RewardBreakdown(shared.universal, game, shared.components)

    def _apply_budgets(self, reward: float, *, advance: bool = True) -> float:
        """Limita ganhos por fatia e janela, preservando sempre o custo de tempo."""
        cfg = self.config
        if advance:
            self._episode_steps += 1
            if self._episode_steps % cfg.episode_slice_steps == 1:
                self._slice_rewards.clear()
        positive = max(0.0, reward)
        slice_remaining = max(0.0, cfg.episode_slice_budget - sum(self._slice_rewards))
        rolling_remaining = max(0.0, cfg.rolling_window_budget - sum(self._rolling_rewards))
        allowed = min(positive, slice_remaining, rolling_remaining)
        self._slice_rewards.append(allowed)
        self._rolling_rewards.append(allowed)
        while len(self._rolling_rewards) > cfg.rolling_window_steps:
            self._rolling_rewards.popleft()
        return reward - positive + allowed

    def _stuck_component(
        self,
        room: str,
        cell: tuple[int, int] | None,
        components: dict[str, float],
    ) -> None:
        if cell is None:
            return
        place = (room, *cell)
        self._steps_by_place[place] = self._steps_by_place.get(place, 0) + 1
        if self._steps_by_place[place] >= self.stuck_steps and place not in self._paid_stuck:
            self._paid_stuck.add(place)
            components["stuck"] = 0.0


def _cell(position: object, width: int, height: int) -> tuple[int, int] | None:
    if not isinstance(position, (tuple, list)) or len(position) < 2:
        return None
    try:
        x, y = float(position[0]), float(position[1])
    except (TypeError, ValueError):
        return None
    return (
        min(width - 1, max(0, int(x * width))),
        min(height - 1, max(0, int(y * height))),
    )


def reward_zero_alert(rewards: Sequence[float], threshold: float = 0.9) -> bool:
    """Retorna verdadeiro quando a janela tem proporção alta de recompensas zero."""
    if not 0 < threshold <= 1 or not rewards:
        return False
    return sum(float(value) == 0.0 for value in rewards) / len(rewards) > threshold
