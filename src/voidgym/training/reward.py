"""Combina partes de recompensa e mantém telemetria separada."""

from collections.abc import Mapping
from typing import Any

from voidgym.core.reward import RewardBreakdown, UniversalReward
from voidgym.gyms.platform_gym.reward import GymReward


class CombinedTrainingReward:
    def __init__(self) -> None:
        self.universal = UniversalReward()
        self.game = GymReward()
        self.last = RewardBreakdown(0.0, 0.0, {})

    def reset(self) -> None:
        self.universal.reset()
        self.game.reset()
        self.last = RewardBreakdown(0.0, 0.0, {})

    def __call__(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> float:
        universal = self.universal.compute(before, after)
        game_reward = self.game(before, after)
        self.last = RewardBreakdown(universal.universal, game_reward, universal.components)
        return self.last.total
