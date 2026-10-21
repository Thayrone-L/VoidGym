"""Adaptador falso de tela, para testar a fronteira sem carregar um jogo."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from voidgym.core.actions import UNIVERSAL_ACTIONS, UniversalAction


class FakeAdapter:
    game_id = "fake"

    def __init__(self) -> None:
        self.position = 20
        self.last_action = 0
        self._frame = np.zeros((108, 192, 3), dtype=np.uint8)
        self._labels: dict[str, Any] | None = None

    def action_table(self) -> Sequence[str]:
        return tuple(item.name for item in UNIVERSAL_ACTIONS[:5])

    def to_universal(self, game_action_index: int) -> UniversalAction:
        return UNIVERSAL_ACTIONS[game_action_index]

    def from_universal(self, action: UniversalAction) -> int | None:
        names = self.action_table()
        try:
            return names.index(action.name)
        except ValueError:
            return None

    def frame_rgb(self) -> np.ndarray:
        return self._frame.copy()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        del seed, options
        self.position = 20
        self.last_action = 0

    def apply(self, game_action_index: int) -> None:
        self.last_action = game_action_index
        self.position = (self.position + (game_action_index - 1) * 3) % 192
        self._frame.fill(0)
        self._frame[48:56, self.position : self.position + 8] = 255

    def action_mask(self) -> np.ndarray:
        return np.ones(5, dtype=np.bool_)

    def signals(self) -> Mapping[str, Any]:
        return {"position": self.position}

    def training_labels(self) -> Mapping[str, Any] | None:
        return self._labels

    def game_time_scale(self) -> float:
        return 1.0
