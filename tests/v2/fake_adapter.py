"""Adaptador mínimo determinístico usado em testes sem jogo."""

from typing import Any

import numpy as np


class FakeAdapter:
    game_id = "fake_platformer"

    def __init__(self) -> None:
        self.x = 0
        self._frame = np.zeros((108, 192, 3), dtype=np.uint8)
        self._mask = np.array([True, True, False, True, True], dtype=np.bool_)

    def action_table(self) -> tuple[str, ...]:
        return ("NOOP", "LEFT", "RIGHT", "JUMP", "INTERACT")

    def frame_rgb(self) -> np.ndarray:
        self._frame.fill(0)
        self._frame[50:55, max(0, min(187, self.x)):max(5, min(192, self.x + 5))] = 255
        return self._frame

    def apply(self, game_action_index: int) -> None:
        self.x = max(0, min(187, self.x + (-1 if game_action_index == 1 else 1 if game_action_index == 2 else 0)))

    def action_mask(self) -> np.ndarray:
        return self._mask.copy()

    def signals(self) -> dict[str, Any]:
        return {"position": self.x}

    def training_labels(self) -> None:
        return None

    def game_time_scale(self) -> float:
        return 1.0

    def to_universal(self, game_action_index: int) -> Any:
        from voidgym.core.actions import UNIVERSAL_ACTIONS
        return UNIVERSAL_ACTIONS[game_action_index]

    def from_universal(self, action: Any) -> int | None:
        return next((index for index, item in enumerate(UNIVERSAL_ACTIONS[:5]) if item == action), None)


from voidgym.core.actions import UNIVERSAL_ACTIONS
