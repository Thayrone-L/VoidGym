"""Interface entre o núcleo genérico e um jogo concreto."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from voidgym.core.actions import UniversalAction


@dataclass(frozen=True, slots=True)
class AdapterInfo:
    game_id: str
    version: str
    supports_training_labels: bool
    supports_time_scale: bool


class GameAdapter(Protocol):
    """signals() e training_labels() nunca entram na entrada da política."""

    game_id: str

    def action_table(self) -> Sequence[str]: ...
    def to_universal(self, game_action_index: int) -> UniversalAction: ...
    def from_universal(self, action: UniversalAction) -> int | None: ...
    def frame_rgb(self) -> np.ndarray: ...
    def apply(self, game_action_index: int) -> None: ...
    def action_mask(self) -> np.ndarray: ...
    def signals(self) -> Mapping[str, Any]: ...
    def training_labels(self) -> Mapping[str, Any] | None: ...
    def game_time_scale(self) -> float: ...
