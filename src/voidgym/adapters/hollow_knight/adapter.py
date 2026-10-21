"""Wrapper de compatibilidade do ambiente atual do Hollow Knight."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

# V1-SHIM: portar no prompt 86 — mantém os nomes e a ordem originais.
from hk_agent.scene_change_task import (  # V1-SHIM: portar no prompt 86
    SCENE_CHANGE_INTERACTION_ACTION_NAMES,
    SCENE_CHANGE_INTERACTION_ACTION_TEMPLATES,
)
from voidgym.core.actions import UNIVERSAL_ACTION_BY_NAME, UniversalAction


class HollowKnightAdapter:
    """Adapta um ExplorationTask já configurado, sem alterar a v1."""

    game_id = "hollow_knight"

    def __init__(self, task: Any) -> None:
        self.task = task
        self._last_observation: Mapping[str, np.ndarray] | None = None
        self._last_info: Mapping[str, Any] = {}
        self._actions = tuple(SCENE_CHANGE_INTERACTION_ACTION_TEMPLATES)
        self._names = tuple(SCENE_CHANGE_INTERACTION_ACTION_NAMES)
        if len(self._actions) != len(self._names):
            raise ValueError("A tabela de ações HK está inconsistente")

    def action_table(self) -> Sequence[str]:
        return self._names

    def to_universal(self, game_action_index: int) -> UniversalAction:
        try:
            name = self._names[game_action_index]
            return UNIVERSAL_ACTION_BY_NAME[name]
        except (IndexError, KeyError) as error:
            raise ValueError("Índice de ação HK fora da tabela universal") from error

    def from_universal(self, action: UniversalAction) -> int | None:
        try:
            return self._names.index(action.name)
        except ValueError:
            return None

    def frame_rgb(self) -> NDArray[np.uint8]:
        if self._last_observation is None:
            raise RuntimeError("O adaptador HK precisa ser resetado antes da captura")
        pixels = np.asarray(self._last_observation["pixels"])
        return np.asarray(pixels[-1], dtype=np.uint8)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> None:
        del seed
        self._last_observation, self._last_info = self.task.reset(options=options)

    def apply(self, game_action_index: int) -> None:
        if not 0 <= game_action_index < len(self._actions):
            raise ValueError("Índice de ação HK inválido")
        (
            self._last_observation,
            _reward,
            terminated,
            truncated,
            info,
        ) = self.task.step(game_action_index)
        self._last_info = dict(info)
        self._last_info = {
            **self._last_info,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    def action_mask(self) -> np.ndarray:
        return np.asarray(self.task.action_masks(), dtype=np.bool_)

    def signals(self) -> Mapping[str, Any]:
        return self._last_info

    def training_labels(self) -> None:
        return None

    def game_time_scale(self) -> float:
        return 1.0
