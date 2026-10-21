"""Gymnasium Env que adapta qualquer GameAdapter para o contrato universal."""

from collections import deque
from collections.abc import Callable, Mapping
from typing import Any, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from voidgym.core.adapter import GameAdapter
from voidgym.core.observation import (
    FRAME_STACK,
    UNIVERSAL_ACTION_HISTORY_KEY,
    UNIVERSAL_IMAGE_KEY,
    fit_to_universal,
    universal_observation_space,
)

RewardFn = Callable[[Mapping[str, Any], Mapping[str, Any]], float]


class AdapterEnv(gym.Env[dict[str, NDArray[Any]], int]):
    """Ambiente universal; sinais e rótulos ficam apenas no info."""

    def __init__(self, adapter: GameAdapter, reward_fn: RewardFn) -> None:
        super().__init__()
        self.adapter = adapter
        self.reward_fn = reward_fn
        names = tuple(adapter.action_table())
        if not names:
            raise ValueError("O adaptador precisa expor ao menos uma ação")
        self._action_count = len(names)
        self.action_space = cast(gym.Space[int], gym.spaces.Discrete(self._action_count))
        self.observation_space = cast(
            gym.Space[dict[str, NDArray[Any]]],
            universal_observation_space(action_count=self._action_count),
        )
        self._frames: deque[NDArray[np.uint8]] = deque(maxlen=FRAME_STACK)
        self._history: deque[int] = deque(maxlen=FRAME_STACK - 1)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
        super().reset(seed=seed)
        adapter_reset = getattr(self.adapter, "reset", None)
        if callable(adapter_reset):
            adapter_reset(seed=seed, options=options)
        frame = self._frame()
        self._frames.clear()
        self._frames.extend([frame] * FRAME_STACK)
        self._history.clear()
        self._history.extend([0] * (FRAME_STACK - 1))
        return self._observation(), self._info()

    def step(
        self, action: int
    ) -> tuple[dict[str, NDArray[Any]], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError("Índice de ação fora da tabela do adaptador")
        before = dict(self.adapter.signals())
        self.adapter.apply(int(action))
        after = dict(self.adapter.signals())
        self._frames.append(self._frame())
        self._history.append(int(action))
        info = self._info()
        info["signals"] = after
        labels = self.adapter.training_labels()
        if labels is not None:
            info["training_labels"] = dict(labels)
        return (
            self._observation(),
            float(self.reward_fn(before, after)),
            bool(after.get("terminated", False)),
            bool(after.get("truncated", False)),
            info,
        )

    def action_masks(self) -> NDArray[np.bool_]:
        mask = np.asarray(self.adapter.action_mask(), dtype=np.bool_)
        if mask.shape != (self._action_count,):
            raise ValueError("A máscara precisa ter o mesmo tamanho da tabela de ações")
        return mask

    def render(self) -> NDArray[np.uint8]:
        return self._frame()

    def _frame(self) -> NDArray[np.uint8]:
        frame = np.asarray(self.adapter.frame_rgb())
        if frame.dtype != np.uint8:
            raise ValueError("frame_rgb deve retornar uint8")
        return fit_to_universal(frame)

    def _observation(self) -> dict[str, NDArray[Any]]:
        stacked = np.stack(tuple(self._frames), axis=0)
        image = np.transpose(stacked, (0, 3, 1, 2)).reshape(12, 108, 192)
        return {
            UNIVERSAL_IMAGE_KEY: image,
            UNIVERSAL_ACTION_HISTORY_KEY: np.asarray(self._history, dtype=np.int64),
        }

    def _info(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "game_id": self.adapter.game_id,
            "signals": dict(self.adapter.signals()),
        }
        labels = self.adapter.training_labels()
        if labels is not None:
            result["training_labels"] = dict(labels)
        return result
