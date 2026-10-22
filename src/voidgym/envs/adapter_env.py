"""Gymnasium Env genérico para qualquer GameAdapter."""

from collections import deque
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

from voidgym.core.adapter import GameAdapter
from voidgym.core.observation import FRAME_STACK, fit_to_universal, universal_observation_space


class AdapterEnv(gym.Env[dict[str, np.ndarray], int]):
    """Empacota frames/histórico para a política e mantém sinais apenas no info."""

    def __init__(self, adapter: GameAdapter, reward_fn: Callable[[Any, Any], float],
                 *, frame_stack: int = FRAME_STACK, history_len: int = 3) -> None:
        super().__init__()
        self.adapter = adapter
        self.metadata = {"render_modes": []}
        self.reward_fn = reward_fn
        self.frame_stack = frame_stack
        self.history_len = history_len
        self.actions = tuple(adapter.action_table())
        if not self.actions:
            raise ValueError("O adaptador precisa declarar ao menos uma ação")
        self.action_space = gym.spaces.Discrete(len(self.actions))  # type: ignore[assignment]
        self.observation_space = universal_observation_space(
            frame_stack=frame_stack, history_len=history_len
        )  # type: ignore[assignment]
        self._frames: deque[np.ndarray] = deque(maxlen=frame_stack)
        self._history: deque[int] = deque([0] * history_len, maxlen=history_len)

    def _observation(self) -> dict[str, np.ndarray]:
        frames = np.stack(tuple(self._frames), axis=0)
        image = frames.transpose(0, 3, 1, 2).reshape(-1, 108, 192)
        return {"pixels": image, "action_history": np.asarray(self._history, dtype=np.int64)}

    def _info(self) -> dict[str, Any]:
        return {"signals": dict(self.adapter.signals()),
                "training_labels": self.adapter.training_labels()}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None
              ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        reset = getattr(self.adapter, "reset", None)
        if callable(reset):
            reset(seed=seed, options=options)
        frame = fit_to_universal(self.adapter.frame_rgb())
        self._frames.clear()
        self._frames.extend([frame] * self.frame_stack)
        self._history = deque([0] * self.history_len, maxlen=self.history_len)
        return self._observation(), self._info()

    def action_masks(self) -> np.ndarray:
        mask = np.asarray(self.adapter.action_mask(), dtype=np.bool_)
        if mask.shape != (len(self.actions),):
            raise ValueError("A máscara do adaptador tem tamanho incompatível com as ações")
        return mask

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError("Índice de ação fora da tabela do adaptador")
        before = dict(self.adapter.signals())
        self.adapter.apply(int(action))
        self._frames.append(fit_to_universal(self.adapter.frame_rgb()))
        after = dict(self.adapter.signals())
        self._history.append(int(action))
        return self._observation(), float(self.reward_fn(before, after)), False, False, self._info()
