"""Contrato de imagem e histórico fixo para políticas universais."""

from collections.abc import Mapping

import gymnasium as gym
import numpy as np

from voidgym.core.errors import VoidGymValueError as ValueError

UNIVERSAL_IMAGE_HW = (108, 192)
FRAME_STACK = 4
MODEL_CONTRACT_VERSION = "2.0"
STEP_GAME_MS = 80  # Um timestep do learner v1 usa lease-base 80 ms; action_repeat=3 só vale para o probe (3 passos = 240 ms).
UNIVERSAL_IMAGE_KEY = "pixels"  # nome real do Hollow Knight.
UNIVERSAL_ACTION_HISTORY_KEY = "action_history"


def fit_to_universal(frame_rgb: np.ndarray) -> np.ndarray:
    """Redimensiona com letterbox preto para RGB 108×192 por amostragem uniforme."""
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3 or frame_rgb.dtype != np.uint8:
        raise ValueError("frame_rgb deve ter shape (H,W,3) e dtype uint8")
    height, width = frame_rgb.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("frame_rgb não pode estar vazio")
    out_h, out_w = UNIVERSAL_IMAGE_HW
    scale = min(out_h / height, out_w / width)
    new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
    ys = np.linspace(0, height - 1, new_h).round().astype(int)
    xs = np.linspace(0, width - 1, new_w).round().astype(int)
    resized = frame_rgb[ys[:, None], xs[None, :]]
    result = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    top, left = (out_h - new_h) // 2, (out_w - new_w) // 2
    result[top:top + new_h, left:left + new_w] = resized
    return result


def split_observation(obs: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Separa imagem/histórico universais dos campos específicos do jogo."""
    keys = {UNIVERSAL_IMAGE_KEY, UNIVERSAL_ACTION_HISTORY_KEY}
    universal = {key: value for key, value in obs.items() if key in keys}
    specific = {key: value for key, value in obs.items() if key not in keys}
    return universal, specific


def universal_observation_space(height: int = 108, width: int = 192,
                                frame_stack: int = FRAME_STACK, history_len: int = 3) -> gym.spaces.Dict:
    return gym.spaces.Dict({
        UNIVERSAL_IMAGE_KEY: gym.spaces.Box(0, 255, shape=(frame_stack * 3, height, width), dtype=np.uint8),
        UNIVERSAL_ACTION_HISTORY_KEY: gym.spaces.MultiDiscrete(np.full(history_len, 256, dtype=np.int64)),
    })
