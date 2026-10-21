"""Contrato fixo da observação universal da v2."""

from collections.abc import Mapping

import gymnasium as gym
import numpy as np
from PIL import Image

UNIVERSAL_IMAGE_HW = (108, 192)
FRAME_STACK = 4
MODEL_CONTRACT_VERSION = "2.0"
STEP_GAME_MS = 240  # lease-base 80 ms × action_repeat 3; confirmar no jogo principal.
UNIVERSAL_IMAGE_KEY = "pixels"  # Nome usado pela observação atual do HK.
UNIVERSAL_ACTION_HISTORY_KEY = "action_history"


def fit_to_universal(frame_rgb: np.ndarray) -> np.ndarray:
    """Redimensiona mantendo proporção e centraliza a imagem em faixas pretas."""
    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3 or frame_rgb.dtype != np.uint8:
        raise ValueError("A tela deve ser um array RGB uint8 com shape (H, W, 3)")
    height, width = frame_rgb.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("A tela não pode ter dimensões vazias")
    target_h, target_w = UNIVERSAL_IMAGE_HW
    scale = min(target_w / width, target_h / height)
    resized_w = max(1, round(width * scale))
    resized_h = max(1, round(height * scale))
    resized = np.asarray(
        Image.fromarray(frame_rgb).resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    )
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0, y0 = (target_w - resized_w) // 2, (target_h - resized_h) // 2
    result[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    return result


def split_observation(
    obs: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Separa pixels/histórico universais dos extras próprios do jogo."""
    universal_names = {UNIVERSAL_IMAGE_KEY, UNIVERSAL_ACTION_HISTORY_KEY}
    universal = {key: value for key, value in obs.items() if key in universal_names}
    extras = {key: value for key, value in obs.items() if key not in universal_names}
    return universal, extras


def universal_observation_space(
    height: int = 108,
    width: int = 192,
    frame_stack: int = FRAME_STACK,
    history_len: int = FRAME_STACK - 1,
    action_count: int = 43,
) -> gym.spaces.Dict:
    """Constrói o espaço universal em canais concatenados (RGB × quadros)."""
    if (height, width) != UNIVERSAL_IMAGE_HW or frame_stack != FRAME_STACK:
        raise ValueError("A v2 exige imagem 192×108 e pilha de quatro quadros")
    return gym.spaces.Dict(
        {
            UNIVERSAL_IMAGE_KEY: gym.spaces.Box(
                0, 255, shape=(frame_stack * 3, height, width), dtype=np.uint8
            ),
            UNIVERSAL_ACTION_HISTORY_KEY: gym.spaces.MultiDiscrete(
                np.full(history_len, action_count, dtype=np.int64)
            ),
        }
    )
