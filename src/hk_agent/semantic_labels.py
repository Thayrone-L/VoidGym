"""Compact transport codec for optional semantic labels in bridge FRAMEs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

IGNORE_NIBBLE = 15


def unpack_u4(payload: bytes, *, width: int, height: int) -> NDArray[np.uint8]:
    """Unpack two 4-bit class IDs per byte; nibble 15 maps to IGNORE (255)."""
    if width <= 0 or height <= 0:
        raise ValueError("semantic grid dimensions must be positive")
    cells = width * height
    if len(payload) != (cells + 1) // 2:
        raise ValueError("packed semantic payload has the wrong length")
    packed = np.frombuffer(payload, dtype=np.uint8)
    values = np.empty(packed.size * 2, dtype=np.uint8)
    values[0::2] = packed >> 4
    values[1::2] = packed & 0x0F
    values = values[:cells]
    if np.any((values > 9) & (values != IGNORE_NIBBLE)):
        raise ValueError("packed semantic payload contains an unknown class ID")
    values[values == IGNORE_NIBBLE] = 255
    return values.reshape(height, width).copy()


def pack_u4(labels: NDArray[np.uint8]) -> bytes:
    """Pack class IDs 0–9 and IGNORE (255) for bridge tests and tooling."""
    if labels.ndim != 2 or not labels.size:
        raise ValueError("semantic labels must be a non-empty 2D grid")
    values = labels.astype(np.uint8, copy=False).reshape(-1).copy()
    values[values == 255] = IGNORE_NIBBLE
    if np.any((values > 9) & (values != IGNORE_NIBBLE)):
        raise ValueError("semantic labels contain an unknown class ID")
    if values.size % 2:
        values = np.pad(values, (0, 1), constant_values=0)
    return ((values[0::2] << 4) | values[1::2]).tobytes()
