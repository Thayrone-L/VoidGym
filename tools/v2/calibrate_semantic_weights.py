"""Estima pesos por frequência invertida com 50 mil quadros procedurais."""

import argparse
import ctypes
import json
import os
import random
from pathlib import Path

import numpy as np

from voidgym.core.semantics import Sem
from voidgym.gyms.platform_gym.generator import _build_room
from voidgym.gyms.platform_gym.labels import labels
from voidgym.gyms.platform_gym.world import Player


def _priority() -> None:
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.windll.kernel32
        kernel.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.SetPriorityClass.restype = wintypes.BOOL
        kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00004000)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=50_000)
    parser.add_argument("--output", default="src/voidgym/models/semantic_weights.json")
    args = parser.parse_args()
    _priority()
    counts = np.zeros(len(Sem), dtype=np.int64)
    for index in range(args.frames):
        level = index % 8
        room = _build_room(level, index + 70_000)
        rng = random.Random(index)
        player = Player(rng.uniform(1, max(1.1, room.width - 1)),
                        rng.uniform(.5, min(4, room.height - 1)))
        semantic = np.asarray(labels(room, player)["sem"], dtype=np.uint8)
        values, frequencies = np.unique(semantic, return_counts=True)
        for value, count in zip(values, frequencies, strict=True):
            if value != 255:
                counts[int(value)] += int(count)
    frequency = counts / max(1, int(counts.sum()))
    weights = np.where(counts > 0, 1 / np.sqrt(np.maximum(frequency, 1e-9)), 0)
    weights = weights / max(float(weights.mean()), 1e-9)
    payload = {label.name: float(weights[int(label)]) for label in Sem}
    Path(args.output).write_text(json.dumps({"frames": args.frames, "counts": counts.tolist(),
        "weights_by_class": payload}, indent=2), encoding="utf-8")
    print(json.dumps({"frames": args.frames, "counts": counts.tolist(), "weights_by_class": payload}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
