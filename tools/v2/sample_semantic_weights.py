"""Estimate semantic class weights from exactly 50,000 generated gym frames."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from voidgym.core.semantics import SEM_GRID, Sem
from voidgym.gyms.platform_gym.generator import generate_room
from voidgym.gyms.platform_gym.labels import labels
from voidgym.gyms.platform_gym.world import Player
from voidgym.models.perception import inverse_frequency_weights


def main() -> int:
    samples = 50_000
    counts = np.zeros(len(Sem), dtype=np.int64)
    rng = np.random.default_rng(8500)
    room_count = 500
    per_room = samples // room_count
    for index in range(room_count):
        room = generate_room(index % 7, 50_000 + index)
        player = Player(*room.spawn, on_ground=True)
        for frame in range(per_room):
            player.x = float(rng.uniform(.5, max(.51, room.width - .5)))
            player.y = room.spawn[1] + float(rng.uniform(0, min(2.0, room.height - room.spawn[1] - 1)))
            label = labels(room, player)["sem"]
            counts += np.bincount(np.asarray(label).reshape(-1), minlength=len(Sem))
    result = {
        "frames": samples, "grid": list(SEM_GRID), "classes": [item.name for item in Sem],
        "cell_counts": counts.tolist(),
        "weights": inverse_frequency_weights(counts.tolist()).tolist(),
    }
    output = PROJECT_ROOT / "configs" / "v2" / "semantic_weights_50000.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"saved {output}; frames={samples}; total_cells={int(counts.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
