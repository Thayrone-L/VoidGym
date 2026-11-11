"""Mede custo no laço do ator e compressão de 1.000 quadros RGB."""

from __future__ import annotations

import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from voidgym.core.experience_store import ExperienceWriter


def main() -> int:
    rng = np.random.default_rng(7)
    latencies_ms: list[float] = []
    with tempfile.TemporaryDirectory(prefix="voidgym-experience-bench-") as temporary:
        root = Path(temporary)
        writer = ExperienceWriter(root, "benchmark", "1000-steps", source="test",
                                  block_steps=1024, action_table=("NOOP",))
        for step in range(1000):
            frame = rng.integers(0, 256, size=(108, 192, 3), dtype=np.uint8)
            start = time.perf_counter()
            writer.append({
                "frame": frame, "action": 0, "action_mask": np.array([True]),
                "reward_universal": -0.001, "reward_game": 0.0,
                "done": False, "truncated": False, "discontinuity": step == 0,
                "scene": "random-noise-worst-case",
            })
            latencies_ms.append((time.perf_counter() - start) * 1000)
        writer.close()
        block = next((root / "benchmark" / "1000-steps" / "blocks").rglob("*.npz"))
        with np.load(block, allow_pickle=False) as archive:
            stored_steps = int(archive["frame"].shape[0])
        if stored_steps != 1000:
            raise RuntimeError(f"o store salvou {stored_steps} de 1000 passos")
        result = {
            "steps": len(latencies_ms),
            "stored_steps": stored_steps,
            "append_ms_mean": statistics.mean(latencies_ms),
            "append_ms_p50": statistics.median(latencies_ms),
            "append_ms_p95": sorted(latencies_ms)[int(len(latencies_ms) * 0.95) - 1],
            "append_ms_max": max(latencies_ms),
            "compressed_block_bytes": block.stat().st_size,
            "compressed_kib_per_step": block.stat().st_size / len(latencies_ms) / 1024,
            "note": "quadros aleatórios são um limite de compressão pior que cenas reais",
        }
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
