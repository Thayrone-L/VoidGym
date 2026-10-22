"""Persistent success-rate curriculum for platform gym episodes."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from voidgym.gyms.platform_gym.params import split_for_seed


class Curriculum:
    def __init__(self, *, seed: int = 0) -> None:
        self.level = 0
        self.seed = seed
        self.current: deque[bool] = deque(maxlen=100)
        self.history: dict[int, deque[bool]] = {i: deque(maxlen=300) for i in range(8)}
        self.episodes = 0

    def record(self, success: bool) -> None:
        self.current.append(bool(success))
        self.history[self.level].append(bool(success))
        self.episodes += 1
        if len(self.current) == 100 and sum(self.current) / 100 > 0.7 and self.level < 7:
            self.level += 1
            self.current.clear()
        elif len(self.history[self.level]) == 300 and sum(self.history[self.level]) / 300 < 0.2 and self.level > 0:
            self.level -= 1
            self.current.clear()

    def sample_level(self, rng_seed: int | None = None) -> int:
        import numpy as np
        rng = np.random.default_rng(self.seed if rng_seed is None else rng_seed)
        return self.level if self.level == 0 or rng.random() < .7 else int(rng.integers(0, self.level))

    def style_split(self, seed: int) -> str:
        return split_for_seed(seed)

    def save(self, path: str | Path) -> None:
        payload = {"level": self.level, "seed": self.seed, "episodes": self.episodes,
                   "current": list(self.current), "history": {str(k): list(v) for k, v in self.history.items()}}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Curriculum:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        result = cls(seed=int(data["seed"]))
        result.level = int(data["level"])
        result.episodes = int(data["episodes"])
        result.current.extend(bool(x) for x in data["current"])
        result.history = {i: deque((bool(x) for x in data.get("history", {}).get(str(i), [])), maxlen=300)
                          for i in range(8)}
        return result
