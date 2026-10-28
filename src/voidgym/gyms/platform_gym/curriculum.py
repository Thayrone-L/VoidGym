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

    def record(self, success: bool, level: int | None = None) -> None:
        sampled_level = self.level if level is None else int(level)
        if not 0 <= sampled_level <= 7:
            raise ValueError("level must be in [0, 7]")
        self.history[sampled_level].append(bool(success))
        if sampled_level == self.level:
            self.current.append(bool(success))
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
        choice = rng.random()
        if self.level == 0:
            return 1 if choice >= .9 else 0
        if choice < .6:
            return self.level
        if choice < .9:
            return int(rng.integers(0, self.level))
        return self.level if self.level == 7 else self.level + 1

    def success_rate(self, level: int) -> float:
        values = self.history[level]
        return sum(values) / len(values) if values else 0.0

    def style_split(self, seed: int) -> str:
        return split_for_seed(seed)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def to_dict(self) -> dict[str, object]:
        return {"level": self.level, "seed": self.seed, "episodes": self.episodes,
                "current": list(self.current),
                "history": {str(k): list(v) for k, v in self.history.items()}}

    @classmethod
    def load(cls, path: str | Path) -> Curriculum:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        result = cls(seed=int(data["seed"]))
        result.level = int(data["level"])
        result.episodes = int(data["episodes"])
        old_current = [bool(x) for x in data.get("current", [])]
        result.current.extend(old_current)
        saved_history = data.get("history", {})
        result.history = {i: deque((bool(x) for x in saved_history.get(str(i), [])), maxlen=300)
                          for i in range(8)}
        if not saved_history and 0 <= result.level <= 7:
            result.history[result.level].extend(old_current)
        return result
