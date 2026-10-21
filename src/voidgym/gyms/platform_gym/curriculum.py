"""Currículo adaptativo persistível em JSON."""

import json
import random
from collections import deque
from pathlib import Path


class Curriculum:
    def __init__(self, path: Path, level: int = 0) -> None:
        self.path = path
        self.level = level
        self.results: dict[int, deque[bool]] = {number: deque(maxlen=300) for number in range(8)}

    def record(self, success: bool) -> int:
        history = self.results[self.level]
        history.append(success)
        if len(history) >= 100 and sum(list(history)[-100:]) / 100 > .7 and self.level < 7:
            self.level += 1
        elif len(history) >= 300 and sum(history) / 300 < .2 and self.level > 0:
            self.level -= 1
        return self.level

    def sample_level(self, seed: int) -> int:
        rng = random.Random(seed)
        return rng.randrange(self.level) if self.level and rng.random() < .3 else self.level

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"level": self.level, "results": {str(k): list(v) for k, v in self.results.items()}}), encoding="utf-8")

    def load(self) -> None:
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.level = int(payload["level"])
            self.results = {int(k): deque(v, maxlen=300) for k, v in payload["results"].items()}
