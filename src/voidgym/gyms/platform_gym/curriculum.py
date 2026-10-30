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
        self.history: dict[int, deque[bool]] = {i: deque(maxlen=300) for i in range(9)}
        self.episodes = 0
        self.best_window_rate: dict[int, float] = {}
        self.last_improvement_step: dict[int, int] = {}

    def window_success_rate(self, level: int, window: int = 100) -> float | None:
        if window < 1:
            raise ValueError("window must be positive")
        values = self.history[level]
        if len(values) < window:
            return None
        return sum(list(values)[-window:]) / window

    def training_goal_met(self, *, window: int = 100, levels_0_7_target: float = .70,
                          level_8_target: float = .30) -> bool:
        return all((rate := self.window_success_rate(level, window)) is not None
                   and rate >= (level_8_target if level == 8 else levels_0_7_target)
                   for level in range(9))

    def update_progress_watch(self, steps: int, *, window: int = 100,
                              levels_0_7_target: float = .70, level_8_target: float = .30,
                              no_improvement_steps: int = 3_000_000) -> int | None:
        for level in range(9):
            rate = self.window_success_rate(level, window)
            if rate is None:
                continue
            target = level_8_target if level == 8 else levels_0_7_target
            if rate >= target:
                self.best_window_rate.pop(level, None)
                self.last_improvement_step.pop(level, None)
                continue
            best = self.best_window_rate.get(level)
            if best is None or rate > best:
                self.best_window_rate[level] = rate
                self.last_improvement_step[level] = steps
                continue
            last_improvement = self.last_improvement_step.get(level, steps)
            if steps - last_improvement >= no_improvement_steps:
                return level
        return None

    def record(self, success: bool, level: int | None = None) -> None:
        sampled_level = self.level if level is None else int(level)
        if not 0 <= sampled_level <= 8:
            raise ValueError("level must be in [0, 8]")
        self.history[sampled_level].append(bool(success))
        if sampled_level == self.level:
            self.current.append(bool(success))
        self.episodes += 1
        if len(self.current) == 100 and sum(self.current) / 100 > 0.7 and self.level < 8:
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
        return self.level if self.level == 8 else self.level + 1

    def success_rate(self, level: int) -> float:
        values = self.history[level]
        return sum(values) / len(values) if values else 0.0

    def style_split(self, seed: int) -> str:
        return split_for_seed(seed)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": 2, "level": self.level, "seed": self.seed,
                "episodes": self.episodes,
                "current": list(self.current),
                "history": {str(k): list(v) for k, v in self.history.items()},
                "progress_watch": {
                    "best_window_rate": {str(k): v for k, v in self.best_window_rate.items()},
                    "last_improvement_step": {str(k): v for k, v in self.last_improvement_step.items()},
                }}

    @classmethod
    def load(cls, path: str | Path) -> Curriculum:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        result = cls(seed=int(data["seed"]))
        result.level = int(data["level"])
        legacy = int(data.get("schema_version", 1)) < 2
        result.episodes = int(data["episodes"])
        old_current = [bool(x) for x in data.get("current", [])]
        result.current.extend(old_current)
        saved_history = data.get("history", {})
        # Pre-86g stored level 6 as the mandatory-break room and level 7 as
        # the multi-room challenge. Shift those to 7 and 8; level 6 is new.
        result.history = {i: deque(maxlen=300) for i in range(9)}
        for old_level in range(9 if not legacy else 8):
            new_level = old_level if not legacy or old_level < 6 else old_level + 1
            result.history[new_level].extend(
                bool(x) for x in saved_history.get(str(old_level), [])
            )
        if not saved_history and 0 <= result.level <= (8 if not legacy else 7):
            migrated_level = (result.level if not legacy or result.level < 6
                              else result.level + 1)
            result.history[migrated_level].extend(old_current)
        if legacy and result.level >= 6:
            # The inserted combat lesson must be completed before the legacy
            # breakable and multi-room lessons count toward the final target.
            result.level = 6
            result.current.clear()
        progress_watch = data.get("progress_watch", {})
        if isinstance(progress_watch, dict):
            raw_best = progress_watch.get("best_window_rate", {})
            raw_steps = progress_watch.get("last_improvement_step", {})
            if isinstance(raw_best, dict):
                result.best_window_rate = {
                    (int(key) + 1 if legacy and int(key) >= 6 else int(key)): float(value)
                    for key, value in raw_best.items()
                }
            if isinstance(raw_steps, dict):
                result.last_improvement_step = {
                    (int(key) + 1 if legacy and int(key) >= 6 else int(key)): int(value)
                    for key, value in raw_steps.items()
                }
        return result
