"""Bounded, v2-only discovery rewards for hits and confirmed combat events."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DiscoveryRewardConfig:
    first_geo_rock_hit: float = 0.05
    geo_rock_broken: float = 0.1
    first_enemy_kill: float = 0.15
    room_budget: float = 0.5
    episode_slice_budget: float = 2.0

    def validate(self) -> None:
        values = (self.first_geo_rock_hit, self.geo_rock_broken,
                  self.first_enemy_kill, self.room_budget, self.episode_slice_budget)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("discovery reward values must be finite and non-negative")


class DiscoveryReward:
    """Pay once per scene/object for discovery events, within room and slice caps."""

    def __init__(self, config: DiscoveryRewardConfig | None = None) -> None:
        self.config = config or DiscoveryRewardConfig()
        self.config.validate()
        self._seen: set[tuple[str, str, str]] = set()
        self.reset_slice()

    def reset_slice(self) -> None:
        self._slice_paid = 0.0
        self._room_paid: dict[str, float] = {}

    def __call__(self, before: dict[str, Any], after: dict[str, Any]) -> float:
        del before  # The ledger is event-driven; pixels and geo amounts are not rewarded.
        reward = 0.0
        events = after.get("combat_events", ())
        if not isinstance(events, (list, tuple)):
            return 0.0
        for event in events:
            if not isinstance(event, dict):
                continue
            kind, scene, object_id = (event.get("kind"), event.get("scene_name"),
                                      event.get("object_id"))
            if not (isinstance(kind, str) and kind
                    and isinstance(scene, str) and scene
                    and isinstance(object_id, str) and object_id):
                continue
            values = {"hit_geo_rock": self.config.first_geo_rock_hit,
                      "geo_rock_broken": self.config.geo_rock_broken,
                      "kill_enemy": self.config.first_enemy_kill}
            amount = values.get(kind)
            if amount is None:
                continue
            key = (kind, scene, object_id)
            if key in self._seen:
                continue
            self._seen.add(key)
            room_paid = self._room_paid.get(scene, 0.0)
            remaining = min(self.config.room_budget - room_paid,
                            self.config.episode_slice_budget - self._slice_paid)
            award = max(0.0, min(amount, remaining))
            self._room_paid[scene] = room_paid + award
            self._slice_paid += award
            reward += award
        return reward
