from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

GRID_WIDTH = 16
GRID_HEIGHT = 9
ABILITY_FLAGS: tuple[tuple[str, tuple[float, ...]], ...] = (
    # acquired, category, effect, action; descriptors are numeric and anonymous.
    ("hasDash", (1, 1, 1, 1)),
    ("hasWallJump", (1, 1, 2, 2)),
    ("hasDoubleJump", (1, 1, 3, 3)),
    ("hasSuperDash", (1, 1, 4, 4)),
    ("hasDreamNail", (1, 3, 5, 5)),
)
KEY_ITEM_FLAGS: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("hasShadowDash", (1, 6, 1, 1)),
    ("hasAcidArmour", (4, 7, 0, 0)),
    ("hasLantern", (4, 8, 0, 0)),
    ("hasTramPass", (4, 9, 0, 0)),
    ("hasQuill", (4, 10, 0, 0)),
    ("hasMap", (4, 11, 0, 0)),
    ("hasCityKey", (4, 12, 0, 0)),
    ("hasSlykey", (4, 13, 0, 0)),
    ("hasWhiteKey", (4, 14, 0, 0)),
    ("hasMenderKey", (4, 15, 0, 0)),
    ("hasWaterwaysKey", (4, 16, 0, 0)),
    ("hasSpaKey", (4, 17, 0, 0)),
    ("hasLoveKey", (4, 18, 0, 0)),
    ("hasKingsBrand", (4, 19, 0, 0)),
    ("hasXunFlower", (4, 20, 0, 0)),
    ("hasDreamGate", (4, 21, 0, 0)),
    ("hasGodfinder", (4, 22, 0, 0)),
    ("hasNailArt", (2, 23, 5, 0)),
    ("hasCyclone", (2, 24, 5, 0)),
    ("hasDashSlash", (2, 25, 5, 0)),
    ("hasUpwardSlash", (2, 26, 5, 0)),
)
CHARM_CATEGORIES = (
    4, 4, 5, 3, 3, 2, 3, 3, 3, 4, 5, 2, 2, 2, 2, 2, 5, 2, 5, 5,
    5, 6, 3, 4, 2, 2, 3, 1, 3, 3, 5, 2, 5, 3, 2, 5, 1, 3, 6, 6,
)
CHARM_EFFECT_CODES = tuple(range(1, 41))  # Stable anonymous slots in the local effect table.
CHARM_FLAGS = tuple(
    (f"gotCharm_{index}", (category, effect, 0, 0))
    for index, (category, effect) in enumerate(zip(CHARM_CATEGORIES, CHARM_EFFECT_CODES, strict=True), start=1)
)
ITEM_FLAGS = ABILITY_FLAGS + KEY_ITEM_FLAGS + CHARM_FLAGS
ITEM_FLAG_NAMES = frozenset(flag for flag, _ in ITEM_FLAGS)
CHARM_FLAG_NAMES = frozenset(flag for flag, _ in CHARM_FLAGS)


@dataclass(slots=True)
class PlayerKnowledge:
    """Bounded first-person memory derived from events and positions already lived."""

    visited: dict[str, set[tuple[int, int]]] = field(default_factory=dict)
    exits: dict[str, set[tuple[int, int]]] = field(default_factory=dict)
    exit_records: dict[str, list[tuple[int, int, int, int, int]]] = field(default_factory=dict)
    points: dict[str, set[tuple[int, int, int]]] = field(default_factory=dict)
    last_scene: str | None = None
    last_cell: tuple[int, int] | None = None
    known_flags: dict[str, bool] = field(default_factory=dict)
    charm_costs: dict[str, int] = field(default_factory=dict)
    skill_epoch: int = 0

    @classmethod
    def from_payload(cls, value: object) -> PlayerKnowledge:
        memory = cls()
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            return memory
        flags, visited, exits, points = (value.get(key) for key in ("flags", "visited", "exits", "points"))
        if isinstance(flags, dict):
            memory.known_flags = {
                key: val for key, val in flags.items()
                if key in ITEM_FLAG_NAMES and isinstance(val, bool)
            }
        costs = value.get("charm_costs")
        if isinstance(costs, dict):
            memory.charm_costs = {
                key: cost for key, cost in costs.items()
                if key in CHARM_FLAG_NAMES and isinstance(cost, int) and not isinstance(cost, bool) and 0 <= cost <= 6
            }
        epoch = value.get("skill_epoch", 0)
        memory.skill_epoch = epoch if isinstance(epoch, int) and not isinstance(epoch, bool) and epoch >= 0 else 0
        if isinstance(visited, dict):
            for room, cells in list(visited.items())[:256]:
                if isinstance(room, str) and isinstance(cells, list):
                    memory.visited[room] = {(int(c[0]), int(c[1])) for c in cells[:144] if isinstance(c, list) and len(c) == 2 and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v < limit for v, limit in zip(c, (GRID_WIDTH, GRID_HEIGHT), strict=True))}
        if isinstance(exits, dict):
            for room, cells in list(exits.items())[:256]:
                if isinstance(room, str) and isinstance(cells, list):
                    memory.exits[room] = {(int(c[0]), int(c[1])) for c in cells[:8] if isinstance(c, list) and len(c) == 2 and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v < limit for v, limit in zip(c, (GRID_WIDTH, GRID_HEIGHT), strict=True))}
        if isinstance(points, dict):
            for room, events in list(points.items())[:256]:
                if isinstance(room, str) and isinstance(events, list):
                    memory.points[room] = {(int(e[0]), int(e[1]), int(e[2])) for e in events[:8] if isinstance(e, list) and len(e) == 3 and all(isinstance(v, int) and not isinstance(v, bool) for v in e) and 0 <= e[0] < GRID_WIDTH and 0 <= e[1] < GRID_HEIGHT and 1 <= e[2] <= 3}
        if isinstance(value.get("exit_records"), dict):
            for room, records in list(value["exit_records"].items())[:256]:
                if isinstance(room, str) and isinstance(records, list):
                    memory.exit_records[room] = [tuple(record) for record in records[-8:] if isinstance(record, list) and len(record) == 5 and all(isinstance(v, int) and not isinstance(v, bool) for v in record) and 0 <= record[0] < GRID_WIDTH and 0 <= record[1] < GRID_HEIGHT and 0 <= record[2] <= 3 and record[3] in (0, 1) and record[4] >= 0]
        last_scene, last_cell = value.get("last_scene"), value.get("last_cell")
        if (
            isinstance(last_scene, str)
            and isinstance(last_cell, list)
            and len(last_cell) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) for v in last_cell)
            and 0 <= last_cell[0] < GRID_WIDTH
            and 0 <= last_cell[1] < GRID_HEIGHT
        ):
            memory.last_scene = last_scene
            memory.last_cell = (last_cell[0], last_cell[1])
        return memory

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "flags": dict(self.known_flags),
            "charm_costs": dict(self.charm_costs),
            "visited": {room: [list(cell) for cell in sorted(cells)[:144]] for room, cells in sorted(self.visited.items())[:256]},
            "exits": {room: [list(cell) for cell in sorted(cells)[:8]] for room, cells in sorted(self.exits.items())[:256]},
            "points": {room: [list(event) for event in sorted(events)[:8]] for room, events in sorted(self.points.items())[:256]},
            "exit_records": {room: [list(record) for record in records[-8:]] for room, records in sorted(self.exit_records.items())[:256]},
            "skill_epoch": self.skill_epoch,
            "last_scene": self.last_scene,
            "last_cell": list(self.last_cell) if self.last_cell else None,
        }

    def update(self, info: dict[str, Any]) -> None:
        telemetry = info.get("operator_telemetry")
        if not isinstance(telemetry, dict):
            return
        scene = telemetry.get("scene_name")
        bounds = telemetry.get("scene_bounds")
        x, y = telemetry.get("x"), telemetry.get("y")
        if not isinstance(scene, str) or len(scene) > 256 or not isinstance(bounds, list) or len(bounds) != 4:
            return
        samples = (*bounds, x, y)
        if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in samples):
            return
        x0, y0, x1, y1 = (float(v) for v in bounds)
        assert isinstance(x, (int, float)) and isinstance(y, (int, float))
        if x1 <= x0 or y1 <= y0:
            return
        col = int(np.clip((float(x) - x0) / (x1 - x0) * GRID_WIDTH, 0, GRID_WIDTH - 1))
        row = int(np.clip((float(y) - y0) / (y1 - y0) * GRID_HEIGHT, 0, GRID_HEIGHT - 1))
        cell = (col, row)
        previous_scene, previous_cell = self.last_scene, self.last_cell
        previously_known = int(scene in self.visited)
        flags = telemetry.get("progress_flags")
        if isinstance(flags, dict):
            clean_flags = {
                key: value for key, value in flags.items()
                if key in ITEM_FLAG_NAMES and isinstance(value, bool)
            }
            self.skill_epoch += sum(1 for k, v in clean_flags.items() if v and not self.known_flags.get(k, False))
            self.known_flags.update(clean_flags)
        costs = telemetry.get("charm_costs")
        if isinstance(costs, dict):
            for key, cost in costs.items():
                flag = f"gotCharm_{key}" if isinstance(key, int) and not isinstance(key, bool) else key
                if flag in CHARM_FLAG_NAMES and isinstance(cost, int) and not isinstance(cost, bool) and 0 <= cost <= 6:
                    self.charm_costs[flag] = cost
        if previous_scene is not None and scene != previous_scene and previous_cell is not None:
            self.exits.setdefault(previous_scene, set()).add(previous_cell)
            col0, row0 = previous_cell
            direction = 0 if col0 <= 1 else 1 if col0 >= GRID_WIDTH - 2 else 2 if row0 <= 1 else 3
            records = self.exit_records.setdefault(previous_scene, [])
            records.append((col0, row0, direction, previously_known, self.skill_epoch))
            del records[:-8]
        self.visited.setdefault(scene, set()).add(cell)
        self.last_scene, self.last_cell = scene, cell
        events = telemetry.get("milestone_events_since_last_frame")
        if isinstance(events, list):
            for event in events:
                if isinstance(event, dict) and event.get("category") in {"death", "stuck", "novelty"}:
                    self.points.setdefault(scene, set()).add((col, row, {"death": 1, "stuck": 2, "novelty": 3}[event["category"]]))

    def vector(self) -> NDArray[np.float32]:
        values: list[float] = []
        for flag, descriptor in ITEM_FLAGS:
            acquired = self.known_flags.get(flag, False)
            values.append(1.0 if acquired else 0.0)
            if flag in CHARM_FLAG_NAMES:
                category, effect, _, action = descriptor
                charm_values = (category / 6.0, effect / 40.0, self.charm_costs.get(flag, 0) / 6.0, action / 5.0)
                values.extend(charm_values if acquired else (0.0,) * 4)
            else:
                category, effect, action, auxiliary = descriptor
                values.extend(
                    (category / 6.0, effect / 40.0, action / 6.0, auxiliary / 5.0)
                    if acquired else (0.0,) * 4
                )
        values.extend([0.0] * (GRID_WIDTH * GRID_HEIGHT))
        if self.last_scene is not None:
            for col, row in self.visited.get(self.last_scene, set()):
                values[len(values) - GRID_WIDTH * GRID_HEIGHT + row * GRID_WIDTH + col] = 1.0
        values.extend([0.0] * (8 * 6))
        offset = len(values) - 8 * 6
        if self.last_scene is not None:
            for index, (_col, _row, direction, known, epoch) in enumerate(self.exit_records.get(self.last_scene, ())[-8:]):
                values[offset + index * 6 + direction] = 1.0
                values[offset + index * 6 + 4] = float(known)
                values[offset + index * 6 + 5] = float(epoch == self.skill_epoch)
        values.extend([0.0] * (8 * 5))
        offset = len(values) - 8 * 5
        if self.last_scene is not None:
            for index, (col, row, kind) in enumerate(sorted(self.points.get(self.last_scene, set()))[-8:]):
                values[offset + index * 5] = col / (GRID_WIDTH - 1)
                values[offset + index * 5 + 1] = row / (GRID_HEIGHT - 1)
                values[offset + index * 5 + 2 + kind - 1] = 1.0
        values.extend([0.0] * (GRID_WIDTH * GRID_HEIGHT))
        if self.last_cell is not None:
            col, row = self.last_cell
            values[len(values) - GRID_WIDTH * GRID_HEIGHT + row * GRID_WIDTH + col] = 1.0
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def space() -> gym.spaces.Box:
        item_size = sum(1 + len(descriptor) for _, descriptor in ITEM_FLAGS)
        return gym.spaces.Box(0.0, 1.0, (item_size + GRID_WIDTH * GRID_HEIGHT * 2 + 8 * 6 + 8 * 5,), dtype=np.float32)
