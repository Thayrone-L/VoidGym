from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from hk_agent.bestiary import ENEMY_VECTOR_SIZE, MAX_ENEMY_SLOTS, BestiaryMemory
from hk_agent.story_objectives import (
    OBJECTIVE_BY_ID,
    STORY_OBJECTIVES,
    is_target_scene,
)

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
PLACE_DESCRIPTORS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("bench", (1 / 4, 1 / 4, 1 / 4)),
    ("stag_station", (2 / 4, 2 / 4, 2 / 4)),
    ("shop", (3 / 4, 3 / 4, 3 / 4)),
    ("map_shop", (1.0, 1.0, 1.0)),
)
PLACE_TYPES = frozenset(name for name, _ in PLACE_DESCRIPTORS)
STORY_TRIGGER_FLAGS = frozenset(item.reveal_flag for item in STORY_OBJECTIVES)
STATION_OPEN_FLAGS = frozenset({
    "openedCrossroads", "openedGreenpath", "openedRestingGrounds", "openedStagNest",
    "openedHiddenStation", "openedGardensStagStation", "openedPalaceGrounds",
})


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
    shop_prices: dict[str, int] = field(default_factory=dict)
    special_places: dict[str, dict[str, set[tuple[int, int]]]] = field(default_factory=dict)
    seen_places: set[str] = field(default_factory=set)
    current_objective: str | None = None
    revealed_objectives: set[str] = field(default_factory=set)
    completed_objectives: set[str] = field(default_factory=set)
    arrived_targets: set[str] = field(default_factory=set)
    claimed_tutorial_rewards: set[str] = field(default_factory=set)
    game_flags: dict[str, bool] = field(default_factory=dict)
    new_events: list[dict[str, Any]] = field(default_factory=list)
    bestiary: BestiaryMemory = field(default_factory=BestiaryMemory)

    @classmethod
    def from_payload(cls, value: object) -> PlayerKnowledge:
        memory = cls()
        if not isinstance(value, dict) or value.get("schema_version") not in {1, 2, 3}:
            return memory
        memory.bestiary = BestiaryMemory.from_payload(value)
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
        for field_name, allowed in (
            ("seen_places", PLACE_TYPES),
            ("revealed_objectives", frozenset(OBJECTIVE_BY_ID)),
            ("completed_objectives", frozenset(OBJECTIVE_BY_ID)),
            ("arrived_targets", frozenset(item.target for item in STORY_OBJECTIVES)),
            ("claimed_tutorial_rewards", None),
        ):
            raw = value.get(field_name)
            if isinstance(raw, list):
                items = {item for item in raw if isinstance(item, str) and (allowed is None or item in allowed)}
                setattr(memory, field_name, items)
        current = value.get("current_objective")
        if isinstance(current, str) and current in OBJECTIVE_BY_ID:
            memory.current_objective = current
        raw_flags = value.get("game_flags")
        if isinstance(raw_flags, dict):
            memory.game_flags = {
                key: flag for key, flag in raw_flags.items()
                if key in STORY_TRIGGER_FLAGS or key in STATION_OPEN_FLAGS
                if isinstance(flag, bool)
            }
        raw_prices = value.get("shop_prices")
        if isinstance(raw_prices, dict):
            memory.shop_prices = {
                key: cost for key, cost in list(raw_prices.items())[:64]
                if isinstance(key, str) and len(key) <= 128
                and isinstance(cost, int) and not isinstance(cost, bool) and 0 <= cost <= 999_999
            }
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
        raw_places = value.get("special_places")
        if isinstance(raw_places, dict):
            for room, kinds in list(raw_places.items())[:256]:
                if not isinstance(room, str) or not isinstance(kinds, dict):
                    continue
                clean: dict[str, set[tuple[int, int]]] = {}
                for kind, cells in kinds.items():
                    if kind in PLACE_TYPES and isinstance(cells, list):
                        clean[kind] = {
                            (cell[0], cell[1]) for cell in cells[:8]
                            if isinstance(cell, list) and len(cell) == 2
                            and all(isinstance(v, int) and not isinstance(v, bool) for v in cell)
                            and 0 <= cell[0] < GRID_WIDTH and 0 <= cell[1] < GRID_HEIGHT
                        }
                if clean:
                    memory.special_places[room] = clean
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
            "schema_version": 3,
            "flags": dict(self.known_flags),
            "charm_costs": dict(self.charm_costs),
            "visited": {room: [list(cell) for cell in sorted(cells)[:144]] for room, cells in sorted(self.visited.items())[:256]},
            "exits": {room: [list(cell) for cell in sorted(cells)[:8]] for room, cells in sorted(self.exits.items())[:256]},
            "points": {room: [list(event) for event in sorted(events)[:8]] for room, events in sorted(self.points.items())[:256]},
            "exit_records": {room: [list(record) for record in records[-8:]] for room, records in sorted(self.exit_records.items())[:256]},
            "skill_epoch": self.skill_epoch,
            "shop_prices": dict(sorted(self.shop_prices.items())[:64]),
            "special_places": {
                room: {kind: [list(cell) for cell in sorted(cells)[:8]] for kind, cells in kinds.items()}
                for room, kinds in sorted(self.special_places.items())[:256]
            },
            "seen_places": sorted(self.seen_places),
            "current_objective": self.current_objective,
            "revealed_objectives": sorted(self.revealed_objectives),
            "completed_objectives": sorted(self.completed_objectives),
            "arrived_targets": sorted(self.arrived_targets),
            "claimed_tutorial_rewards": sorted(self.claimed_tutorial_rewards),
            "game_flags": dict(self.game_flags),
            "bestiary": self.bestiary.to_payload(),
            "last_scene": self.last_scene,
            "last_cell": list(self.last_cell) if self.last_cell else None,
        }

    def update(self, info: dict[str, Any]) -> None:
        self.new_events = []
        telemetry = info.get("operator_telemetry")
        if not isinstance(telemetry, dict):
            return
        self.bestiary.update(
            telemetry,
            hp_lost=info.get("hp_lost_since_last_frame", 0),
            events=info.get("bestiary_events_since_last_frame"),
        )
        scene = telemetry.get("scene_name")
        bounds = telemetry.get("scene_bounds")
        x, y = telemetry.get("x"), telemetry.get("y")
        valid_position = (
            isinstance(scene, str) and len(scene) <= 256
            and isinstance(bounds, list) and len(bounds) == 4
            and all(isinstance(v, (int, float)) and np.isfinite(v) for v in (*bounds, x, y))
        )
        col = row = 0
        if valid_position:
            assert isinstance(bounds, list) and len(bounds) == 4
            x0, y0, x1, y1 = (float(v) for v in bounds)
            assert isinstance(x, (int, float)) and isinstance(y, (int, float))
            valid_position = x1 > x0 and y1 > y0
            if valid_position:
                col = int(np.clip((float(x) - x0) / (x1 - x0) * GRID_WIDTH, 0, GRID_WIDTH - 1))
                row = int(np.clip((float(y) - y0) / (y1 - y0) * GRID_HEIGHT, 0, GRID_HEIGHT - 1))
        flags = telemetry.get("progress_flags")
        clean_game_flags: dict[str, bool] = {}
        if isinstance(flags, dict):
            clean_game_flags = {
                key: value for key, value in flags.items()
                if (key in STORY_TRIGGER_FLAGS or key in STATION_OPEN_FLAGS) and isinstance(value, bool)
            }
        self.game_flags.update(clean_game_flags)
        self._observe_story(scene if isinstance(scene, str) else None, clean_game_flags)
        transitions = telemetry.get("milestone_events_since_last_frame")
        if isinstance(transitions, list):
            for event in transitions:
                if not isinstance(event, dict):
                    continue
                if event.get("transition_kind") in {"stag", "station"}:
                    self._claim_reward("first-fast-travel", "first_fast_travel")
                if event.get("category") == "flag" and event.get("key") in STATION_OPEN_FLAGS:
                    self._claim_reward(f"station-open:{event['key']}", "station_opened")
        if valid_position and isinstance(scene, str):
            self._observe_special_places(scene, telemetry.get("visible_places"), bounds)
            if telemetry.get("at_bench") is True:
                self._claim_reward(f"bench-rest:{scene}:{col}:{row}", "first_bench_rest")
        if not valid_position or not isinstance(scene, str):
            return
        cell = (col, row)
        previous_scene, previous_cell = self.last_scene, self.last_cell
        previously_known = int(scene in self.visited)
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
        prices = telemetry.get("shop_prices_visible")
        if isinstance(prices, list):
            for price in prices[:64]:
                if not isinstance(price, dict):
                    continue
                item_id, geo = price.get("item_id"), price.get("geo")
                if isinstance(item_id, str) and len(item_id) <= 128 and isinstance(geo, int) and not isinstance(geo, bool) and 0 <= geo <= 999_999:
                    self.shop_prices[item_id] = geo

    def _claim_reward(self, key: str, event_type: str) -> None:
        if key in self.claimed_tutorial_rewards:
            return
        self.claimed_tutorial_rewards.add(key)
        self.new_events.append({"type": event_type, "key": key})

    def _observe_special_places(self, scene: str, visible: object, bounds: object) -> None:
        if not isinstance(visible, list) or not isinstance(bounds, list) or len(bounds) != 4:
            return
        x0, y0, x1, y1 = (float(value) for value in bounds)
        if x1 <= x0 or y1 <= y0:
            return
        for place in visible[:8]:
            if not isinstance(place, dict):
                continue
            kind, px, py = place.get("kind"), place.get("world_x"), place.get("world_y")
            if kind not in PLACE_TYPES or not isinstance(px, (int, float)) or not isinstance(py, (int, float)):
                continue
            if not np.isfinite(px) or not np.isfinite(py) or not x0 <= px <= x1 or not y0 <= py <= y1:
                continue
            place_col = int(np.clip((float(px) - x0) / (x1 - x0) * GRID_WIDTH, 0, GRID_WIDTH - 1))
            place_row = int(np.clip((float(py) - y0) / (y1 - y0) * GRID_HEIGHT, 0, GRID_HEIGHT - 1))
            self.seen_places.add(kind)
            self.special_places.setdefault(scene, {}).setdefault(kind, set()).add((place_col, place_row))
            if kind in {"shop", "map_shop"}:
                prices = place.get("prices")
                if isinstance(prices, list):
                    for price in prices[:32]:
                        if isinstance(price, dict):
                            item, cost = price.get("item_id"), price.get("geo")
                            if isinstance(item, str) and len(item) <= 128 and isinstance(cost, int) and not isinstance(cost, bool) and 0 <= cost <= 999_999:
                                self.shop_prices[item] = cost

    def _observe_story(self, scene: str | None, flags: dict[str, bool]) -> None:
        if self.current_objective is None:
            next_index = len(self.revealed_objectives)
            if next_index < len(STORY_OBJECTIVES):
                objective = STORY_OBJECTIVES[next_index]
                if flags.get(objective.reveal_flag, self.game_flags.get(objective.reveal_flag, False)):
                    self.current_objective = objective.objective_id
                    self.revealed_objectives.add(objective.objective_id)
                    self.new_events.append({"type": "objective_revealed", "key": objective.objective_id})
        if self.current_objective is None:
            return
        objective = OBJECTIVE_BY_ID[self.current_objective]
        reached = scene is not None and is_target_scene(objective, scene)
        completed = reached if objective.kind == "reach_area" else bool(
            objective.target_flag and self.game_flags.get(objective.target_flag, False)
        )
        if not completed:
            return
        self.completed_objectives.add(objective.objective_id)
        self._claim_reward(f"objective:{objective.objective_id}", "objective_completed")
        self.current_objective = None
        if objective.kind == "reach_area" and objective.target not in self.arrived_targets:
            self.arrived_targets.add(objective.target)
            self._claim_reward(f"area-arrival:{objective.target}", "first_target_arrival")

    def vector(self, *, hint_level: int = 0) -> NDArray[np.float32]:
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
        for place_kind, descriptor in PLACE_DESCRIPTORS:
            seen = place_kind in self.seen_places
            values.extend((1.0, *descriptor) if seen else (0.0, 0.0, 0.0, 0.0))
        for place_kind, _descriptor in PLACE_DESCRIPTORS:
            values.extend([0.0] * (GRID_WIDTH * GRID_HEIGHT))
            if self.last_scene is not None:
                offset = len(values) - GRID_WIDTH * GRID_HEIGHT
                for col, row in self.special_places.get(self.last_scene, {}).get(place_kind, set()):
                    values[offset + row * GRID_WIDTH + col] = 1.0
        objective = OBJECTIVE_BY_ID.get(self.current_objective or "")
        values.extend([0.0] * 8)
        if objective is not None:
            values[-8] = 1.0
            values[-7 + STORY_OBJECTIVES.index(objective)] = 1.0
            values[-4 + {"reach_area": 0, "defeat_boss": 1, "obtain_ability": 2}.get(objective.kind, 0)] = 1.0
            target_reached = objective.target in self.arrived_targets
            if objective.target_flag:
                target_reached = self.game_flags.get(objective.target_flag, False)
            values[-1] = float(target_reached)
        values.append(float(hint_level))
        for item_id in sorted(self.shop_prices)[:64]:
            price = self.shop_prices[item_id]
            values.extend(
                (1.0, float(np.log1p(price) / np.log1p(999_999)))
                if 0 <= price <= 999_999 else (0.0, 0.0)
            )
        values.extend([0.0] * (2 * (64 - min(64, len(self.shop_prices)))))
        values.extend(self.bestiary.vector())
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def space() -> gym.spaces.Box:
        item_size = sum(1 + len(descriptor) for _, descriptor in ITEM_FLAGS)
        return gym.spaces.Box(
            0.0, 1.0,
            (item_size + GRID_WIDTH * GRID_HEIGHT * 2 + 8 * 6 + 8 * 5
             + len(PLACE_DESCRIPTORS) * (4 + GRID_WIDTH * GRID_HEIGHT) + 8 + 128 + 1
             + MAX_ENEMY_SLOTS * ENEMY_VECTOR_SIZE,),
            dtype=np.float32,
        )
