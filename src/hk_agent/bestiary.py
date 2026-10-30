from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, cast

MAX_ENEMY_SLOTS = 6
ENEMY_TYPE_ID = re.compile(r"^e_[0-9a-f]{10}$")
ENEMY_VECTOR_SIZE = 13


def _enemy_id(name: str) -> str:
    canonical = name.casefold()
    return "e_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]


BESTIARY_DISPLAY_NAMES = {
    _enemy_id(name): label
    for name, label in (
        ("crawlid", "Rastejante"), ("vengefly", "Mosca Vingativa"),
        ("vengefly king", "Rei Mosca Vingativa"), ("aspid hunter", "Caçador Aspídico"),
        ("mosscreep", "Rastejante de Musgo"), ("moss charger", "Investida de Musgo"),
        ("moss knight", "Cavaleiro de Musgo"), ("husk hornhead", "Invólucro Chifrudo"),
        ("husk guard", "Guarda Invólucro"), ("husk sentry", "Sentinela Invólucro"),
        ("mantis youth", "Jovem Louva-a-Deus"), ("mantis warrior", "Guerreiro Louva-a-Deus"),
        ("soul twister", "Torcedor de Alma"), ("false knight", "Falso Cavaleiro"),
        ("gruz mother", "Mãe Gruz"), ("baldur", "Baldur"),
        ("elder baldur", "Baldur Ancião"), ("heavy fool", "Tolo Pesado"),
        ("great husk sentry", "Grande Sentinela Invólucro"),
    )
}


@dataclass(slots=True)
class EnemyExperience:
    encounters: int = 0
    defeated: int = 0
    total_hits_to_defeat: int = 0
    damage_taken: int = 0
    times_killed: int = 0
    total_geo: int = 0
    geo_defeats: int = 0
    gave_item: bool = False
    fought_encounters: int = 0

    @property
    def never_seen(self) -> bool:
        return self.encounters == 0

    def vector(self) -> tuple[float, ...]:
        wins = min(1.0, self.defeated / self.encounters) if self.encounters else 0.0
        mean_hits = self.total_hits_to_defeat / self.defeated if self.defeated else 0.0
        mean_geo = self.total_geo / self.geo_defeats if self.geo_defeats else 0.0
        return (
            float(self.never_seen),
            min(1.0, math.log1p(self.encounters) / math.log1p(1000)),
            wins,
            min(1.0, mean_hits / 100.0),
            min(1.0, math.log1p(self.damage_taken) / math.log1p(100)),
            min(1.0, self.times_killed / max(1, self.encounters)),
            min(1.0, math.log1p(mean_geo) / math.log1p(999_999)),
            float(self.gave_item),
            min(1.0, self.fought_encounters / self.encounters) if self.encounters else 0.0,
        )


class BestiaryMemory:
    """Anonymous enemy memory derived only from visible encounters and lived events."""

    def __init__(self) -> None:
        self.experience: dict[str, EnemyExperience] = {}
        self._active_instances: dict[str, tuple[str, int]] = {}
        self._missing_ticks: dict[str, int] = {}
        self._hits_seen: dict[str, int] = {}
        self.slots: list[tuple[str, float, float, int]] = []

    @classmethod
    def from_payload(cls, value: object) -> BestiaryMemory:
        memory = cls()
        if not isinstance(value, dict):
            return memory
        raw = value.get("bestiary")
        if isinstance(raw, dict):
            for enemy_id, fields in list(raw.items())[:256]:
                if not isinstance(enemy_id, str) or not ENEMY_TYPE_ID.fullmatch(enemy_id) or not isinstance(fields, dict):
                    continue
                clean = {
                    key: val for key, val in fields.items()
                    if key in EnemyExperience.__dataclass_fields__
                    and isinstance(val, (int, bool)) and not isinstance(val, float)
                }
                try:
                    memory.experience[enemy_id] = EnemyExperience(**cast(Any, clean))
                except TypeError:
                    continue
        return memory

    def to_payload(self) -> dict[str, dict[str, int | bool]]:
        return {enemy_id: asdict(stats) for enemy_id, stats in sorted(self.experience.items())[:256]}

    def update(self, telemetry: object, *, hp_lost: object = 0, events: object = None) -> None:
        if not isinstance(telemetry, dict):
            self.slots = []
            return
        raw_slots = telemetry.get("visible_enemies")
        current: dict[str, tuple[str, int]] = {}
        slots: list[tuple[str, float, float, int]] = []
        if isinstance(raw_slots, list):
            for slot in raw_slots[:MAX_ENEMY_SLOTS]:
                if not isinstance(slot, dict):
                    continue
                enemy_id = slot.get("enemy_type_id")
                x, y, hits = slot.get("screen_x"), slot.get("screen_y"), slot.get("struck_this_encounter")
                instance_id = slot.get("instance_id")
                if (
                    not isinstance(enemy_id, str) or not ENEMY_TYPE_ID.fullmatch(enemy_id)
                    or not isinstance(x, (int, float)) or not isinstance(y, (int, float))
                    or not math.isfinite(x) or not math.isfinite(y)
                    or not 0 <= x <= 1 or not 0 <= y <= 1
                    or not isinstance(hits, int) or isinstance(hits, bool) or not 0 <= hits <= 100_000
                    or not isinstance(instance_id, int) or isinstance(instance_id, bool)
                ):
                    continue
                key = str(instance_id)
                current[key] = (enemy_id, hits)
                stats = self.experience.setdefault(enemy_id, EnemyExperience())
                self._hits_seen[key] = max(self._hits_seen.get(key, 0), hits)
                if key not in self._active_instances:
                    stats.encounters += 1
                slots.append((enemy_id, float(x), float(y), hits))
        for instance_id, (enemy_id, _hits) in list(self._active_instances.items()):
            if instance_id in current:
                continue
            self._missing_ticks[instance_id] = self._missing_ticks.get(instance_id, 0) + 1
            if self._missing_ticks[instance_id] >= 10:
                del self._active_instances[instance_id]
                self._missing_ticks.pop(instance_id, None)
                if self._hits_seen.pop(instance_id, 0) > 0:
                    self.experience.setdefault(enemy_id, EnemyExperience()).fought_encounters += 1
        for instance_id, slot in current.items():
            self._active_instances[instance_id] = slot
            self._missing_ticks.pop(instance_id, None)
        self.slots = slots
        active_types = {enemy_id for enemy_id, _hits in current.values()}
        if isinstance(hp_lost, int) and not isinstance(hp_lost, bool) and hp_lost > 0 and len(active_types) == 1:
            only_type = next(iter(active_types))
            self.experience.setdefault(only_type, EnemyExperience()).damage_taken += min(hp_lost, 99)
        if isinstance(events, list):
            for event in events[:128]:
                if not isinstance(event, dict):
                    continue
                enemy_id = event.get("enemy_type_id")
                if not isinstance(enemy_id, str) or not ENEMY_TYPE_ID.fullmatch(enemy_id):
                    continue
                stats = self.experience.setdefault(enemy_id, EnemyExperience())
                stats.defeated += 1
                strikes = event.get("hits_dealt")
                if isinstance(strikes, int) and not isinstance(strikes, bool) and strikes >= 0:
                    stats.total_hits_to_defeat += strikes
                geo = event.get("geo_awarded")
                if isinstance(geo, int) and not isinstance(geo, bool) and 0 <= geo <= 999_999:
                    stats.total_geo += geo
                    stats.geo_defeats += 1
                stats.gave_item |= event.get("item_awarded") is True
                if event.get("killed_player") is True:
                    stats.times_killed += 1

    def vector(self) -> list[float]:
        values: list[float] = []
        for enemy_id, x, y, hits in self.slots:
            stats = self.experience.get(enemy_id, EnemyExperience())
            type_code = int(hashlib.sha256(enemy_id.encode("ascii")).hexdigest()[:8], 16) / 0xFFFFFFFF
            values.extend((x, y, type_code, min(1.0, hits / 100.0), *stats.vector()))
        values.extend([0.0] * ((MAX_ENEMY_SLOTS - len(self.slots)) * ENEMY_VECTOR_SIZE))
        return values
