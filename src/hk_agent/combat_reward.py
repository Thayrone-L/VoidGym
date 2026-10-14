from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

COMBAT_REWARD_VERSION = "combat_events_v1"


@dataclass(frozen=True, slots=True)
class CombatRewardConfig:
    """Reward contract for confirmed kills and health actually lost.

    Progress is deliberately configured alongside combat shaping so the
    contract can reject a campaign where farming combat is worth more than
    reaching a new scenario.
    """

    progress_reward: float = 10.0
    enemy_kill_reward: float = 0.5
    boss_kill_multiplier: float = 4.0
    repeated_kill_decay: float = 0.25
    repeated_identity_cap: float = 0.25
    repeated_scene_budget: float = 0.5
    repeated_episode_budget: float = 1.0
    damage_penalty_per_health: float = -0.25
    death_penalty: float = -2.0
    run_failed_penalty: float = -4.0
    progress_dominance_ratio: float = 2.0

    def validate(self) -> None:
        positive = (
            "progress_reward",
            "enemy_kill_reward",
            "boss_kill_multiplier",
            "progress_dominance_ratio",
        )
        if any(getattr(self, name) <= 0 for name in positive):
            raise ValueError("positive combat reward settings must be greater than zero")
        if not 0 <= self.repeated_kill_decay < 1:
            raise ValueError("repeated_kill_decay must be in [0, 1)")
        if any(
            value < 0
            for value in (
                self.repeated_identity_cap,
                self.repeated_scene_budget,
                self.repeated_episode_budget,
            )
        ):
            raise ValueError("combat shaping caps must not be negative")
        if self.damage_penalty_per_health > 0:
            raise ValueError("damage_penalty_per_health must not be positive")
        if self.death_penalty > 0 or self.run_failed_penalty > 0:
            raise ValueError("death penalties must not be positive")
        maximum_repeat_shaping = min(
            self.repeated_scene_budget,
            self.repeated_episode_budget,
        )
        if self.progress_reward < self.progress_dominance_ratio * maximum_repeat_shaping:
            raise ValueError("progress_reward must dominate repeatable combat shaping")
        if self.progress_reward < self.enemy_kill_reward * self.boss_kill_multiplier:
            raise ValueError("progress_reward must exceed a confirmed boss-kill reward")


@dataclass(frozen=True, slots=True)
class ConfirmedKillEvent:
    """A normalized bridge event emitted only after confirmed enemy death."""

    process_session_id: str
    event_sequence: int
    scene_id: str
    stable_enemy_id: str | None
    runtime_enemy_id: str | None
    boss_confirmed: bool = False

    def validate(self) -> None:
        if not self.process_session_id or not self.scene_id:
            raise ValueError("kill event requires process_session_id and scene_id")
        if self.event_sequence < 0:
            raise ValueError("event_sequence must not be negative")
        if self.stable_enemy_id == "" or self.runtime_enemy_id == "":
            raise ValueError("enemy identity fields must be non-empty when present")


@dataclass(frozen=True, slots=True)
class CombatStep:
    """Normalized cumulative supervisor signals for one environment step."""

    process_session_id: str
    health_lost_total: int
    death_event_sequence: int
    player_dead: bool = False
    run_failed: bool = False

    def validate(self) -> None:
        if not self.process_session_id:
            raise ValueError("combat step requires process_session_id")
        if self.health_lost_total < 0 or self.death_event_sequence < 0:
            raise ValueError("combat counters must not be negative")
        if self.run_failed and not self.player_dead:
            raise ValueError("run_failed requires player_dead")


@dataclass(frozen=True, slots=True)
class CombatRewardResult:
    total: float
    components: Mapping[str, float]
    accepted_kills: int
    duplicate_kills: int
    identity_fallbacks: int
    health_lost_delta: int


class CombatRewardLedger:
    """Stateful, restart-serializable combat reward and anti-farm ledger.

    Transport dedupe uses ``(process_session_id, event_sequence)``. Anti-farm
    accounting prefers the bridge's stable identity (scene + hierarchy/name +
    quantized spawn position). Runtime instance identity is an explicit
    process-local fallback and therefore cannot protect across a relaunch.

    First kills are never suppressed by repeat budgets. Only later deaths of
    the same stable identity are diminished and capped by identity, scene and
    episode, so discovering a genuinely new enemy remains rewarding.
    """

    def __init__(self, config: CombatRewardConfig | None = None) -> None:
        self.config = config or CombatRewardConfig()
        self.config.validate()
        self._gameplay_run_id: str | None = None
        self._episode_id: str | None = None
        self._seen_transport: set[tuple[str, int]] = set()
        self._identity_kills: dict[str, int] = {}
        self._identity_repeat_paid: dict[str, float] = {}
        self._scene_repeat_paid: dict[str, float] = {}
        self._episode_repeat_paid = 0.0
        self._health_counters: dict[str, int] = {}
        self._seen_deaths: set[tuple[str, int]] = set()

    def begin_gameplay_run(self, gameplay_run_id: str) -> None:
        if not gameplay_run_id:
            raise ValueError("gameplay_run_id must not be empty")
        self._gameplay_run_id = gameplay_run_id
        self._episode_id = None
        self._scene_repeat_paid.clear()
        self._episode_repeat_paid = 0.0

    def begin_episode(self, episode_id: str) -> None:
        if self._gameplay_run_id is None:
            raise RuntimeError("begin_gameplay_run() is required before begin_episode()")
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        self._episode_id = episode_id
        self._episode_repeat_paid = 0.0

    def apply(
        self,
        step: CombatStep,
        kills: Iterable[ConfirmedKillEvent] = (),
    ) -> CombatRewardResult:
        if self._episode_id is None:
            raise RuntimeError("begin_episode() is required before apply()")
        step.validate()
        components = {
            "confirmed_enemy_kill": 0.0,
            "confirmed_boss_kill_bonus": 0.0,
            "repeated_enemy_kill": 0.0,
            "health_lost": 0.0,
            "player_death": 0.0,
            "gameplay_run_failed": 0.0,
        }
        accepted_kills = 0
        duplicate_kills = 0
        identity_fallbacks = 0

        for event in kills:
            event.validate()
            transport_key = (event.process_session_id, event.event_sequence)
            if transport_key in self._seen_transport:
                duplicate_kills += 1
                continue
            self._seen_transport.add(transport_key)
            accepted_kills += 1
            identity_key, used_fallback = self._identity_key(event)
            identity_fallbacks += int(used_fallback)
            prior_kills = self._identity_kills.get(identity_key, 0)
            self._identity_kills[identity_key] = prior_kills + 1
            base = self.config.enemy_kill_reward
            if prior_kills == 0:
                components["confirmed_enemy_kill"] += base
                if event.boss_confirmed:
                    components["confirmed_boss_kill_bonus"] += (
                        base * (self.config.boss_kill_multiplier - 1.0)
                    )
                continue

            candidate = base * self.config.repeated_kill_decay**prior_kills
            identity_available = (
                self.config.repeated_identity_cap
                - self._identity_repeat_paid.get(identity_key, 0.0)
            )
            scene_key = self._scene_key(event.scene_id)
            scene_available = (
                self.config.repeated_scene_budget
                - self._scene_repeat_paid.get(scene_key, 0.0)
            )
            episode_available = (
                self.config.repeated_episode_budget - self._episode_repeat_paid
            )
            paid = max(0.0, min(candidate, identity_available, scene_available, episode_available))
            components["repeated_enemy_kill"] += paid
            self._identity_repeat_paid[identity_key] = (
                self._identity_repeat_paid.get(identity_key, 0.0) + paid
            )
            self._scene_repeat_paid[scene_key] = (
                self._scene_repeat_paid.get(scene_key, 0.0) + paid
            )
            self._episode_repeat_paid += paid

        previous_health_lost = self._health_counters.get(step.process_session_id, 0)
        if step.health_lost_total < previous_health_lost:
            raise ValueError("health_lost_total moved backwards within one process session")
        health_lost_delta = step.health_lost_total - previous_health_lost
        self._health_counters[step.process_session_id] = step.health_lost_total

        death_key = (step.process_session_id, step.death_event_sequence)
        new_death = step.player_dead and death_key not in self._seen_deaths
        if new_death:
            self._seen_deaths.add(death_key)
            components["player_death"] = self.config.death_penalty
            if step.run_failed:
                components["gameplay_run_failed"] = self.config.run_failed_penalty
        elif not step.player_dead:
            components["health_lost"] = (
                health_lost_delta * self.config.damage_penalty_per_health
            )
        # On a terminal death step the death penalty replaces the health-loss
        # penalty. This avoids charging the lethal mask twice.

        return CombatRewardResult(
            total=float(sum(components.values())),
            components=components,
            accepted_kills=accepted_kills,
            duplicate_kills=duplicate_kills,
            identity_fallbacks=identity_fallbacks,
            health_lost_delta=health_lost_delta,
        )

    def state(self) -> dict[str, Any]:
        """Return JSON-serializable state for atomic supervisor persistence."""

        return {
            "schema_version": 1,
            "reward_version": COMBAT_REWARD_VERSION,
            "config": asdict(self.config),
            "gameplay_run_id": self._gameplay_run_id,
            "episode_id": self._episode_id,
            "seen_transport": [list(item) for item in sorted(self._seen_transport)],
            "identity_kills": dict(self._identity_kills),
            "identity_repeat_paid": dict(self._identity_repeat_paid),
            "scene_repeat_paid": dict(self._scene_repeat_paid),
            "episode_repeat_paid": self._episode_repeat_paid,
            "health_counters": dict(self._health_counters),
            "seen_deaths": [list(item) for item in sorted(self._seen_deaths)],
        }

    @classmethod
    def from_state(cls, payload: Mapping[str, Any]) -> CombatRewardLedger:
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported combat reward state schema")
        raw_config = payload.get("config")
        if not isinstance(raw_config, Mapping):
            raise TypeError("combat reward state has no config")
        ledger = cls(CombatRewardConfig(**raw_config))
        ledger._gameplay_run_id = _optional_text(payload.get("gameplay_run_id"))
        ledger._episode_id = _optional_text(payload.get("episode_id"))
        ledger._seen_transport = _pair_set(payload.get("seen_transport"), "seen_transport")
        ledger._identity_kills = _string_int_map(payload.get("identity_kills"), "identity_kills")
        ledger._identity_repeat_paid = _string_float_map(
            payload.get("identity_repeat_paid"), "identity_repeat_paid"
        )
        ledger._scene_repeat_paid = _string_float_map(
            payload.get("scene_repeat_paid"), "scene_repeat_paid"
        )
        episode_paid = payload.get("episode_repeat_paid")
        if isinstance(episode_paid, bool) or not isinstance(episode_paid, (int, float)):
            raise TypeError("episode_repeat_paid must be numeric")
        ledger._episode_repeat_paid = float(episode_paid)
        ledger._health_counters = _string_int_map(
            payload.get("health_counters"), "health_counters"
        )
        ledger._seen_deaths = _pair_set(payload.get("seen_deaths"), "seen_deaths")
        return ledger

    def _identity_key(self, event: ConfirmedKillEvent) -> tuple[str, bool]:
        if event.stable_enemy_id is not None:
            return f"stable:{event.scene_id}:{event.stable_enemy_id}", False
        if event.runtime_enemy_id is not None:
            return (
                f"runtime:{event.process_session_id}:{event.scene_id}:{event.runtime_enemy_id}",
                True,
            )
        return f"transport:{event.process_session_id}:{event.event_sequence}", True

    def _scene_key(self, scene_id: str) -> str:
        return f"{self._gameplay_run_id}:{scene_id}"


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("optional identifier must be non-empty text")
    return value


def _pair_set(value: object, name: str) -> set[tuple[str, int]]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    result: set[tuple[str, int]] = set()
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or isinstance(item[1], bool)
            or not isinstance(item[1], int)
            or item[1] < 0
        ):
            raise ValueError(f"{name} contains an invalid pair")
        result.add((item[0], item[1]))
    return result


def _string_int_map(value: object, name: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    result: dict[str, int] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key
            or isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
        ):
            raise ValueError(f"{name} contains an invalid entry")
        result[key] = item
    return result


def _string_float_map(value: object, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    result: dict[str, float] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key
            or isinstance(item, bool)
            or not isinstance(item, (int, float))
            or item < 0
        ):
            raise ValueError(f"{name} contains an invalid entry")
        result[key] = float(item)
    return result
