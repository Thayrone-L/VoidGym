"""Generic isolated challenge snapshots, repeated practice, and bounded memory."""
from __future__ import annotations

import json
import re
import shutil
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

ChallengeKind = Literal["boss", "traversal", "item", "area"]
_SAVE_FILE = re.compile(r"^user\d+(?:_[0-9.]+)?\.(?:dat|modded\.json)(?:\.bak\d*)?$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ChallengeSnapshot:
    challenge_id: str
    kind: ChallengeKind
    source: Literal["arena_entrance", "challenge_room", "last_bench"]
    save_path: str
    created_ns: int


def create_challenge_snapshot(
    *, challenge_id: str, kind: ChallengeKind,
    source: Literal["arena_entrance", "challenge_room", "last_bench"],
    save_root: Path, snapshot_root: Path, personal_root: Path,
    max_snapshots: int = 32,
) -> ChallengeSnapshot:
    """Copy only at approved boundaries and refuse personal/baseline save roots."""
    if source not in {"arena_entrance", "challenge_room", "last_bench"}:
        raise ValueError("snapshot source is not an approved challenge boundary")
    if (
        not challenge_id or len(challenge_id) > 128
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", challenge_id)
        or max_snapshots < 1
    ):
        raise ValueError("invalid challenge snapshot settings")
    source_root = save_root.resolve(strict=True)
    personal = personal_root.resolve(strict=False)
    if source_root == personal or personal in source_root.parents or source_root in personal.parents:
        raise ValueError("challenge snapshots cannot use the personal save tree")
    target_root = snapshot_root.resolve(strict=False)
    if target_root == source_root or source_root in target_root.parents:
        raise ValueError("snapshot target cannot be inside the live save tree")
    if target_root == personal or personal in target_root.parents or target_root in personal.parents:
        raise ValueError("challenge snapshots must be outside the personal save tree")
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / f"{time.time_ns()}-{challenge_id}"
    target.mkdir()
    copied = 0
    for save_file in source_root.iterdir():
        if not _SAVE_FILE.fullmatch(save_file.name) or not save_file.is_file() or save_file.is_symlink():
            continue
        shutil.copy2(save_file, target / save_file.name)
        copied += 1
    if copied == 0:
        target.rmdir()
        raise ValueError("isolated save root contains no supported game save files")
    snapshots = sorted(
        (path for path in target_root.iterdir() if path.is_dir() and not path.is_symlink()),
        key=lambda path: path.stat().st_mtime_ns,
    )
    for stale in snapshots[:-max_snapshots]:
        if stale.is_dir() and not stale.is_symlink():
            shutil.rmtree(stale)
        else:
            stale.unlink(missing_ok=True)
    return ChallengeSnapshot(challenge_id, kind, source, str(target), time.time_ns())


@dataclass(slots=True)
class ChallengeAttempt:
    challenge_id: str
    kind: ChallengeKind
    version: str
    duration_seconds: float = 0.0
    hits: int = 0
    progress: float = 0.0
    success: bool = False


@dataclass(slots=True)
class ChallengeMemory:
    attempts: int = 0
    wins: int = 0
    best_progress: float = 0.0
    best_hits: int = 0
    best_survival_seconds: float = 0.0
    versions: set[str] = field(default_factory=set)

    def record(self, attempt: ChallengeAttempt) -> None:
        if attempt.duration_seconds < 0 or attempt.hits < 0 or attempt.progress < 0:
            raise ValueError("challenge metrics must be non-negative")
        self.attempts += 1
        self.wins += int(attempt.success)
        self.best_progress = max(self.best_progress, attempt.progress)
        if attempt.success and (not self.best_hits or attempt.hits < self.best_hits):
            self.best_hits = attempt.hits
        self.best_survival_seconds = max(self.best_survival_seconds, attempt.duration_seconds)
        self.versions.add(attempt.version)

    def payload(self) -> dict[str, Any]:
        return {**asdict(self), "versions": sorted(self.versions)}


class ChallengeSignalTracker:
    """Consumes explicit bridge boss classifications and emits bounded attempt events."""

    def __init__(self) -> None:
        self.active: dict[str, dict[str, Any]] = {}
        self.scene: str | None = None

    def update(self, info: dict[str, Any]) -> list[dict[str, Any]]:
        telemetry = info.get("operator_telemetry")
        if not isinstance(telemetry, dict):
            return []
        now = time.monotonic_ns()
        scene = telemetry.get("scene_name")
        visible = telemetry.get("visible_enemies", [])
        events: list[dict[str, Any]] = []
        active_now: dict[str, dict[str, Any]] = {}
        if isinstance(visible, list):
            for item in visible[:6]:
                if not isinstance(item, dict) or item.get("boss_confirmed") is not True:
                    continue
                key, version = item.get("boss_key"), item.get("boss_version")
                if not isinstance(key, str) or not isinstance(version, str):
                    continue
                previous = self.active.get(key, {"started_ns": now, "hits": 0, "rewarded_hits": 0, "version": version, "scene": scene})
                hits = int(item.get("struck_this_encounter", 0))
                delta_hits = min(
                    max(0, hits - int(previous["hits"])),
                    max(0, 3 - int(previous.get("rewarded_hits", 0))),
                )
                previous["rewarded_hits"] = int(previous.get("rewarded_hits", 0)) + delta_hits
                previous["hits"] = max(int(previous["hits"]), hits)
                active_now[key] = previous
                if key not in self.active:
                    events.append({
                        "event": "challenge_started", "challenge_id": key, "kind": "boss",
                        "version": version, "scene": scene, "snapshot_requested": True,
                    })
                if delta_hits:
                    events.append({
                        "event": "challenge_hit", "challenge_id": key,
                        "hit_delta": min(delta_hits, 100), "version": version,
                    })
        self.active.update(active_now)
        if self.scene is not None and isinstance(scene, str) and scene != self.scene:
            for key, record in list(self.active.items()):
                if record.get("scene") == self.scene and key not in active_now:
                    events.append(self._attempt(key, record, success=False, now=now))
                    self.active.pop(key, None)
        self.scene = scene if isinstance(scene, str) else self.scene

        killed: set[str] = set()
        raw_kills = info.get("bestiary_events_since_last_frame", [])
        if isinstance(raw_kills, (list, tuple)):
            for event in raw_kills[:32]:
                if not isinstance(event, dict) or event.get("boss_confirmed") is not True:
                    continue
                key = event.get("boss_key")
                if isinstance(key, str):
                    record = self.active.pop(key, {"started_ns": now, "hits": 0, "rewarded_hits": 0, "version": event.get("boss_version", "unknown"), "scene": scene})
                    record["hits"] = max(int(record.get("hits", 0)), int(event.get("hits_dealt", 0)))
                    events.append(self._attempt(key, record, success=True, now=now))
                    killed.add(key)
        supervisor = info.get("supervisor", {})
        player_dead = info.get("player_dead") is True or (
            isinstance(supervisor, dict) and supervisor.get("player_dead") is True
        )
        if player_dead:
            for key, record in list(self.active.items()):
                events.append(self._attempt(key, record, success=False, now=now))
                self.active.pop(key, None)
        return events

    @staticmethod
    def _attempt(key: str, record: dict[str, Any], *, success: bool, now: int) -> dict[str, Any]:
        elapsed = max(0.0, (now - int(record["started_ns"])) / 1_000_000_000)
        return {
            "event": "challenge_attempt", "challenge_id": key, "kind": "boss",
            "version": record.get("version", "unknown"), "attempts": 1,
            "wins": int(success), "success": success, "hits": int(record.get("hits", 0)),
            "duration_seconds": elapsed, "progress": 1.0 if success else 0.0,
            "survival_improvement": False,
        }


class ChallengePracticeController:
    """Optional second-actor practice; all collected attempts use shared learner callback."""

    def __init__(
        self, *, main_actor: str, practice_actor: str, required_wins: int = 2,
        recent_attempts: int = 5, time_limit_seconds: float = 3600.0,
    ) -> None:
        if main_actor == practice_actor or required_wins < 1 or recent_attempts < required_wins:
            raise ValueError("invalid practice actor or exit criteria")
        if time_limit_seconds <= 0:
            raise ValueError("time limit must be positive")
        self.main_actor = main_actor
        self.practice_actor = practice_actor
        self.required_wins = required_wins
        self.recent: deque[bool] = deque(maxlen=recent_attempts)
        self.time_limit_seconds = time_limit_seconds
        self.deadline = 0.0
        self.snapshot: ChallengeSnapshot | None = None
        self.active = False
        self.memory: dict[str, ChallengeMemory] = {}
        self._submit: Any = None

    def start(self, snapshot: ChallengeSnapshot, submit_to_shared_learner: Any) -> None:
        if self.active:
            raise RuntimeError("challenge practice is already active")
        if not callable(submit_to_shared_learner):
            raise TypeError("shared learner submission callback is required")
        self.snapshot = snapshot
        self._submit = submit_to_shared_learner
        self.recent.clear()
        self.deadline = time.monotonic() + self.time_limit_seconds
        self.active = True

    def record_attempt(self, attempt: ChallengeAttempt) -> bool:
        if not self.active or self.snapshot is None:
            raise RuntimeError("challenge practice is not active")
        if attempt.challenge_id != self.snapshot.challenge_id or attempt.kind != self.snapshot.kind:
            raise ValueError("attempt belongs to a different challenge context")
        self.memory.setdefault(attempt.challenge_id, ChallengeMemory()).record(attempt)
        self.recent.append(attempt.success)
        should_exit = (
            sum(self.recent) >= self.required_wins or time.monotonic() >= self.deadline
        )
        if should_exit:
            self.active = False
        return should_exit

    def submit_rollout(self, rollout: Any) -> None:
        if not self.active or self._submit is None:
            raise RuntimeError("practice rollout outside active challenge mode")
        self._submit(rollout)

    def should_exit(self) -> bool:
        if not self.active:
            return True
        return sum(self.recent) >= self.required_wins or time.monotonic() >= self.deadline

    def status(self) -> dict[str, Any]:
        name = self.snapshot.challenge_id if self.snapshot else None
        memory = self.memory.get(name or "", ChallengeMemory())
        return {
            "mode": "challenge_practice" if self.active else "main_playthrough",
            "actor": self.practice_actor if self.active else self.main_actor,
            "challenge": name,
            "label": f"Treinando: {name}" if self.active and name else None,
            "attempts": memory.attempts,
            "best_progress": memory.best_progress,
            "wins": memory.wins,
        }


def challenge_reward(
    *, kind: ChallengeKind, hit_delta: int = 0, survival_improvement: bool = False,
    success: bool = False, already_rewarded_hits: int = 0, hit_cap: int = 3,
) -> dict[str, float]:
    """Bounded progress reward; repeated attempts never reward the attempt itself."""
    if min(hit_delta, already_rewarded_hits, hit_cap) < 0:
        raise ValueError("challenge reward counters must be non-negative")
    hit = min(hit_delta, max(0, hit_cap - already_rewarded_hits)) * 0.05
    improve = 0.1 if survival_improvement else 0.0
    win = 2.0 if success else 0.0
    return {"challenge_hit": hit, "survival_record": improve, "challenge_success": win}


def save_challenge_memory(path: Path, memory: dict[str, ChallengeMemory]) -> None:
    payload = {"schema_version": 1, "challenges": {key: value.payload() for key, value in memory.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, allow_nan=False), encoding="utf-8")

