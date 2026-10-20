"""Bounded, operator-only campaign journey records and schematic map helpers."""

from __future__ import annotations

import gzip
import hashlib
import html
import io
import itertools
import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, cast
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

SCHEMA_VERSION = 1
MAX_CAPTURE_COUNT = 32
MAX_CAPTURE_BYTES = 24 * 1024 * 1024
MAX_RUNS = 5000
MAX_DIARY_BYTES = 2 * 1024 * 1024
MAX_CALIBRATION_THUMBNAILS = 12
MAX_CALIBRATION_THUMBNAIL_BYTES = 2 * 1024 * 1024
CALIBRATION_THUMBNAIL_MAX_BYTES = 192 * 1024
MEDIAFIRE_STABLE_PAGE = "https://www.mediafire.com/file/mo42xgq0f4dw8nx/"
MAP_USER_AGENT = "VoidLabJourney/1.0 (local operator dashboard; map cache downloader)"
SCENE_NAMES = {
    "Crossroads_01": "Encruzilhada Esquecida",
    "Town": "Dirtmouth",
    "Tutorial_01": "Passagem do Rei",
    "Crossroads_05": "Encruzilhada Esquecida",
    "Crossroads_07": "Encruzilhada Esquecida",
    "Fungus1_01": "Caminho Verde",
    "Fungus1_02": "Caminho Verde",
    "Fungus1_05": "Caminho Verde",
    "Fungus2_01": "Ermos Fúngicos",
    "Fungus2_03": "Ermos Fúngicos",
    "Fungus2_14": "Ermos Fúngicos",
    "Mines_01": "Terra do Repouso",
    "Mines_24": "Pico de Cristal",
    "Mines_34": "Pico de Cristal",
    "Ruins1_01": "Cidade das Lágrimas",
    "Ruins1_03": "Cidade das Lágrimas",
    "Waterways_01": "Canais Reais",
    "Deepnest_01": "Ninho Profundo",
    "RestingGrounds_01": "Terra do Repouso",
    "Abyss_01": "O Abismo",
    "Hive_01": "Colmeia",
    "Cliffs_01": "Penhascos Uivantes",
    "Fungus3_01": "Jardins da Rainha",
    "Dream_01": "Palácio Branco",
}
ZONE_COLORS = {
    "Town": "#d9b979",
    "Crossroads": "#92a7c2",
    "Greenpath": "#76a58a",
    "Fungal Wastes": "#87a56f",
    "Queen's Gardens": "#6d9d76",
    "City of Tears": "#69aeb3",
    "Waterways": "#5d9e9a",
    "Deepnest": "#6f667d",
    "Mines": "#a78cbd",
    "RestingGrounds": "#aab6cc",
    "Abyss": "#596574",
    "Hive": "#c29248",
    "Howling Cliffs": "#b0a07e",
    "White Palace": "#c4c7d6",
}


def friendly_scene(scene: str | None) -> str:
    return SCENE_NAMES.get(scene or "", scene or "Desconhecida")


def scene_zone(scene: str) -> str:
    prefix = scene.split("_", 1)[0]
    zones = {
        "Town": "Town",
        "Crossroads": "Crossroads",
        "Fungus1": "Greenpath",
        "Fungus2": "Fungal Wastes",
        "Fungus3": "Queen's Gardens",
        "Ruins1": "City of Tears",
        "Waterways": "Waterways",
        "Deepnest": "Deepnest",
        "Mines": "Mines",
        "RestingGrounds": "RestingGrounds",
        "Abyss": "Abyss",
        "Hive": "Hive",
        "Cliffs": "Howling Cliffs",
        "Dream": "White Palace",
    }
    return zones.get(prefix, prefix)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(6):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                # Dashboard readers and file scanners can briefly hold the old
                # destination open on Windows; retry the atomic swap, not the write.
                time.sleep(0.02 * (attempt + 1))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Small cross-process lock for shared campaign indexes on Windows and Unix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]


def increment_campaign_count(path: Path, key: str, *, limit: int = 4096) -> int:
    with file_lock(path.with_suffix(path.suffix + ".lock")):
        counts: dict[str, int] = {}
        if path.exists():
            payload = json.loads(path.read_text("utf-8"))
            raw = payload.get("counts") if isinstance(payload, dict) else None
            if not isinstance(raw, dict):
                raise ValueError("campaign memory has an unsupported schema")
            counts = {
                str(name): int(value)
                for name, value in raw.items()
                if isinstance(name, str)
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 1
            }
        prior = counts.get(key, 0)
        counts[key] = prior + 1
        if len(counts) > limit:
            for old_key in list(counts)[: len(counts) - limit]:
                del counts[old_key]
        atomic_json(path, {"schema_version": 1, "counts": counts})
        return prior


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "runs": [],
            "scenes": {},
            "edges": {},
            "records": {},
            "achievements": [],
            "events": [],
        }
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported journey schema")
    return value


def _event_key(event: Mapping[str, Any]) -> str:
    category, identity = (
        str(event.get("category", "")),
        str(event.get("key", event.get("value", ""))),
    )
    return f"{category}:{identity}"[:512]


@dataclass
class RunDiary:
    path: Path
    campaign_root: Path
    actor: str
    episode: int
    started_unix_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    sample_interval_seconds: float = 0.5
    capture_png: Callable[[], bytes] | None = None
    data: dict[str, Any] = field(default_factory=dict)
    _last_sample: float = -math.inf
    _last_scene: str | None = None
    _captures: int = 0
    _capture_bytes: int = 0
    _new_keys: set[str] = field(default_factory=set)
    _last_map_view: dict[str, Any] = field(default_factory=dict)
    _scene_spawns: dict[str, tuple[float, float]] = field(default_factory=dict)
    _scene_farthest: dict[str, float] = field(default_factory=dict)
    _last_thumbnail_seconds: dict[str, float] = field(default_factory=dict)
    _exit_capture_scenes: set[str] = field(default_factory=set)
    _last_persist_monotonic: float = 0.0
    last_persist_ms: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.sample_interval_seconds:
            raise ValueError("sample interval must be positive")
        self.data = {
            "schema_version": SCHEMA_VERSION,
            "campaign": self.campaign_root.name,
            "actor": self.actor,
            "episode": self.episode,
            "started_unix_ms": self.started_unix_ms,
            "route": [],
            "positions": [],
            "calibration_points": [],
            "events": [],
            "termination_reason": None,
            "geo_current": None,
            "geo_collected": 0,
            "charm_loadouts": [],
        }
        self.path = self.path.with_suffix(".json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._persist(force=True)

    def _persist(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_persist_monotonic < 2.0:
            return
        raw = json.dumps(self.data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if len(raw.encode("utf-8")) > MAX_DIARY_BYTES:
            raise ValueError("run diary reached its size limit")
        started = time.perf_counter()
        atomic_json(self.path, self.data)
        self.last_persist_ms = (time.perf_counter() - started) * 1000
        self._last_persist_monotonic = now

    def flush(self) -> None:
        self._persist(force=True)

    def observe(
        self, telemetry: Mapping[str, Any], *, monotonic_seconds: float | None = None
    ) -> list[dict[str, Any]]:
        now = time.monotonic() if monotonic_seconds is None else monotonic_seconds
        ts = int(time.time() * 1000)
        scene = telemetry.get("scene_name")
        confirmed: list[dict[str, Any]] = []
        source_events = telemetry.get("events", telemetry.get("milestone_events_since_last_frame", []))
        geo_delta = telemetry.get("geo_gained_since_last_frame", 0)
        if isinstance(geo_delta, int) and not isinstance(geo_delta, bool) and geo_delta > 0:
            self.data["geo_collected"] = int(self.data.get("geo_collected", 0)) + geo_delta
        geo_current = telemetry.get("geo_current")
        if isinstance(geo_current, int) and not isinstance(geo_current, bool) and geo_current >= 0:
            self.data["geo_current"] = geo_current
        loadout_mask = telemetry.get("equipped_charm_mask")
        notch_used = telemetry.get("charm_notches_used")
        notch_total = telemetry.get("charm_notches_total")
        if (isinstance(loadout_mask, int) and not isinstance(loadout_mask, bool)
                and 0 <= loadout_mask < (1 << 40)
                and isinstance(notch_used, int) and not isinstance(notch_used, bool)
                and isinstance(notch_total, int) and not isinstance(notch_total, bool)
                and 0 <= notch_used <= 12 and 0 <= notch_total <= 12):
            loadouts = self.data["charm_loadouts"]
            loadout = {"scene": scene if isinstance(scene, str) else None,
                       "equipped_mask": loadout_mask, "notches_used": notch_used,
                       "notches_total": notch_total}
            changed = not loadouts or any(
                loadouts[-1][key] != loadout[key]
                for key in ("scene", "equipped_mask", "notches_used", "notches_total")
            )
            active_segment = not changed
            if changed and len(loadouts) < 256:
                loadouts.append({**loadout, "unix_ms": ts, "frames": 0,
                                 "geo_gained": 0, "damage_taken": 0,
                                 "enemy_kills": 0, "deaths": 0})
                active_segment = True
            if loadouts and active_segment:
                segment = loadouts[-1]
                segment["frames"] = min(1_000_000, int(segment["frames"]) + 1)
                for field_name, telemetry_name in (
                    ("geo_gained", "geo_gained_since_last_frame"),
                    ("damage_taken", "hp_lost_since_last_frame"),
                    ("enemy_kills", "enemy_kills_since_last_frame"),
                ):
                    amount = telemetry.get(telemetry_name, 0)
                    if isinstance(amount, int) and not isinstance(amount, bool) and amount > 0:
                        segment[field_name] = min(1_000_000, int(segment[field_name]) + amount)
                if isinstance(source_events, list) and any(
                    isinstance(event, Mapping) and event.get("category") == "death"
                    for event in source_events[:64]
                ):
                    segment["deaths"] = min(1_000_000, int(segment["deaths"]) + 1)
        self._last_map_view = {
            key: telemetry[key]
            for key in ("camera_x", "camera_y", "scene_bounds")
            if key in telemetry and isinstance(telemetry[key], (int, float, list, tuple))
        }
        if isinstance(scene, str) and scene:
            if not self.data.get("started_from"):
                self.data["started_from"] = scene
            if scene != self._last_scene:
                self._last_sample = -math.inf
                transition = {"from": self._last_scene, "to": scene, "unix_ms": ts}
                if isinstance(source_events, list):
                    evidence = next((event for event in source_events
                                     if isinstance(event, Mapping)
                                     and event.get("category") == "scene_transition"
                                     and event.get("from") == self._last_scene
                                     and event.get("to") == scene), None)
                    if evidence is not None:
                        for key in ("spatial", "transition_kind", "gate", "exit_x", "exit_y",
                                    "entry_x", "entry_y", "from_map_zone", "to_map_zone"):
                            if key in evidence:
                                transition[key] = evidence[key]
                    else:
                        transition["spatial"] = False
                        transition["transition_kind"] = "missing_bridge_evidence"
                else:
                    transition["spatial"] = False
                    transition["transition_kind"] = "missing_bridge_evidence"
                self.data["route"].append(transition)
                self._last_scene = scene
                confirmed.append({"category": "scene", "key": scene, "scene": scene, "unix_ms": ts})
            if now - self._last_sample >= self.sample_interval_seconds:
                hero_x, hero_y = telemetry.get("x"), telemetry.get("y")
                camera_x, camera_y = telemetry.get("camera_x"), telemetry.get("camera_y")
                bounds = telemetry.get("scene_bounds")

                def numeric_pair(px: Any, py: Any) -> tuple[float, float] | None:
                    if (
                        isinstance(px, (int, float))
                        and not isinstance(px, bool)
                        and math.isfinite(float(px))
                        and isinstance(py, (int, float))
                        and not isinstance(py, bool)
                        and math.isfinite(float(py))
                    ):
                        return float(px), float(py)
                    return None

                hero_point = numeric_pair(hero_x, hero_y)
                camera_point = numeric_pair(camera_x, camera_y)
                bounds_values: tuple[float, float, float, float] | None = None
                if (
                    isinstance(bounds, (list, tuple))
                    and len(bounds) == 4
                    and all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        for value in bounds
                    )
                ):
                    candidate_bounds = cast(
                        tuple[float, float, float, float],
                        tuple(float(value) for value in bounds),
                    )
                    if candidate_bounds[2] > candidate_bounds[0] and candidate_bounds[3] > candidate_bounds[1]:
                        bounds_values = candidate_bounds
                hero_in_bounds = hero_point is not None and (
                    bounds_values is None
                    or bounds_values[0] <= hero_point[0] <= bounds_values[2]
                    and bounds_values[1] <= hero_point[1] <= bounds_values[3]
                )
                x: float | None
                y: float | None
                position_source: str | None
                if hero_point is not None and (hero_in_bounds or camera_point is None):
                    x, y = hero_point
                    position_source = "hero"
                elif camera_point is not None:
                    x, y = camera_point
                    position_source = "camera"
                else:
                    x = y = None
                    position_source = None
                if x is not None and y is not None:
                    position = {
                        "scene": scene,
                        "x": x,
                        "y": y,
                        "unix_ms": ts,
                        "position_source": position_source,
                    }
                    if hero_point is not None:
                        position.update({"hero_x": hero_point[0], "hero_y": hero_point[1]})
                    self.data["positions"].append(position)
                    for key, value in self._last_map_view.items():
                        self.data["positions"][-1][key] = value
                    self._last_sample = now
                    self._capture_calibration_point(scene, x, y, now, ts)
        raw_events = source_events
        if isinstance(geo_delta, int) and not isinstance(geo_delta, bool) and geo_delta > 0:
            raw_events = list(raw_events) if isinstance(raw_events, list) else []
            raw_events.append(
                {"category": "geo", "key": f"geo_{self.data['geo_collected']}", "confirmed": True}
            )
        if isinstance(raw_events, list):
            for raw in raw_events[:64]:
                if not isinstance(raw, Mapping):
                    continue
                event = dict(raw)
                if event.get("confirmed") is not True:
                    continue
                event.setdefault("scene", scene)
                event.setdefault("unix_ms", ts)
                for key in ("x", "y", "camera_x", "camera_y", "scene_bounds"):
                    value = telemetry.get(key)
                    if isinstance(value, (int, float, list, tuple)) and not isinstance(value, bool):
                        event.setdefault(key, value)
                key = _event_key(event)
                self.data["events"].append(event)
                confirmed.append(event)
                if key and key not in self._new_keys:
                    self._new_keys.add(key)
        self._persist(force=bool(confirmed))
        return confirmed

    def _capture_calibration_point(
        self, scene: str, x: float, y: float, now: float, timestamp_ms: int,
        *, force_kind: str | None = None,
    ) -> None:
        if self.capture_png is None:
            return
        points = self.data["calibration_points"]
        if len(points) >= MAX_CALIBRATION_THUMBNAILS:
            return
        spawn = self._scene_spawns.setdefault(scene, (x, y))
        distance = math.hypot(x - spawn[0], y - spawn[1])
        prior_farthest = self._scene_farthest.get(scene, -1.0)
        kind = force_kind or ("spawn" if prior_farthest < 0 else "farthest")
        if kind == "farthest" and (
            distance <= prior_farthest
            or now - self._last_thumbnail_seconds.get(scene, -math.inf) < 30.0
        ):
            return
        self._last_thumbnail_seconds[scene] = now
        try:
            payload = self.capture_png()
            from PIL import Image

            validate_png_header(io.BytesIO(payload))
            with Image.open(io.BytesIO(payload)) as source:
                source.load()
                source.thumbnail((320, 180), Image.Resampling.LANCZOS)
                output = io.BytesIO()
                source.convert("RGB").save(output, format="PNG", optimize=True)
            thumbnail = output.getvalue()
        except (ImportError, OSError, ValueError, RuntimeError):
            return
        current_size = sum(int(point.get("bytes", 0)) for point in points)
        if len(thumbnail) > CALIBRATION_THUMBNAIL_MAX_BYTES or current_size + len(thumbnail) > MAX_CALIBRATION_THUMBNAIL_BYTES:
            return
        if kind == "farthest":
            old_points = [point for point in points
                          if point.get("scene") == scene and point.get("kind") == "farthest"]
            for point in old_points:
                old_path = (self.campaign_root / str(point.get("capture", ""))).resolve()
                try:
                    old_path.relative_to((self.path.parent / f"{self.path.stem}-calibration").resolve())
                except ValueError:
                    continue
                old_path.unlink(missing_ok=True)
            points[:] = [point for point in points
                         if not (point.get("scene") == scene and point.get("kind") in {"farthest", "exit"})]
        directory = self.path.parent / f"{self.path.stem}-calibration"
        directory.mkdir(parents=True, exist_ok=True)
        safe_scene = "".join(char if char.isalnum() or char in "-_" else "_" for char in scene)[:80]
        name = f"{safe_scene}-{kind}-{timestamp_ms}.png"
        target = directory / name
        target.write_bytes(thumbnail)
        points.append({"scene": scene, "kind": kind, "x": x, "y": y,
                       "distance_from_spawn": distance, "unix_ms": timestamp_ms,
                       "bytes": len(thumbnail),
                       "capture": target.relative_to(self.campaign_root).as_posix()})
        self._scene_farthest[scene] = distance
        self._last_thumbnail_seconds[scene] = now

    def capture_before_exit(self, buttons: Sequence[str], *, monotonic_seconds: float | None = None) -> None:
        """Save a local reference before an action likely to cross a room boundary."""
        if self.capture_png is None or not self.data["positions"]:
            return
        position = self.data["positions"][-1]
        scene = position.get("scene")
        bounds = position.get("scene_bounds")
        if not isinstance(scene, str) or scene in self._exit_capture_scenes:
            return
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 4
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(float(value)) for value in bounds)):
            return
        x, y = position.get("x"), position.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return
        left, bottom, right, top = (float(value) for value in bounds)
        near_exit = any((button == "LEFT" and x <= left + 8)
                        or (button == "RIGHT" and x >= right - 8)
                        or (button == "DOWN" and y <= bottom + 8)
                        or (button == "UP" and y >= top - 8) for button in buttons)
        if not near_exit:
            return
        self._exit_capture_scenes.add(scene)
        now = time.monotonic() if monotonic_seconds is None else monotonic_seconds
        self._capture_calibration_point(scene, float(x), float(y), now,
                                        int(time.time() * 1000), force_kind="exit")

    def finish(
        self,
        reason: str | None,
        *,
        geo: int | None = None,
        deaths: Sequence[Mapping[str, Any]] = (),
        shade: Mapping[str, int] | None = None,
    ) -> dict[str, Any]:
        self.data.update(
            {
                "ended_unix_ms": int(time.time() * 1000),
                "termination_reason": reason,
                "geo": geo,
                "deaths": list(deaths),
                "shade": dict(shade or {}),
            }
        )
        self._persist(force=True)
        return self.data

    def capture_firsts(self, events: Sequence[Mapping[str, Any]]) -> list[Path]:
        if not events or self.capture_png is None:
            return []
        with file_lock(self.campaign_root / ".journey-index.lock"):
            return self._capture_firsts_locked(events)

    def _capture_firsts_locked(self, events: Sequence[Mapping[str, Any]]) -> list[Path]:
        saved: list[Path] = []
        known_path = self.campaign_root / "campaign-progress.json"
        index = load_json(known_path)
        seen = set(index.get("event_keys", []))
        directory = self.campaign_root / "marcos"
        existing_captures = list(directory.glob("*.png")) if directory.exists() else []
        self._captures = len(existing_captures)
        self._capture_bytes = sum(path.stat().st_size for path in existing_captures)
        for event in events:
            key = _event_key(event)
            if (
                not key
                or key in seen
                or self.capture_png is None
                or self._captures >= MAX_CAPTURE_COUNT
            ):
                continue
            try:
                payload = self.capture_png()
            except (OSError, ValueError, RuntimeError):
                continue
            if (
                not isinstance(payload, bytes)
                or not payload.startswith(b"\x89PNG\r\n\x1a\n")
                or self._capture_bytes + len(payload) > MAX_CAPTURE_BYTES
            ):
                continue
            directory.mkdir(parents=True, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)[:96]
            target = directory / f"{int(event.get('unix_ms', time.time() * 1000))}-{safe}.png"
            target.write_bytes(payload)
            seen.add(key)
            index["event_keys"] = sorted(seen)
            scene_name = friendly_scene(str(event.get("scene", event.get("key", ""))))
            category = str(event.get("category", "marco"))
            phrase = (
                f"Run {self.episode} chegou pela primeira vez em {scene_name}!"
                if category == "scene"
                else f"Run {self.episode}: {category} confirmado em {scene_name}."
            )
            index.setdefault("milestones", []).append(
                {
                    **dict(event),
                    "scene_name_friendly": scene_name,
                    "phrase": phrase,
                    "run": self.episode,
                    "actor": self.actor,
                    "map_view": self._last_map_view,
                    "capture": str(target.relative_to(self.campaign_root)),
                }
            )
            self._captures += 1
            self._capture_bytes += len(payload)
            saved.append(target)
        if saved:
            atomic_json(known_path, index)
        return saved


def rebuild_progress(campaign_root: Path) -> dict[str, Any]:
    """Aggregate bounded per-actor diaries into one atomic campaign index."""
    with file_lock(campaign_root / ".journey-index.lock"):
        return _rebuild_progress_locked(campaign_root)


def _rebuild_progress_locked(campaign_root: Path) -> dict[str, Any]:
    path = campaign_root / "campaign-progress.json"
    old = load_json(path)
    scenes: dict[str, dict[str, Any]] = {}
    edges: dict[str, int] = {}
    runs: list[dict[str, Any]] = []
    achievements = set(old.get("achievements", []))
    diary_paths = sorted(campaign_root.glob("*/episode-*.json"), key=lambda p: p.stat().st_mtime)[
        -MAX_RUNS:
    ]
    for diary_path in diary_paths:
        try:
            run = json.loads(diary_path.read_text("utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(run, dict) or run.get("schema_version") != SCHEMA_VERSION:
            continue
        run_id = f"{run.get('actor', diary_path.parent.name)}/{run.get('episode', diary_path.stem)}"
        events = [event for event in run.get("events", []) if isinstance(event, Mapping)]
        categories = {
            category: sum(1 for event in events if event.get("category") == category)
            for category in (
                "grub",
                "boss",
                "item",
                "death",
                "shade_created",
                "shade_recovered",
                "shade_lost",
            )
        }
        run_deaths = max(len(run.get("deaths", [])), categories["death"])
        runs.append(
            {
                "id": run_id,
                "actor": run.get("actor"),
                "episode": run.get("episode"),
                "distinct_scenes": len(
                    {edge.get("to") for edge in run.get("route", []) if edge.get("to")}
                ),
                "deaths": run_deaths,
                "geo_current": run.get("geo_current"),
                "geo_collected": run.get("geo_collected", 0),
                "charm_loadouts": (
                    run.get("charm_loadouts", [])[-256:]
                    if isinstance(run.get("charm_loadouts", []), list) else []
                ),
                "grubs": categories["grub"],
                "bosses": categories["boss"],
                "items": categories["item"],
                "shade_created": categories["shade_created"],
                "shade_recovered": categories["shade_recovered"],
                "shade_lost": categories["shade_lost"],
                "termination_reason": run.get("termination_reason"),
                "route": run.get("route", []),
                "positions": run.get("positions", [])[-200:],
                "calibration_points": run.get("calibration_points", [])[-MAX_CALIBRATION_THUMBNAILS:],
                "events": run.get("events", [])[-200:],
                "ended_unix_ms": run.get("ended_unix_ms"),
            }
        )
        for edge in run.get("route", []):
            target = edge.get("to")
            if isinstance(target, str) and target:
                scenes.setdefault(
                    target,
                    {"first_seen": edge.get("unix_ms"), "run": run_id, "actor": run.get("actor")},
                )
            source = edge.get("from")
            if isinstance(source, str) and isinstance(target, str):
                key = f"{source}\u001f{target}"
                edges[key] = edges.get(key, 0) + 1
        for event in run.get("events", []):
            if isinstance(event, dict) and event.get("category") in {"grub", "boss", "item"}:
                achievements.add(_event_key(event))
    max_scenes = max((r["distinct_scenes"] for r in runs), default=0)
    furthest = max((r["distinct_scenes"] for r in runs), default=0)
    fewest_deaths = min((r["deaths"] for r in runs), default=0)
    previous_comparison = None
    if len(runs) >= 2:
        current, previous = runs[-1], runs[-2]
        previous_comparison = {
            "run": current["id"],
            "previous_run": previous["id"],
            "scene_delta": current["distinct_scenes"] - previous["distinct_scenes"],
            "death_delta": current["deaths"] - previous["deaths"],
            "geo_delta": current["geo_collected"] - previous["geo_collected"],
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "runs": runs[-MAX_RUNS:],
        "scenes": scenes,
        "edges": edges,
        "records": {"max_scenes": max_scenes, "furthest": furthest, "fewest_deaths": fewest_deaths},
        "previous_comparison": previous_comparison,
        "achievements": sorted(achievements),
        "event_keys": old.get("event_keys", []),
        "milestones": old.get("milestones", []),
    }
    atomic_json(path, result)
    rotate_runs(campaign_root)
    return result


def rotate_runs(campaign_root: Path) -> None:
    paths = sorted(
        campaign_root.glob("*/episode-*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for path in paths[MAX_RUNS:]:
        gz = path.with_suffix(path.suffix + ".gz")
        if not gz.exists():
            with path.open("rb") as source, gzip.open(gz, "wb", compresslevel=6) as target:
                target.writelines(source)
        path.unlink(missing_ok=True)


def schematic_svg(
    progress: Mapping[str, Any], *, route: Sequence[str] = (), width: int = 900, height: int = 520
) -> str:
    """Stable own schematic of visited scenes only; unknown rooms get deterministic placement."""
    scenes = progress.get("scenes", {})
    if not isinstance(scenes, Mapping) or not scenes:
        return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><text x="24" y="40">O mapa vai se revelar durante a exploração.</text></svg>'
    layout_path = (
        Path(str(progress.get("_layout_path", ""))) if progress.get("_layout_path") else None
    )
    layout: dict[str, Any] = {}
    if layout_path and layout_path.exists():
        try:
            layout = json.loads(layout_path.read_text("utf-8"))
        except (OSError, ValueError):
            layout = {}
    names = sorted(str(name) for name in scenes)
    names.sort(key=lambda name: (scene_zone(name), name))
    occupied = {
        (float(node.get("x", 0)), float(node.get("y", 0)))
        for node in layout.values()
        if isinstance(node, Mapping)
    }
    for name in names:
        if name not in layout:
            zone = scene_zone(name)
            i = len(occupied)
            while (float(90 + (i % 8) * 105), float(90 + (i // 8) * 100)) in occupied:
                i += 1
            x, y = 90 + (i % 8) * 105, 90 + (i // 8) * 100
            layout[name] = {"x": x, "y": y, "zone": zone}
            occupied.add((float(x), float(y)))
    if layout_path:
        atomic_json(layout_path, layout)
    height = max(height, int(max(float(layout[name]["y"]) for name in names) + 80))
    route_edges = set(itertools.pairwise(route))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Mapa das salas visitadas">',
        '<rect width="100%" height="100%" fill="#f3f6f5"/>',
    ]
    edges = progress.get("edges", {})
    if isinstance(edges, Mapping):
        for key in edges:
            if not isinstance(key, str) or "\u001f" not in key:
                continue
            a, b = key.split("\u001f", 1)
            if a not in layout or b not in layout:
                continue
            x1, y1 = layout[a]["x"], layout[a]["y"]
            x2, y2 = layout[b]["x"], layout[b]["y"]
            active = (a, b) in route_edges
            parts.append(
                f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{"#e26d5a" if active else "#aeb9bb"}" stroke-width="{5 if active else 2}"/>'
            )
    for name in names:
        node = layout[name]
        color = ZONE_COLORS.get(node.get("zone"), "#88a7a5")
        parts.append(
            f'<g><circle cx="{node["x"]}" cy="{node["y"]}" r="30" fill="{color}" stroke="#fff" stroke-width="3"/><text x="{node["x"]}" y="{node["y"] + 45}" text-anchor="middle" font-size="12">{html.escape(friendly_scene(name))}</text></g>'
        )
    parts.append("</svg>")
    return "".join(parts)


def fit_affine(
    points: Sequence[tuple[float, float, float, float]],
) -> tuple[float, float, float, float, float, float]:
    """Least-squares world→map affine transform from at least three point pairs."""
    if len(points) < 3:
        raise ValueError("at least three point pairs are required for an affine transform")
    import numpy as np

    source = np.asarray([[x, y, 1.0] for x, y, _, _ in points], dtype=np.float64)
    target = np.asarray([[u, v] for _, _, u, v in points], dtype=np.float64)
    if np.linalg.matrix_rank(source) < 3:
        raise ValueError("calibration points must not be collinear")
    coefficients, *_ = np.linalg.lstsq(source, target, rcond=None)
    values = tuple(
        float(x)
        for x in (
            coefficients[0, 0],
            coefficients[1, 0],
            coefficients[2, 0],
            coefficients[0, 1],
            coefficients[1, 1],
            coefficients[2, 1],
        )
    )
    return (values[0], values[1], values[2], values[3], values[4], values[5])


def build_transition_graph(transitions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate spatial passage observations and reject inconsistent outliers."""
    import numpy as np

    grouped: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    excluded_tokens = ("dream", "stag", "station", "elevator", "respawn", "load", "menu", "title")
    for item in transitions:
        source, target = item.get("from"), item.get("to")
        gate = item.get("gate", "")
        if (item.get("spatial") is not True or not isinstance(source, str) or not source
                or not isinstance(target, str) or not target or source == target
                or item.get("transition_kind") not in (None, "spatial")):
            continue
        gate_name = gate if isinstance(gate, str) else ""
        if any(token in (gate_name + " " + source + " " + target).lower() for token in excluded_tokens):
            continue
        exit_x, exit_y = item.get("exit_x"), item.get("exit_y")
        entry_x, entry_y = item.get("entry_x"), item.get("entry_y")
        coordinates = (exit_x, exit_y, entry_x, entry_y)
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) for value in coordinates):
            continue
        exit_x, exit_y, entry_x, entry_y = (float(cast(float, value)) for value in coordinates)
        delta = (exit_x - entry_x, exit_y - entry_y)
        grouped.setdefault((source, target, gate_name), []).append(delta)

    edges: list[dict[str, Any]] = []
    for (source, target, gate), samples in sorted(grouped.items()):
        values: np.ndarray = np.asarray(samples, dtype=np.float64)
        center: np.ndarray = np.median(values, axis=0)
        residual = np.linalg.norm(values - center, axis=1)
        mad = float(np.median(np.abs(residual - np.median(residual))))
        threshold = max(1.0, float(np.median(residual)) + 3.0 * 1.4826 * mad)
        accepted = values[residual <= threshold]
        if not len(accepted):
            continue
        median = np.median(accepted, axis=0)
        edges.append({"from": source, "to": target, "gate": gate or None,
                      "dx": float(median[0]), "dy": float(median[1]),
                      "samples": len(accepted), "outliers": len(values) - len(accepted)})
    return edges


def solve_room_origins(edges: Sequence[Mapping[str, Any]], anchor_scene: str) -> dict[str, tuple[float, float]]:
    """Propagate room origins from an anchor; fit cycles by anchored least squares."""
    import numpy as np

    adjacency: dict[str, list[tuple[str, float, float]]] = {}
    valid_edges: list[Mapping[str, Any]] = []
    for edge in edges:
        source, target = edge.get("from"), edge.get("to")
        dx, dy = edge.get("dx"), edge.get("dy")
        if (not isinstance(source, str) or not isinstance(target, str)
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(float(value)) for value in (dx, dy))):
            continue
        edge_dx, edge_dy = float(cast(float, dx)), float(cast(float, dy))
        adjacency.setdefault(source, []).append((target, edge_dx, edge_dy))
        adjacency.setdefault(target, []).append((source, -edge_dx, -edge_dy))
        valid_edges.append(edge)
    if anchor_scene not in adjacency:
        return {anchor_scene: (0.0, 0.0)}

    seeds: dict[str, tuple[float, float]] = {anchor_scene: (0.0, 0.0)}
    queue = [anchor_scene]
    for current in queue:
        for neighbor, dx, dy in adjacency.get(current, []):
            if neighbor not in seeds:
                origin = seeds[current]
                seeds[neighbor] = (origin[0] + dx, origin[1] + dy)
                queue.append(neighbor)

    nodes = sorted(seeds)
    variables = [node for node in nodes if node != anchor_scene]
    indices = {node: index for index, node in enumerate(variables)}
    rows: list[list[float]] = []
    targets: list[tuple[float, float]] = []
    weights: list[float] = []
    for edge in valid_edges:
        source, target = str(edge["from"]), str(edge["to"])
        if source not in seeds or target not in seeds:
            continue
        row = [0.0] * len(variables)
        if source != anchor_scene:
            row[indices[source]] -= 1.0
        if target != anchor_scene:
            row[indices[target]] += 1.0
        dx, dy = float(edge["dx"]), float(edge["dy"])
        targets.append((dx, dy))
        rows.append(row)
        weights.append(math.sqrt(max(1, int(edge.get("samples", 1)))))
    if variables and rows:
        matrix = np.asarray(rows, dtype=np.float64)
        weight_array = np.asarray(weights, dtype=np.float64)[:, None]
        rhs = np.asarray(targets, dtype=np.float64)
        solution = np.linalg.lstsq(matrix * weight_array, rhs * weight_array, rcond=None)[0]
        seeds.update({node: (float(solution[index, 0]), float(solution[index, 1]))
                      for node, index in indices.items()})
    return seeds


def propagate_room_calibrations(
    transitions: Sequence[Mapping[str, Any]], *, anchor_scene: str,
    anchor_transform: Sequence[float],
) -> dict[str, dict[str, Any]]:
    """Copy the anchor scale/orientation to connected rooms using passage offsets."""
    edges = build_transition_graph(transitions)
    origins = solve_room_origins(edges, anchor_scene)
    a, b, c, d, e, f = (float(value) for value in anchor_transform)
    result: dict[str, dict[str, Any]] = {}
    for scene, (origin_x, origin_y) in origins.items():
        transform = (a, b, c + a * origin_x + b * origin_y,
                     d, e, f + d * origin_x + e * origin_y)
        confidence = 1.0 if scene == anchor_scene else 0.95
        result[scene] = {"affine": list(transform), "confidence": confidence,
                         "positioned": True,
                         "source": "manual_anchor" if scene == anchor_scene else "observed_passages"}
    return result


def apply_affine(transform: Sequence[float], x: float, y: float) -> tuple[float, float]:
    if len(transform) != 6:
        raise ValueError("affine transform must contain six values")
    a, b, c, d, e, f = transform
    return a * x + b * y + c, d * x + e * y + f


def fit_two_point_calibration(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """Manual two-anchor similarity transform (uniform scale, rotation, translation)."""
    x1, y1, u1, v1 = first
    x2, y2, u2, v2 = second
    sx, sy, tx, ty = x2 - x1, y2 - y1, u2 - u1, v2 - v1
    source_length, target_length = math.hypot(sx, sy), math.hypot(tx, ty)
    if source_length <= 1e-9 or target_length <= 1e-9:
        raise ValueError("manual calibration anchors must be distinct")
    scale = target_length / source_length
    cosine = (sx * tx + sy * ty) / (source_length * target_length)
    sine = (sx * ty - sy * tx) / (source_length * target_length)
    a, b, d, e = scale * cosine, -scale * sine, scale * sine, scale * cosine
    return (a, b, u1 - a * x1 - b * y1, d, e, v1 - d * x1 - e * y1)


def save_scene_calibration(
    path: Path,
    *,
    map_sha256: str,
    scene: str,
    transform: Sequence[float],
    confidence: float,
    candidate: Mapping[str, float] | None = None,
) -> None:
    if len(map_sha256) != 64 or any(char not in "0123456789abcdef" for char in map_sha256):
        raise ValueError("map SHA-256 must be 64 lowercase hex characters")
    if not scene or len(scene) > 256 or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("scene calibration identity/confidence is invalid")
    if len(transform) != 6 or any(not math.isfinite(value) for value in transform):
        raise ValueError("affine transform must contain six finite values")
    payload: dict[str, Any] = {"schema_version": 1, "map_sha256": map_sha256, "transforms": {}}
    if path.exists():
        existing = json.loads(path.read_text("utf-8"))
        if (
            isinstance(existing, dict)
            and existing.get("schema_version") == 1
            and existing.get("map_sha256") == map_sha256
            and isinstance(existing.get("transforms"), dict)
        ):
            payload = existing
    payload["map_sha256"] = map_sha256
    record: dict[str, Any] = {
        "affine": [float(value) for value in transform],
        "confidence": float(confidence),
        "positioned": confidence >= 0.75,
    }
    if candidate is not None:
        keys = ("x", "y", "width", "height")
        if any(key not in candidate or not math.isfinite(candidate[key]) or abs(candidate[key]) > 10_000_000
               for key in keys):
            raise ValueError("candidate region must contain finite bounded map coordinates")
        record["candidate"] = {key: float(candidate[key]) for key in keys}
    payload["transforms"][scene] = record
    atomic_json(path, payload)


def load_scene_calibration(path: Path, *, map_sha256: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text("utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("map_sha256") != map_sha256
        or not isinstance(payload.get("transforms"), dict)
    ):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for scene, record in payload["transforms"].items():
        if not isinstance(scene, str) or not isinstance(record, dict):
            continue
        transform = record.get("affine")
        confidence = record.get("confidence")
        if (
            not isinstance(transform, list)
            or len(transform) != 6
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in transform
            )
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
        ):
            continue
        result[scene] = {
            "affine": [float(value) for value in transform],
            "confidence": float(confidence),
            "positioned": record.get("positioned") is True and confidence >= 0.75,
        }
        raw_candidate = record.get("candidate")
        if isinstance(raw_candidate, dict) and all(
            isinstance(raw_candidate.get(key), (int, float))
            and not isinstance(raw_candidate.get(key), bool)
            and math.isfinite(float(raw_candidate[key]))
            and abs(float(raw_candidate[key])) <= 10_000_000
            for key in ("x", "y", "width", "height")
        ):
            result[scene]["candidate"] = {key: float(raw_candidate[key])
                                           for key in ("x", "y", "width", "height")}
    return result


def validate_png_header(
    stream: Any,
    *,
    max_width: int = 100_000,
    max_height: int = 100_000,
    max_pixels: int = 4_000_000_000,
) -> tuple[int, int]:
    stream.seek(0)
    header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("arquivo não é PNG válido")
    width, height = int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")
    if (
        not width
        or not height
        or width > max_width
        or height > max_height
        or width * height > max_pixels
    ):
        raise ValueError("dimensões do PNG fora do limite seguro")
    stream.seek(0)
    return width, height


def cache_map_file(source: Path, runtime_root: Path) -> tuple[Path, str, tuple[int, int]]:
    """Validate a local PNG and place it in a content-addressed private cache."""
    with source.open("rb") as stream:
        dimensions = validate_png_header(stream)
    hasher = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    target_dir = runtime_root / "cache" / "map-art" / digest
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "map.png"
    if not target.exists():
        tmp = target.with_suffix(".tmp")
        with source.open("rb") as src, tmp.open("wb") as dst:
            while chunk := src.read(1024 * 1024):
                dst.write(chunk)
        os.replace(tmp, target)
    return target, digest, dimensions


class _DownloadLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        values = dict(attrs)
        href = values.get("href")
        identity = " ".join(
            (values.get("id") or "", values.get("class") or "", values.get("aria-label") or "")
        ).lower()
        host = (urlsplit(href).hostname or "").lower() if href else ""
        if href and ("download" in identity or "download" in host):
            self.links.append(href)


def download_mediafire_map(
    runtime_root: Path,
    *,
    page_url: str = MEDIAFIRE_STABLE_PAGE,
    opener: Callable[..., Any] = urlopen,
    allowed_hosts: Sequence[str] = ("mediafire.com",),
    maximum_bytes: int = 1_073_741_824,
    progress: Callable[[int, int | None], None] | None = None,
    disk_free: Callable[[Path], int] | None = None,
) -> tuple[Path, str, tuple[int, int], bool]:
    """Resolve MediaFire's current button URL and stream a validated PNG into cache."""
    page_request = Request(page_url, headers={"User-Agent": MAP_USER_AGENT})
    with opener(page_request, timeout=30) as response:
        page = response.read(2 * 1024 * 1024 + 1)
    if len(page) > 2 * 1024 * 1024:
        raise ValueError("página do mapa excede o limite")
    links = _DownloadLinks()
    links.feed(page.decode("utf-8", errors="replace"))
    direct_url = next(
        (
            resolved
            for link in links.links
            if (resolved := urljoin(page_url, link))
            and urlsplit(resolved).scheme in {"https", "http"}
            and any(
                (urlsplit(resolved).hostname or "").lower() == host
                or (urlsplit(resolved).hostname or "").lower().endswith("." + host)
                for host in allowed_hosts
            )
        ),
        None,
    )
    if direct_url is None:
        raise ValueError("não encontrei o botão de download; importe o PNG baixado pelo navegador")
    root = runtime_root / "cache" / "map-art"
    root.mkdir(parents=True, exist_ok=True)
    pending = root / ".map-download.partial"
    total = pending.stat().st_size if pending.exists() else 0
    request_headers = {"User-Agent": MAP_USER_AGENT}
    if total:
        request_headers["Range"] = f"bytes={total}-"
    with opener(Request(direct_url, headers=request_headers), timeout=60) as response:
        if total and getattr(response, "status", None) != 206:
            total = 0
        mode = "ab" if total else "wb"
        declared = response.headers.get("Content-Length")
        declared_total = total + int(declared) if declared and int(declared) > 0 else None
        if declared_total and declared_total > maximum_bytes:
            raise ValueError("arquivo do mapa excede o limite de download")
        if declared_total and disk_free is not None:
            required = declared_total * 6 + 512 * 1024 * 1024
            available = disk_free(root)
            if available < required:
                raise OSError(
                    f"espaço insuficiente: precisa de ~{required // (1024 * 1024)} MiB, há {available // (1024 * 1024)} MiB livres"
                )
        with pending.open(mode) as output:
            if progress:
                progress(total, declared_total)
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > maximum_bytes:
                    raise ValueError("arquivo do mapa excede o limite de download")
                output.write(chunk)
                if progress:
                    progress(total, declared_total)
            output.flush()
            os.fsync(output.fileno())
    with pending.open("rb") as stream:
        dimensions = validate_png_header(stream)
    hasher = hashlib.sha256()
    with pending.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    target_dir = root / digest
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "map.png"
    os.replace(pending, target)
    reference = root / "reference.json"
    previous = json.loads(reference.read_text("utf-8")) if reference.exists() else {}
    changed = bool(previous.get("sha256") and previous.get("sha256") != digest)
    atomic_json(
        reference,
        {
            "schema_version": 1,
            "sha256": previous.get("sha256", digest),
            "latest_sha256": digest,
            "source_page": page_url,
            "attribution": "The Embraced One — hallownest.net; Hollow Knight © Team Cherry",
        },
    )
    return target, digest, dimensions, changed
