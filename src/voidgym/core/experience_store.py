"""Armazém único de experiência, com blocos NPZ comprimidos e índice JSONL."""

from __future__ import annotations

import json
import math
import os
import queue
import threading
import uuid
from collections import deque
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from voidgym.core.errors import ExperienceStoreError
from voidgym.core.paths import runtime_root
from voidgym.core.semantics import IGNORE, SEM_GRID

MAX_BLOCK_STEPS = 1024
EXPERIENCE_SOURCES = {
    "actor",
    "evaluation",
    "demonstration",
    "self_imitation",
    "platform_gym",
    "test",
}


class ExperienceWriter:
    """Acumula passos leves e grava blocos completos numa thread de fundo."""

    def __init__(
        self,
        root: Path | None,
        game_id: str,
        campaign_id: str,
        *,
        source: str = "actor",
        action_table: tuple[str, ...] = (),
        config_hash: str = "",
        model_contract: str = "2.0",
    ) -> None:
        if source not in EXPERIENCE_SOURCES:
            raise ValueError("source de experiência inválida")
        self.root = (root or runtime_root()) / "v2" / "experience" / game_id / campaign_id
        self.blocks_root = self.root / "blocks"
        self.index_path = self.root / "index.jsonl"
        self.game_id = game_id
        self.campaign_id = campaign_id
        self.source = source
        self.action_table = action_table
        self.config_hash = config_hash
        self.model_contract = model_contract
        self._pending: list[dict[str, Any]] = []
        self._episode_id = uuid.uuid4().hex
        self._queue: queue.Queue[tuple[str, list[dict[str, Any]]] | None] = queue.Queue()
        self._errors: list[Exception] = []
        self._index_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._write_loop, name="voidgym-experience-writer", daemon=True
        )
        self._thread.start()

    def append(self, step_dict: dict[str, Any]) -> None:
        """Guarda cópias dos arrays para que o ator possa reutilizá-los."""
        if self._closed:
            raise ExperienceStoreError("O escritor do armazém já foi fechado")
        step = {
            key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
            for key, value in step_dict.items()
        }
        self._pending.append(step)
        if len(self._pending) == MAX_BLOCK_STEPS:
            self._queue.put((self._episode_id, self._pending))
            self._pending = []

    def end_episode(self) -> None:
        if self._pending:
            self._queue.put((self._episode_id, self._pending))
            self._pending = []
        self._queue.join()
        self._raise_writer_error()
        self._episode_id = uuid.uuid4().hex

    def flush(self) -> None:
        if not self._closed:
            if self._pending:
                self._queue.put((self._episode_id, self._pending))
                self._pending = []
            self._queue.join()
            self._closed = True
            self._queue.put(None)
            self._thread.join()
        self._raise_writer_error()

    def _raise_writer_error(self) -> None:
        if self._errors:
            error = self._errors[0]
            raise ExperienceStoreError(f"Falha ao gravar experiência: {error}") from error

    def _write_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                episode_id, block = item
                self._write_block(block, episode_id)
            except (OSError, ValueError, TypeError, KeyError) as error:
                self._errors.append(error)
            finally:
                self._queue.task_done()

    def _write_block(self, steps: list[dict[str, Any]], episode_id: str) -> None:
        now = datetime.now(UTC)
        day = now.strftime("%Y%m%d")
        self.blocks_root.joinpath(day).mkdir(parents=True, exist_ok=True)
        block_id = uuid.uuid4().hex
        destination = self.blocks_root / day / f"{block_id}.npz"
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        arrays = _pack_steps(steps, self.action_table)
        scenes = [str(step.get("scene", "")) for step in steps]
        tags = sorted({tag for step in steps for tag in step.get("tags", [])})
        meta = {
            "game_id": self.game_id,
            "campaign_id": self.campaign_id,
            "episode_id": episode_id,
            "source": self.source,
            "scenes": scenes,
            "action_table": list(self.action_table),
            "model_contract": self.model_contract,
            "config_hash": self.config_hash,
        }
        arrays["meta"] = np.asarray(json.dumps(meta, ensure_ascii=False))
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **cast(Any, arrays))
            os.replace(temporary, destination)
            entry = {
                "id": block_id,
                "file": str(destination.relative_to(self.root)),
                "steps": len(steps),
                "scenes": sorted(set(scenes)),
                "source": self.source,
                "tags": tags,
                "priority": max((float(step.get("priority", 0)) for step in steps), default=0),
                "config_hash": self.config_hash,
                "model_contract": self.model_contract,
                "created_at": now.isoformat(),
                "bytes": destination.stat().st_size,
                "episode_id": episode_id,
            }
            _append_index_atomic(self.index_path, entry, self._index_lock)
        except (OSError, ValueError, TypeError, KeyError):
            temporary.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            raise


def _pack_steps(
    steps: list[dict[str, Any]], action_table: tuple[str, ...]
) -> dict[str, np.ndarray]:
    count = len(steps)
    frames = np.stack([np.asarray(row["frame"], dtype=np.uint8) for row in steps])
    actions = np.asarray([row.get("action", 0) for row in steps], dtype=np.int16)
    masks = [
        np.asarray(row.get("action_mask", np.ones(len(action_table), dtype=bool)), dtype=bool)
        for row in steps
    ]
    mask_width = max((mask.size for mask in masks), default=len(action_table))
    mask_matrix = np.stack([np.pad(mask, (0, mask_width - mask.size)) for mask in masks])
    sem = np.stack(
        [np.asarray(row.get("sem", np.full(SEM_GRID, IGNORE)), dtype=np.uint8) for row in steps]
    )
    motion = np.stack(
        [np.asarray(row.get("hero_motion", [np.nan] * 3), dtype=np.float32) for row in steps]
    )
    return {
        "frame": frames,
        "action": actions,
        "action_mask": np.packbits(mask_matrix, axis=1),
        "action_mask_width": np.asarray(mask_width, dtype=np.int16),
        "reward_universal": np.asarray(
            [row.get("reward_universal", 0) for row in steps], dtype=np.float32
        ),
        "reward_game": np.asarray([row.get("reward_game", 0) for row in steps], dtype=np.float32),
        "done": np.asarray([row.get("done", False) for row in steps], dtype=bool),
        "truncated": np.asarray([row.get("truncated", False) for row in steps], dtype=bool),
        "discontinuity": np.asarray([row.get("discontinuity", False) for row in steps], dtype=bool),
        "sem": sem.reshape(count, *SEM_GRID),
        "hero_motion": motion.reshape(count, 3),
        "privileged": _pack_privileged(steps),
        **_pack_game_extras(steps),
    }


def _pack_game_extras(steps: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    keys = sorted({key for row in steps for key in row.get("game_extras", {})})
    packed: dict[str, np.ndarray] = {}
    for key in keys:
        values = [row.get("game_extras", {}).get(key) for row in steps]
        present = [np.asarray(value) for value in values if value is not None]
        if not present:
            continue
        width = max(value.size for value in present)
        dtype = np.result_type(*(value.dtype for value in present), np.float32)
        output = np.full((len(steps), width), np.nan, dtype=dtype)
        for index, value in enumerate(values):
            if value is not None:
                flat = np.asarray(value, dtype=dtype).reshape(-1)
                output[index, : flat.size] = flat
        packed[f"extra_{key}"] = output
    return packed


def _pack_privileged(steps: list[dict[str, Any]]) -> np.ndarray:
    values = [row.get("privileged") for row in steps]
    present = [
        np.asarray(value, dtype=np.float32).reshape(-1) for value in values if value is not None
    ]
    if not present:
        return np.empty((len(steps), 0), dtype=np.float32)
    width = max(value.size for value in present)
    output = np.full((len(steps), width), np.nan, dtype=np.float32)
    for index, value in enumerate(values):
        if value is not None:
            row = np.asarray(value, dtype=np.float32).reshape(-1)
            output[index, : row.size] = row
    return output


def _append_index_atomic(path: Path, entry: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        previous = path.read_text(encoding="utf-8") if path.exists() else ""
        pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        pending.write_text(
            previous + json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(pending, path)


class ExperienceReader:
    """Itera blocos filtrados e remonta pilhas de quadros por episódio."""

    def __init__(self, root: Path, filters: dict[str, Any] | None = None) -> None:
        self.root = root
        self.index_path = root / "index.jsonl"
        self.filters = filters or {}

    def iter_blocks(self) -> Iterator[tuple[dict[str, Any], dict[str, np.ndarray]]]:
        if not self.index_path.exists():
            return
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if not _matches(entry, self.filters):
                continue
            path = self.root / entry["file"]
            with np.load(path, allow_pickle=False) as file:
                arrays = {key: file[key].copy() for key in file.files}
            yield entry, arrays

    def iter_steps(self) -> Iterator[dict[str, Any]]:
        histories: dict[str, deque[np.ndarray]] = {}
        for entry, arrays in self.iter_blocks():
            episode_id = str(entry["episode_id"])
            history = histories.setdefault(episode_id, deque(maxlen=4))
            width = int(arrays["action_mask_width"])
            masks = np.unpackbits(arrays["action_mask"], axis=1)[:, :width].astype(bool)
            for index, frame in enumerate(arrays["frame"]):
                history.append(frame)
                while len(history) < 4:
                    history.appendleft(frame)
                yield {
                    "frame_stack": np.stack(tuple(history)),
                    "frame": frame,
                    "action": arrays["action"][index],
                    "action_mask": masks[index],
                    "reward_universal": arrays["reward_universal"][index],
                    "reward_game": arrays["reward_game"][index],
                    "done": arrays["done"][index],
                    "truncated": arrays["truncated"][index],
                    "discontinuity": arrays["discontinuity"][index],
                    "sem": arrays["sem"][index],
                    "hero_motion": arrays["hero_motion"][index],
                    "privileged": arrays["privileged"][index],
                    "meta": json.loads(str(arrays["meta"])),
                }


def _matches(entry: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key in ("source", "config_hash", "model_contract", "game_id", "campaign_id"):
        expected = filters.get(key)
        if expected is not None and entry.get(key) != expected:
            return False
    expected_tags = set(filters.get("tags", ()))
    if expected_tags and not expected_tags.intersection(entry.get("tags", [])):
        return False
    expected_scene = filters.get("scene")
    if expected_scene is not None and expected_scene not in entry.get("scenes", []):
        return False
    created = str(entry.get("created_at", ""))
    date_from = filters.get("date_from")
    date_to = filters.get("date_to")
    if date_from is not None and created[:10] < str(date_from):
        return False
    return date_to is None or created[:10] <= str(date_to)


class Retention:
    """Aplica retenção sem apagar demonstrações e ajudas resolvidas."""

    def __init__(self, budget_gb: float) -> None:
        self.budget_bytes = int(budget_gb * 1024**3)

    def apply(self, root: Path) -> list[str]:
        index = root / "index.jsonl"
        if not index.exists():
            return []
        entries = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
        total = sum(int(item.get("bytes", 0)) for item in entries)
        if total <= int(self.budget_bytes * 0.7):
            return []
        keep_actor = _actor_sample(entries)
        candidates = sorted(
            (item for item in entries if _deletable(item, keep_actor)),
            key=lambda item: (
                _retention_tier(item),
                int(item.get("priority", 0)),
                item.get("created_at", ""),
            ),
        )
        removed: list[str] = []
        for entry in candidates:
            if total <= int(self.budget_bytes * 0.7):
                break
            (root / entry["file"]).unlink(missing_ok=True)
            total -= int(entry.get("bytes", 0))
            removed.append(str(entry["id"]))
        if removed:
            remaining = [item for item in entries if item["id"] not in set(removed)]
            pending = index.with_name(f".{index.name}.{uuid.uuid4().hex}.tmp")
            pending.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in remaining),
                encoding="utf-8",
            )
            os.replace(pending, index)
        return removed


def _actor_sample(entries: list[dict[str, Any]]) -> set[str]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        if entry.get("source") != "actor":
            continue
        day = str(entry.get("created_at", ""))[:10]
        scenes = entry.get("scenes") or [""]
        for scene in scenes:
            groups.setdefault((day, str(scene)), []).append(entry)
    keep: set[str] = set()
    for group in groups.values():
        ordered = sorted(group, key=lambda item: item.get("created_at", ""), reverse=True)
        keep.update(str(item["id"]) for item in ordered[: max(1, math.ceil(len(ordered) * 0.1))])
    return keep


def _deletable(entry: dict[str, Any], keep_actor: set[str]) -> bool:
    tags = set(entry.get("tags", []))
    if entry.get("source") == "demonstration" or "help_resolved" in tags:
        return False
    return not (entry.get("source") == "actor" and str(entry.get("id")) in keep_actor)


def _retention_tier(entry: dict[str, Any]) -> int:
    tags = set(entry.get("tags", []))
    if any(tag.startswith("rare_event:") for tag in tags) or {
        "record_progress",
        "self_imitation",
    }.intersection(tags | {str(entry.get("source", ""))}):
        return 2
    return 0 if entry.get("source") == "actor" else 1
