"""Armazenamento assíncrono, atômico e sem pickle de blocos de experiência."""

import json
import os
import queue
import threading
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import Any, Self

import numpy as np

from voidgym.core.errors import VoidGymRuntimeError as RuntimeError
from voidgym.core.errors import VoidGymValueError as ValueError


class ExperienceWriter:
    """Agrupa transições por episódio e grava blocos NPZ em uma thread de fundo."""

    def __init__(self, root: Path, game_id: str, campaign_id: str, *, source: str = "actor",
                 config_hash: str = "", model_contract: str = "2.0", block_steps: int = 1024,
                 action_table: Sequence[str] = ()) -> None:
        if source not in {"actor", "evaluation", "demonstration", "self_imitation", "platform_gym", "test"}:
            raise ValueError("source de experiência inválida")
        self.root = root / game_id / campaign_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.game_id, self.campaign_id, self.source = game_id, campaign_id, source
        self.config_hash, self.model_contract = config_hash, model_contract
        self.action_table = tuple(action_table)
        self.block_steps = min(1024, max(1, block_steps))
        self._buffer: list[dict[str, Any]] = []
        self._queue: queue.Queue[list[dict[str, Any]] | None] = queue.Queue()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    def append(self, step_dict: dict[str, Any]) -> None:
        if self._error:
            raise RuntimeError("falha ao gravar armazém de experiência") from self._error
        self._buffer.append(dict(step_dict))
        if len(self._buffer) >= self.block_steps or step_dict.get("done") or step_dict.get("truncated"):
            self._enqueue()

    def end_episode(self) -> None:
        self._enqueue()

    def _enqueue(self) -> None:
        if self._buffer:
            block, self._buffer = self._buffer, []
            self._queue.put(block)

    def _write_loop(self) -> None:
        while True:
            block = self._queue.get()
            try:
                if block is None:
                    return
                self._write_block(block)
            except Exception as error:  # noqa: BLE001 - persist worker failures for flush().
                self._error = error
            finally:
                self._queue.task_done()

    def _write_block(self, block: list[dict[str, Any]]) -> None:
        now = datetime.now(UTC).strftime("%Y%m%d")
        block_id = uuid.uuid4().hex
        directory = self.root / "blocks" / now
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{block_id}.npz"
        temporary = target.with_suffix(".npz.tmp")
        names = set.intersection(*(set(item) for item in block))
        dtypes = {"frame": np.uint8, "action": np.int16, "action_mask": np.bool_,
                  "action_history": np.int16,
                  "reward_universal": np.float32, "reward_game": np.float32,
                  "done": np.bool_, "truncated": np.bool_, "discontinuity": np.bool_,
                  "sem": np.uint8, "hero_motion": np.float32, "privileged": np.float32,
                  "interact_prompt": np.int8, "interact_prompt_valid": np.bool_}
        arrays: dict[str, np.ndarray] = {}
        for name in names - {"scene", "tags", "priority"}:
            if name not in dtypes and not name.startswith("extra_"):
                continue
            values = [item[name] for item in block]
            if name == "frame" and any(np.asarray(value).shape != (108, 192, 3) for value in values):
                raise ValueError("frame deve ter shape (108, 192, 3)")
            if name == "sem" and any(np.asarray(value).shape != (27, 48) for value in values):
                raise ValueError("sem deve ter shape (27, 48)")
            arrays[name] = np.asarray(values, dtype=dtypes.get(name))
        if any(array.dtype.hasobject for array in arrays.values()):
            raise ValueError("arrays de experiência não podem ter dtype object/pickle")
        if "action_mask" in arrays:
            arrays["action_mask"] = np.packbits(arrays["action_mask"].astype(np.bool_), axis=-1)
        scenes = [str(item.get("scene", "")) for item in block]
        scene_runs: list[list[Any]] = []
        for step, scene in enumerate(scenes):
            if scene_runs and scene_runs[-1][0] == scene:
                scene_runs[-1][2] += 1
            else:
                scene_runs.append([scene, step, 1])
        tags = sorted({tag for item in block for tag in item.get("tags", [])})
        meta = {"game_id": self.game_id, "campaign_id": self.campaign_id,
                "scene_runs": scene_runs, "action_table": list(self.action_table),
                "source": self.source, "tags": tags, "model_contract": self.model_contract,
                "config_hash": self.config_hash, "created_at": datetime.now(UTC).isoformat()}
        with temporary.open("wb") as stream:
            payload: dict[str, Any] = {**arrays, "meta": np.asarray(json.dumps(meta))}
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
        record = {"id": block_id, "file": str(target.relative_to(self.root)), "steps": len(block),
                  "scenes": sorted(set(scenes)), "source": self.source, "tags": tags,
                  "priority": max((float(item.get("priority", 0.0)) for item in block), default=0.0),
                  "config_hash": self.config_hash, "model_contract": self.model_contract,
                  "created_at": meta["created_at"], "bytes": target.stat().st_size}
        index = self.root / "index.jsonl"
        with index.open("ab") as stream:
            stream.write((json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())

    def flush(self) -> None:
        self._enqueue()
        self._queue.join()
        if self._error:
            raise RuntimeError("falha ao gravar armazém de experiência") from self._error

    def close(self) -> None:
        self.flush()
        self._queue.put(None)
        self._queue.join()
        self._thread.join(timeout=5)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ExperienceReader:
    def __init__(self, root: Path, *, game_id: str | None = None, campaign_id: str | None = None,
                 source: str | None = None, tag: str | None = None, scene: str | None = None,
                 model_contract: str | None = None, date_from: str | None = None,
                 date_to: str | None = None) -> None:
        self.root, self.game_id, self.campaign_id = root, game_id, campaign_id
        self.source, self.tag, self.scene = source, tag, scene
        self.model_contract = model_contract
        self.date_from, self.date_to = date_from, date_to

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in self.root.rglob("index.jsonl"):
            for line in index.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if self.source and record["source"] != self.source:
                    continue
                if self.tag and self.tag not in record["tags"]:
                    continue
                if self.scene and self.scene not in record["scenes"]:
                    continue
                if self.model_contract and record["model_contract"] != self.model_contract:
                    continue
                created = record["created_at"][:10]
                if self.date_from and created < self.date_from:
                    continue
                if self.date_to and created > self.date_to:
                    continue
                if self.game_id and index.parent.parent.name != self.game_id:
                    continue
                if self.campaign_id and index.parent.name != self.campaign_id:
                    continue
                with np.load(index.parent / record["file"], allow_pickle=False) as archive:
                    block = {key: archive[key].copy() for key in archive.files}
                meta = json.loads(str(block.pop("meta")))
                yield {"record": record, "arrays": block, "meta": meta}

    def iter_stacked(self, frame_stack: int = 4) -> Iterator[dict[str, Any]]:
        """Remonta imagens empilhadas por bloco, reiniciando em descontinuidades."""
        if frame_stack < 1:
            raise ValueError("frame_stack deve ser positivo")
        for block in self:
            arrays = block["arrays"]
            frames = arrays.get("frame")
            if frames is None:
                continue
            history: list[np.ndarray] = []
            stacked = []
            discontinuity = arrays.get("discontinuity", np.zeros(len(frames), dtype=np.bool_))
            for index, frame in enumerate(frames):
                if bool(discontinuity[index]):
                    history.clear()
                history.append(frame)
                history = history[-frame_stack:]
                padded = [history[0]] * (frame_stack - len(history)) + history
                stacked.append(np.stack(padded))
            block["arrays"]["frames_stacked"] = np.stack(stacked)
            yield block


class Retention:
    """Aplica um limite removendo blocos antigos de atores primeiro."""

    def __init__(self, budget_gb: float) -> None:
        if budget_gb <= 0:
            raise ValueError("budget_gb deve ser positivo")
        self.budget_bytes = int(budget_gb * 1024**3)

    def apply(self, root: Path) -> int:
        rows: list[tuple[Path, dict[str, Any], Path]] = []
        for index in root.rglob("index.jsonl"):
            for line in index.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                rows.append((index, row, index.parent / row["file"]))
        total = sum(path.stat().st_size for _, _, path in rows if path.exists())
        day_counts: dict[str, int] = {}
        scene_counts: dict[str, int] = {}
        for _, row, _ in rows:
            day = str(row["created_at"])[:10]
            day_counts[day] = day_counts.get(day, 0) + 1
            for scene in set(row["scenes"]):
                scene_counts[str(scene)] = scene_counts.get(str(scene), 0) + 1
        day_minimum = {day: max(1, ceil(count * 0.1)) for day, count in day_counts.items()}
        scene_minimum = {scene: max(1, ceil(count * 0.1)) for scene, count in scene_counts.items()}
        candidates = sorted(rows, key=lambda item: (
            "rare_event:" in " ".join(item[1]["tags"]),
            "record_progress" in item[1]["tags"],
            item[1]["source"] == "self_imitation",
            item[1]["source"] == "demonstration",
            item[1]["created_at"],
        ))
        removed = 0
        for index, row, path in candidates:
            if total <= int(self.budget_bytes * 0.7):
                break
            if row["source"] == "demonstration" or "help_resolved" in row["tags"]:
                continue
            if row["source"] not in {"actor", "evaluation", "platform_gym", "self_imitation"}:
                continue
            day = str(row["created_at"])[:10]
            scenes = {str(scene) for scene in row["scenes"]}
            if day_counts.get(day, 0) - 1 < day_minimum.get(day, 1):
                continue
            if any(scene_counts.get(scene, 0) - 1 < scene_minimum.get(scene, 1)
                   for scene in scenes):
                continue
            size = path.stat().st_size if path.exists() else 0
            path.unlink(missing_ok=True)
            total -= size
            removed += 1
            day_counts[day] -= 1
            for scene in scenes:
                scene_counts[scene] -= 1
            remaining = [line for line in index.read_text(encoding="utf-8").splitlines()
                         if json.loads(line).get("id") != row["id"]]
            temporary = index.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write("\n".join(remaining) + ("\n" if remaining else ""))
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(index)
        return removed


def storage_usage_by_source(root: Path) -> dict[str, int]:
    usage: dict[str, int] = {}
    for index in root.rglob("index.jsonl"):
        for line in index.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            usage[item["source"]] = usage.get(item["source"], 0) + int(item["bytes"])
    return usage
