"""Round trip, atomicidade, filtros, retenção e custo do caminho do ator."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from voidgym.core.errors import ExperienceStoreError
from voidgym.core.experience_store import ExperienceReader, ExperienceWriter, Retention
from voidgym.core.semantics import IGNORE


def _step(index: int, *, tags: list[str] | None = None) -> dict[str, object]:
    frame = np.full((8, 12, 3), index % 255, dtype=np.uint8)
    return {
        "frame": frame,
        "action": index % 3,
        "action_mask": np.asarray([True, True, False]),
        "reward_universal": 0.25,
        "reward_game": 0.5,
        "scene": "room-a",
        "tags": tags or [],
        "game_extras": {"coins": np.asarray([index, index + 1], dtype=np.float32)},
    }


def test_three_episodes_round_trip_filters_and_rebuilds_stack(tmp_path: Path) -> None:
    writer = ExperienceWriter(
        tmp_path, "fake", "campaign-a", source="actor", action_table=("a", "b", "c")
    )
    for episode in range(3):
        for index in range(5):
            writer.append(_step(episode * 5 + index))
        writer.end_episode()
    writer.flush()
    root = tmp_path / "v2" / "experience" / "fake" / "campaign-a"
    blocks = list(
        ExperienceReader(
            root,
            {"source": "actor", "scene": "room-a", "date_from": "2000-01-01"},
        ).iter_blocks()
    )
    steps = list(ExperienceReader(root).iter_steps())
    assert len(blocks) == 3
    assert len(steps) == 15
    assert steps[0]["frame_stack"].shape == (4, 8, 12, 3)
    assert steps[0]["action_mask"].tolist() == [True, True, False]
    assert np.all(steps[0]["sem"] == IGNORE)
    assert np.isnan(steps[0]["hero_motion"]).all()
    assert steps[0]["meta"]["source"] == "actor"
    assert list(blocks[0][1]["extra_coins"][0]) == [0.0, 1.0]


def test_atomic_failure_does_not_publish_partial_index(tmp_path: Path, monkeypatch) -> None:
    writer = ExperienceWriter(tmp_path, "fake", "campaign", action_table=("a",))

    def fail(*_args, **_kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(np, "savez_compressed", fail)
    writer.append(_step(0))
    with pytest.raises(ExperienceStoreError, match="simulated write failure"):
        writer.flush()
    root = tmp_path / "v2" / "experience" / "fake" / "campaign"
    assert not (root / "index.jsonl").exists()
    assert list((root / "blocks").rglob("*.npz")) == []


def test_retention_preserves_demos_help_and_actor_sample(tmp_path: Path) -> None:
    writer = ExperienceWriter(tmp_path, "fake", "campaign", source="actor", action_table=("a",))
    for tags in ([], [], [], ["rare_event:death"], ["help_resolved"]):
        for index in range(40):
            row = _step(index, tags=tags)
            row["scene"] = "test-scene"
            writer.append(row)
        writer.end_episode()
    writer.flush()
    demo = ExperienceWriter(
        tmp_path, "fake", "campaign", source="demonstration", action_table=("a",)
    )
    for index in range(40):
        demo.append(_step(index))
    demo.flush()
    actor_root = tmp_path / "v2" / "experience" / "fake" / "campaign"
    index_path = actor_root / "index.jsonl"
    entries = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    for entry in entries:
        entry["bytes"] = 10_000_000
    index_path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
    removed = Retention(0.001).apply(actor_root)
    assert removed
    remaining = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    assert any(item["source"] == "demonstration" for item in remaining)
    assert any("help_resolved" in item["tags"] for item in remaining)
    assert sum(item["source"] == "actor" for item in remaining) <= 2


def test_append_latency_is_below_two_milliseconds_per_step(tmp_path: Path) -> None:
    writer = ExperienceWriter(tmp_path, "fake", "speed", action_table=("a", "b", "c"))
    started = time.perf_counter()
    for index in range(1000):
        writer.append(_step(index))
    elapsed = time.perf_counter() - started
    writer.flush()
    assert elapsed / 1000 < 0.002
