from __future__ import annotations

import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hk_agent.map_art import template_match_ncc
from hk_agent.run_journey import (
    RunDiary,
    apply_affine,
    atomic_json,
    cache_map_file,
    download_mediafire_map,
    fit_affine,
    fit_two_point_calibration,
    load_scene_calibration,
    rebuild_progress,
    save_scene_calibration,
    schematic_svg,
    validate_png_header,
)


def tiny_png(width: int = 1, height: int = 1) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )


def test_diary_samples_at_most_two_hz_and_aggregates(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    diary = RunDiary(root / "actor-1" / "episode-1", root, "actor-1", 1)
    diary.observe({"scene_name": "Crossroads_01", "x": 1, "y": 2}, monotonic_seconds=0)
    diary.observe({"scene_name": "Crossroads_01", "x": 2, "y": 3}, monotonic_seconds=0.2)
    diary.observe({"scene_name": "Town", "x": 3, "y": 4}, monotonic_seconds=0.5)
    diary.data["events"].append({"category": "grub", "key": "grub_1", "confirmed": True})
    diary.finish("time_limit", deaths=[{"scene": "Town"}])
    index = rebuild_progress(root)
    assert len(diary.data["positions"]) == 2
    assert index["scenes"].keys() == {"Crossroads_01", "Town"}
    assert index["records"]["max_scenes"] == 2
    assert schematic_svg(index, route=["Crossroads_01", "Town"]).count("<circle") == 2


def test_diary_uses_camera_when_hero_is_outside_scene_bounds_and_records_geo_event(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    diary = RunDiary(root / "actor-1" / "episode-1", root, "actor-1", 1)
    diary.observe(
        {
            "scene_name": "Tutorial_01",
            "x": -32.9,
            "y": -2000,
            "camera_x": 14.6,
            "camera_y": 8.5,
            "scene_bounds": [0, 0, 30, 20],
        },
        monotonic_seconds=0.0,
    )
    diary.observe(
        {
            "scene_name": "Tutorial_01",
            "x": -32.9,
            "y": -2000,
            "camera_x": 14.7,
            "camera_y": 8.5,
            "scene_bounds": [0, 0, 30, 20],
            "geo_gained_since_last_frame": 3,
        },
        monotonic_seconds=0.5,
    )
    assert diary.data["positions"][0]["position_source"] == "camera"
    assert diary.data["positions"][0]["x"] == 14.6
    assert diary.data["positions"][0]["hero_y"] == -2000
    assert diary.data["events"][0]["category"] == "geo"
    assert diary.data["events"][0]["confirmed"] is True


def test_atomic_json_retries_transient_windows_replace_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hk_agent.run_journey as journey

    real_replace = os.replace
    attempts = 0

    def replace_once_locked(source: str, target: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("temporary read lock")
        real_replace(source, target)

    monkeypatch.setattr(journey.os, "replace", replace_once_locked)
    destination = tmp_path / "journal.json"
    atomic_json(destination, {"ok": True})
    assert attempts == 2
    assert json.loads(destination.read_text("utf-8")) == {"ok": True}


def test_first_campaign_milestone_png_limits_and_dedup(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    diary = RunDiary(root / "a" / "episode-1", root, "a", 1, capture_png=tiny_png)
    event = {"category": "grub", "key": "grub-7", "confirmed": True}
    assert len(diary.capture_firsts([event])) == 1
    assert diary.capture_firsts([event]) == []
    assert len(list((root / "marcos").glob("*.png"))) == 1


def test_campaign_capture_count_is_capped_and_event_counts_are_atomic(tmp_path: Path) -> None:
    from hk_agent.run_journey import MAX_CAPTURE_COUNT, increment_campaign_count

    root = tmp_path / "campaign"
    captures = root / "marcos"
    captures.mkdir(parents=True)
    for index in range(MAX_CAPTURE_COUNT):
        (captures / f"{index}.png").write_bytes(tiny_png())
    diary = RunDiary(root / "a" / "episode-2", root, "a", 2, capture_png=tiny_png)
    assert (
        diary.capture_firsts([{"category": "boss", "key": "hive-knight", "confirmed": True}]) == []
    )

    counts = root / "lifelong-milestones.json"
    with ThreadPoolExecutor(max_workers=6) as pool:
        prior_counts = list(
            pool.map(
                lambda _index: increment_campaign_count(counts, "boss:Watcher_Knight"), range(12)
            )
        )
    assert sorted(prior_counts) == list(range(12))


def test_no_game_map_art_is_bundled_in_runtime_code() -> None:
    repo = Path(__file__).resolve().parents[1]
    asset_roots = [repo / name for name in ("src", "bridge", "tools", "docs")]
    image_files = [
        path
        for root in asset_roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    assert image_files == []
    assert not any(
        path.name.lower() == "combined_100%.png" for path in repo.rglob("Combined_100%.png")
    )


def test_affine_fit_maps_points_and_rejects_degenerate() -> None:
    transform = fit_affine([(0, 0, 10, 20), (1, 0, 12, 20), (0, 1, 10, 23)])
    assert apply_affine(transform, 2, 2) == pytest.approx((14, 26))
    with pytest.raises(ValueError):
        fit_affine([(0, 0, 0, 0), (1, 1, 1, 1)])


def test_manual_two_point_calibration_maps_both_anchors() -> None:
    transform = fit_two_point_calibration((0, 0, 10, 20), (1, 0, 10, 22))
    assert apply_affine(transform, 0, 0) == pytest.approx((10, 20))
    assert apply_affine(transform, 1, 0) == pytest.approx((10, 22))


def test_scene_transform_is_versioned_by_map_sha_and_low_confidence_is_unplaced(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scene-calibration.json"
    map_hash = "a" * 64
    save_scene_calibration(
        path,
        map_sha256=map_hash,
        scene="Crossroads_01",
        transform=(1, 0, 10, 0, 1, 20),
        confidence=0.92,
    )
    save_scene_calibration(
        path, map_sha256=map_hash, scene="Town", transform=(1, 0, 0, 0, 1, 0), confidence=0.2
    )
    calibration = load_scene_calibration(path, map_sha256=map_hash)
    assert calibration["Crossroads_01"]["positioned"] is True
    assert calibration["Town"]["positioned"] is False
    assert load_scene_calibration(path, map_sha256="b" * 64) == {}


def test_png_validation_and_content_addressed_cache(tmp_path: Path) -> None:
    source = tmp_path / "art.png"
    source.write_bytes(tiny_png(640, 480))
    with source.open("rb") as stream:
        assert validate_png_header(stream) == (640, 480)
    cached, digest, dimensions = cache_map_file(source, tmp_path / "runtime")
    assert cached.read_bytes() == source.read_bytes()
    assert digest in str(cached) and dimensions == (640, 480)
    assert json.loads(json.dumps({"ok": True}))
    source.write_bytes(b"not png")
    with pytest.raises(ValueError):
        cache_map_file(source, tmp_path / "runtime")


def test_download_map_resolves_current_link_sets_identifiable_user_agent_and_caches(
    tmp_path: Path,
) -> None:
    observed: list[str | None] = []
    requested: list[str] = []
    png = tiny_png(1024, 512)

    class Response(io.BytesIO):
        def __init__(self, content: bytes) -> None:
            super().__init__(content)
            self.headers = {"Content-Length": str(len(png))}

    def fake_opener(request, *, timeout):
        del timeout
        requested.append(request.full_url)
        observed.append(request.get_header("User-agent"))
        body = (
            b'<a id="downloadButton" href="/download-fresh">Download</a>'
            if request.full_url.endswith("/page")
            else png
        )
        return Response(body)

    page = "http://127.0.0.1:43210/page"
    path, digest, dimensions, changed = download_mediafire_map(
        tmp_path,
        page_url=page,
        allowed_hosts=("127.0.0.1",),
        opener=fake_opener,
    )
    assert path.exists() and digest in str(path)
    assert dimensions == (1024, 512) and changed is False
    assert requested == [page, "http://127.0.0.1:43210/download-fresh"]
    assert all(value and "VoidLabJourney" in value for value in observed)


def test_numpy_fft_template_match_finds_known_scaled_offset() -> None:
    import numpy as np

    rng = np.random.default_rng(7)
    template = rng.normal(size=(18, 22)).astype(np.float32)
    ys = (np.arange(27) * 18 / 27).astype(int)
    xs = (np.arange(33) * 22 / 33).astype(int)
    scaled = template[ys[:, None], xs[None, :]]
    image = rng.normal(size=(100, 150)).astype(np.float32)
    image[31:58, 54:87] = scaled
    result = template_match_ncc(image, template, scales=(0.75, 1.0, 1.5))
    assert result is not None
    assert (result["x"], result["y"]) == (54.0, 31.0)
    assert (result["width"], result["height"]) == (33.0, 27.0)
    assert result["confidence"] > 0.99


def test_deepzoom_tiles_are_streamed_and_restartable(tmp_path: Path) -> None:
    pyvips = pytest.importorskip("pyvips")
    from hk_agent.map_art import generate_map_tiles, read_map_tile

    source = tmp_path / "synthetic-large.png"
    cache = tmp_path / "cache" / ("a" * 64)
    pyvips.Image.black(8192, 4096, bands=3).write_to_file(str(source))
    updates: list[float] = []
    metadata = generate_map_tiles(source, cache, progress=updates.append)
    assert metadata is not None and metadata["complete"] is True
    assert (cache / "thumbnail.jpg").is_file()
    assert (cache / "tiles" / "map.dzi").is_file()
    assert updates and updates[-1] == 100.0
    assert read_map_tile(cache, metadata["max_level"], 0, 0).startswith(b"\xff\xd8")
    (cache / "tiles.json").unlink()
    restarted = generate_map_tiles(source, cache)
    assert restarted is not None and restarted["complete"] is True
    assert (cache / "tiles" / "map_files" / str(metadata["max_level"]) / "0_0.jpg").is_file()
