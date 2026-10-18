from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hk_agent.run_journey import (
    RunDiary,
    apply_affine,
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
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + width.to_bytes(4, "big") + height.to_bytes(4, "big")


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
    assert diary.capture_firsts([{"category": "boss", "key": "hive-knight", "confirmed": True}]) == []

    counts = root / "lifelong-milestones.json"
    with ThreadPoolExecutor(max_workers=6) as pool:
        prior_counts = list(pool.map(lambda _index: increment_campaign_count(counts, "boss:Watcher_Knight"), range(12)))
    assert sorted(prior_counts) == list(range(12))


def test_no_game_map_art_is_bundled_in_runtime_code() -> None:
    repo = Path(__file__).resolve().parents[1]
    asset_roots = [repo / name for name in ("src", "bridge", "tools", "docs")]
    image_files = [path for root in asset_roots if root.exists() for path in root.rglob("*")
                   if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    assert image_files == []
    assert not any(path.name.lower() == "combined_100%.png" for path in repo.rglob("Combined_100%.png"))


def test_affine_fit_maps_points_and_rejects_degenerate() -> None:
    transform = fit_affine([(0, 0, 10, 20), (1, 0, 12, 20), (0, 1, 10, 23)])
    assert apply_affine(transform, 2, 2) == pytest.approx((14, 26))
    with pytest.raises(ValueError):
        fit_affine([(0, 0, 0, 0), (1, 1, 1, 1)])


def test_manual_two_point_calibration_maps_both_anchors() -> None:
    transform = fit_two_point_calibration((0, 0, 10, 20), (1, 0, 10, 22))
    assert apply_affine(transform, 0, 0) == pytest.approx((10, 20))
    assert apply_affine(transform, 1, 0) == pytest.approx((10, 22))


def test_scene_transform_is_versioned_by_map_sha_and_low_confidence_is_unplaced(tmp_path: Path) -> None:
    path = tmp_path / "scene-calibration.json"
    map_hash = "a" * 64
    save_scene_calibration(path, map_sha256=map_hash, scene="Crossroads_01",
                           transform=(1, 0, 10, 0, 1, 20), confidence=0.92)
    save_scene_calibration(path, map_sha256=map_hash, scene="Town",
                           transform=(1, 0, 0, 0, 1, 0), confidence=0.2)
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


def test_download_map_resolves_current_link_sets_identifiable_user_agent_and_caches(tmp_path: Path) -> None:
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
        body = (b'<a id="downloadButton" href="/download-fresh">Download</a>'
                if request.full_url.endswith("/page") else png)
        return Response(body)

    page = "http://127.0.0.1:43210/page"
    path, digest, dimensions, changed = download_mediafire_map(
        tmp_path, page_url=page, allowed_hosts=("127.0.0.1",), opener=fake_opener,
    )
    assert path.exists() and digest in str(path)
    assert dimensions == (1024, 512) and changed is False
    assert requested == [page, "http://127.0.0.1:43210/download-fresh"]
    assert all(value and "VoidLabJourney" in value for value in observed)
