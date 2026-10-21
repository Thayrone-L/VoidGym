from __future__ import annotations

import json
from pathlib import Path

import pytest

from hk_agent.map_layout import (
    fit_global_calibration,
    game_map_to_art,
    load_game_map_layout,
    room_calibration_offsets,
    world_to_game_map,
)


def test_game_map_layout_validation_and_world_transform(tmp_path: Path) -> None:
    path = tmp_path / "layout.json"
    path.write_text(json.dumps({"schema_version": 1, "game_version": "1.5.78.11833", "game_assembly_sha256": "a" * 64, "scenes": [
        {"scene": "Tutorial_01", "area": "King's Pass", "has_sprite": True,
         "rect": {"x": 10, "y": 20, "width": 30, "height": 15}},
        {"scene": "Town", "area": "Dirtmouth", "has_sprite": False, "rect": None},
    ]}), encoding="utf-8")
    layout = load_game_map_layout(path)
    assert world_to_game_map("Tutorial_01", 5, 2.5, [0, 0, 10, 5], layout) == (25, 27.5)
    assert world_to_game_map("Town", 1, 1, [0, 0, 10, 10], layout) is None
    path.write_text(json.dumps({"schema_version": 1, "game_version": "1.5", "game_assembly_sha256": "bad", "scenes": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_game_map_layout(path)
    assert world_to_game_map("Tutorial_01", 1, 1, [0, 0, 0, 1], layout) is None


def test_global_similarity_calibration_and_room_nudge() -> None:
    result = fit_global_calibration([(0, 0, 100, 400), (100, 0, 300, 400), (0, 100, 100, 200)])
    assert result["scale"] == pytest.approx(2)
    assert result["rmse_px"] == pytest.approx(0)
    assert game_map_to_art(result, 50, 50, (3, -4)) == pytest.approx((203, 296))
    with pytest.raises(ValueError):
        fit_global_calibration([(0, 0, 1, 1)])


def test_legacy_room_calibration_migration_keeps_pixel_delta() -> None:
    offsets = room_calibration_offsets(
        {"Tutorial_01": {"positioned": True, "affine": [2, 0, 5, 0, -2, 9], "anchor_world": [10, 5]}},
        {"Tutorial_01": [24, -1]},
    )
    assert offsets["Tutorial_01"] == [1, 0]


def test_control_plane_saves_global_calibration_and_room_offsets(tmp_path: Path, monkeypatch) -> None:
    from hk_agent.control_plane import ControlPlane

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    plane = ControlPlane(tmp_path, observer=True)
    monkeypatch.setattr(plane, "map_art_status", lambda: {"sha256": "a" * 64, "tiles_ready": True})
    calibration = plane.save_global_map_calibration({"anchors": [
        {"map_x": 0, "map_y": 0, "art_x": 50, "art_y": 400},
        {"map_x": 100, "map_y": 0, "art_x": 250, "art_y": 400},
        {"map_x": 0, "map_y": 100, "art_x": 50, "art_y": 200},
    ]})
    assert calibration["rmse_px"] == pytest.approx(0)
    saved = plane.save_map_room_offset({"scene": "Tutorial_01", "dx": 3, "dy": -2})
    assert saved["offset"] == [3.0, -2.0]
    root = tmp_path / "local" / "VoidLab" / "cache" / "map-art" / ("a" * 64)
    assert json.loads((root / "global-calibration.json").read_text("utf-8"))["anchor_count"] == 3
    assert json.loads((root / "room-offsets.json").read_text("utf-8"))["offsets"]["Tutorial_01"] == [3.0, -2.0]
