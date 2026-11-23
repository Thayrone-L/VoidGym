"""Validated operator-only map layout and global coordinate transforms."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def load_game_map_layout(path: Path) -> dict[str, Any]:
    """Read and strictly validate the bridge's GameMap dump."""
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("GameMap layout file is missing or too large")
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported GameMap layout schema")
    game_version = value.get("game_version")
    assembly_hash = value.get("game_assembly_sha256")
    if not isinstance(game_version, str) or not game_version or not isinstance(assembly_hash, str):
        raise ValueError("GameMap layout is missing game version or assembly hash")
    if len(assembly_hash) != 64 or any(character not in "0123456789abcdef" for character in assembly_hash.casefold()):
        raise ValueError("GameMap assembly hash must be a SHA-256")
    scenes = value.get("scenes")
    if not isinstance(scenes, list) or not scenes or len(scenes) > 20_000:
        raise ValueError("GameMap layout must contain scenes")
    seen: set[str] = set()
    for row in scenes:
        if not isinstance(row, dict):
            raise TypeError("GameMap scene entry must be an object")
        name = row.get("scene")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("GameMap scene names must be non-empty and unique")
        seen.add(name)
        if not isinstance(row.get("area"), str) or not isinstance(row.get("has_sprite"), bool):
            raise TypeError(f"GameMap scene metadata is invalid: {name}")
        rect = row.get("rect")
        if row["has_sprite"]:
            if not isinstance(rect, dict):
                raise ValueError(f"GameMap scene rectangle is missing: {name}")
            numbers = [rect.get(key) for key in ("x", "y", "width", "height")]
            if not all(isinstance(number, (int, float)) and math.isfinite(number) for number in numbers):
                raise ValueError(f"GameMap scene rectangle is invalid: {name}")
            if rect["width"] <= 0 or rect["height"] <= 0:
                raise ValueError(f"GameMap scene rectangle is empty: {name}")
        elif rect is not None:
            raise ValueError(f"sprite-less scene must not have a rectangle: {name}")
    return value


def world_to_game_map(scene: str, x: float, y: float, world_bounds: Sequence[float],
                      layout: Mapping[str, Any]) -> tuple[float, float] | None:
    """Map a world point through matching room bounds into the game's map space."""
    row = next((item for item in layout.get("scenes", [])
                if isinstance(item, Mapping) and item.get("scene") == scene), None)
    if not isinstance(row, Mapping) or row.get("has_sprite") is not True:
        return None
    if len(world_bounds) != 4:
        return None
    wx1, wy1, wx2, wy2 = (float(value) for value in world_bounds)
    rect = row.get("rect")
    if not isinstance(rect, Mapping):
        return None
    rx, ry, rw, rh = (float(rect[key]) for key in ("x", "y", "width", "height"))
    if wx2 <= wx1 or wy2 <= wy1:
        return None
    return rx + (x - wx1) * rw / (wx2 - wx1), ry + (y - wy1) * rh / (wy2 - wy1)


def fit_global_calibration(anchors: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Fit one uniform scale and translation from game-map points to art pixels."""
    from hk_agent.run_journey import fit_manual_calibration

    pairs = [(float(point[0]), float(point[1]), float(point[2]), float(point[3])) for point in anchors
             if len(point) == 4]
    transform, scale, rmse = fit_manual_calibration(pairs)
    return {"scale": scale, "affine": list(transform), "rmse_px": rmse,
            "anchor_count": len(pairs), "schema_version": 1}


def game_map_to_art(calibration: Mapping[str, Any], x: float, y: float,
                    room_offset: Sequence[float] = (0.0, 0.0)) -> tuple[float, float]:
    """Apply the fitted global map transform and a persisted per-room pixel nudge."""
    transform = calibration.get("affine")
    if not isinstance(transform, Sequence) or len(transform) != 6 or len(room_offset) != 2:
        raise ValueError("global map calibration is invalid")
    a, b, c, d, e, f = (float(value) for value in transform)
    return a * x + b * y + c + float(room_offset[0]), d * x + e * y + f + float(room_offset[1])


def room_calibration_offsets(calibrations: Mapping[str, Any], automatic: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Migrate calibrated room centers into offsets from their automatic positions."""
    migrated: dict[str, list[float]] = {}
    for scene, old in calibrations.items():
        if not isinstance(old, Mapping) or old.get("positioned") is not True:
            continue
        affine = old.get("affine")
        anchor = old.get("anchor_world")
        if not isinstance(affine, Sequence) or len(affine) != 6 or not isinstance(anchor, Sequence) or len(anchor) != 2:
            continue
        expected = automatic.get(scene)
        if expected is None or len(expected) != 2:
            continue
        from hk_agent.run_journey import apply_affine
        x, y = apply_affine(affine, float(anchor[0]), float(anchor[1]))
        migrated[scene] = [x - float(expected[0]), y - float(expected[1])]
    return migrated
