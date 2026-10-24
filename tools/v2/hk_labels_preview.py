"""Render captured Hollow Knight semantic labels beside their source frames."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from voidgym.core.experience_store import ExperienceReader
from voidgym.core.semantics import IGNORE, Sem

PALETTE = np.asarray([
    (0, 0, 0), (60, 100, 255), (50, 220, 230), (255, 40, 40), (230, 40, 230),
    (255, 255, 255), (255, 150, 10), (255, 235, 20), (20, 245, 120), (210, 210, 210),
], dtype=np.uint8)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experience-root", type=Path,
                        default=PROJECT_ROOT / "runtime" / "v2" / "experience")
    parser.add_argument("--scene", default="Tutorial_01")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "relatorios" / "86_amostras" / "hk_gabarito.png")
    return parser.parse_args()


def _overlay(frame: np.ndarray, labels: np.ndarray) -> np.ndarray:
    scaled = Image.fromarray(labels).resize((frame.shape[1], frame.shape[0]), Image.Resampling.NEAREST)
    ids = np.asarray(scaled)
    color = np.zeros_like(frame)
    valid = (ids != IGNORE) & (ids < len(PALETTE))
    color[valid] = PALETTE[ids[valid]]
    result = frame.copy()
    result[valid] = (0.45 * frame[valid] + 0.55 * color[valid]).astype(np.uint8)
    return result


def _collect(root: Path, scene: str, count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    for block in ExperienceReader(root, game_id="hollow_knight", source="actor", scene=scene):
        arrays: dict[str, Any] = block["arrays"]
        frames, labels = arrays.get("frame"), arrays.get("sem")
        if frames is None or labels is None:
            continue
        for frame, sem in zip(frames, labels, strict=True):
            if sem.shape != (27, 48):
                continue
            rows.append((frame, sem))
            if len(rows) >= count:
                return rows
    return rows


def render(args: argparse.Namespace) -> dict[str, Any]:
    if args.count < 1 or args.count > 120:
        raise ValueError("count must be between 1 and 120")
    rows = _collect(args.experience_root, args.scene, args.count)
    if len(rows) < args.count:
        raise RuntimeError(f"found {len(rows)}/{args.count} labeled frames in scene {args.scene}")
    tile_w, tile_h, columns = 384, 448, 6
    canvas = Image.new("RGB", (columns * tile_w, ((len(rows) + columns - 1) // columns) * tile_h),
                       (12, 12, 16))
    draw = ImageDraw.Draw(canvas)
    for index, (frame, labels) in enumerate(rows):
        x, y = (index % columns) * tile_w, (index // columns) * tile_h
        for offset, image in ((0, frame), (224, _overlay(frame, labels))):
            tile = Image.fromarray(image).resize((tile_w, 216), Image.Resampling.NEAREST)
            canvas.paste(tile, (x, y + offset))
        draw.text((x + 5, y + 2), f"{args.scene} · tela", fill="white")
        draw.text((x + 5, y + 226), "gabarito sobreposto", fill="white")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    return {"output": str(args.output), "scene": args.scene, "frames": len(rows),
            "classes": [item.name for item in Sem], "ignore_id": IGNORE}


def main() -> int:
    import json

    print(json.dumps(render(_parse_args()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
