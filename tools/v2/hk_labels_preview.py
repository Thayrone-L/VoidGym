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
    parser.add_argument("--source", choices=("actor", "test"), default="actor")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "relatorios" / "86_amostras" / "hk_gabarito.png")
    return parser.parse_args()


def _largest_connected_region(rows: list[tuple[np.ndarray, np.ndarray]],
                             semantic: Sem) -> tuple[int, int]:
    largest_cells = 0
    largest_frame = -1
    for frame_index, (_, labels) in enumerate(rows):
        mask = labels == int(semantic)
        visited = np.zeros(mask.shape, dtype=np.bool_)
        for y, x in np.argwhere(mask):
            if visited[y, x]:
                continue
            visited[y, x] = True
            stack = [(int(y), int(x))]
            size = 0
            while stack:
                current_y, current_x = stack.pop()
                size += 1
                for next_y, next_x in (
                    (current_y - 1, current_x), (current_y + 1, current_x),
                    (current_y, current_x - 1), (current_y, current_x + 1),
                ):
                    if (0 <= next_y < mask.shape[0] and 0 <= next_x < mask.shape[1]
                            and mask[next_y, next_x] and not visited[next_y, next_x]):
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))
            if size > largest_cells:
                largest_cells = size
                largest_frame = frame_index
    return largest_cells, largest_frame


def _is_transition_frame(frame: np.ndarray) -> bool:
    """Discard fully faded/black frames while preserving dark gameplay scenes."""
    if frame.ndim != 3 or frame.shape[-1] < 3:
        return True
    dark = np.all(frame[..., :3] <= 10, axis=-1)
    return float(dark.mean()) >= 0.98


def _overlay(frame: np.ndarray, labels: np.ndarray) -> np.ndarray:
    scaled = Image.fromarray(labels).resize((frame.shape[1], frame.shape[0]), Image.Resampling.NEAREST)
    ids = np.asarray(scaled)
    color = np.zeros_like(frame)
    valid = (ids != IGNORE) & (ids < len(PALETTE))
    color[valid] = PALETTE[ids[valid]]
    result = frame.copy()
    result[valid] = (0.45 * frame[valid] + 0.55 * color[valid]).astype(np.uint8)
    return result


def _collect(root: Path, scene: str, source: str = "actor") -> list[tuple[np.ndarray, np.ndarray]]:
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    for block in ExperienceReader(root, game_id="hollow_knight", source=source, scene=scene):
        arrays: dict[str, Any] = block["arrays"]
        frames, labels = arrays.get("frame"), arrays.get("sem")
        if frames is None or labels is None:
            continue
        for frame, sem in zip(frames, labels, strict=True):
            if sem.shape != (27, 48):
                continue
            rows.append((frame, sem))
    return rows


def render(args: argparse.Namespace) -> dict[str, Any]:
    if args.count < 1 or args.count > 120:
        raise ValueError("count must be between 1 and 120")
    captured_rows = _collect(args.experience_root, args.scene, args.source)
    rows = [row for row in captured_rows if not _is_transition_frame(row[0])]
    discarded_transition_frames = len(captured_rows) - len(rows)
    if len(rows) > args.count:
        indices = np.linspace(0, len(rows) - 1, args.count, dtype=np.int64)
        rows = [rows[int(index)] for index in indices]
    if len(rows) < args.count:
        raise RuntimeError(f"found {len(rows)}/{args.count} in-game labeled frames in scene "
                           f"{args.scene}; discarded {discarded_transition_frames} transition frames")
    class_counts = np.zeros(len(Sem), dtype=np.int64)
    ignored = 0
    for _, labels in rows:
        valid = labels != IGNORE
        ignored += int((~valid).sum())
        class_counts += np.bincount(labels[valid].astype(np.int64), minlength=len(Sem))[:len(Sem)]
    total_cells = len(rows) * 27 * 48
    fractions = {item.name: float(class_counts[index] / total_cells)
                 for index, item in enumerate(Sem)}
    fractions["IGNORE"] = float(ignored / total_cells)
    print("Fração das classes no gabarito:")
    for name, fraction in fractions.items():
        print(f"  {name}: {fraction:.1%}")
    largest_regions = {
        item: _largest_connected_region(rows, item)
        for item in (Sem.HAZARD, Sem.ENEMY, Sem.INTERACTIVE)
    }
    largest_enemy_cells, largest_enemy_frame = largest_regions[Sem.ENEMY]
    largest_enemy_fraction = largest_enemy_cells / (27 * 48)
    warnings: list[str] = []
    if fractions.get("SOLID", 0.0) > 0.70:
        warnings.append("SOLID excede 70% das células médias")
    if largest_enemy_fraction > 0.06:
        warnings.append(
            f"região ENEMY conectada ocupa {largest_enemy_fraction:.1%} da tela "
            f"no quadro {largest_enemy_frame}"
        )
    if fractions.get("ENEMY", 0.0) > 0.10:
        warnings.append(f"ENEMY total excede 10% da amostra ({fractions['ENEMY']:.1%})")
    for semantic in (Sem.HAZARD, Sem.INTERACTIVE):
        cells, frame_index = largest_regions[semantic]
        fraction = cells / (27 * 48)
        if fraction > 0.08:
            warnings.append(f"região {semantic.name} conectada ocupa {fraction:.1%} da tela "
                            f"no quadro {frame_index}; revise o colisor/gabarito")
    for warning in warnings:
        print(f"\033[91mAVISO: {warning}; confira o gabarito.\033[0m")
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
            "captured_frames_before_transition_filter": len(captured_rows),
            "discarded_transition_frames": discarded_transition_frames,
            "classes": [item.name for item in Sem], "class_fractions": fractions,
            "largest_enemy_region_cells": largest_enemy_cells,
            "largest_enemy_region_screen_fraction": largest_enemy_fraction,
            "largest_enemy_region_frame": largest_enemy_frame,
            "largest_regions": {item.name: {"cells": cells, "frame": frame,
                                             "screen_fraction": cells / (27 * 48)}
                                for item, (cells, frame) in largest_regions.items()},
            "warnings": warnings, "source": args.source, "ignore_id": IGNORE}


def main() -> int:
    import json

    print(json.dumps(render(_parse_args()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
