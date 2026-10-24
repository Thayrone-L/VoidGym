"""Evaluate the frozen universal semantic head on stored Hollow Knight frames."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hk_agent.temporal_actions import PLAYER_INTERACTION_ACTIONS, TEMPORAL_COMBAT_ACTIONS
from voidgym.core.experience_store import ExperienceReader
from voidgym.core.semantics import IGNORE, SEM_GRID, Sem
from voidgym.models.policy import UniversalPolicy
from voidgym.training.checkpoints import load_universal

PALETTE = np.asarray([
    (0, 0, 0), (60, 100, 255), (50, 220, 230), (255, 40, 40), (230, 40, 230),
    (255, 255, 255), (255, 150, 10), (255, 235, 20), (20, 245, 120), (210, 210, 210),
], dtype=np.uint8)
HK_ACTION_TABLE = tuple(item.name for item in TEMPORAL_COMBAT_ACTIONS + PLAYER_INTERACTION_ACTIONS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experience-root", type=Path,
                        default=PROJECT_ROOT / "runtime" / "v2" / "experience")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "relatorios" / "86_amostras")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--scenes", default="Tutorial_01,Room_Town_Street",
                        help="cenas candidatas (Kings Pass e, se houver, Dirtmouth)")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.count < 1:
        raise ValueError("count must be positive")
    wanted_scenes = tuple(item.strip() for item in args.scenes.split(",") if item.strip())
    if not wanted_scenes:
        raise ValueError("at least one scene is required")

    policy = UniversalPolicy().cpu().eval()
    manifest = load_universal(args.checkpoint, policy, HK_ACTION_TABLE)
    confusion = np.zeros((len(Sem), len(Sem)), dtype=np.int64)
    previews: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    scene_counts = {scene: 0 for scene in wanted_scenes}
    history_available = True
    processed = 0

    reader = ExperienceReader(args.experience_root, game_id="hollow_knight")
    with torch.inference_mode():
        for block in reader.iter_stacked():
            arrays = block["arrays"]
            frames = arrays.get("frames_stacked")
            labels = arrays.get("sem")
            if frames is None or labels is None:
                continue
            history = arrays.get("action_history")
            if history is None:
                history_available = False
                history = np.zeros((len(frames), 3), dtype=np.int64)
            scenes = _expand_scene_runs(block.get("meta", {}).get("scene_runs", []), len(frames))
            recurrent_state = policy.initial_state(1, torch.device("cpu"))
            previous_scene: str | None = None
            for index in range(len(frames)):
                scene = scenes[index]
                selected = next((item for item in wanted_scenes
                                 if item.casefold() in scene.casefold()), None)
                if selected is None:
                    previous_scene = None
                    recurrent_state = policy.initial_state(1, torch.device("cpu"))
                    continue
                if processed >= args.count:
                    break
                scene_counts[selected] += 1
                if bool(arrays.get("discontinuity", np.zeros(len(frames), dtype=bool))[index]) or scene != previous_scene:
                    recurrent_state = policy.initial_state(1, torch.device("cpu"))
                stacked = frames[index]
                image = stacked.transpose(0, 3, 1, 2).reshape(1, 12, 108, 192)
                observation = {
                    "pixels": torch.as_tensor(image, dtype=torch.uint8),
                    "action_history": torch.as_tensor(history[index:index + 1], dtype=torch.int64),
                }
                output = policy(observation, recurrent_state,
                                torch.tensor([previous_scene != scene]))
                semantic_logits = output["semantic_logits"]
                next_state = output["state"]
                assert isinstance(semantic_logits, Tensor) and isinstance(next_state, tuple)
                recurrent_state = next_state
                predicted = semantic_logits.argmax(dim=1).squeeze(0).cpu().numpy()
                target = np.asarray(labels[index], dtype=np.uint8)
                if target.shape != SEM_GRID:
                    raise ValueError(f"semantic label shape mismatch: {target.shape}")
                valid = target != IGNORE
                np.add.at(confusion, (target[valid].astype(int), predicted[valid].astype(int)), 1)
                if len(previews) < 12:
                    previews.append((stacked[-1].copy(), predicted.copy(), target.copy(), scene))
                processed += 1
                previous_scene = scene
            if processed >= args.count:
                break

    if processed != args.count:
        available = ", ".join(f"{scene}={count}" for scene, count in scene_counts.items())
        raise RuntimeError(f"found {processed}/{args.count} labeled frames in requested scenes ({available})")

    args.output.mkdir(parents=True, exist_ok=True)
    _write_confusion(args.output / "zero_shot_confusion.csv", confusion)
    _write_previews(args.output / "zero_shot.png", previews)
    iou = {item.name: _iou(confusion, index) for index, item in enumerate(Sem)}
    result = {
        "checkpoint": str(args.checkpoint),
        "trained_steps": manifest.get("trained_steps"),
        "sampled_frames": processed,
        "requested_scenes": wanted_scenes,
        "frames_per_scene": scene_counts,
        "action_history_available": history_available,
        "iou": iou,
        "confusion": "zero_shot_confusion.csv",
        "preview": "zero_shot.png",
    }
    (args.output / "zero_shot_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _expand_scene_runs(raw: object, size: int) -> list[str]:
    scenes = [""] * size
    if not isinstance(raw, list):
        return scenes
    for row in raw:
        if (not isinstance(row, list) or len(row) != 3 or not isinstance(row[0], str)
                or not all(isinstance(value, int) and not isinstance(value, bool) for value in row[1:])):
            continue
        name, start, count = row
        if 0 <= start <= size and 0 <= count <= size - start:
            scenes[start:start + count] = [name] * count
    return scenes


def _iou(confusion: np.ndarray, index: int) -> float | None:
    union = int(confusion[index].sum() + confusion[:, index].sum() - confusion[index, index])
    return float(confusion[index, index] / union) if union else None


def _write_confusion(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/predicted", *(item.name for item in Sem)])
        writer.writerows([[item.name, *row.tolist()] for item, row in zip(Sem, matrix, strict=True)])


def _write_previews(path: Path,
                    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]) -> None:
    tile_w, tile_h = 384, 270
    canvas = Image.new("RGB", (tile_w * 3, tile_h * max(1, len(rows))), (12, 12, 16))
    draw = ImageDraw.Draw(canvas)
    for index, (frame, predicted, target, scene) in enumerate(rows):
        row_y = index * tile_h
        for column, image in enumerate((frame, PALETTE[predicted], PALETTE[target])):
            tile = Image.fromarray(image).resize((tile_w, 216), Image.Resampling.NEAREST)
            canvas.paste(tile, (column * tile_w, row_y + 24))
        draw.text((8, row_y + 5), f"{scene}: tela / previsão / gabarito", fill="white")
    canvas.save(path)


def main() -> int:
    print(json.dumps(evaluate(_parse_args()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
