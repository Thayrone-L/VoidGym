"""Run an offline semantic transfer check on the corrected 86f recapture."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hk_agent.semantic_labels import unpack_u4
from hk_agent.temporal_actions import PLAYER_INTERACTION_ACTIONS, TEMPORAL_COMBAT_ACTIONS
from voidgym.core.observation import fit_to_universal
from voidgym.core.semantics import IGNORE, SEM_GRID, Sem
from voidgym.models.policy import UniversalPolicy
from voidgym.training.checkpoints import load_universal


def _iou(matrix: np.ndarray, class_index: int) -> float | None:
    union = int(matrix[class_index].sum() + matrix[:, class_index].sum() -
                matrix[class_index, class_index])
    return float(matrix[class_index, class_index] / union) if union else None


def _capture_paths(
    capture_dir: Path, capture_manifest: Path | None,
) -> tuple[list[Path], int, int]:
    if capture_manifest is None:
        paths = sorted(capture_dir.glob("*-capture.png"))
        return paths, len(paths), 0
    data = json.loads(capture_manifest.read_text(encoding="utf-8"))
    rows = data.get("captures") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise TypeError("capture manifest must contain a captures list")
    paths: list[Path] = []
    menu_frames = 0
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise TypeError("capture manifest has an invalid frame row")
        if row.get("scene_name") == "Menu_Title":
            menu_frames += 1
            continue
        path = Path(row["path"]).resolve(strict=True)
        if path.suffix.lower() != ".png" or not path.is_file():
            raise ValueError("capture manifest frame must reference an existing PNG")
        paths.append(path)
    return paths, len(rows), menu_frames


def evaluate(checkpoint: Path, capture_dir: Path, output: Path, *, device_name: str,
             skip_menu_frames: int, black_mean_below: float,
             capture_manifest: Path | None = None) -> dict[str, Any]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    action_table = tuple(item.name for item in TEMPORAL_COMBAT_ACTIONS + PLAYER_INTERACTION_ACTIONS)
    model = UniversalPolicy().to(device).eval()
    checkpoint_manifest = load_universal(checkpoint, model, action_table)
    paths, manifest_frames, menu_scene_frames = _capture_paths(capture_dir, capture_manifest)
    if not paths:
        raise FileNotFoundError(f"no capture frames under {capture_dir}")
    confusion = np.zeros((len(Sem), len(Sem)), dtype=np.int64)
    digest = hashlib.sha256()
    history: list[np.ndarray] = []
    state = model.initial_state(1, device)
    episode_start = True
    rows = 0
    skipped_menu = 0
    skipped_black = 0
    solid_empty = 0
    solid_support = 0
    for path in paths:
        frame_index = int(path.name.split("-", maxsplit=1)[0])
        if frame_index <= skip_menu_frames:
            skipped_menu += 1
            continue
        with Image.open(path) as source:
            frame = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        if float(frame.mean()) < black_mean_below:
            skipped_black += 1
            history.clear()
            state = model.initial_state(1, device)
            episode_start = True
            continue
        labels_path = path.with_name(path.stem + ".sem.u4")
        target = unpack_u4(labels_path.read_bytes(), width=48, height=27)
        if target.shape != SEM_GRID:
            raise ValueError(f"unexpected semantic grid in {labels_path}: {target.shape}")
        fitted = fit_to_universal(frame)
        history.append(fitted)
        history = history[-4:]
        stacked = [history[0]] * (4 - len(history)) + history
        pixels = np.stack(stacked).transpose(0, 3, 1, 2).reshape(1, 12, 108, 192)
        observation = {
            "pixels": torch.as_tensor(pixels, dtype=torch.uint8, device=device),
            "action_history": torch.zeros((1, 3), dtype=torch.int64, device=device),
        }
        with torch.inference_mode():
            result = model(observation, state, torch.tensor([episode_start], device=device))
        logits = result["semantic_logits"]
        next_state = result["state"]
        if not isinstance(logits, Tensor) or not isinstance(next_state, tuple):
            raise TypeError("policy output is missing semantic logits or recurrent state")
        predicted = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        valid = target != IGNORE
        np.add.at(confusion, (target[valid].astype(int), predicted[valid].astype(int)), 1)
        solid = target == int(Sem.SOLID)
        solid_support += int(solid.sum())
        solid_empty += int(np.count_nonzero(solid & (predicted == int(Sem.EMPTY))))
        digest.update(path.name.encode("utf-8"))
        digest.update(frame.tobytes())
        digest.update(target.tobytes())
        state = next_state
        episode_start = False
        rows += 1
        if rows % 100 == 0:
            print(f"HK recapture: {rows} frames", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    names = [item.name for item in Sem]
    with (output / "confusion.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/predicted", *names])
        writer.writerows([[item.name, *confusion[index].tolist()]
                          for index, item in enumerate(Sem)])
    with (output / "solid_row.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true", *names])
        writer.writerow(["SOLID", *confusion[int(Sem.SOLID)].tolist()])
    report = {
        "checkpoint": str(checkpoint),
        "trained_steps": checkpoint_manifest.get("trained_steps"),
        "capture_dir": str(capture_dir),
        "capture_manifest": str(capture_manifest) if capture_manifest is not None else None,
        "device": str(device),
        "capture_frames_available": len(paths),
        "capture_frames_in_manifest": manifest_frames,
        "menu_scene_frames_excluded": menu_scene_frames,
        "menu_frames_excluded_by_order": skipped_menu,
        "almost_black_frames_excluded": skipped_black,
        "frames_evaluated": rows,
        "action_history_available": False,
        "action_history_used": "zero-filled (the manual recapture stores no decisions)",
        "sample_sha256": digest.hexdigest(),
        "iou": {item.name: _iou(confusion, index) for index, item in enumerate(Sem)},
        "solid_support_cells": solid_support,
        "solid_predicted_empty_cells": solid_empty,
        "solid_predicted_empty_fraction": solid_empty / max(1, solid_support),
        "confusion_csv": str(output / "confusion.csv"),
        "solid_row_csv": str(output / "solid_row.csv"),
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-menu-frames", type=int, default=21)
    parser.add_argument("--black-mean-below", type=float, default=1.0)
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.capture_dir, args.output,
                      device_name=args.device, skip_menu_frames=args.skip_menu_frames,
                      black_mean_below=args.black_mean_below,
                      capture_manifest=args.capture_manifest)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
