"""Evaluate a universal checkpoint on held-out platform-gym styles."""

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
from torch.distributions import Categorical
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from voidgym.core.semantics import IGNORE, SEM_GRID, Sem
from voidgym.envs.adapter_env import AdapterEnv
from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter
from voidgym.models.policy import UniversalPolicy
from voidgym.training.checkpoints import load_universal

PALETTE = np.asarray([
    (0, 0, 0), (60, 100, 255), (50, 220, 230), (255, 40, 40), (230, 40, 230),
    (255, 255, 255), (255, 150, 10), (255, 235, 20), (20, 245, 120), (210, 210, 210),
], dtype=np.uint8)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "relatorios" / "85_eval")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--level7-sets", type=int, default=30)
    parser.add_argument("--perception-frames", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=85_000)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cpu")
    model = UniversalPolicy().to(device)
    action_table = tuple(PlatformGymAdapter(level=0, split="test").action_table())
    manifest = load_universal(args.checkpoint, model, action_table)
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "gifs").mkdir(exist_ok=True)
    confusion = np.zeros((len(Sem), len(Sem)), dtype=np.int64)
    previews: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
    perception_count = [0]
    progress_path = args.output / "progress_eval.json"
    stats: list[dict[str, Any]] = []
    completed: set[tuple[str, int]] = set()
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        stats = progress["stats"]
        completed = {(row["mode"], row["level"]) for row in stats}
        confusion = np.asarray(progress["confusion"], dtype=np.int64)
        perception_count[0] = int(progress["perception_count"])
        preview_path = args.output / "progress_previews.npz"
        if preview_path.exists():
            with np.load(preview_path) as saved:
                previews.extend((saved[f"frame_{i}"], saved[f"pred_{i}"],
                                 saved[f"target_{i}"], int(saved["levels"][i]))
                                for i in range(len(saved["levels"])))
    for stochastic in (False, True):
        mode = "stochastic" if stochastic else "deterministic"
        for level in range(8):
            if (mode, level) in completed:
                continue
            episodes = args.level7_sets if level == 7 else args.episodes
            rows = [_episode(model, level, args.seed + level * 10_000 + index * 10,
                             stochastic=stochastic, max_steps=args.max_steps,
                             confusion=confusion if perception_count[0] < args.perception_frames else None,
                             preview_target=previews, preview_limit=12,
                             perception_count=perception_count,
                             perception_limit=args.perception_frames, output_root=args.output,
                             capture=(not stochastic and level in {3, 4, 6} and index == 0))
                    for index in range(episodes)]
            stats.append({"mode": mode, "level": level, "episodes": episodes,
                          "success_rate": float(np.mean([row["success"] for row in rows])),
                          "mean_steps": float(np.mean([row["steps"] for row in rows])),
                          "deaths": int(sum(row["deaths"] for row in rows)),
                          "stuck": int(sum(row["stuck"] for row in rows))})
            _save_progress(progress_path, stats, confusion, perception_count[0], previews)
            completed.add((mode, level))
            print(f"Avaliação: {mode}, nível {level} concluído.", flush=True)
    _save_confusion(args.output / "confusion.csv", confusion)
    _save_perception(args.output / "percepcao.png", previews)
    report = _format_report(stats, confusion)
    (args.output / "tabela.md").write_text(report, encoding="utf-8", newline="\n")
    (args.output / "manifest_eval.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "trained_steps": manifest.get("trained_steps"),
        "episodes_per_level": args.episodes, "level7_sets": args.level7_sets,
        "perception_frames": int(confusion.sum()),
    }, indent=2) + "\n", encoding="utf-8")
    return {"stats": stats, "confusion": confusion, "output": args.output}


def _save_progress(path: Path, stats: list[dict[str, Any]], confusion: np.ndarray,
                   perception_count: int,
                   previews: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]) -> None:
    preview_path = path.with_name("progress_previews.npz")
    preview_data: dict[str, np.ndarray] = {"levels": np.asarray([item[3] for item in previews])}
    for index, (frame, predicted, target, _) in enumerate(previews):
        preview_data[f"frame_{index}"] = frame
        preview_data[f"pred_{index}"] = predicted
        preview_data[f"target_{index}"] = target
    np.savez_compressed(preview_path, **preview_data)  # type: ignore[arg-type]
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"stats": stats, "confusion": confusion.tolist(),
                                     "perception_count": perception_count}, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _episode(model: UniversalPolicy, level: int, seed: int, *, stochastic: bool,
             max_steps: int, confusion: np.ndarray | None,
             preview_target: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]],
             preview_limit: int, perception_count: list[int], perception_limit: int,
             output_root: Path, capture: bool) -> dict[str, Any]:
    adapter = PlatformGymAdapter(level=level, seed=seed, split="test")
    env = AdapterEnv(adapter, lambda _before, _after: 0.0)
    observation, info = env.reset(seed=seed)
    state = model.initial_state(1, torch.device("cpu"))
    starts = torch.ones(1, dtype=torch.bool)
    deaths = 0
    stuck = False
    success = False
    frames: list[Image.Image] = []
    steps = 0
    for steps in range(1, max_steps + 1):
        tensors = {key: torch.as_tensor(value[None]) for key, value in observation.items()}
        mask = torch.as_tensor(env.action_masks()[None])
        with torch.no_grad():
            policy_output = model(tensors, state, starts, mask)
            logits = policy_output["logits"]
            semantic_logits = policy_output["semantic_logits"]
            next_state = policy_output["state"]
            assert isinstance(logits, Tensor) and isinstance(semantic_logits, Tensor)
            assert isinstance(next_state, tuple)
            distribution = Categorical(logits=logits)
            action = int(distribution.sample().item()) if stochastic else int(logits.argmax(-1).item())  # type: ignore[no-untyped-call]
            state = next_state
            predicted = semantic_logits.argmax(dim=-3).squeeze(0).cpu().numpy()
        labels = info.get("training_labels", {})
        target = np.asarray(labels.get("sem", np.full(SEM_GRID, IGNORE)), dtype=np.uint8)
        if (confusion is not None and perception_count[0] < perception_limit and
                np.asarray(target).shape == SEM_GRID):
            valid = target != IGNORE
            np.add.at(confusion, (target[valid].astype(int), predicted[valid].astype(int)), 1)
            if len(preview_target) < preview_limit and perception_count[0] % 417 == 0:
                preview_target.append((adapter.frame_rgb().copy(), predicted.copy(), target.copy(), level))
            perception_count[0] += 1
        observation, _, terminated, truncated, info = env.step(action)
        starts = torch.tensor([terminated or truncated])
        next_signals = info.get("signals", {})
        deaths += sum(getattr(event, "kind", None) == "death"
                      for event in next_signals.get("events", ()))
        stuck |= bool(next_signals.get("stuck", False))
        if level == 7:
            success = terminated
        else:
            success |= bool(next_signals.get("room_changed", False))
        if capture and steps % 5 == 0:
            frames.append(Image.fromarray(adapter.frame_rgb()).resize((384, 216), Image.Resampling.NEAREST))
        if success or terminated or truncated:
            break
    env.close()
    if capture and frames:
        path = output_root / "gifs" / f"nivel_{level}.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=60, loop=0)
    return {"success": success, "steps": steps, "deaths": deaths, "stuck": stuck}


def _save_confusion(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/predicted", *(item.name for item in Sem)])
        writer.writerows([[item.name, *row.tolist()] for item, row in zip(Sem, matrix, strict=True)])


def _save_perception(path: Path, previews: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]) -> None:
    tile_w, tile_h = 384, 270
    canvas = Image.new("RGB", (tile_w * 3, tile_h * max(1, len(previews))), (12, 12, 16))
    draw = ImageDraw.Draw(canvas)
    for index, (frame, predicted, target, level) in enumerate(previews):
        row_y = index * tile_h
        for column, image in enumerate((frame, PALETTE[predicted], PALETTE[target])):
            tile = Image.fromarray(image).resize((tile_w, 216), Image.Resampling.NEAREST)
            canvas.paste(tile, (column * tile_w, row_y + 24))
        draw.text((8, row_y + 5), f"nível {level}: tela / previsão / gabarito", fill="white")
    canvas.save(path)


def _format_report(stats: list[dict[str, Any]], confusion: np.ndarray) -> str:
    lines = ["# Avaliação do pré-treino v2", "", "| Modo | Nível | Episódios | Sucesso | Passos médios | Mortes | Travadas |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in stats:
        lines.append(f"| {row['mode']} | {row['level']} | {row['episodes']} | "
                     f"{row['success_rate']:.1%} | {row['mean_steps']:.1f} | "
                     f"{row['deaths']} | {row['stuck']} |")
    total = confusion.sum(axis=1)
    lines.extend(["", "## IoU e matriz de confusão", "", "| Classe | IoU | Suporte |", "|---|---:|---:|"])
    for index, item in enumerate(Sem):
        union = total[index] + confusion[:, index].sum() - confusion[index, index]
        iou = confusion[index, index] / union if union else float("nan")
        lines.append(f"| {item.name} | {iou:.3f} | {total[index]} |")
    lines.extend(["", "Matriz completa: `confusion.csv`.", "Prévia: `percepcao.png`.",
                  "GIFs: `gifs/nivel_3.gif`, `gifs/nivel_4.gif`, `gifs/nivel_6.gif`.", ""])
    return "\n".join(lines)


def main() -> int:
    args = _args()
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
