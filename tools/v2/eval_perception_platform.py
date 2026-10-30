"""Measure semantic perception on a level-balanced, time-spread sample.

Actions are generated independently from the checkpoint, so identical seed and
split arguments replay the same states for every checkpoint being compared.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from voidgym.core.semantics import IGNORE, SEM_GRID, Sem
from voidgym.envs.adapter_env import AdapterEnv
from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter
from voidgym.gyms.platform_gym.params import Split
from voidgym.models.policy import UniversalPolicy
from voidgym.training.checkpoints import load_universal

PALETTE = np.asarray([
    (0, 0, 0), (60, 100, 255), (50, 220, 230), (255, 40, 40), (230, 40, 230),
    (255, 255, 255), (255, 150, 10), (255, 235, 20), (20, 245, 120), (210, 210, 210),
], dtype=np.uint8)
ROUTE = ("RIGHT", "RIGHT", "RIGHT", "RIGHT+JUMP_HOLD", "RIGHT", "RIGHT+JUMP_TAP",
         "RIGHT", "RIGHT+ATTACK", "RIGHT", "RIGHT+DASH", "RIGHT")
REQUIRED_SUPPORT = {5: (Sem.HAZARD, Sem.ENEMY), 6: (Sem.HAZARD, Sem.ENEMY)}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--compare-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=5000,
                        help="total sample size, divided evenly over levels 0–7")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--seed", type=int, default=185_000)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--episodes-per-level", type=int, default=100)
    parser.add_argument("--max-episodes-per-level", type=int, default=500)
    parser.add_argument("--stride", type=int, default=8,
                        help="sample candidate observations at this step interval")
    return parser.parse_args()


def _frame_quotas(total: int, levels: int = 8) -> dict[int, int]:
    if total < levels or levels < 1:
        raise ValueError("frames must provide at least one frame per level")
    base, remainder = divmod(total, levels)
    return {level: base + int(level < remainder) for level in range(levels)}


def _episode_quotas(total: int, episodes: int) -> list[int]:
    if total < 0 or episodes < 1:
        raise ValueError("invalid frame or episode quota")
    base, remainder = divmod(total, episodes)
    return [base + int(index < remainder) for index in range(episodes)]


def _iou(confusion: np.ndarray, index: int) -> float | None:
    union = int(confusion[index].sum() + confusion[:, index].sum() - confusion[index, index])
    return float(confusion[index, index] / union) if union else None


def _valid_target(info: dict[str, Any]) -> np.ndarray | None:
    labels = info.get("training_labels", {})
    target = np.asarray(labels.get("sem", np.full(SEM_GRID, IGNORE)), dtype=np.uint8)
    return target if target.shape == SEM_GRID and np.any(target != IGNORE) else None


def _new_sample(observation: dict[str, np.ndarray], mask: np.ndarray,
                target: np.ndarray, frame: np.ndarray, episode: int, step: int,
                seed: int) -> dict[str, Any]:
    return {"observation": {key: np.asarray(value).copy()
                            for key, value in observation.items()},
            "mask": np.asarray(mask).copy(), "target": target.copy(),
            "frame": frame.copy(), "episode": episode, "step": step, "seed": seed}


def _choose_action(action_table: Sequence[str], mask: np.ndarray, step: int,
                   episode: int) -> int:
    desired = ROUTE[(step + episode) % len(ROUTE)]
    action_index = action_table.index(desired)
    if bool(mask[action_index]):
        return action_index
    right = action_table.index("RIGHT")
    if bool(mask[right]):
        return right
    valid = np.flatnonzero(mask)
    if not len(valid):
        raise RuntimeError("platform adapter returned an empty action mask")
    return int(valid[(step + episode) % len(valid)])


def _collect_level(level: int, split: Split, args: argparse.Namespace,
                   needed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_quotas = _episode_quotas(needed, args.episodes_per_level)
    samples: list[dict[str, Any]] = []
    support_candidates: dict[Sem, dict[str, Any]] = {}
    candidates_seen = 0
    episode = 0
    while episode < args.max_episodes_per_level:
        if len(samples) >= needed and all(
            item in support_candidates for item in REQUIRED_SUPPORT.get(level, ())
        ):
            break
        remaining = max(0, needed - len(samples))
        quota = (base_quotas[episode] if episode < len(base_quotas)
                 else min(max(base_quotas, default=1), remaining))
        episode_seed = args.seed + level * 10_000 + episode * 10
        adapter = PlatformGymAdapter(level=level, seed=episode_seed, split=split)
        env = AdapterEnv(adapter, lambda _before, _after: 0.0)
        observation, info = env.reset(seed=episode_seed)
        rng = np.random.default_rng(episode_seed ^ (level << 24) ^ 0x86F)
        offset = int(rng.integers(0, args.stride))
        reservoir: list[dict[str, Any]] = []
        seen = 0
        for step in range(args.max_steps):
            mask = np.asarray(env.action_masks(), dtype=bool)
            target = _valid_target(info)
            if target is not None and step % args.stride == offset:
                candidates_seen += 1
                sample = _new_sample(observation, mask, target, adapter.frame_rgb(),
                                     episode, step, episode_seed)
                for semantic in REQUIRED_SUPPORT.get(level, ()):
                    if semantic not in support_candidates and np.any(target == int(semantic)):
                        support_candidates[semantic] = sample
                seen += 1
                if quota > 0 and len(reservoir) < quota:
                    reservoir.append(sample)
                elif quota > 0:
                    replace_at = int(rng.integers(0, seen))
                    if replace_at < quota:
                        reservoir[replace_at] = sample
            action = _choose_action(adapter.action_table(), mask, step, episode)
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()
        if quota > 0:
            samples.extend(reservoir[:needed - len(samples)])
        episode += 1
        if episode % 25 == 0:
            print(f"Amostra {split} L{level}: episódios={episode}, quadros={len(samples)}/{needed}, "
                  f"candidatos={candidates_seen}.", flush=True)

    if len(samples) != needed:
        raise RuntimeError(f"L{level}/{split}: coletados {len(samples)} de {needed} quadros "
                           f"em {episode} episódios")
    for semantic in REQUIRED_SUPPORT.get(level, ()):
        if any(np.any(item["target"] == int(semantic)) for item in samples):
            continue
        candidate = support_candidates.get(semantic)
        if candidate is None:
            raise RuntimeError(f"L{level}/{split}: não foi observado suporte de {semantic.name} "
                               f"em {episode} episódios")
        replacement = next((index for index in range(len(samples) - 1, -1, -1)
                            if not any(np.any(samples[index]["target"] == int(required))
                                        for required in REQUIRED_SUPPORT.get(level, ()))), None)
        if replacement is None:
            replacement = len(samples) - 1
        samples[replacement] = candidate

    metadata = {"level": level, "split": split, "requested_frames": needed,
                "sampled_frames": len(samples), "episodes": episode,
                "episodes_in_sample": len({item["episode"] for item in samples}),
                "candidate_frames": candidates_seen,
                "support_candidates": {item.name: item in support_candidates
                                       for item in REQUIRED_SUPPORT.get(level, ())}}
    return samples, metadata


def _infer_sample(model: UniversalPolicy, sample: dict[str, Any],
                  device: torch.device) -> np.ndarray:
    tensors = {key: torch.as_tensor(value[None], device=device)
               for key, value in sample["observation"].items()}
    mask = torch.as_tensor(sample["mask"][None], device=device)
    state = model.initial_state(1, device)
    starts = torch.ones(1, dtype=torch.bool, device=device)
    with torch.no_grad():
        output = model(tensors, state, starts, mask)
        logits = output["semantic_logits"]
        assert isinstance(logits, Tensor)
        return logits.argmax(dim=-3).squeeze(0).cpu().numpy().astype(np.uint8)


def _accumulate(confusion: np.ndarray, target: np.ndarray, predicted: np.ndarray) -> None:
    valid = target != IGNORE
    np.add.at(confusion, (target[valid].astype(int), predicted[valid].astype(int)), 1)


def _write_confusion(path: Path, confusion: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/predicted", *(item.name for item in Sem)])
        writer.writerows([[item.name, *row.tolist()]
                          for item, row in zip(Sem, confusion, strict=True)])


def _write_by_level(path: Path, confusion: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["level", "true/predicted", *(item.name for item in Sem)])
        for level, matrix in enumerate(confusion):
            writer.writerows([[level, item.name, *row.tolist()]
                              for item, row in zip(Sem, matrix, strict=True)])


def _write_previews(path: Path,
                    previews: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]) -> None:
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


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.frames < 8 or args.stride < 1 or args.episodes_per_level < 1:
        raise ValueError("frames, stride and episodes-per-level must be positive")
    torch.set_num_threads(3)
    device = torch.device("cpu")
    action_table = tuple(PlatformGymAdapter(level=0, split=args.split).action_table())
    models: list[tuple[str, Path, UniversalPolicy, dict[str, Any]]] = []
    for checkpoint in [args.checkpoint, *args.compare_checkpoint]:
        model = UniversalPolicy().to(device).eval()
        manifest = load_universal(checkpoint, model, action_table)
        label = checkpoint.name.removeprefix("universal-")
        models.append((label, checkpoint, model, manifest))
    args.output.mkdir(parents=True, exist_ok=True)
    quotas = _frame_quotas(args.frames)
    per_model_confusion = {
        label: np.zeros((8, len(Sem), len(Sem)), dtype=np.int64)
        for label, _, _, _ in models
    }
    previews: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
    level_metadata: list[dict[str, Any]] = []
    sample_hashes: dict[str, str] = {}

    for level, quota in quotas.items():
        samples, metadata = _collect_level(level, args.split, args, quota)
        digest = hashlib.sha256()
        for index, sample in enumerate(samples):
            digest.update(f"{level}:{sample['seed']}:{sample['step']};".encode("ascii"))
            digest.update(sample["target"].tobytes())
            digest.update(sample["observation"]["pixels"].tobytes())
            for model_index, (label, _, model, _) in enumerate(models):
                predicted = _infer_sample(model, sample, device)
                _accumulate(per_model_confusion[label][level], sample["target"], predicted)
                if model_index == 0 and index in {0, len(samples) - 1}:
                    previews.append((sample["frame"], predicted, sample["target"], level))
        sample_hashes[str(level)] = digest.hexdigest()
        level_metadata.append(metadata)
        print(f"Percepção {args.split} L{level}: {quota} quadros em "
              f"{metadata['episodes_in_sample']} episódios.", flush=True)

    results: dict[str, dict[str, Any]] = {}
    for label, checkpoint, _, checkpoint_manifest in models:
        per_level_confusion = per_model_confusion[label]
        confusion = per_level_confusion.sum(axis=0)
        output_dir = args.output / label if len(models) > 1 else args.output
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_confusion(output_dir / "confusion.csv", confusion)
        _write_by_level(output_dir / "confusion_by_level.csv", per_level_confusion)
        if label == models[0][0]:
            _write_previews(output_dir / "percepcao.png", previews)
        class_support = {
            str(level): {item.name: int(per_level_confusion[level, int(item)].sum())
                         for item in Sem}
            for level in range(8)
        }
        report = {
            "checkpoint": str(checkpoint),
            "trained_steps": checkpoint_manifest.get("trained_steps"),
            "requested_frames": args.frames,
            "sampled_frames": int(sum(meta["sampled_frames"] for meta in level_metadata)),
            "frames_by_level": {str(meta["level"]): meta["sampled_frames"]
                                for meta in level_metadata},
            "split": args.split,
            "sample_policy": {"action_route": list(ROUTE), "stride": args.stride,
                              "seed": args.seed, "max_steps": args.max_steps,
                              "episodes_per_level": args.episodes_per_level,
                              "stateless_per_frame": True},
            "sample_id_sha256_by_level": sample_hashes,
            "class_support_by_level": class_support,
            "iou": {item.name: _iou(confusion, index) for index, item in enumerate(Sem)},
            "iou_by_level": {str(level): {item.name: _iou(per_level_confusion[level], index)
                                          for index, item in enumerate(Sem)}
                             for level in range(8)},
            "levels": level_metadata,
        }
        (output_dir / "perception_manifest.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        results[label] = report
    return {"split": args.split, "sampled_frames": int(sum(
                meta["sampled_frames"] for meta in level_metadata)),
            "frames_by_level": {str(meta["level"]): meta["sampled_frames"]
                                for meta in level_metadata},
            "sample_id_sha256_by_level": sample_hashes,
            "checkpoints": results}


def main() -> int:
    result = evaluate(_args())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
