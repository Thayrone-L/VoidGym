"""Avaliação em famílias de estilo holdout e métricas semânticas."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical
from PIL import Image

from voidgym.core.actions import UNIVERSAL_ACTIONS
from voidgym.envs.adapter_env import AdapterEnv
from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter
from voidgym.gyms.platform_gym.reward import GymReward
from voidgym.models.policy import UniversalPolicy
from voidgym.training.checkpoints import load_universal


def _lower_priority() -> None:
    if os.name == "nt":
        from ctypes import wintypes

        kernel = ctypes.windll.kernel32
        kernel.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.SetPriorityClass.restype = wintypes.BOOL
        kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00004000)


def _episode(model: UniversalPolicy, level: int, seed: int, deterministic: bool,
             device: str, capture: bool = False) -> tuple[bool, int, int, bool, list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    env = AdapterEnv(PlatformGymAdapter(split="test"), GymReward())
    obs, _ = env.reset(seed=seed, options={"level": level})
    state = None
    deaths = 0
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    initial_frame = env.render().copy()
    success = False
    stuck = False
    stagnant_steps = 0
    previous_position: tuple[float, float] | None = None
    previous_room: str | None = None
    step_count = 0
    for step_count in range(1500):
        if capture and step_count % 10 == 0:
            frames.append(env.render().copy())
        sample_semantics = step_count % 20 == 0
        with torch.no_grad():
            image = torch.as_tensor(obs["pixels"], device=device).unsqueeze(0)
            history = torch.as_tensor(obs["action_history"], device=device).unsqueeze(0)
            mask = torch.as_tensor(env.action_masks(), device=device).unsqueeze(0)
            logits, _, state, sem, _ = model(
                image, history, state, mask, compute_auxiliary=sample_semantics,
            )
            dist = Categorical(logits=logits[:, -1])
            action = logits[:, -1].argmax(-1) if deterministic else dist.sample()  # type: ignore[no-untyped-call]
        if sample_semantics:
            predictions.append(sem[0].argmax(0).cpu().numpy().astype(np.uint8))
            labels = env._info().get("training_labels", {})
            truths.append(np.asarray(labels.get("sem", np.full((27, 48), 255)), dtype=np.uint8))
        obs, _, terminated, truncated, info = env.step(int(action.item()))
        deaths += int(bool(info.get("signals", {}).get("died", False)))
        signals = info.get("signals", {})
        position = signals.get("player_pos_world")
        room = str(signals.get("room_id", ""))
        if position is not None:
            current_position = (float(position[0]), float(position[1]))
            if (previous_position is None or room != previous_room or
                    np.linalg.norm(np.subtract(current_position, previous_position)) > 1e-3 or
                    signals.get("died", False)):
                stagnant_steps = 0
            else:
                stagnant_steps += 1
                stuck = stuck or stagnant_steps >= 300
            previous_position, previous_room = current_position, room
        if terminated:
            success = True
            break
        if truncated:
            break
    env.close()
    return success, step_count + 1, deaths, not success and stuck, predictions, truths, frames, initial_frame


def main() -> int:
    _lower_priority()
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("relatorios/85_eval"))
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    table = tuple(a.name for a in UNIVERSAL_ACTIONS)
    model = load_universal(args.checkpoint, table, device).eval()
    out = args.output
    (out / "gifs").mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    conf = np.zeros((10, 10), dtype=np.int64)
    sample_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    colors = np.asarray([(25,25,35),(120,120,130),(60,180,80),(240,70,70),(240,140,40),
                         (255,240,80),(170,90,220),(60,210,220),(240,210,100),(255,255,255),
                         (255,0,255)], dtype=np.uint8)
    episodes = 2 if args.quick else args.episodes
    workers = max(1, min(6, (os.cpu_count() or 2) - 2))
    for level in range(8):
        count = 2 if args.quick and level == 7 else episodes
        if level == 7 and not args.quick:
            count = min(count, 30)
        for deterministic in (False, True):
            successes: list[bool] = []
            steps: list[int] = []
            deaths: list[int] = []
            stuck: list[bool] = []
            tasks = []
            gif_indices: set[int] = set()
            for index in range(count):
                seed = (index * 10 + 1000) if index % 10 == 0 else index * 10 + 1
                gif_capture = deterministic and index == 0 and level in (3, 4, 6)
                if gif_capture:
                    gif_indices.add(index)
                sample_capture = level == 0 and not deterministic and index < 12
                tasks.append((index, seed, gif_capture or sample_capture))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending = {
                    pool.submit(
                        _episode, model, level, seed, deterministic, device, capture,
                    ): index
                    for index, seed, capture in tasks
                }
                results = {}
                for completed_count, future in enumerate(as_completed(pending), start=1):
                    results[pending[future]] = future.result()
                    if completed_count % 10 == 0 or completed_count == count:
                        print(json.dumps({"level": level,
                                          "policy": "deterministic" if deterministic else "stochastic",
                                          "episodes_done": completed_count, "episodes_total": count}),
                              flush=True)
            for index, result in sorted(results.items()):
                success, nsteps, nd, ns, pred, truth, frames, initial_frame = result
                successes.append(success); steps.append(nsteps); deaths.append(nd); stuck.append(ns)
                if len(sample_pairs) < 12:
                    sample_pairs.append((initial_frame, pred[0], truth[0]))
                if frames and index in gif_indices:
                    Image.fromarray(frames[0]).save(out / "gifs" / f"nivel_{level}.gif", save_all=True,
                        append_images=[Image.fromarray(frame) for frame in frames[1:]], duration=50, loop=0, optimize=True)
                for p, t in zip(pred[-5:], truth[-5:], strict=False):
                    valid = t != 255
                    np.add.at(conf, (t[valid], p[valid]), 1)
            records.append({"level": level, "policy": "deterministic" if deterministic else "stochastic",
                            "episodes": count, "success": float(np.mean(successes)),
                            "steps": float(np.mean(steps)), "deaths": int(sum(deaths)),
                            "stuck": int(sum(stuck))})
            print(json.dumps(records[-1]), flush=True)
    lines = ["| Nível | Política | Salas | Sucesso | Passos médios | Mortes | Travadas |", "|---:|---|---:|---:|---:|---:|---:|"]
    lines.extend(f"| {r['level']} | {r['policy']} | {r['episodes']} | {r['success']:.1%} | {r['steps']:.1f} | {r['deaths']} | {r['stuck']} |" for r in records)
    (out / "tabela.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    iou = np.diag(conf) / np.maximum(1, conf.sum(0) + conf.sum(1) - np.diag(conf))
    (out / "perception_metrics.json").write_text(json.dumps({"iou_by_class": iou.tolist(), "confusion": conf.tolist()}, indent=2), encoding="utf-8")
    canvas = Image.new("RGB", (576, max(1, len(sample_pairs)) * 108), (20, 20, 20))
    for row, (frame, predicted, target) in enumerate(sample_pairs):
        canvas.paste(Image.fromarray(frame), (0, row * 108))
        for col, grid in enumerate((predicted, target), start=1):
            indexed = np.where(grid == 255, 10, grid)
            panel = Image.fromarray(colors[indexed]).resize((192, 108), Image.Resampling.NEAREST)
            canvas.paste(panel, (col * 192, row * 108))
    canvas.save(out / "percepcao.png")
    manifest_path = Path(args.checkpoint) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("metrics", {})["evaluation"] = {
        "episodes": records, "iou_by_class": iou.tolist(), "confusion": conf.tolist(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
