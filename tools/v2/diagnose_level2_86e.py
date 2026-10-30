"""Classify deterministic level-2 failures and measure solver margins."""

from __future__ import annotations

from copy import deepcopy
import json
import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter
from voidgym.gyms.platform_gym.physics import step_physics
from voidgym.gyms.platform_gym.solver import DT, _buttons
from voidgym.envs.adapter_env import AdapterEnv


def _max_jump_reach(room: object) -> float:
    physics = room.physics  # type: ignore[attr-defined]
    return physics.run_speed * (
        physics.jump_hold_max_seconds + 2 * physics.jump_v0 / physics.gravity
    )


def _solver_trace(room: object, limit: int = 20_000) -> tuple[bool, int]:
    from voidgym.gyms.platform_gym.world import Player

    room = deepcopy(room)
    player = Player(*room.spawn, on_ground=True)
    player.last_safe_pos = room.spawn
    for step in range(1, limit + 1):
        events = step_physics(room, player, _buttons(room, player), DT)
        if any(event.kind == "exit" for event in events):
            return True, step
        if player.y < -2 or player.x > room.width + 2 or player.hp <= 0:
            return False, step
    return False, limit


def _classify(room: object, history: list[tuple[float, float]],
              terminal: str, steps: int, limit: int) -> str:
    floors = sorted((s.rect for s in room.solids
                     if s.rect.h == .6 and abs(s.rect.top - room.spawn[1]) < 1e-6),
                    key=lambda rect: rect.x)
    gap_left, gap_right = floors[0].right, floors[1].x
    wall = next(s for s in room.solids
                if s.rect.y == room.spawn[1] and s.rect.h > .3)
    x, y = history[-1]
    direction = 1 if room.exits[0].rect.x > room.spawn[0] else -1
    progress = direction * (x - room.spawn[0])
    gap_start = direction * ((gap_left if direction > 0 else gap_right) - room.spawn[0])
    gap_end = direction * ((gap_right if direction > 0 else gap_left) - room.spawn[0])
    wall_edges = (direction * (wall.rect.x - room.spawn[0]),
                  direction * (wall.rect.right - room.spawn[0]))
    wall_start, wall_end = min(wall_edges), max(wall_edges)
    if terminal == "death":
        if progress < gap_start - .8:
            return "fell_before_gap"
        if progress <= gap_end + .8:
            return "fell_in_gap"
        if progress < wall_start:
            return "fell_after_gap_before_barrier"
        return "fell_at_barrier"
    if steps >= limit or terminal == "timeout":
        if progress < gap_start - .8:
            return "stuck_before_gap"
        if wall_start - 1.0 <= progress <= wall_end + 1.0:
            return "stuck_at_barrier"
        return "timeout"
    if progress < gap_end + .8:
        return "failed_to_clear_gap"
    if wall_start - 1.0 <= progress <= wall_end + 1.0 and y <= room.spawn[1] + room.physics.player_height:
        return "blocked_by_wall"
    if any(abs(a[0] - b[0]) < .05 for a, b in zip(history[-60:], history[-59:])):
        return "stuck_after_gap"
    return "other"


def diagnose(count: int = 200, seed: int = 8600, max_steps: int = 1200) -> dict[str, object]:
    output_dir = PROJECT_ROOT / "relatorios" / "86e_level2_diagnosis"
    gif_dir = output_dir / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    for old_gif in gif_dir.glob("nivel2_*.gif"):
        old_gif.unlink()
    rows: list[dict[str, object]] = []
    exemplars: dict[str, tuple[list[Image.Image], dict[str, object]]] = {}
    fixed = PlatformGymAdapter(level=2, seed=seed + 2 * 1_000_000, split="test")
    action_table = fixed.action_table()
    try:
        import torch
        from torch import Tensor
        from voidgym.models.policy import UniversalPolicy
        from voidgym.training.checkpoints import load_universal
        checkpoint = Path(r"E:\VoidGymRuntime\v2\pretrain\pretrain-86d-long-2026-10-29-cuda\universal-009235520")
        model = UniversalPolicy().to("cpu").eval()
        load_universal(checkpoint, model, tuple(action_table))
        torch.set_num_threads(2)
    except Exception as exc:
        raise RuntimeError(f"could not load 86e checkpoint: {exc}") from exc

    for index in range(count):
        room_seed = seed + 2_000_000 + index
        adapter = PlatformGymAdapter(level=2, seed=room_seed, split="test")
        room = deepcopy(adapter.room)
        floors = sorted((s.rect for s in room.solids
                         if s.rect.h == .6 and abs(s.rect.top - room.spawn[1]) < 1e-6),
                        key=lambda rect: rect.x)
        gap_left, gap_right = floors[0].right, floors[1].x
        wall = next(s for s in room.solids
                    if abs(s.rect.y - room.spawn[1]) < 1e-6 and
                    s.rect.h > .6 and s.rect.w <= .7)
        runup_distance = (wall.rect.x - gap_right if room.spawn[0] < gap_left
                          else gap_left - wall.rect.right)
        solver_ok, solver_steps = _solver_trace(room)
        report_row: dict[str, object] = {
            "seed": room_seed, "solver_success": solver_ok, "solver_steps": solver_steps,
            "gap_width_ratio_to_physical_reach": round((gap_right - gap_left) / _max_jump_reach(room), 4),
            "gap_width_ratio_to_85pct_limit": round((gap_right - gap_left) / (.85 * _max_jump_reach(room)), 4),
            "wall_height_ratio_to_jump_max": round(wall.rect.h / room.physics.jump_max_height, 4),
            "wall_height_ratio_to_85pct_limit": round(wall.rect.h / (.85 * room.physics.jump_max_height), 4),
            "runup_distance": round(runup_distance, 4),
            "gap_to_wall_landing_distance": round(runup_distance, 4),
        }
        for mode in ("deterministic",):
            env = AdapterEnv(adapter, lambda _before, _after: 0.0)
            obs, info = env.reset(seed=room_seed)
            state = model.initial_state(1, __import__("torch").device("cpu"))
            starts = __import__("torch").ones(1, dtype=__import__("torch").bool)
            history: list[tuple[float, float]] = [(adapter.player.x, adapter.player.y)]
            frames: list[Image.Image] = []
            terminal = ""
            done = False
            for action_index in range(1, max_steps + 1):
                tensors = {key: __import__("torch").as_tensor(value[None]) for key, value in obs.items()}
                mask = __import__("torch").as_tensor(env.action_masks()[None])
                with __import__("torch").no_grad():
                    output = model(tensors, state, starts, mask)
                    logits = output["logits"]
                    next_state = output["state"]
                    assert isinstance(logits, Tensor) and isinstance(next_state, tuple)
                    action = int(logits.argmax(-1).item())
                    state = next_state
                obs, _, terminated, truncated, info = env.step(action)
                player = adapter.player
                history.append((player.x, player.y))
                if action_index % 6 == 0:
                    frame = Image.fromarray(adapter.frame_rgb()).resize((384, 216), Image.Resampling.NEAREST)
                    frames.append(frame)
                signals = info.get("signals", {})
                if signals.get("died") or player.y < -2 or player.hp <= 0:
                    terminal = "death"
                    done = True
                    break
                if terminated or signals.get("room_changed"):
                    terminal = "success"
                    done = True
                    break
                if truncated:
                    terminal = "timeout"
                    done = True
                    break
            if not done:
                terminal = "timeout"
            category = "success" if terminal == "success" else _classify(room, history, terminal, action_index, max_steps)
            report_row[f"{mode}_success"] = terminal == "success"
            report_row[f"{mode}_steps"] = action_index
            report_row[f"{mode}_final_x"] = round(history[-1][0], 4)
            report_row[f"{mode}_final_y"] = round(history[-1][1], 4)
            report_row[f"{mode}_failure_type"] = None if terminal == "success" else category
            if mode == "deterministic" and category not in {"success"} and category not in exemplars:
                exemplars[category] = (frames, {"seed": room_seed, "category": category,
                                                       "steps": action_index, "trace": history})
            env.close()
        rows.append(report_row)
        if (index + 1) % 20 == 0:
            print(f"level2 diagnostic {index + 1}/{count}", flush=True)

    types = sorted({str(row["deterministic_failure_type"]) for row in rows if row["deterministic_failure_type"]})
    for category, (frames, _) in exemplars.items():
        if frames:
            frames[0].save(gif_dir / f"nivel2_{category}.gif", save_all=True,
                           append_images=frames[1:], duration=70, loop=0)
    summary: dict[str, object] = {
        "checkpoint": "universal-009235520", "split": "test", "seed_base": seed,
        "rooms": count, "max_steps": max_steps, "failure_types_observed": types,
        "failure_counts": {category: sum(row["deterministic_failure_type"] == category for row in rows)
                           for category in types},
        "solver_successes": sum(bool(row["solver_success"]) for row in rows),
        "solver_steps_mean": float(np.mean([int(row["solver_steps"]) for row in rows])),
        "gap_width_ratio_to_85pct_limit_mean": float(np.mean([float(row["gap_width_ratio_to_85pct_limit"]) for row in rows])),
        "wall_height_ratio_mean": float(np.mean([float(row["wall_height_ratio_to_jump_max"]) for row in rows])),
        "wall_height_ratio_to_85pct_limit_mean": float(np.mean([float(row["wall_height_ratio_to_85pct_limit"]) for row in rows])),
        "exemplars": {key: {"seed": val[1]["seed"], "steps": val[1]["steps"]} for key, val in exemplars.items()},
        "results": rows,
    }
    (output_dir / "level2_diagnosis.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=8600)
    parser.add_argument("--max-steps", type=int, default=1200)
    args = parser.parse_args()
    diagnose(count=args.count, seed=args.seed, max_steps=args.max_steps)
