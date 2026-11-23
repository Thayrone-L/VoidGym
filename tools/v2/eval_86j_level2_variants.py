"""Compare checkpoint behavior across controlled level-2 geometry variants."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from voidgym.envs.adapter_env import AdapterEnv
from voidgym.gyms.platform_gym import adapter as adapter_module
from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter
from voidgym.gyms.platform_gym.generator import generate_room as original_generate_room
from voidgym.gyms.platform_gym.world import Rect, Solid
from voidgym.models.policy import UniversalPolicy
from voidgym.training.checkpoints import load_universal

VARIANTS = ("original", "gap_only", "barrier_only", "wide_floor_barrier", "no_attack")


def _mutate_room(room: Any, variant: str) -> tuple[float, float]:
    floors = sorted(
        (solid for solid in room.solids
         if abs(solid.rect.top - room.spawn[1]) < 1e-6 and solid.rect.h <= .7),
        key=lambda solid: solid.rect.x,
    )
    if len(floors) < 2:
        raise RuntimeError(f"seed {room.seed} has no level-2 gap")
    gap_left, gap_right = floors[0].rect.right, floors[1].rect.x
    if room.spawn[0] < gap_left:
        wall = min((solid for solid in room.solids
                    if abs(solid.rect.y - room.spawn[1]) < 1e-6 and .6 <= solid.rect.w <= .7
                    and solid.rect.h > .6 and solid.rect.x >= gap_right),
                   key=lambda solid: solid.rect.x, default=None)
    else:
        wall = max((solid for solid in room.solids
                    if abs(solid.rect.y - room.spawn[1]) < 1e-6 and .6 <= solid.rect.w <= .7
                    and solid.rect.h > .6 and solid.rect.right <= gap_left),
                   key=lambda solid: solid.rect.right, default=None)
    if wall is None:
        raise RuntimeError(f"room has no mandatory thin barrier at spawn={room.spawn}")
    wall_x = wall.rect.x
    if variant == "gap_only":
        room.solids.remove(wall)
    elif variant == "barrier_only":
        room.solids.append(Solid(Rect(gap_left, room.spawn[1] - .6,
                                      gap_right - gap_left, .6)))
    elif variant == "wide_floor_barrier":
        room.solids.remove(wall)
        widened_x = wall.rect.x if room.spawn[0] < gap_left else wall.rect.right - 2.0
        room.solids.append(Solid(Rect(widened_x, wall.rect.y, 2.0, wall.rect.h)))
    elif variant not in {"original", "no_attack"}:
        raise ValueError(f"unknown variant: {variant}")
    return wall_x, (gap_left + gap_right) / 2


def _run_episode(model: UniversalPolicy, seed: int, mode: str, variant: str,
                 max_steps: int, device: torch.device) -> dict[str, Any]:
    wall_x_box: list[float] = []
    gap_mid_box: list[float] = []

    def generate(level: int, room_seed: int, *, allow_vertical: bool = True) -> Any:
        room = original_generate_room(level, room_seed, allow_vertical=allow_vertical)
        if level == 2:
            wall_x, gap_mid = _mutate_room(room, variant)
            wall_x_box.append(wall_x)
            gap_mid_box.append(gap_mid)
        return room

    adapter_module.generate_room = generate
    try:
        adapter = PlatformGymAdapter(level=2, seed=seed, split="test")
        if variant == "no_attack":
            original_mask = adapter.action_mask

            def no_attack_mask() -> np.ndarray:
                mask = original_mask()
                for index, name in enumerate(adapter.action_table()):
                    if "ATTACK" in name or "NAIL_ART" in name:
                        mask[index] = False
                return mask

            adapter.action_mask = no_attack_mask  # type: ignore[method-assign]
        env = AdapterEnv(adapter, lambda _before, _after: 0.0)
        observation, _info = env.reset(seed=seed)
    finally:
        adapter_module.generate_room = original_generate_room
    state = model.initial_state(1, device)
    starts = torch.ones(1, dtype=torch.bool, device=device)
    names = tuple(adapter.action_table())
    attacks: Counter[str] = Counter()
    attack_distances: list[float] = []
    stationary_at_barrier = 0
    stationary_seconds = 0.0
    barrier_steps = 0
    max_x = adapter.player.x
    success = False
    stuck = False
    steps = 0
    for steps in range(1, max_steps + 1):
        x_before = adapter.player.x
        wall_x, gap_mid = wall_x_box[-1], gap_mid_box[-1]
        near_barrier = abs(x_before - wall_x) <= 1.5
        if near_barrier:
            barrier_steps += 1
            if abs(adapter.player.vx) < .08:
                stationary_at_barrier += 1
        tensors = {key: torch.as_tensor(value[None], device=device)
                   for key, value in observation.items()}
        mask = torch.as_tensor(env.action_masks()[None], device=device)
        with torch.inference_mode():
            output = model(tensors, state, starts, mask)
            logits, state = output["logits"], output["state"]
            if not isinstance(logits, torch.Tensor) or not isinstance(state, tuple):
                raise TypeError("policy returned an invalid action or recurrent state")
            action = (int(logits.argmax(-1).item()) if mode == "deterministic"
                      else int(Categorical(logits=logits).sample().item()))
        name = names[action]
        if "ATTACK" in name or "NAIL_ART" in name:
            attacks[name] += 1
            attack_distances.append(abs(x_before - wall_x))
        observation, _reward, terminated, truncated, info = env.step(action)
        if near_barrier and abs(adapter.player.vx) < .08:
            stationary_seconds += float(adapter._last_step_game_ms) / 1000.0
        starts = torch.tensor([terminated or truncated], dtype=torch.bool, device=device)
        signals = info.get("signals", {})
        max_x = max(max_x, adapter.player.x)
        success |= bool(signals.get("room_changed", False))
        stuck |= bool(signals.get("stuck", False))
        if success or terminated or truncated:
            break
    env.close()
    return {
        "seed": seed, "mode": mode, "variant": variant, "success": success,
        "steps": steps, "stuck": stuck, "attack_count": sum(attacks.values()),
        "attack_actions": dict(attacks), "attack_distance_mean": (
            float(np.mean(attack_distances)) if attack_distances else None),
        "attack_distance_median": (
            float(np.median(attack_distances)) if attack_distances else None),
        "barrier_stationary_steps": stationary_at_barrier,
        "barrier_stationary_seconds": stationary_seconds,
        "barrier_steps": barrier_steps,
        "gap_center_x": gap_mid, "barrier_x": wall_x,
        "final_x": adapter.player.x, "max_x": max_x,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=28_600)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--variant", choices=VARIANTS, action="append")
    parser.add_argument("--mode", choices=("deterministic", "stochastic"), action="append")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed_base - 20_000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed_base - 20_000)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = UniversalPolicy().to(device)
    action_table = tuple(PlatformGymAdapter(level=0).action_table())
    load_universal(args.checkpoint.resolve(strict=True), model, action_table)
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    progress = args.output / "variants.jsonl"
    completed: set[tuple[str, str, int]] = set()
    if progress.exists():
        for line in progress.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                completed.add((row["variant"], row["mode"], row["seed"]))
    selected_variants = args.variant or VARIANTS
    selected_modes = args.mode or ("deterministic", "stochastic")
    with progress.open("a", encoding="utf-8", buffering=1) as handle:
        for variant in selected_variants:
            for mode in selected_modes:
                for index in range(args.episodes):
                    seed = args.seed_base + index * 10
                    key = (variant, mode, seed)
                    if key in completed:
                        continue
                    row = _run_episode(model, seed, mode, variant, args.max_steps, device)
                    handle.write(json.dumps(row) + "\n")
                    if (index + 1) % 20 == 0:
                        print(f"{variant} {mode} {index + 1}/{args.episodes}: "
                              f"success={row['success']} steps={row['steps']}", flush=True)
    rows = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines() if line]
    summary: list[dict[str, Any]] = []
    for variant in selected_variants:
        for mode in selected_modes:
            group = [row for row in rows if row["variant"] == variant and row["mode"] == mode]
            if not group:
                continue
            distances = [row["attack_distance_mean"] for row in group
                         if row["attack_distance_mean"] is not None]
            summary.append({
                "variant": variant, "mode": mode, "episodes": len(group),
                "successes": sum(bool(row["success"]) for row in group),
                "success_rate": float(np.mean([bool(row["success"]) for row in group])),
                "attack_actions": sum(int(row["attack_count"]) for row in group),
                "attack_actions_per_episode": float(np.mean([row["attack_count"] for row in group])),
                "episodes_with_attack": sum(int(row["attack_count"] > 0) for row in group),
                "mean_attack_distance_to_barrier": float(np.mean(distances)) if distances else None,
                "mean_stationary_seconds_at_barrier": float(np.mean(
                    [row["barrier_stationary_seconds"] for row in group])),
                "mean_steps": float(np.mean([row["steps"] for row in group])),
                "mean_max_x": float(np.mean([row["max_x"] for row in group])),
            })
    (args.output / "summary.json").write_text(
        json.dumps({"seed_base": args.seed_base, "episodes_per_variant_mode": args.episodes,
                    "checkpoint": str(args.checkpoint), "device": str(device),
                    "results": summary}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
