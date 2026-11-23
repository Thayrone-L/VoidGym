"""Measure free-physics and action-table solvability for the 86i evaluation set."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import pairwise
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from voidgym.gyms.platform_gym.action_solver import ActionSolveResult, solve_with_actions
from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter
from voidgym.gyms.platform_gym.levels import DEAD_END_LEVEL, LEVEL_COUNT, MULTI_ROOM_LEVEL
from voidgym.gyms.platform_gym.solver import solve_room, solve_room_set


def _level2_geometry(adapter: PlatformGymAdapter) -> dict[str, float]:
    room = adapter.room
    floors = sorted((solid.rect for solid in room.solids
                     if abs(solid.rect.top - room.spawn[1]) < 1e-6 and solid.rect.h <= .7),
                    key=lambda rect: rect.x)
    gaps = [(right.x - left.right, left, right)
            for left, right in pairwise(floors) if right.x > left.right]
    gap_width, _, _ = max(gaps, key=lambda item: item[0])
    wall = min((solid.rect for solid in room.solids
                if abs(solid.rect.y - room.spawn[1]) < 1e-6 and solid.rect.h > .6
                and solid.rect.w <= .7), key=lambda rect: rect.x)
    physics = room.physics
    reach = physics.run_speed * (physics.jump_hold_max_seconds +
                                 2 * physics.jump_v0 / physics.gravity)
    return {"gap_width": gap_width,
            "gap_width_pct_of_physical_reach": 100 * gap_width / reach,
            "gap_width_pct_of_85pct_reach_limit": 100 * gap_width / (.85 * reach),
            "barrier_height": wall.h,
            "barrier_height_pct_of_jump_max": 100 * wall.h / physics.jump_max_height,
            "barrier_height_pct_of_85pct_jump_limit":
                100 * wall.h / (.85 * physics.jump_max_height)}


def _solve_level8_by_room(adapter: PlatformGymAdapter, *, max_expansions: int,
                          max_game_seconds: float,
                          heuristic_weight: float) -> ActionSolveResult:
    """Search each room segment separately while preserving the real adapter state."""
    names = tuple(adapter.action_table())
    action_indices = {name: index for index, name in enumerate(names)}
    actions: list[str] = []
    expanded = 0
    duration_ms = 0
    for _ in range(len(adapter._rooms)):
        segment = solve_with_actions(
            adapter, max_expansions=max_expansions, max_game_seconds=max_game_seconds,
            heuristic_weight=heuristic_weight, stop_after_room_change=True,
        )
        expanded += segment.expanded
        if not segment.success:
            return ActionSolveResult(False, (), expanded, duration_ms, segment.reason)
        for name in segment.actions:
            adapter.apply(action_indices[name])
        actions.extend(segment.actions)
        duration_ms += segment.duration_ms
        if adapter._finished:
            return ActionSolveResult(True, tuple(actions), expanded, duration_ms, "exit")
        if segment.reason != "room_exit":
            return ActionSolveResult(False, tuple(actions), expanded, duration_ms,
                                     "unexpected_segment_end")
    return ActionSolveResult(False, tuple(actions), expanded, duration_ms,
                             "room_limit_without_exit")


def _solve_one(task: tuple[int, int, int, int, float, float]) -> dict[str, Any]:
    level, index, seed_base, max_expansions, max_game_seconds, heuristic_weight = task
    seed = seed_base + level * 10_000 + index * 10
    adapter = PlatformGymAdapter(level=level, seed=seed, split="test")
    if level == MULTI_ROOM_LEVEL:
        free_ok = solve_room_set(adapter._rooms)
    else:
        free_ok = solve_room(adapter.room, require_breakables=level == 7)
    result = (_solve_level8_by_room(adapter, max_expansions=max_expansions,
                                    max_game_seconds=max_game_seconds,
                                    heuristic_weight=heuristic_weight)
              if level == MULTI_ROOM_LEVEL else
              solve_with_actions(adapter, max_expansions=max_expansions,
                                 max_game_seconds=max_game_seconds,
                                 heuristic_weight=heuristic_weight))
    row: dict[str, Any] = {
        "level": level, "episode": index, "seed": seed,
        "room_style_seed": adapter.room.style_seed,
        "free_solver_success": free_ok,
        "action_solver_success": result.success,
        "action_solver_reason": result.reason,
        "search_budget_expansions": max_expansions,
        "expanded_nodes": result.expanded,
        "action_solver_duration_ms": result.duration_ms,
        "actions": list(result.actions),
    }
    if level == 2:
        row.update(_level2_geometry(adapter))
    return row


def evaluate(*, count: int, seed: int, output: Path, workers: int,
             max_expansions: int, max_game_seconds: float,
             heuristic_weight: float, retry_failed: bool,
             only_level: int | None) -> dict[str, Any]:
    if count < 1 or workers < 1:
        raise ValueError("count and workers must be positive")
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "action_solvability_progress.json"
    rows: dict[str, dict[str, Any]] = {}
    if progress_path.exists():
        rows = {f"{item['level']}:{item['episode']}": item
                for item in json.loads(progress_path.read_text(encoding="utf-8"))["results"]}
    levels_to_run = range(LEVEL_COUNT) if only_level is None else (only_level,)
    tasks = [(level, episode, seed, max_expansions,
              max_game_seconds * (2 if level in {MULTI_ROOM_LEVEL, DEAD_END_LEVEL} else 1),
              heuristic_weight)
             for level in levels_to_run for episode in range(count)
             if f"{level}:{episode}" not in rows or
             (retry_failed and rows[f"{level}:{episode}"].get("action_solver_reason") == "node_limit")]
    task_runs_done = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_solve_one, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows[f"{row['level']}:{row['episode']}"] = row
            task_runs_done += 1
            if task_runs_done % 10 == 0 or task_runs_done == len(tasks):
                progress = {"seed_base": seed, "rooms_per_level": count,
                            "max_expansions": max_expansions,
                            "max_game_seconds": max_game_seconds,
                            "heuristic_weight": heuristic_weight,
                            "completed_rooms": len(rows),
                            "task_runs_done": task_runs_done,
                            "results": sorted(rows.values(), key=lambda item:
                                               (item["level"], item["episode"]))}
                temporary = progress_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
                temporary.replace(progress_path)
                print(f"action solver: {len(rows)}/{count * LEVEL_COUNT} salas; "
                      f"buscas desta rodada={task_runs_done}/{len(tasks)}", flush=True)

    result_rows = sorted(rows.values(), key=lambda item: (item["level"], item["episode"]))
    levels: dict[str, Any] = {}
    for level in range(LEVEL_COUNT):
        selected = [row for row in result_rows if row["level"] == level]
        levels[str(level)] = {
            "rooms": len(selected),
            "free_solver_successes": sum(row["free_solver_success"] for row in selected),
            "action_solver_successes": sum(row["action_solver_success"] for row in selected),
            "action_solver_unknown_or_failed": {
                reason: sum(row["action_solver_reason"] == reason for row in selected)
                for reason in sorted({row["action_solver_reason"] for row in selected
                                      if not row["action_solver_success"]})
            },
            "mean_expanded_nodes": float(sum(row["expanded_nodes"] for row in selected) /
                                          max(1, len(selected))),
        }
        if level == 2:
            levels["2"]["mean_gap_width_pct_of_reach"] = float(sum(
                row["gap_width_pct_of_physical_reach"] for row in selected) / len(selected))
            levels["2"]["mean_barrier_height_pct_of_jump_max"] = float(sum(
                row["barrier_height_pct_of_jump_max"] for row in selected) / len(selected))
    report = {"seed_base": seed, "rooms_per_level": count,
              "max_expansions": max_expansions, "max_game_seconds": max_game_seconds,
              "heuristic_weight": heuristic_weight,
              "levels": levels, "results": result_rows}
    report_path = output / "action_solvability.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=8600)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-expansions", type=int, default=30_000)
    parser.add_argument("--max-game-seconds", type=float, default=45.0)
    parser.add_argument("--heuristic-weight", type=float, default=2.5)
    parser.add_argument("--retry-failed", action="store_true",
                        help="retry previous rooms whose search hit its node limit")
    parser.add_argument("--only-level", type=int, choices=range(LEVEL_COUNT),
                        help="limit a retry to one level while preserving other saved rows")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "relatorios" / "86i_action_solver")
    args = parser.parse_args()
    result = evaluate(count=args.count, seed=args.seed, output=args.output,
                      workers=args.workers, max_expansions=args.max_expansions,
                      max_game_seconds=args.max_game_seconds,
                      heuristic_weight=args.heuristic_weight,
                      retry_failed=args.retry_failed, only_level=args.only_level)
    print(json.dumps({"completed": sum(row["rooms"] for row in result["levels"].values()),
                      "levels": result["levels"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
