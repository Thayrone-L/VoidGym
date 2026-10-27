"""Measure simple-control baselines on certified procedural platform rooms."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter
from voidgym.gyms.platform_gym.solver import solve_room, solve_room_set

POLICIES = ("RIGHT", "RIGHT+JUMP_TAP", "RANDOM_UNIFORM")


def _rollout(adapter: PlatformGymAdapter, policy: str, seed: int, max_actions: int) -> bool:
    actions = adapter.action_table()
    fixed_action = actions.index(policy) if policy != "RANDOM_UNIFORM" else None
    rng = np.random.default_rng(seed)
    level = adapter.level
    for _ in range(max_actions):
        if fixed_action is None:
            legal = np.flatnonzero(adapter.action_mask())
            action = int(rng.choice(legal))
        else:
            action = fixed_action
        adapter.apply(action)
        signals = adapter.signals()
        if signals.get("terminated") or (level != 7 and signals.get("room_changed")):
            return True
        if signals.get("died") or signals.get("truncated"):
            return False
    return False


def evaluate(*, count: int = 200, seed: int = 8600) -> dict[str, Any]:
    if count < 1:
        raise ValueError("count must be positive")
    report: dict[str, Any] = {"rooms_per_level": count, "policies": list(POLICIES),
                              "levels": {}, "level7_door_without_ability": {}}
    for level in range(2, 8):
        successes = {policy: 0 for policy in POLICIES}
        solver_passes = 0
        no_ability_door_passes = 0
        for index in range(count):
            room_seed = seed + level * 1_000_000 + index
            adapter = PlatformGymAdapter(level=level, seed=room_seed, split="test")
            if level == 7:
                solver_passes += int(solve_room_set(deepcopy(adapter._rooms)))
                without_ability = deepcopy(adapter._rooms)
                for room in without_ability:
                    room.pickups = [item for item in room.pickups
                                    if item.kind != "ability_double_jump"]
                no_ability_door_passes += int(solve_room_set(without_ability))
            else:
                solver_passes += int(solve_room(deepcopy(adapter.room),
                                                require_breakables=level == 6))
            max_actions = 3000 if level == 7 else 1200
            for policy_index, policy in enumerate(POLICIES):
                episode = deepcopy(adapter)
                successes[policy] += int(_rollout(episode, policy,
                                                   room_seed + policy_index * 10_000,
                                                   max_actions))
        failures = {policy: 1.0 - successes[policy] / count for policy in POLICIES}
        report["levels"][str(level)] = {
            "solver_success_rate": solver_passes / count,
            "policy_successes": successes,
            "policy_failure_rates": failures,
            "all_baselines_fail_at_least_80_percent": all(rate >= .8 for rate in failures.values()),
        }
        if level == 7:
            report["level7_door_without_ability"] = {
                "solver_successes": no_ability_door_passes,
                "sets": count,
                "failure_rate": 1.0 - no_ability_door_passes / count,
            }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=8600)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "relatorios" / "86b_politicas_bobas.json")
    args = parser.parse_args()
    report = evaluate(count=args.count, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    criteria_met = (all(row["all_baselines_fail_at_least_80_percent"]
                        and row["solver_success_rate"] == 1.0
                        for row in report["levels"].values())
                    and report["level7_door_without_ability"]["failure_rate"] == 1.0)
    return 0 if criteria_met else 1


if __name__ == "__main__":
    raise SystemExit(main())
