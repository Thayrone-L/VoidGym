from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from hk_agent.process_owner import _write_json_atomic


class BaselineEnvironment(Protocol):
    action_space: Any

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[object, Mapping[str, Any]]: ...

    def step(
        self, action: int
    ) -> tuple[object, float, bool, bool, Mapping[str, Any]]: ...

    def close(self) -> None: ...


def run_random_baseline(
    environment: BaselineEnvironment,
    *,
    seed: int,
    step_budget: int,
    manifest_path: Path,
) -> dict[str, object]:
    """Run one short reproducible random-policy episode.

    The policy samples exclusively from ``environment.action_space``.  It has no
    menu bootstrap, route, macro, reward shaping, or privileged game state.
    """

    if step_budget < 1:
        raise ValueError("step_budget must be positive")
    started = time.monotonic_ns()
    actions: list[int] = []
    valid_steps = 0
    total_reward = 0.0
    termination_reason: object = None
    terminated = truncated = False
    status = "FAILED"
    failure: str | None = None
    try:
        environment.action_space.seed(seed)
        environment.reset(seed=seed)
        for _ in range(step_budget):
            action = int(environment.action_space.sample())
            if not environment.action_space.contains(action):
                raise RuntimeError("action_space.sample returned an invalid action")
            actions.append(action)
            _, reward, terminated, truncated, info = environment.step(action)
            if reward != 0.0:
                raise RuntimeError("phase-2 baseline requires reward 0")
            total_reward += reward
            if info.get("valid") is not True:
                reason = info.get("termination_reason")
                raise RuntimeError(f"invalid environment transition: {reason}")
            valid_steps += 1
            termination_reason = info.get("termination_reason")
            if terminated or truncated:
                break
        status = "COMPLETED"
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            environment.close()
        except Exception as error:  # noqa: BLE001 - report teardown independently
            if failure is None:
                failure = f"close:{type(error).__name__}: {error}"
                status = "FAILED"
        manifest: dict[str, object] = {
            "schema_version": 1,
            "kind": "random_baseline",
            "status": status,
            "seed": seed,
            "step_budget": step_budget,
            "steps_attempted": len(actions),
            "valid_steps": valid_steps,
            "actions": actions,
            "total_reward": total_reward,
            "reward_contract": "constant_zero",
            "terminated": terminated,
            "truncated": truncated,
            "termination_reason": termination_reason,
            "failure": failure,
            "duration_ms": (time.monotonic_ns() - started) / 1_000_000,
            "menu_bootstrap": "external_not_policy",
        }
        _write_json_atomic(manifest_path, manifest)
    return manifest
