from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import pytest

from hk_agent.random_baseline import run_random_baseline


class FakeEnvironment:
    def __init__(
        self, *, fail_step: int | None = None, invalid_step: int | None = None
    ) -> None:
        self.action_space = gym.spaces.Discrete(16)
        self.fail_step = fail_step
        self.invalid_step = invalid_step
        self.actions: list[int] = []
        self.closed = 0
        self.reset_seed: int | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[object, dict[str, Any]]:
        del options
        self.reset_seed = seed
        return object(), {}

    def step(self, action: int) -> tuple[object, float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        if self.fail_step == len(self.actions):
            raise ConnectionError("injected")
        if self.invalid_step == len(self.actions):
            return (
                object(),
                0.0,
                False,
                True,
                {"valid": False, "termination_reason": "operational_failure"},
            )
        return object(), 0.0, False, False, {"valid": True, "termination_reason": None}

    def close(self) -> None:
        self.closed += 1


def test_random_baseline_is_seeded_bounded_and_action_space_only(tmp_path: Path) -> None:
    first = FakeEnvironment()
    second = FakeEnvironment()
    first_result = run_random_baseline(
        first, seed=42, step_budget=5, manifest_path=tmp_path / "first.json"
    )
    run_random_baseline(
        second, seed=42, step_budget=5, manifest_path=tmp_path / "second.json"
    )
    assert first.actions == second.actions
    assert all(first.action_space.contains(action) for action in first.actions)
    assert first_result["status"] == "COMPLETED"
    assert first_result["steps_attempted"] == 5
    assert first_result["valid_steps"] == 5
    assert first_result["total_reward"] == 0.0
    assert first_result["menu_bootstrap"] == "external_not_policy"
    assert first.closed == second.closed == 1


def test_random_baseline_fault_writes_failed_manifest_and_closes(tmp_path: Path) -> None:
    environment = FakeEnvironment(fail_step=2)
    manifest_path = tmp_path / "failed.json"
    with pytest.raises(ConnectionError, match="injected"):
        run_random_baseline(
            environment, seed=1, step_budget=5, manifest_path=manifest_path
        )
    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest["status"] == "FAILED"
    assert manifest["steps_attempted"] == 2
    assert "ConnectionError" in manifest["failure"]
    assert environment.closed == 1


def test_random_baseline_rejects_unbounded_zero_budget(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        run_random_baseline(
            FakeEnvironment(), seed=0, step_budget=0, manifest_path=tmp_path / "x.json"
        )


def test_random_baseline_rejects_invalid_operational_transition(tmp_path: Path) -> None:
    environment = FakeEnvironment(invalid_step=1)
    manifest_path = tmp_path / "invalid.json"
    with pytest.raises(RuntimeError, match="invalid environment transition"):
        run_random_baseline(
            environment, seed=3, step_budget=4, manifest_path=manifest_path
        )
    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest["status"] == "FAILED"
    assert manifest["valid_steps"] == 0
    assert manifest["truncated"] is True
    assert "operational_failure" in manifest["failure"]
