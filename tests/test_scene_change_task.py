from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pytest

from hk_agent.route_memory import TopologicalRouteMemory
from hk_agent.scene_change_probe import (
    SceneChangeProbeConfig,
    completed_probe_result,
    run_scene_change_probe,
)
from hk_agent.scene_change_task import (
    SCENE_CHANGE_ACTION_NAMES,
    SCENE_CHANGE_ACTIONS,
    SCENE_CHANGE_REWARD_VERSION,
    ReachSceneChangeTask,
    SceneChangeTaskConfig,
)
from hk_agent.training_task import InvalidTrainingTransition


def test_scene_task_exposes_nail_spell_and_focus_actions() -> None:
    actions = dict(zip(SCENE_CHANGE_ACTION_NAMES, SCENE_CHANGE_ACTIONS, strict=True))
    assert actions["ATTACK"] == 1 << 5
    assert actions["LEFT+ATTACK"] == (1 << 0) | (1 << 5)
    assert actions["RIGHT+ATTACK"] == (1 << 1) | (1 << 5)
    assert actions["UP+ATTACK"] == (1 << 2) | (1 << 5)
    assert actions["DOWN+ATTACK"] == (1 << 3) | (1 << 5)
    assert actions["JUMP+ATTACK"] == (1 << 4) | (1 << 5)
    assert actions["LEFT+JUMP+ATTACK"] == (1 << 0) | (1 << 4) | (1 << 5)
    assert actions["RIGHT+JUMP+ATTACK"] == (1 << 1) | (1 << 4) | (1 << 5)
    assert actions["CAST"] == 1 << 7
    assert actions["UP+CAST"] == (1 << 2) | (1 << 7)
    assert actions["DOWN+CAST"] == (1 << 3) | (1 << 7)
    assert actions["FOCUS_HOLD"] == 1 << 7
    assert actions["NAIL_ART_CHARGE"] == 1 << 5
    assert actions["UP+NAIL_ART_CHARGE"] == (1 << 2) | (1 << 5)
    assert actions["DOWN+NAIL_ART_CHARGE"] == (1 << 3) | (1 << 5)


class FakeSceneEnvironment:
    metadata: ClassVar[dict[str, object]] = {}

    def __init__(
        self,
        frames: list[np.ndarray[Any, Any]],
        *,
        invalid_at: int | None = None,
        terminated_at: int | None = None,
        truncated_at: int | None = None,
    ):
        self.frames = frames
        self.invalid_at = invalid_at
        self.terminated_at = terminated_at
        self.truncated_at = truncated_at
        self.index = 0
        self.releases = 0
        self.closed = False
        self.lease_ms = 80
        self.last_action: tuple[int, int] | None = None
        height, width, _ = frames[0].shape
        self.observation_space = gym.spaces.Dict(
            {
                "pixels": gym.spaces.Box(0, 255, (2, height, width, 3), np.uint8),
                "action_history": gym.spaces.Box(0, 1023, (1,), np.int64),
            }
        )
        self.backend = SimpleNamespace(release_all=self._release)

    def _release(self) -> None:
        self.releases += 1

    def _observation(self) -> dict[str, np.ndarray[Any, Any]]:
        return {
            "pixels": np.stack([self.frames[self.index], self.frames[self.index]]),
            "action_history": np.zeros(1, dtype=np.int64),
        }

    def reset(self, **_: object) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, object]]:
        self.index = 0
        return self._observation(), {}

    def step(
        self, action: int, *, lease_ms: int | None = None
    ) -> tuple[dict[str, np.ndarray[Any, Any]], float, bool, bool, dict[str, object]]:
        self.last_action = (action, self.lease_ms if lease_ms is None else lease_ms)
        self.index = min(self.index + 1, len(self.frames) - 1)
        invalid = self.invalid_at == self.index
        terminated = self.terminated_at == self.index
        truncated = self.truncated_at == self.index
        return self._observation(), 0.0, terminated, invalid or truncated, {
            "valid": not invalid,
            "termination_reason": (
                "operational_failure"
                if invalid
                else (
                    "player_death"
                    if terminated
                    else ("time_limit" if truncated else None)
                )
            ),
        }

    def close(self) -> None:
        self.closed = True


def _frame(rgb: tuple[int, int, int]) -> np.ndarray[Any, Any]:
    return np.full((24, 32, 3), rgb, dtype=np.uint8)


def _textured(red: int, green: int, blue: int) -> np.ndarray[Any, Any]:
    frame = _frame((red, green, blue))
    frame[::2, ::2] = np.asarray(
        (min(255, red + 35), min(255, green + 35), min(255, blue + 35)),
        dtype=np.uint8,
    )
    frame[1::2, 1::2] = np.asarray(
        (max(0, red - 35), max(0, green - 35), max(0, blue - 35)),
        dtype=np.uint8,
    )
    return frame


def _config() -> SceneChangeTaskConfig:
    return SceneChangeTaskConfig(
        horizon_steps=32,
        warmup_frames=3,
        descriptor_rows=3,
        descriptor_columns=4,
        crop_top_fraction=0,
        crop_bottom_fraction=1,
        crop_left_fraction=0,
        crop_right_fraction=1,
        transition_dark_frames=2,
        stable_frames=3,
        minimum_entropy_bits=0.5,
    )


def _run(task: ReachSceneChangeTask, count: int) -> list[dict[str, Any]]:
    task.reset(seed=4)
    infos: list[dict[str, Any]] = []
    for _ in range(count):
        _, _, terminated, truncated, info = task.step(2)
        infos.append(info)
        if terminated or truncated:
            break
    return infos


def test_stable_post_fade_scene_is_rewarded_once_and_terminates() -> None:
    frames = [_textured(170, 170, 170)] * 3
    frames += [_frame((0, 0, 0))] * 2
    frames += [_textured(170, 35, 35)] * 4
    task = ReachSceneChangeTask(FakeSceneEnvironment(frames), config=_config())  # type: ignore[arg-type]
    infos = _run(task, len(frames) - 1)
    assert infos[-1]["visual_regime_change_detected"] is True
    assert infos[-1]["termination_reason"] == "visual_regime_change_detected"
    assert infos[-1]["reward_components"]["stable_visual_regime_change"] == 10.0
    assert sum(info["reward_components"]["bounded_novelty"] for info in infos) <= 1.0


def test_continuous_scene_reward_does_not_end_episode() -> None:
    frames = [_textured(170, 170, 170)] * 3
    frames += [_frame((0, 0, 0))] * 2
    frames += [_textured(170, 35, 35)] * 4
    task = ReachSceneChangeTask(
        FakeSceneEnvironment(frames),  # type: ignore[arg-type]
        config=replace(_config(), horizon_steps=None, terminate_on_scene_change=False),
    )
    task.reset(seed=4)
    saw_reward = False
    for _ in range(len(frames) - 1):
        _, reward, terminated, truncated, info = task.step(2)
        assert not terminated and not truncated
        if info["visual_regime_change_detected"]:
            saw_reward = True
            assert reward > 0
            assert info["scene_change_event"] is True
            assert info["termination_reason"] is None
    assert saw_reward


def test_continuous_detector_reanchors_for_two_consecutive_scene_changes() -> None:
    room_a = _textured(170, 170, 170)
    room_b = _textured(170, 35, 35)
    room_c = _textured(35, 180, 35)
    frames = [room_a] * 3
    frames += [_frame((0, 0, 0))] * 2 + [room_b] * 7
    frames += [_frame((0, 0, 0))] * 2 + [room_c] * 5
    task = ReachSceneChangeTask(
        FakeSceneEnvironment(frames),  # type: ignore[arg-type]
        config=replace(_config(), horizon_steps=None, terminate_on_scene_change=False),
    )
    task.reset(seed=4)
    events: list[dict[str, Any]] = []
    for _ in range(len(frames) - 1):
        _, _, terminated, truncated, info = task.step(2)
        assert not terminated and not truncated
        if info["scene_change_event"]:
            events.append(info)

    assert len(events) == 2
    assert all(event["termination_reason"] is None for event in events)


def test_scene_events_build_persistent_route_memory_without_changing_reward(
    tmp_path: Path,
) -> None:
    room_a = _textured(170, 170, 170)
    room_b = _textured(170, 35, 35)
    frames = [room_a] * 3 + [_frame((0, 0, 0))] * 2 + [room_b] * 4
    route_memory = TopologicalRouteMemory(
        tmp_path / "routes.json",
        scope={
            "game_profile": "test-profile",
            "save_source": "existing_save",
            "save_slot": 1,
            "campaign_id": "test-campaign",
            "detector_signature": "test-detector-v1",
        },
    )
    task = ReachSceneChangeTask(
        FakeSceneEnvironment(frames),  # type: ignore[arg-type]
        config=_config(),
        route_memory=route_memory,
    )
    infos = _run(task, len(frames) - 1)

    event = infos[-1]["route_memory"]["event"]
    assert event["source"] == "landmark-000001"
    assert event["destination"] == "landmark-000002"
    assert event["new_edge"] is True
    assert infos[-1]["reward_components"]["stable_visual_regime_change"] == 10.0
    assert route_memory.shortest_path(event["source"], event["destination"]) == (
        event["source"],
        event["destination"],
    )
    assert (tmp_path / "routes.json").is_file()


def test_camera_or_animation_change_without_fade_is_not_scene_success() -> None:
    frames = [_textured(170, 170, 170)] * 3 + [_textured(30, 180, 60)] * 8
    task = ReachSceneChangeTask(FakeSceneEnvironment(frames), config=_config())  # type: ignore[arg-type]
    infos = _run(task, len(frames) - 1)
    assert all(info["visual_regime_change_detected"] is False for info in infos)
    assert all(info["detector"]["phase"] == "seek_transition" for info in infos)


def test_single_dark_animation_frame_cannot_arm_detector() -> None:
    frames = [_textured(170, 170, 170)] * 3
    frames += [_frame((0, 0, 0)), _textured(170, 170, 170)]
    frames += [_textured(170, 35, 35)] * 5
    task = ReachSceneChangeTask(FakeSceneEnvironment(frames), config=_config())  # type: ignore[arg-type]
    infos = _run(task, len(frames) - 1)
    assert all(info["visual_regime_change_detected"] is False for info in infos)


@pytest.mark.parametrize("post_frame", [_frame((255, 255, 255)), _frame((170, 35, 35))])
def test_blank_post_transition_frames_are_rejected(post_frame: np.ndarray[Any, Any]) -> None:
    frames = [_textured(170, 170, 170)] * 3
    frames += [_frame((0, 0, 0))] * 2 + [post_frame] * 5
    task = ReachSceneChangeTask(FakeSceneEnvironment(frames), config=_config())  # type: ignore[arg-type]
    infos = _run(task, len(frames) - 1)
    assert all(info["visual_regime_change_detected"] is False for info in infos)
    assert sum(info["reward_components"]["bounded_novelty"] for info in infos) == 0


def test_unstable_warmup_does_not_freeze_baseline_or_arm_transition() -> None:
    frames = [
        _textured(170, 170, 170),
        _textured(30, 180, 60),
        _textured(50, 50, 200),
        _frame((0, 0, 0)),
        _frame((0, 0, 0)),
    ] + [_textured(170, 35, 35)] * 5
    task = ReachSceneChangeTask(FakeSceneEnvironment(frames), config=_config())  # type: ignore[arg-type]
    infos = _run(task, len(frames) - 1)
    assert all(info["detector"]["phase"] == "seek_transition" for info in infos)
    assert all(info["visual_regime_change_detected"] is False for info in infos)


def test_invalid_transition_aborts_before_learner_and_is_logged(tmp_path: Path) -> None:
    environment = FakeSceneEnvironment([_frame((220, 220, 220))] * 3, invalid_at=1)
    log_path = tmp_path / "transitions.jsonl"
    task = ReachSceneChangeTask(environment, config=_config(), task_log_path=log_path)  # type: ignore[arg-type]
    task.reset()
    with pytest.raises(InvalidTrainingTransition, match="discarded before learner"):
        task.step(1)
    record = json.loads(log_path.read_text("utf-8"))
    assert record["valid"] is False
    assert record["discarded_before_learner"] is True
    assert environment.releases == 1


@pytest.mark.parametrize(
    ("end_kind", "expected_reason"),
    [("terminated", "player_death"), ("truncated", "time_limit")],
)
def test_base_end_precedes_visual_candidate_and_pays_no_positive_component(
    end_kind: str, expected_reason: str
) -> None:
    frames = [_textured(170, 170, 170)] * 3
    frames += [_frame((0, 0, 0))] * 2
    frames += [_textured(170, 35, 35)] * 4
    end_arguments = (
        {"terminated_at": len(frames) - 1}
        if end_kind == "terminated"
        else {"truncated_at": len(frames) - 1}
    )
    environment = FakeSceneEnvironment(frames, **end_arguments)
    configured = replace(_config(), novelty_shaping_budget=1.0)
    task = ReachSceneChangeTask(environment, config=configured)  # type: ignore[arg-type]
    infos = _run(task, len(frames) - 1)
    assert infos[-1]["termination_reason"] == expected_reason
    assert infos[-1]["visual_regime_change_detected"] is False
    assert not any(value > 0 for value in infos[-1]["reward_components"].values())


def test_default_shaping_is_disabled() -> None:
    assert SceneChangeTaskConfig().novelty_shaping_budget == 0.0


def test_seeded_probe_captures_only_detector_frames_with_hashes(tmp_path: Path) -> None:
    frames = [_textured(170, 170, 170)] * 3
    frames += [_frame((0, 0, 0))] * 2
    frames += [_textured(170, 35, 35)] * 4
    task = ReachSceneChangeTask(FakeSceneEnvironment(frames), config=_config())  # type: ignore[arg-type]
    summaries, positives = run_scene_change_probe(
        task,
        config=SceneChangeProbeConfig(
            seed=13, episodes=1, horizon_steps=len(frames) - 1, action_repeat=2
        ),
        artifact_root=tmp_path,
    )
    assert positives == 1
    assert summaries[0]["positive_events"] == 1
    assert summaries[0]["termination_reason"] == "visual_regime_change_detected"
    captured = summaries[0]["captured_frames"]
    assert captured
    assert all(Path(item["path"]).read_bytes().startswith(b"\x89PNG") for item in captured)
    assert all(len(item["sha256"]) == 64 for item in captured)


def test_probe_without_positive_is_completed_diagnostic_not_failure(tmp_path: Path) -> None:
    frames = [_textured(170, 170, 170)] * 8
    task = ReachSceneChangeTask(FakeSceneEnvironment(frames), config=_config())  # type: ignore[arg-type]
    summaries, positives = run_scene_change_probe(
        task,
        config=SceneChangeProbeConfig(
            seed=13, episodes=1, horizon_steps=5, action_repeat=2
        ),
        artifact_root=tmp_path,
    )
    result = completed_probe_result(summaries, positives)
    assert result["status"] == "COMPLETED"
    assert result["gate_passed"] is False
    assert result["positive_events"] == 0
    assert result["operational_failures"] == 0
    assert not (tmp_path / "frames").exists()


def test_probe_records_death_and_resets_next_episode_without_operational_failure(
    tmp_path: Path,
) -> None:
    frames = [_textured(170, 170, 170)] * 4
    environment = FakeSceneEnvironment(frames, terminated_at=2)
    task = ReachSceneChangeTask(environment, config=_config())  # type: ignore[arg-type]
    summaries, positives = run_scene_change_probe(
        task,
        config=SceneChangeProbeConfig(seed=8, episodes=2, horizon_steps=4, action_repeat=1),
        artifact_root=tmp_path,
    )
    result = completed_probe_result(summaries, positives)
    assert len(summaries) == 2
    assert all(item["ended_by_death"] is True for item in summaries)
    assert result["death_episodes"] == 2
    assert result["episode_resets_started"] == 2
    assert result["operational_failures"] == 0
    assert result["status"] == "COMPLETED"


def test_probe_honors_stop_request_before_next_episode(tmp_path: Path) -> None:
    stop_file = tmp_path / "stop.request"
    stop_file.write_text("stop", encoding="utf-8")
    task = ReachSceneChangeTask(
        FakeSceneEnvironment([_textured(170, 170, 170)] * 4),  # type: ignore[arg-type]
        config=_config(),
    )

    summaries, positives = run_scene_change_probe(
        task,
        config=SceneChangeProbeConfig(seed=8, episodes=2, horizon_steps=4),
        artifact_root=tmp_path,
        stop_file=stop_file,
    )

    assert summaries == []
    assert positives == 0


def test_contract_has_no_privileged_state() -> None:
    task = ReachSceneChangeTask(FakeSceneEnvironment([_frame((220, 220, 220))]), config=_config())  # type: ignore[arg-type]
    contract = task.task_contract()
    assert contract["reward_version"] == SCENE_CHANGE_REWARD_VERSION
    assert contract["information_boundary"] == ["pixels", "applied_action_history"]
    assert contract["privileged_state"] == []
    assert contract["route_memory"]["enabled"] is False
    assert (
        contract["route_memory"]["config"]["policy_integration"]
        == "external_planner_only"
    )
    assert contract["diagnostic_training"] is True
    assert "holdout" in contract["claim"]


def test_runner_defaults_to_dry_run_and_does_not_launch_game() -> None:
    tool = Path(__file__).resolve().parents[1] / "tools" / "run_scene_change_training.py"
    result = subprocess.run(
        [sys.executable, str(tool), "--timesteps", "256", "--checkpoint-interval", "256"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["mode"] == "dry-run"
    assert plan["will_launch_game"] is False
    assert plan["curriculum"] == SCENE_CHANGE_REWARD_VERSION


def test_probe_runner_defaults_to_dry_run_without_learner_or_checkpoints() -> None:
    tool = Path(__file__).resolve().parents[1] / "tools" / "run_scene_change_probe.py"
    result = subprocess.run(
        [sys.executable, str(tool), "--episodes", "1", "--horizon", "32"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["mode"] == "dry-run"
    assert plan["will_launch_game"] is False
    assert plan["learner"] is None
    assert plan["checkpoints"] is False
