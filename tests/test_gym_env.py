from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from hk_agent.gym_env import (
    ACTION_NAMES,
    AppliedAction,
    BackendStep,
    HollowKnightEnv,
    PixelFrame,
    ResetResult,
)


class FakeBackend:
    run_id = "run-test"
    instance_id = "hk-00"

    def __init__(self) -> None:
        self.frame_id = 0
        self.releases = 0
        self.closes = 0
        self.steps: list[tuple[int, tuple[str, ...], int]] = []
        self.bad_frame: str | None = None
        self.end: str | None = None
        self.episode_id = ""
        self.enemy_kills = 0
        self.hp_lost = 0
        self.geo_gained = 0
        self.geo_spent = 0
        self.semantic_labels: np.ndarray | None = None

    def _frame(self, last_action: int | None = None) -> PixelFrame:
        self.frame_id += 1
        pixels = np.full((2, 3, 3), 7, dtype=np.uint8)
        frame_id = self.frame_id
        duplicate_of = None
        transition = False
        if self.bad_frame == "duplicate":
            duplicate_of = max(0, frame_id - 1)
        elif self.bad_frame == "uncorrelated":
            last_action = 999
        elif self.bad_frame == "shape":
            pixels = np.zeros((1, 1, 3), dtype=np.uint8)
        return PixelFrame(
            episode_id=self.episode_id,
            frame_id=frame_id,
            capture_monotonic_ns=frame_id * 100,
            pixels=pixels,
            scene_transition=transition,
            duplicate_of=duplicate_of,
            last_applied_action_id=last_action,
            enemy_kills_since_last_frame=self.enemy_kills,
            hp_lost_since_last_frame=self.hp_lost,
            geo_gained_since_last_frame=self.geo_gained,
            geo_spent_since_last_frame=self.geo_spent,
            semantic_labels=self.semantic_labels,
            hero_motion=(0.25, -0.5, True) if self.semantic_labels is not None else None,
            interact_prompt=self.semantic_labels is not None,
            interact_prompt_valid=self.semantic_labels is not None,
        )

    def reset(self, *, seed: int | None, options: Mapping[str, Any] | None) -> ResetResult:
        del seed, options
        self.bad_frame = None
        self.episode_id = f"episode-{self.frame_id + 1}"
        return ResetResult(episode_id=self.episode_id, frame=self._frame())

    def step(self, *, action_id: int, buttons: tuple[str, ...], lease_ms: int) -> BackendStep:
        self.steps.append((action_id, buttons, lease_ms))
        return BackendStep(
            frame=self._frame(action_id),
            applied_action=AppliedAction(action_id, buttons, buttons, lease_ms),
            terminated=self.end == "player_death",
            truncated=self.end == "capture_failed",
            termination_reason=self.end,
            timing_ms={"backend": 1.25},
        )

    def release_all(self) -> None:
        self.releases += 1

    def close(self) -> None:
        self.closes += 1


def make_env(backend: FakeBackend, tmp_path: Path, **kwargs: Any) -> HollowKnightEnv:
    return HollowKnightEnv(
        backend,
        frame_height=2,
        frame_width=3,
        frame_stack=3,
        lease_ms=40,
        log_root=tmp_path,
        **kwargs,
    )


def test_env_checker_and_observation_contract(tmp_path: Path) -> None:
    backend = FakeBackend()
    env = make_env(backend, tmp_path)
    check_env(env, skip_render_check=True)
    observation, info = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    assert observation["pixels"].shape == (3, 2, 3, 3)
    assert observation["pixels"].dtype == np.uint8
    assert observation["action_history"].tolist() == [0, 0]
    assert info == {}
    env.close()


def test_frame_validation_preserves_opt_in_training_labels(tmp_path: Path) -> None:
    backend = FakeBackend()
    backend.semantic_labels = np.full((27, 48), 255, dtype=np.uint8)
    backend.semantic_labels[12, 20] = 5
    env = make_env(backend, tmp_path)

    env.reset()

    np.testing.assert_array_equal(env._frames[-1].semantic_labels, backend.semantic_labels)
    assert env._frames[-1].hero_motion == (0.25, -0.5, True)
    assert env._frames[-1].interact_prompt is True
    assert env._frames[-1].interact_prompt_valid is True
    env.close()


def test_step_correlates_frame_stacks_history_and_writes_transition(tmp_path: Path) -> None:
    backend = FakeBackend()
    env = make_env(backend, tmp_path)
    before, _ = env.reset()
    observation, reward, terminated, truncated, info = env.step(ACTION_NAMES.index("JUMP"))

    assert reward == 0.0
    assert not terminated and not truncated
    assert info["valid"] is True
    assert backend.steps == [(1, ("JUMP",), 40)]
    assert observation["action_history"].tolist() == [0, ACTION_NAMES.index("JUMP")]
    assert observation["pixels"][0].tolist() == before["pixels"][1].tolist()

    path = tmp_path / "instances" / "hk-00" / "episodes" / "episode-1" / "transitions.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["obs_before"]["frame_ids"] == [1, 1, 1]
    assert records[0]["obs_after"]["frame_ids"] == [1, 1, 2]
    assert records[0]["requested_action"]["name"] == "JUMP"
    assert records[0]["reward"] == {"total": 0.0, "components": {}}
    assert records[0]["valid"] is True
    env.close()


def test_long_bounded_lease_supports_focus_and_charged_actions(tmp_path: Path) -> None:
    backend = FakeBackend()
    env = make_env(backend, tmp_path)
    env.reset()
    env.step(ACTION_NAMES.index("CAST"), lease_ms=1_500)
    assert backend.steps == [(1, ("CAST",), 1_500)]

    with pytest.raises(ValueError, match="between 1 and 2000"):
        env.step(ACTION_NAMES.index("CAST"), lease_ms=2_001)
    env.close()


def test_combat_signals_are_reward_metadata_not_policy_observations(tmp_path: Path) -> None:
    backend = FakeBackend()
    env = make_env(backend, tmp_path)
    observation, _ = env.reset()
    backend.enemy_kills = 1
    backend.hp_lost = 2
    backend.geo_gained = 11
    backend.geo_spent = 3
    after, _, _, _, info = env.step(0)
    assert set(after) == set(observation) == {"pixels", "action_history"}
    assert info["enemy_kills_since_last_frame"] == 1
    assert info["hp_lost_since_last_frame"] == 2
    assert info["geo_gained_since_last_frame"] == 11
    assert info["geo_spent_since_last_frame"] == 3
    assert info["supervisor"]["enemy_kills_since_last_frame"] == 1
    assert info["supervisor"]["geo_gained_since_last_frame"] == 11
    assert info["supervisor"]["geo_spent_since_last_frame"] == 3
    env.close()


@pytest.mark.parametrize("bad_frame", ["duplicate", "uncorrelated", "shape"])
def test_invalid_frame_truncates_releases_and_is_not_training_data(
    tmp_path: Path, bad_frame: str
) -> None:
    backend = FakeBackend()
    env = make_env(backend, tmp_path)
    observation, _ = env.reset()
    backend.bad_frame = bad_frame
    after, reward, terminated, truncated, info = env.step(1)

    assert reward == 0.0
    assert not terminated and truncated
    assert info == {
        "termination_reason": "operational_failure",
        "valid": False,
    }
    assert np.array_equal(after["pixels"], observation["pixels"])
    assert backend.releases == 1
    assert not list(tmp_path.rglob("transitions.jsonl"))
    operation = json.loads(
        (tmp_path / "instances" / "hk-00" / "operations.jsonl").read_text(encoding="utf-8")
    )
    assert operation["event"] == "transition_rejected"
    with pytest.raises(RuntimeError, match=r"reset\(\)"):
        env.step(1)
    env.close()


@pytest.mark.parametrize(
    ("backend_end", "expected_terminated", "expected_truncated"),
    [("player_death", True, False), ("capture_failed", False, True)],
)
def test_backend_end_semantics(
    tmp_path: Path,
    backend_end: str,
    expected_terminated: bool,
    expected_truncated: bool,
) -> None:
    backend = FakeBackend()
    env = make_env(backend, tmp_path)
    env.reset()
    backend.end = backend_end
    _, _, terminated, truncated, info = env.step(0)
    assert terminated is expected_terminated
    assert truncated is expected_truncated
    assert info["termination_reason"] == backend_end
    env.close()


def test_local_time_limit_is_truncation(tmp_path: Path) -> None:
    backend = FakeBackend()
    env = make_env(backend, tmp_path, max_episode_steps=1)
    env.reset()
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == 0.0
    assert not terminated and truncated
    assert info["termination_reason"] == "time_limit"
    env.close()


def test_close_releases_and_closes_once_even_without_reset(tmp_path: Path) -> None:
    backend = FakeBackend()
    env = make_env(backend, tmp_path)
    env.close()
    env.close()
    assert backend.releases == 1
    assert backend.closes == 1


def test_observation_is_detached_from_backend_buffer(tmp_path: Path) -> None:
    backend = FakeBackend()
    env = make_env(backend, tmp_path)
    observation, _ = env.reset()
    observation["pixels"].fill(99)
    next_observation, *_ = env.step(0)
    assert np.all(next_observation["pixels"][0] == 7)
    env.close()


def test_step_exposes_backend_and_total_timing_in_info(tmp_path: Path) -> None:
    env = make_env(FakeBackend(), tmp_path)
    env.reset()

    *_, info = env.step(0)

    assert info["timing_ms"]["backend"] == pytest.approx(1.25)
    assert info["timing_ms"]["env_step"] >= 0
    env.close()


def test_action_must_be_in_discrete_space(tmp_path: Path) -> None:
    env = make_env(FakeBackend(), tmp_path)
    env.reset()
    with pytest.raises(ValueError, match="invalid action"):
        env.step(len(ACTION_NAMES))
    env.close()


def test_raw_action_bitmask_supports_combinations_and_normalizes_opposites(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    env = make_env(backend, tmp_path)
    env.reset()
    right_jump = (1 << 1) | (1 << 4)
    env.step(right_jump)
    assert list(backend.steps[-1][1]) == ["RIGHT", "JUMP"]

    env.reset()
    left_right_attack = (1 << 0) | (1 << 1) | (1 << 5)
    env.step(left_right_attack)
    assert list(backend.steps[-1][1]) == ["ATTACK"]
    env.close()


def test_jsonl_rejects_non_finite_timing(tmp_path: Path) -> None:
    class BadTimingBackend(FakeBackend):
        def step(self, *, action_id: int, buttons: tuple[str, ...], lease_ms: int) -> BackendStep:
            result = super().step(action_id=action_id, buttons=buttons, lease_ms=lease_ms)
            return BackendStep(
                frame=result.frame,
                applied_action=result.applied_action,
                timing_ms={"bad": float("nan")},
            )

    env = make_env(BadTimingBackend(), tmp_path)
    env.reset()
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == 0.0
    assert not terminated and truncated
    assert info["valid"] is False
    assert not list(tmp_path.rglob("transitions.jsonl"))
    env.close()


@pytest.mark.parametrize(
    "bad_frame",
    ["duplicate", "uncorrelated", "shape", "repeated_id", "old_timestamp", "transition"],
)
def test_all_invalid_frame_classes_are_rejected(tmp_path: Path, bad_frame: str) -> None:
    class ExtendedBadFrameBackend(FakeBackend):
        def _frame(self, last_action: int | None = None) -> PixelFrame:
            frame = super()._frame(last_action)
            if self.bad_frame == "repeated_id":
                return PixelFrame(
                    episode_id=frame.episode_id,
                    frame_id=frame.frame_id - 1,
                    capture_monotonic_ns=frame.capture_monotonic_ns,
                    pixels=frame.pixels,
                    last_applied_action_id=last_action,
                )
            if self.bad_frame == "old_timestamp":
                return PixelFrame(
                    episode_id=frame.episode_id,
                    frame_id=frame.frame_id,
                    capture_monotonic_ns=0,
                    pixels=frame.pixels,
                    last_applied_action_id=last_action,
                )
            if self.bad_frame == "transition":
                return PixelFrame(
                    episode_id=frame.episode_id,
                    frame_id=frame.frame_id,
                    capture_monotonic_ns=frame.capture_monotonic_ns,
                    pixels=frame.pixels,
                    scene_transition=True,
                    last_applied_action_id=last_action,
                )
            return frame

    backend = ExtendedBadFrameBackend()
    env = make_env(backend, tmp_path)
    env.reset()
    backend.bad_frame = bad_frame
    _, _, terminated, truncated, info = env.step(1)
    assert not terminated and truncated
    assert info["valid"] is False
    assert backend.releases == 1
    env.close()


@pytest.mark.parametrize("defect", ["action_id", "requested", "unknown", "duplicate", "lease"])
def test_invalid_applied_action_is_rejected(tmp_path: Path, defect: str) -> None:
    class BadActionBackend(FakeBackend):
        def step(self, *, action_id: int, buttons: tuple[str, ...], lease_ms: int) -> BackendStep:
            result = super().step(action_id=action_id, buttons=buttons, lease_ms=lease_ms)
            applied = result.applied_action
            if defect == "action_id":
                applied = AppliedAction(99, buttons, buttons, lease_ms)
            elif defect == "requested":
                applied = AppliedAction(action_id, ("RIGHT",), buttons, lease_ms)
            elif defect == "unknown":
                applied = AppliedAction(action_id, buttons, ("PAUSE",), lease_ms)
            elif defect == "duplicate":
                applied = AppliedAction(action_id, buttons, ("LEFT", "LEFT"), lease_ms)
            elif defect == "lease":
                applied = AppliedAction(action_id, buttons, buttons, 0)
            return BackendStep(frame=result.frame, applied_action=applied)

    backend = BadActionBackend()
    env = make_env(backend, tmp_path)
    env.reset()
    _, _, terminated, truncated, info = env.step(1)
    assert not terminated and truncated
    assert info["valid"] is False
    assert backend.releases == 1
    env.close()


def test_invalid_reset_frame_releases_and_raises(tmp_path: Path) -> None:
    class BadResetBackend(FakeBackend):
        def reset(self, *, seed: int | None, options: Mapping[str, Any] | None) -> ResetResult:
            result = super().reset(seed=seed, options=options)
            return ResetResult(
                episode_id=result.episode_id,
                frame=PixelFrame(result.episode_id, 1, 1, np.zeros((1, 1, 3), dtype=np.uint8)),
            )

    backend = BadResetBackend()
    env = make_env(backend, tmp_path)
    with pytest.raises(Exception, match="pixels must"):
        env.reset()
    assert backend.releases == 1
    env.close()


def test_close_still_closes_backend_when_release_fails(tmp_path: Path) -> None:
    class ReleaseFailureBackend(FakeBackend):
        def release_all(self) -> None:
            self.releases += 1
            raise RuntimeError("release failed")

    backend = ReleaseFailureBackend()
    env = make_env(backend, tmp_path)
    with pytest.raises(RuntimeError, match="release failed"):
        env.close()
    assert backend.releases == 1
    assert backend.closes == 1
    env.close()


def test_frame_from_another_episode_is_rejected(tmp_path: Path) -> None:
    class WrongEpisodeBackend(FakeBackend):
        def step(self, *, action_id: int, buttons: tuple[str, ...], lease_ms: int) -> BackendStep:
            result = super().step(action_id=action_id, buttons=buttons, lease_ms=lease_ms)
            frame = result.frame
            return BackendStep(
                frame=PixelFrame(
                    "wrong-episode",
                    frame.frame_id,
                    frame.capture_monotonic_ns,
                    frame.pixels,
                    last_applied_action_id=action_id,
                ),
                applied_action=result.applied_action,
            )

    backend = WrongEpisodeBackend()
    env = make_env(backend, tmp_path)
    env.reset()
    _, _, terminated, truncated, info = env.step(1)
    assert not terminated and truncated
    assert info["valid"] is False
    env.close()


def test_reset_rejects_reused_episode_id(tmp_path: Path) -> None:
    class ReusedEpisodeBackend(FakeBackend):
        def reset(self, *, seed: int | None, options: Mapping[str, Any] | None) -> ResetResult:
            result = super().reset(seed=seed, options=options)
            self.episode_id = "fixed"
            return ResetResult(
                "fixed",
                PixelFrame(
                    "fixed",
                    result.frame.frame_id,
                    result.frame.capture_monotonic_ns,
                    result.frame.pixels,
                ),
            )

    backend = ReusedEpisodeBackend()
    env = make_env(backend, tmp_path)
    env.reset()
    with pytest.raises(ValueError, match="reused"):
        env.reset()
    assert backend.releases >= 2
    with pytest.raises(RuntimeError, match=r"reset\(\) is required"):
        env.step(0)
    env.close()


def test_transition_log_failure_truncates_before_observation_advances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend()
    env = make_env(backend, tmp_path)
    before, _ = env.reset()

    def fail_log(**_: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(env, "_log_transition", fail_log)
    after, reward, terminated, truncated, info = env.step(1)
    assert reward == 0.0
    assert not terminated and truncated
    assert info["termination_reason"] == "transition_log_failure"
    assert info["valid"] is False
    assert np.array_equal(after["pixels"], before["pixels"])
    assert backend.releases == 1
    env.close()



def test_extended_length_path_is_a_noop_off_windows_and_idempotent() -> None:
    from hk_agent.gym_env import extended_length_path

    assert extended_length_path("/tmp/x", os_name="posix") == "/tmp/x"
    prefixed = "\\\\?\\C:\\already\\long"
    assert extended_length_path(prefixed, os_name="nt") == prefixed


@pytest.mark.skipif(os.name != "nt", reason="Windows MAX_PATH behaviour")
def test_extended_length_path_prefixes_drive_and_unc_paths() -> None:
    from hk_agent.gym_env import extended_length_path

    assert extended_length_path("C:\\data\\file.jsonl") == "\\\\?\\C:\\data\\file.jsonl"
    assert extended_length_path("\\\\server\\share\\f.jsonl") == "\\\\?\\UNC\\server\\share\\f.jsonl"


@pytest.mark.skipif(os.name != "nt", reason="Windows MAX_PATH behaviour")
def test_jsonl_writer_survives_paths_beyond_max_path(tmp_path: Path) -> None:
    """Regression: per-launch runtime roots made the experience tree exceed
    MAX_PATH (WinError 206) and every transition was rejected."""
    from hk_agent.gym_env import AtomicJsonlWriter

    deep = tmp_path
    for index in range(8):
        deep = deep / f"episode-51042e97-4d58-4dab-a796-b5502883b9cf-{index}"
    target = deep / "transitions.jsonl"
    assert len(str(target)) > 300
    writer = AtomicJsonlWriter(target)
    writer.append({"step": 1})
    writer.append({"step": 2})
    with open("\\\\?\\" + str(target), encoding="utf-8") as stream:
        assert [line for line in stream.read().splitlines() if line] == [
            '{"step":1}',
            '{"step":2}',
        ]
