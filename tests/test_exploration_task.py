from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from numpy.typing import NDArray

import hk_agent.exploration_task as exploration_task_module
from hk_agent.action_masking import ActionAvailability
from hk_agent.distributed_knowledge import portable_compatibility_fingerprint
from hk_agent.exploration_task import (
    ExplorationTask,
    ExplorationTaskConfig,
    calibrate_novelty,
    load_lifelong_visits,
)
from hk_agent.player_knowledge import PlayerKnowledge
from hk_agent.scene_change_task import scene_descriptor


class _Backend:
    def release_all(self) -> None:
        pass


class _Environment(gym.Env):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Dict(
            {
                "pixels": gym.spaces.Box(0, 255, (4, 90, 160, 3), dtype=np.uint8),
                "action_history": gym.spaces.MultiDiscrete(np.full((8,), 27, dtype=np.int64)),
            }
        )
        self.action_space = gym.spaces.Discrete(30)
        self.lease_ms = 80
        self.latest_action_availability = None
        self.backend = _Backend()
        self.frame = self._frame(0)
        self.terminal = False
        self.truncated = False
        self.extra = {}
        self.initial_extra = {}
        self.initial_scene = "Room-A"

    @staticmethod
    def _frame(shift: int) -> NDArray[np.uint8]:
        y, x = np.indices((90, 160))
        pattern = ((x // 8 + y // 8 + shift) % 2 * 120 + 60).astype(np.uint8)
        return np.stack((pattern, np.roll(pattern, shift * 10, axis=1), 180 - pattern), axis=-1)

    def _obs(self) -> dict[str, NDArray[np.generic]]:
        return {
            "pixels": np.repeat(self.frame[None, ...], 4, axis=0),
            "action_history": np.zeros((8,), dtype=np.int64),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.frame = self._frame(0)
        self.terminal = False
        self.truncated = False
        return self._obs(), {"scene_name": self.initial_scene, **self.initial_extra}

    def step(self, action: int, *, lease_ms: int):
        if action == 1:
            self.frame = self._frame(1)
        elif action == 2:
            self.frame = self._frame(3)
        terminal = self.terminal
        info = {
            "valid": True,
            "termination_reason": "player_dead" if terminal else None,
            "enemy_kills_since_last_frame": 1 if terminal else 0,
            "hp_lost_since_last_frame": 0,
            "geo_gained_since_last_frame": 10,
            "geo_spent_since_last_frame": 0,
            "shade_hits_since_last_frame": 0,
            "hp_restored_since_last_frame": 0,
            "supervisor": {"player_dead": terminal},
            **self.extra,
        }
        return self._obs(), 0.0, terminal, self.truncated, info


def test_noop_static_has_no_novelty_and_camera_shift_does() -> None:
    environment = _Environment()
    task = ExplorationTask(environment, config=ExplorationTaskConfig(horizon_steps=20))
    task.reset()
    _, noop_reward, _, _, noop_info = task.step(0)
    assert noop_info["reward_components"]["novelty"] == 0
    _, active_reward, _, _, active_info = task.step(1)
    assert active_info["reward_components"]["novelty"] == 0.05
    assert active_reward > noop_reward


def test_novelty_budget_and_archive_are_bounded() -> None:
    environment = _Environment()
    config = ExplorationTaskConfig(horizon_steps=20, novelty_budget=0.05, archive_max=1)
    task = ExplorationTask(environment, config=config)
    _, _ = task.reset()
    _, _, _, _, info = task.step(1)
    assert info["reward_components"]["novelty"] == 0.05
    assert info["archive_size"] == 1
    _, _, _, _, repeated = task.step(1)
    assert repeated["reward_components"]["novelty"] == 0


def test_fifo_archive_forgets_old_descriptors() -> None:
    environment = _Environment()
    task = ExplorationTask(
        environment, config=ExplorationTaskConfig(horizon_steps=20, archive_max=1)
    )
    task.reset()
    _, _, _, _, first = task.step(1)
    _, _, _, _, second = task.step(2)
    _, _, _, _, revisit = task.step(1)
    assert first["reward_components"]["novelty"] == 0.05
    assert second["reward_components"]["novelty"] == 0.05
    assert revisit["reward_components"]["novelty"] == 0.05
    assert revisit["archive_size"] == 1


def test_stable_scene_change_pays_once_without_terminating() -> None:
    environment = _Environment()
    task = ExplorationTask(environment, config=ExplorationTaskConfig(horizon_steps=20))
    task.reset()
    environment.frame = np.zeros((90, 160, 3), dtype=np.uint8)
    task.step(0)
    environment.frame = np.zeros((90, 160, 3), dtype=np.uint8)
    task.step(0)
    events = []
    for _ in range(4):
        environment.frame = _Environment._frame(1)
        events.append(task.step(0))
    paid = [item[4]["reward_components"]["scene_change"] for item in events]
    assert 5.0 in paid
    assert sum(value > 0 for value in paid) == 1
    assert all(not item[2] and not item[3] for item in events)


def test_death_suppresses_positive_rewards_and_terminates() -> None:
    environment = _Environment()
    task = ExplorationTask(environment)
    task.reset()
    environment.terminal = True
    _, reward, terminated, truncated, info = task.step(1)
    assert terminated and not truncated
    assert info["reward_components"]["death"] == -5
    assert info["reward_components"]["novelty"] == 0
    assert info["reward_components"]["scene_change"] == 0
    assert info["reward_components"]["confirmed_enemy_kills"] == 0
    assert reward < 0


def test_base_truncation_suppresses_positive_rewards() -> None:
    environment = _Environment()
    task = ExplorationTask(environment)
    task.reset()
    environment.truncated = True
    _, _, terminated, truncated, info = task.step(1)
    assert not terminated and truncated
    assert info["reward_components"]["novelty"] == 0
    assert info["reward_components"]["scene_change"] == 0


def test_black_and_white_frames_do_not_pay_novelty() -> None:
    for value in (0, 255):
        environment = _Environment()
        task = ExplorationTask(environment)
        task.reset()
        environment.frame = np.full((90, 160, 3), value, dtype=np.uint8)
        _, _, _, _, info = task.step(0)
        assert info["reward_components"]["novelty"] == 0


def test_unavailable_action_is_masked_and_audited() -> None:
    environment = _Environment()
    environment.latest_action_availability = ActionAvailability(
        mp_charge=0,
        spell_cost=33,
        focus_cost=33,
        fireball_level=0,
        quake_level=0,
        scream_level=0,
        has_nail_art=False,
        has_cyclone=False,
        has_dash_slash=False,
        has_upward_slash=False,
    )
    task = ExplorationTask(environment)
    task.reset()
    index = __import__(
        "hk_agent.scene_change_task", fromlist=["SCENE_CHANGE_ACTION_NAMES"]
    ).SCENE_CHANGE_ACTION_NAMES.index("FOCUS_HOLD")
    _, _, _, _, info = task.step(index)
    assert info["masked_action_substituted"] is True
    assert info["executed_task_action"] == 0
    assert index in info["action_mask"]


def test_novelty_calibration_requires_noop_floor_and_active_motion() -> None:
    base = np.zeros((144,), dtype=np.float32)
    noop = [base.copy() for _ in range(10)]
    active = [np.full((144,), 0.1 + index * 0.001, dtype=np.float32) for index in range(10)]
    threshold = calibrate_novelty(noop, active)
    assert threshold == 0.02
    try:
        calibrate_novelty(noop, [base.copy() for _ in range(10)])
    except ValueError as error:
        assert "active exploration" in str(error)
    else:
        raise AssertionError("uninformative active calibration must fail")


def test_exploration_contract_is_pixel_only_and_fingerprint_stable() -> None:
    from hk_agent.distributed_knowledge import _public_task_contract

    task = ExplorationTask(_Environment())
    contract = task.task_contract()
    assert contract["reward_version"] == "explore_v1"
    assert contract["information_boundary"] == ["pixels", "applied_action_history"]
    assert contract["privileged_state"] == []
    assert _public_task_contract(contract, from_manifest=False)["reward_version"] == "explore_v1"
    assert portable_compatibility_fingerprint(contract, {}) == portable_compatibility_fingerprint(
        contract, {}
    )
    assert scene_descriptor(
        _Environment._frame(0),
        __import__(
            "hk_agent.scene_change_task", fromlist=["SceneChangeTaskConfig"]
        ).SceneChangeTaskConfig(),
    ).shape == (144, 3)


def _v2_config(**kwargs):
    return ExplorationTaskConfig(
        reward_version="explore_v2",
        geo_gain_reward=0,
        geo_spend_reward=0,
        shade_recovery_reward=4.0,
        **kwargs,
    )


def _v3_config(**kwargs):
    return ExplorationTaskConfig(
        reward_version="explore_v3",
        geo_gain_reward=0,
        geo_spend_reward=0,
        shade_recovery_reward=4.0,
        **kwargs,
    )


def _v4_config(**kwargs):
    return ExplorationTaskConfig(
        reward_version="explore_v4",
        geo_gain_reward=0,
        geo_spend_reward=0,
        shade_recovery_reward=4.0,
        **kwargs,
    )


def test_v4_confirmed_milestone_rewards_decay_and_persist(tmp_path: Path) -> None:
    from hk_agent.exploration_task import load_campaign_counts

    memory = tmp_path / "campaign-memory"
    environment = _Environment()
    environment.extra = {
        "milestone_events_since_last_frame": [
            {"category": "boss", "key": "False_Knight", "confirmed": True},
            {"category": "grub", "key": "grub_1", "confirmed": True},
        ]
    }
    task = ExplorationTask(environment, config=_v4_config(), lifelong_memory_root=memory)
    task.reset()
    _, reward_first, _, _, info_first = task.step(0)
    assert info_first["reward_components"]["milestone_novelty"] == pytest.approx(18.0)
    assert reward_first > 0
    first_counts = load_campaign_counts(memory / "lifelong-milestones.json")
    assert first_counts == {"boss:False_Knight": 1, "grub:grub_1": 1}

    repeated = _Environment()
    repeated.extra = environment.extra
    task2 = ExplorationTask(repeated, config=_v4_config(), lifelong_memory_root=memory)
    task2.reset()
    _, _, _, _, info_repeated = task2.step(0)
    expected = 15 / (2**0.5) + 3 / (2**0.5)
    assert info_repeated["reward_components"]["milestone_novelty"] == pytest.approx(expected)


def test_v4_first_contact_and_story_tutorial_rewards_are_one_time() -> None:
    environment = _Environment()
    environment.extra = {"operator_telemetry": {
        "scene_name": "Crossroads_01", "scene_bounds": [0, 0, 16, 9], "x": 4, "y": 4,
        "at_bench": True,
        "progress_flags": {"enteredTutorialFirstTime": True},
        "visible_places": [{"kind": "bench", "world_x": 4, "world_y": 4}],
        "milestone_events_since_last_frame": [
            {"category": "flag", "key": "openedCrossroads"},
            {"category": "scene_transition", "transition_kind": "stag"},
        ],
    }}
    task = ExplorationTask(environment, config=_v4_config())
    task.reset()
    observation, _, _, _, first = task.step(0)
    assert first["reward_components"]["tutorial_reward"] == pytest.approx(4.0)
    assert task.observation_space.contains(observation)
    _, _, _, _, repeated = task.step(0)
    assert repeated["reward_components"]["tutorial_reward"] == 0
    assert repeated["player_knowledge_events"] == []


def test_v4_contract_adds_player_knowledge_and_keeps_raw_telemetry_out() -> None:
    environment = _Environment()
    environment.extra = {"operator_telemetry": {"scene_name": "Room-A", "x": 4.0, "y": 8.0}}
    task = ExplorationTask(environment, config=_v4_config())
    contract = task.task_contract()
    assert contract["reward_version"] == "explore_v4"
    assert contract["task_contract_schema_version"] == 8
    assert contract["observation_version"] == "pixels_action_history_player_knowledge_charms_v1"
    assert contract["information_boundary"] == ["pixels", "applied_action_history", "player_knowledge"]
    assert "milestone_events_since_last_frame" in contract["reward_only_signals"]
    assert task.config.maximum_positive_episode_reward() <= 75.2
    observation, _ = task.reset()
    observation, _, _, _, info = task.step(0)
    assert "operator_telemetry" in info
    assert task.observation_space.contains(observation)
    assert set(observation) == {"pixels", "action_history", "player_knowledge"}


def test_v4_novelty_key_changes_when_player_learns_an_ability() -> None:
    from hk_agent.exploration_task import _ability_aware_novelty_hash
    from hk_agent.player_knowledge import PlayerKnowledge

    descriptor = np.zeros(16, dtype=np.float32)
    player = PlayerKnowledge()
    before = _ability_aware_novelty_hash(descriptor, player)
    player.known_flags["hasDash"] = True
    after = _ability_aware_novelty_hash(descriptor, player)
    assert before != after


def test_initial_scene_is_captured_for_map_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    journey_root = tmp_path / "journey"
    monkeypatch.setenv("HK_AGENT_JOURNEY_ROOT", str(journey_root))
    environment = _Environment()
    environment.backend.capture_png = lambda: b"\x89PNG\r\n\x1a\nsynthetic"
    environment.initial_extra = {
        "operator_telemetry": {
            "scene_name": "Room-A",
            "x": 4.0,
            "y": 8.0,
            "camera_x": 3.0,
            "camera_y": 7.0,
            "scene_bounds": [0.0, 0.0, 20.0, 15.0],
        }
    }
    task = ExplorationTask(environment, config=_v4_config())
    task.reset()
    progress = json.loads((journey_root / "campaign-progress.json").read_text("utf-8"))
    initial = next(row for row in progress["milestones"] if row["category"] == "scene")
    assert initial["scene"] == "Room-A"
    assert initial["map_view"]["scene_bounds"] == [0.0, 0.0, 20.0, 15.0]
    episode = json.loads((journey_root / "actor-0" / "episode-1.json").read_text("utf-8"))
    assert episode["positions"][0]["x"] == 4.0
    assert Path(initial["capture"]).name.endswith("scene_Room-A.png")
    assert (journey_root / initial["capture"]).is_file()


def test_diary_records_confirmed_damage_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    journey_root = tmp_path / "journey"
    monkeypatch.setenv("HK_AGENT_JOURNEY_ROOT", str(journey_root))
    environment = _Environment()
    environment.extra = {"hp_lost_since_last_frame": 1}
    task = ExplorationTask(environment, config=_v4_config())
    task.reset()
    task.step(0)
    episode = json.loads((journey_root / "actor-0" / "episode-1.json").read_text("utf-8"))
    assert episode["events"][0]["category"] == "damage"
    assert episode["events"][0]["confirmed"] is True


def test_v3_scene_and_enemy_rewards_decay_and_persist(tmp_path: Path) -> None:
    from hk_agent.exploration_task import load_campaign_counts

    memory = tmp_path / "actor-memory"
    expected_scene = (10.0, 2.0 + 8.0 / np.sqrt(2))
    expected_enemy = (1.0, 1.0 / np.sqrt(2))
    for scene_reward, enemy_reward in zip(expected_scene, expected_enemy, strict=True):
        env = _Environment()
        env.extra = {
            "scene_name": "Room-B",
            "enemy_events_since_last_frame": [{"scene_name": "Room-B", "enemy_type": "Crawler"}],
        }
        task = ExplorationTask(env, config=_v3_config(), lifelong_memory_root=memory)
        task.reset()
        _, _, _, _, info = task.step(0)
        assert info["reward_components"]["scene_change"] == pytest.approx(scene_reward)
        assert info["reward_components"]["confirmed_enemy_kills"] == pytest.approx(enemy_reward)
        task.close()
    assert load_campaign_counts(memory / "lifelong-scenes.json")["Room-B"] == 2
    assert load_campaign_counts(memory / "lifelong-enemies.json")["Room-B|Crawler"] == 2


def test_v3_stuck_truncates_after_threshold_but_progress_resets_timer(tmp_path: Path) -> None:
    env = _Environment()
    task = ExplorationTask(
        env,
        config=_v3_config(stuck_seconds=0.12, horizon_steps=30),
        lifelong_memory_root=tmp_path / "memory",
    )
    task.reset()
    _, _, _, truncated, _first = task.step(0)
    assert not truncated
    env.extra = {"scene_name": "Room-B"}
    _, _, _, truncated, room = task.step(0)
    assert not truncated
    assert room["reward_components"]["scene_change"] > 0
    env.extra = {"scene_name": "Room-B"}
    _, _, _, truncated, _next_step = task.step(0)
    assert not truncated
    assert task._progress_elapsed_seconds == pytest.approx(0.08)
    _, _, _, truncated, final = task.step(0)
    assert truncated
    assert final["termination_reason"] == "stuck"
    assert final["episode_progress"]["termination_reason"] == "stuck"


def test_v3_alert_thresholds_and_return_bounds() -> None:
    task = ExplorationTask(_Environment(), config=_v3_config(horizon_steps=3000))
    assert task.config.maximum_positive_episode_reward() == pytest.approx(55.2)
    assert task.config.minimum_negative_episode_reward() == pytest.approx(-35.5)
    for _ in range(1999):
        assert task._record_farm_window({"novelty": 0.1}, "A", "LEFT") == []
    alerts = task._record_farm_window({"novelty": 0.1}, "A", "LEFT")
    assert "component_over_60_percent" in alerts
    assert "farm_component:novelty" in alerts
    task = ExplorationTask(_Environment(), config=_v3_config(horizon_steps=1000))
    alerts = []
    for index in range(2000):
        scene, action = ("A", "LEFT") if index < 1200 else (f"scene-{index}", "RIGHT")
        alerts.extend(task._record_farm_window({}, scene, action))
    assert "same_scene_action_over_40_percent_without_novelty" in alerts


def test_continuous_reward_budget_survives_slice_and_expires_from_rolling_window() -> None:
    task = object.__new__(ExplorationTask)
    task.config = ExplorationTaskConfig(reward_version="explore_v4", continuous_playthrough=True)
    task._continuous_budget_state = {
        "schema_version": 1,
        "game_steps": 64,
        "segment_paid": {"confirmed_enemy_kills": 4.8},
        "buckets": {"1": {"confirmed_enemy_kills": 4.8}},
    }
    task._pending_continuous_budget = {}
    task._continuous_budget_dirty_steps = 0
    task._continuous_budget_path = None
    assert task._remaining_budget("confirmed_enemy_kills", 5.0, 0.0) == pytest.approx(0.2)
    task._continuous_budget_state["segment_paid"] = {}
    task._continuous_budget_state["game_steps"] = 16_000
    assert task._remaining_budget("confirmed_enemy_kills", 5.0, 0.0) == pytest.approx(5.0)


def test_continuous_enemy_kill_reward_decays_without_episode_floor() -> None:
    from hk_agent.exploration_task import enemy_kill_reward_for_visit

    rewards = [enemy_kill_reward_for_visit(visits, continuous=True) for visits in (0, 1, 5, 10)]
    assert rewards == pytest.approx([1.0, 0.25, 1 / 36, 1 / 121])
    assert rewards[-1] < 0.01


def test_continuous_playthrough_flag_does_not_change_shared_task_contract() -> None:
    common = {
        "reward_version": "explore_v4",
        "geo_gain_reward": 0.0,
        "geo_spend_reward": 0.0,
        "shade_recovery_reward": 4.0,
        "shade_hit_reward": 0.3,
    }
    regular = ExplorationTask(
        _Environment(), config=ExplorationTaskConfig(**common)
    )
    continuous = ExplorationTask(
        _Environment(),
        config=ExplorationTaskConfig(**common, continuous_playthrough=True),
    )
    assert regular.task_contract() == continuous.task_contract()


def test_v3_caps_damage_and_shade_components_per_episode(tmp_path: Path) -> None:
    env = _Environment()
    env.extra = {
        "scene_name": "Room-A",
        "shade_pending": True,
        "shade_distance": 100.0,
        "shade_hits_since_last_frame": 4,
        "hp_lost_since_last_frame": 9,
    }
    task = ExplorationTask(
        env, config=_v3_config(horizon_steps=40), lifelong_memory_root=tmp_path / "memory"
    )
    task.reset()
    components = []
    for distance in list(range(99, 79, -1)) + list(range(80, 100)):
        env.extra = {
            "scene_name": "Room-A",
            "shade_pending": True,
            "shade_distance": float(distance),
            "shade_hits_since_last_frame": 4,
            "hp_lost_since_last_frame": 9,
        }
        components.append(task.step(0)[4]["reward_components"])
    assert sum(item["shade_hits"] for item in components) == pytest.approx(1.2)
    assert sum(item["damage_taken"] for item in components) == pytest.approx(-4.5)
    assert sum(max(0.0, item["shade_approach"]) for item in components) <= 1.0
    assert sum(min(0.0, item["shade_approach"]) for item in components) >= -1.0


def test_v3_rejects_reward_budgets_that_break_theoretical_bound() -> None:
    with pytest.raises(ValueError, match="fixed by the reward version"):
        _v3_config(time_cost=-0.01).validate()
    with pytest.raises(ValueError, match="stable episode-return contract"):
        _v3_config(scene_reward_budget=100).validate()


def test_v3_time_cost_has_an_episode_cap(tmp_path: Path) -> None:
    task = ExplorationTask(
        _Environment(),
        config=_v3_config(time_cost_budget=0.02, horizon_steps=50),
        lifelong_memory_root=tmp_path / "memory",
    )
    task.reset()
    paid = sum(task.step(0)[4]["reward_components"]["time_cost"] for _ in range(50))
    assert paid == pytest.approx(-0.02)


def test_v3_contract_exposes_progress_signals_only_to_reward(tmp_path: Path) -> None:
    from hk_agent.distributed_knowledge import _public_task_contract

    task = ExplorationTask(
        _Environment(), config=_v3_config(), lifelong_memory_root=tmp_path / "memory"
    )
    observation, _ = task.reset()
    contract = task.task_contract()
    assert contract["task_contract_schema_version"] == 3
    assert "scene_name" in contract["reward_only_signals"]
    assert "enemy_events_since_last_frame" in contract["reward_only_signals"]
    assert "scene_name" not in observation
    assert _public_task_contract(contract, from_manifest=False)["reward_version"] == "explore_v3"


def test_v2_masks_hud_and_requires_persistent_scene_change(tmp_path: Path) -> None:
    env = _Environment()
    task = ExplorationTask(
        env, config=_v2_config(), task_log_path=tmp_path / "phase" / "task.jsonl"
    )
    task.reset()
    base = env.frame.copy()
    hud = base.copy()
    hud[:27, :64] = 255 - hud[:27, :64]
    env.frame = hud
    _, _, _, _, info = task.step(0)
    assert info["reward_components"]["novelty"] == 0
    env.frame = _Environment._frame(1)
    _, _, _, _, flash = task.step(0)
    assert flash["reward_components"]["novelty"] == 0
    env.frame = base
    task.step(0)
    env.frame = _Environment._frame(3)
    task.step(0)
    slightly_changed = _Environment._frame(3).copy()
    slightly_changed[60:62, 100:102] = np.clip(
        slightly_changed[60:62, 100:102].astype(np.int16) + 4, 0, 255
    ).astype(np.uint8)
    env.frame = slightly_changed
    _, _, _, _, persistent = task.step(0)
    assert persistent["reward_components"]["novelty"] == 0.05


def test_v2_lifelong_novelty_decays_and_persists_across_task_processes(tmp_path: Path) -> None:
    path = tmp_path / "run" / "phase" / "task.jsonl"
    config = _v2_config(novelty_reward=0.05)
    for expected in (0.05, 0.05 / np.sqrt(2)):
        env = _Environment()
        task = ExplorationTask(env, config=config, task_log_path=path)
        task.reset()
        env.frame = _Environment._frame(1)
        task.step(0)
        _, reward, _, _, info = task.step(0)
        assert info["reward_components"]["novelty"] == pytest.approx(expected)
        assert reward > 0
        task.close()
    assert len(load_lifelong_visits(tmp_path / "run" / "lifelong-novelty.json", 4096)) == 1


def test_v2_contract_and_policy_observation_preserve_full_hud(tmp_path: Path) -> None:
    from hk_agent.distributed_knowledge import portable_compatibility_fingerprint

    env1, env2 = _Environment(), _Environment()
    v1 = ExplorationTask(env1)
    v2 = ExplorationTask(env2, config=_v2_config(), task_log_path=tmp_path / "phase" / "task.jsonl")
    obs1, _ = v1.reset(seed=5)
    obs2, _ = v2.reset(seed=5)
    assert np.array_equal(obs1["pixels"], obs2["pixels"])
    c1, c2 = v1.task_contract(), v2.task_contract()
    assert c2["reward_version"] == "explore_v2"
    assert c1["task_contract_schema_version"] == 1 and c2["task_contract_schema_version"] == 2
    assert c2["information_boundary"] == c1["information_boundary"]
    from hk_agent.distributed_knowledge import _public_task_contract

    assert _public_task_contract(c2, from_manifest=False)["reward_version"] == "explore_v2"
    assert portable_compatibility_fingerprint(c1, {}) != portable_compatibility_fingerprint(c2, {})
    assert c2["config"]["geo_gain_reward"] == c2["config"]["geo_spend_reward"] == 0


def test_v2_geo_hp_shade_rewards_are_bounded_and_reward_only(tmp_path: Path) -> None:
    env = _Environment()
    env.extra = {
        "hp_restored_since_last_frame": 2,
        "shade_hits_since_last_frame": 9,
        "shade_distance": 9.0,
        "scene_name": "Room",
        "shade_pending": True,
    }
    task = ExplorationTask(
        env, config=_v2_config(), task_log_path=tmp_path / "phase" / "task.jsonl"
    )
    task.reset()
    observation, _, _, _, info = task.step(0)
    components = info["reward_components"]
    assert components["confirmed_geo_gained"] == components["confirmed_geo_spent"] == 0
    assert components["hp_restored"] == 0.8
    assert components["shade_hits"] == 1.2
    assert "scene_name" in task.task_contract()["reward_only_signals"]
    assert set(observation) == {"pixels", "action_history"}


def test_v2_shade_potential_is_capped_and_requires_same_scene(tmp_path: Path) -> None:
    env = _Environment()
    env.extra = {"shade_distance": 10.0, "scene_name": "Room", "shade_pending": True}
    task = ExplorationTask(
        env, config=_v2_config(), task_log_path=tmp_path / "phase" / "task.jsonl"
    )
    task.reset()
    _, _, _, _, first = task.step(0)
    assert first["reward_components"]["shade_approach"] == 0
    env.extra = {"shade_distance": 1.0, "scene_name": "Room", "shade_pending": True}
    _, _, _, _, closer = task.step(0)
    assert closer["reward_components"]["shade_approach"] == 0.05
    env.extra = {"shade_distance": 0.0, "scene_name": "Other", "shade_pending": True}
    _, _, _, _, switched = task.step(0)
    assert switched["reward_components"]["shade_approach"] == 0
    assert switched["reward_components"]["time_cost"] == pytest.approx(-0.00108)


def test_v2_stationary_attack_cannot_farm_novelty_but_movement_can(tmp_path: Path) -> None:
    from hk_agent.scene_change_task import SCENE_CHANGE_ACTION_NAMES

    env = _Environment()
    task = ExplorationTask(
        env, config=_v2_config(), task_log_path=tmp_path / "phase" / "task.jsonl"
    )
    task.reset()
    env.frame = _Environment._frame(1)
    attack = SCENE_CHANGE_ACTION_NAMES.index("ATTACK")
    _, _, _, _, first = task.step(attack)
    _, _, _, _, repeat = task.step(attack)
    assert first["reward_components"]["novelty"] == 0
    assert repeat["reward_components"]["novelty"] == 0
    env.frame = _Environment._frame(3)
    move = SCENE_CHANGE_ACTION_NAMES.index("RIGHT")
    task.step(move)
    env.frame = _Environment._frame(3)
    _, _, _, _, moved = task.step(move)
    assert moved["reward_components"]["novelty"] == pytest.approx(0.05)


def test_v2_reward_safety_constraints_are_enforced() -> None:
    with pytest.raises(ValueError, match="healing reward"):
        _v2_config(damage_penalty=-0.4).validate()
    with pytest.raises(ValueError, match="does not reward Geo"):
        ExplorationTaskConfig(reward_version="explore_v2", shade_recovery_reward=4.0).validate()


@pytest.mark.skipif(exploration_task_module.os.name != "nt", reason="Windows can transiently lock replace targets")
def test_player_knowledge_persistence_retries_transient_windows_replace_denial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = object.__new__(ExplorationTask)
    target = tmp_path / "player-knowledge.json"
    task._knowledge_path = target
    task._player_knowledge = PlayerKnowledge()
    original_replace = exploration_task_module.os.replace
    attempts = 0

    def transient_replace(source: object, destination: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 4:
            raise PermissionError("transient sharing violation")
        original_replace(source, destination)

    monkeypatch.setattr(exploration_task_module.os, "replace", transient_replace)
    task._persist_player_knowledge()

    assert attempts == 4
    assert target.exists()
    assert not list(tmp_path.glob(".player-knowledge.json.*.tmp"))


def test_challenge_features_disabled_by_default_without_snapshot_or_contract_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HK_AGENT_CHALLENGE_PRACTICE_ENABLED", raising=False)
    monkeypatch.delenv("HK_AGENT_HELP_REQUESTS_ENABLED", raising=False)
    disabled = ExplorationTask(_Environment())
    live = tmp_path / "live"
    snapshots = tmp_path / "snapshots"
    personal = tmp_path / "personal"
    live.mkdir()
    personal.mkdir()
    (live / "user1.dat").write_bytes(b"isolated")
    disabled._challenge_save_root = live
    disabled._challenge_snapshot_root = snapshots
    disabled._personal_save_root = personal
    disabled._capture_challenge_snapshots(
        [{"event": "challenge_started", "challenge_id": "False_Knight",
          "kind": "boss", "snapshot_requested": True}],
        {},
    )
    disabled._player_knowledge.update({
        "operator_telemetry": {"scene_name": "Room_Test", "visible_enemies": [{
            "boss_confirmed": True, "boss_key": "False_Knight",
            "boss_version": "normal", "struck_this_encounter": 0,
        }]},
    }, challenge_tracking_enabled=disabled._challenge_practice_enabled)
    assert disabled._challenge_practice_enabled is False
    assert disabled._player_knowledge.challenge_memory == {}
    assert not snapshots.exists()

    monkeypatch.setenv("HK_AGENT_CHALLENGE_PRACTICE_ENABLED", "1")
    monkeypatch.setenv("HK_AGENT_HELP_REQUESTS_ENABLED", "1")
    enabled = ExplorationTask(_Environment())
    assert enabled._challenge_practice_enabled is True
    assert enabled.task_contract() == disabled.task_contract()
    assert enabled.observation_space.spaces.keys() == disabled.observation_space.spaces.keys()
    assert {
        name: (space.shape, space.dtype.str)
        for name, space in enabled.observation_space.spaces.items()
    } == {
        name: (space.shape, space.dtype.str)
        for name, space in disabled.observation_space.spaces.items()
    }
