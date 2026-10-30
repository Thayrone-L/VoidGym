import pytest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests.test_gym_env import FakeBackend, make_env
from voidgym.adapters.hollow_knight.adapter import HollowKnightAdapter
from voidgym.core.observation import split_observation


class EmptyEnv:
    def __init__(self) -> None:
        self._frames = []
        self.latest_action_availability = None


def test_action_order_matches_current_temporal_table() -> None:
    adapter = HollowKnightAdapter(EmptyEnv())
    assert adapter.action_table() == (
        "NOOP", "LEFT", "RIGHT", "JUMP_TAP", "JUMP_HOLD", "LEFT+JUMP_TAP",
        "LEFT+JUMP_HOLD", "RIGHT+JUMP_TAP", "RIGHT+JUMP_HOLD", "JUMP_MAX",
        "LEFT+JUMP_MAX", "RIGHT+JUMP_MAX", "ATTACK", "LEFT+ATTACK", "RIGHT+ATTACK",
        "UP+ATTACK", "DOWN+ATTACK", "JUMP+ATTACK", "LEFT+JUMP+ATTACK",
        "RIGHT+JUMP+ATTACK", "CAST", "UP+CAST", "DOWN+CAST", "FOCUS_HOLD",
        "NAIL_ART_CHARGE", "LEFT+NAIL_ART_CHARGE", "RIGHT+NAIL_ART_CHARGE",
        "UP+NAIL_ART_CHARGE", "DOWN+NAIL_ART_CHARGE", "DASH", "LEFT+DASH",
        "RIGHT+DASH", "DOWN+DASH", "SUPER_DASH_HOLD", "DREAM_NAIL", "INTERACT",
        "MENU_UP", "MENU_DOWN", "MENU_LEFT", "MENU_RIGHT", "MENU_SUBMIT",
        "MENU_CANCEL", "OPEN_CHARMS",
    )


def test_every_hk_action_round_trips() -> None:
    adapter = HollowKnightAdapter(EmptyEnv())
    for index in range(len(adapter.action_table())):
        assert adapter.from_universal(adapter.to_universal(index)) == index


def test_hk_observation_split_preserves_keys_shapes_and_dtypes(tmp_path) -> None:
    env = make_env(FakeBackend(), tmp_path)
    observation, _info = env.reset(seed=12)
    universal, specific = split_observation(observation)
    assert set(universal) == {"pixels", "action_history"}
    assert specific == {}
    assert all(observation[key].shape == universal[key].shape for key in observation)
    assert all(observation[key].dtype == universal[key].dtype for key in observation)
    env.close()


def test_hk_adapter_exposes_reward_signals_separately_from_policy_input() -> None:
    class SignalEnv(EmptyEnv):
        def step(self, action: int, *, lease_ms: int):
            assert action == 0
            assert lease_ms == 80
            frame = SimpleNamespace(
                pixels=None,
                scene_name="Crossroads_01",
                scene_transition=False,
                shade_pending=True,
                shade_recovered=False,
                run_failed=False,
                enemy_kills_since_last_frame=1,
                hp_lost_since_last_frame=2,
                geo_gained_since_last_frame=3,
                geo_spent_since_last_frame=4,
                hp_restored_since_last_frame=1,
                shade_hits_since_last_frame=2,
                shade_distance=24.0,
                enemy_events_since_last_frame=(("Crossroads_01", "e_0123456789"),),
                bestiary_events=(),
                operator_telemetry={
                    "x": 12.0,
                    "y": 34.0,
                    "milestone_events_since_last_frame": [
                        {"category": "item", "key": "test", "confirmed": True}
                    ],
                },
                action_availability=None,
            )
            self._frames = [frame]
            return {}, 0.0, False, False, {
                "geo_spent_since_last_frame": 4,
                "map_zone": "Crossroads",
            }

    adapter = HollowKnightAdapter(SignalEnv())
    adapter.apply(0)
    signals = adapter.signals()
    assert signals["scene_name"] == "Crossroads_01"
    assert signals["hp_lost_since_last_frame"] == 2
    assert signals["geo_spent_since_last_frame"] == 4
    assert signals["map_zone"] == "Crossroads"
    assert signals["x"] == 12.0
    assert signals["position"]["y"] == 34.0
    assert signals["milestone_events_since_last_frame"][0]["confirmed"] is True
    assert adapter.training_labels() is None


def test_hk_adapter_reads_and_steps_through_explore_task() -> None:
    names = HollowKnightAdapter(EmptyEnv()).action_table()
    frame = SimpleNamespace(
        pixels=np.zeros((8, 12, 3), dtype=np.uint8),
        scene_name="Crossroads_01", operator_telemetry=None,
    )

    class Task:
        _action_names = names
        config = SimpleNamespace(tutorial_hint_level=0)
        _player_knowledge = SimpleNamespace(vector=lambda *, hint_level: np.asarray([hint_level]))

        def __init__(self) -> None:
            self.environment = SimpleNamespace(_frames=[frame])

        def action_masks(self):
            return np.ones(len(names), dtype=np.bool_)

        def step(self, action):
            assert action == 3
            return {}, 1.25, True, False, {"scene_name": "Crossroads_01"}

    task = Task()
    adapter = HollowKnightAdapter(task)

    np.testing.assert_array_equal(adapter.frame_rgb(), frame.pixels)
    adapter.apply(3)

    assert adapter.last_reward == 1.25
    assert adapter.signals()["scene_name"] == "Crossroads_01"
    assert adapter.signals()["terminated"] is True


def test_v2_discovery_reward_is_opt_in_on_hk_adapter() -> None:
    names = HollowKnightAdapter(EmptyEnv()).action_table()
    frame = SimpleNamespace(pixels=np.zeros((8, 12, 3), dtype=np.uint8),
                            scene_name="Room", operator_telemetry=None,
                            combat_events=())

    class Task:
        _action_names = names
        config = SimpleNamespace(tutorial_hint_level=0)
        _player_knowledge = SimpleNamespace(vector=lambda *, hint_level: np.asarray([hint_level]))

        def __init__(self) -> None:
            self.environment = SimpleNamespace(_frames=[frame])

        def action_masks(self):
            return np.ones(len(names), dtype=np.bool_)

        def step(self, _action):
            self.environment._frames[-1] = SimpleNamespace(
                pixels=frame.pixels, scene_name="Room", operator_telemetry=None,
                combat_events=({"kind": "hit_geo_rock", "scene_name": "Room",
                                "object_id": "rock-1", "amount": 0},),
            )
            return {}, 1.0, False, False, {}

    default_adapter = HollowKnightAdapter(Task())
    default_adapter.apply(0)
    assert default_adapter.last_reward == 1.0

    v2_adapter = HollowKnightAdapter(Task(), enable_discovery_reward=True)
    v2_adapter.apply(0)
    assert v2_adapter.last_reward == pytest.approx(1.05)


def test_hk_adapter_exposes_opt_in_labels_only_as_training_targets() -> None:
    env = EmptyEnv()
    sem = np.zeros((27, 48), dtype=np.uint8)
    sem[4, 8] = 5
    env._frames = [SimpleNamespace(
        semantic_labels=sem, hero_motion=(0.25, -0.5, True),
        interact_prompt=False, interact_prompt_valid=False,
    )]
    adapter = HollowKnightAdapter(env)

    labels = adapter.training_labels()

    assert labels is not None
    np.testing.assert_array_equal(labels["sem"], sem)
    np.testing.assert_array_equal(
        labels["hero_velocity"], np.asarray([0.25, -0.5], dtype=np.float32)
    )
    assert labels["on_ground"] == 1
    assert labels["interact_prompt"] == 0
    assert labels["interact_prompt_valid"] is False


def test_latest_campaign_checkpoint_contract(tmp_path) -> None:
    root = Path(__file__).parents[2]
    checkpoint_dir = root / "runtime" / "shared" / "jogada-1-repeat-1c3b" / "checkpoints"
    candidates = sorted(checkpoint_dir.glob("round-*.zip")) if checkpoint_dir.is_dir() else []
    if not candidates:
        pytest.skip(f"checkpoint da campanha não existe em {checkpoint_dir}")

    from hk_agent.exploration_task import ExplorationTask, ExplorationTaskConfig
    from hk_agent.recurrent_sb3 import MaskableRecurrentPPO
    from hk_agent.training_task import ChannelsFirstObservation

    from tests.test_gym_env import FakeBackend, make_env

    raw_env = make_env(FakeBackend(), tmp_path)
    task = ExplorationTask(raw_env, config=ExplorationTaskConfig(reward_version="explore_v4"))
    current_env = ChannelsFirstObservation(task, target_height=90, target_width=160)
    try:
        model = MaskableRecurrentPPO.load(str(candidates[-1]), device="cpu")
        assert model.observation_space == current_env.observation_space
        assert model.action_space == current_env.action_space
    finally:
        current_env.close()
