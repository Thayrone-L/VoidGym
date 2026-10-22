import pytest
from pathlib import Path
from types import SimpleNamespace

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
