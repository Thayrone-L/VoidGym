from __future__ import annotations

import hashlib
import json

import gymnasium as gym
import numpy as np
import pytest

from hk_agent.shared_learner import (
    SharedLearnerError,
    SharedRoundStore,
    _shared_ppo_parameters,
    _space_from_payload,
    _space_to_payload,
    actor_join_round,
    train_shared_round,
)

FINGERPRINT = "a" * 64


def _publish_arrays(root, *, round_number=1, fingerprint=FINGERPRINT):
    store = SharedRoundStore(root)
    folder = root / "rollouts" / f"round-{round_number:08d}"
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {
        "obs__image": np.zeros((2, 3), dtype=np.float32),
        "obs__history": np.array([[0, 1], [1, 2]], dtype=np.int64),
        "actions": np.array([0, 1], dtype=np.int64),
        "rewards": np.ones(2, dtype=np.float32),
        "episode_starts": np.zeros(2, dtype=np.float32),
        "values": np.zeros(2, dtype=np.float32),
        "log_probs": np.zeros(2, dtype=np.float32),
        "advantages": np.ones(2, dtype=np.float32),
        "returns": np.ones(2, dtype=np.float32),
        "action_masks": np.ones((2, 2), dtype=np.bool_),
    }
    artifact = folder / "slot-0.npz"
    np.savez_compressed(artifact, **arrays)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (folder / "slot-0.json").write_text(json.dumps({
        "schema_version": 1, "policy_round": round_number, "slot": 0,
        "launch_id": "actor-test", "n_steps": 2, "sha256": digest,
        "task_contract_fingerprint": fingerprint,
    }), encoding="utf-8")
    return store


def _spaces():
    return gym.spaces.Dict({
        "image": gym.spaces.Box(0, 1, (3,), np.float32),
        "history": gym.spaces.MultiDiscrete([3, 4]),
    }), 2


def test_dict_multidiscrete_space_round_trips_strictly():
    source = gym.spaces.Dict({"history": gym.spaces.MultiDiscrete([3, 4], start=[1, 0])})
    restored = _space_from_payload(_space_to_payload(source))
    assert restored == source


def test_rollout_round_trip_and_rejects_old_round(tmp_path):
    store = _publish_arrays(tmp_path)
    observation_space, action_count = _spaces()
    loaded = store.read_rollout(
        round_number=1, slot=0, fingerprint=FINGERPRINT, n_steps=2,
        observation_space=observation_space, action_count=action_count,
    )
    assert loaded["obs__image"].shape == (2, 3)
    stale_folder = tmp_path / "rollouts" / "round-00000002"
    stale_folder.mkdir(parents=True)
    source_folder = tmp_path / "rollouts" / "round-00000001"
    (stale_folder / "slot-0.npz").write_bytes((source_folder / "slot-0.npz").read_bytes())
    (stale_folder / "slot-0.json").write_bytes((source_folder / "slot-0.json").read_bytes())
    with pytest.raises(SharedLearnerError, match="different policy round"):
        store.read_rollout(
            round_number=2, slot=0, fingerprint=FINGERPRINT, n_steps=2,
            observation_space=observation_space, action_count=action_count,
        )


def test_rollout_rejects_fingerprint_and_corrupt_sha(tmp_path):
    store = _publish_arrays(tmp_path)
    observation_space, action_count = _spaces()
    kwargs = {"round_number": 1, "slot": 0, "n_steps": 2, "observation_space": observation_space, "action_count": action_count}
    with pytest.raises(SharedLearnerError, match="fingerprint"):
        store.read_rollout(**kwargs, fingerprint="b" * 64)
    artifact = tmp_path / "rollouts" / "round-00000001" / "slot-0.npz"
    artifact.write_bytes(artifact.read_bytes() + b"bad")
    with pytest.raises(SharedLearnerError, match="sha256"):
        store.read_rollout(**kwargs, fingerprint=FINGERPRINT)


def test_rollout_rejects_empty_mask_and_unavailable_action(tmp_path):
    store = _publish_arrays(tmp_path)
    artifact = tmp_path / "rollouts" / "round-00000001" / "slot-0.npz"
    with np.load(artifact, allow_pickle=False) as packed:
        arrays = {name: packed[name] for name in packed.files}
    arrays["action_masks"][0] = False
    np.savez_compressed(artifact, **arrays)
    sidecar = artifact.with_suffix(".json")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    meta["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    sidecar.write_text(json.dumps(meta), encoding="utf-8")
    observation_space, action_count = _spaces()
    with pytest.raises(SharedLearnerError, match="empty action mask"):
        store.read_rollout(
            round_number=1, slot=0, fingerprint=FINGERPRINT, n_steps=2,
            observation_space=observation_space, action_count=action_count,
        )


def test_one_actor_satisfies_quorum_after_timeout(tmp_path):
    store = _publish_arrays(tmp_path)
    observation_space, action_count = _spaces()
    rollouts, slots = store.wait_rollouts(
        round_number=1, actors=2, fingerprint=FINGERPRINT, n_steps=2,
        observation_space=observation_space, action_count=action_count, timeout=0.01,
    )
    assert slots == [0]
    assert len(rollouts) == 1


def test_restarted_actor_joins_current_learner_round():
    assert actor_join_round({"round": 7}) == 7
    assert actor_join_round({"round": True}) == 0


def test_exploration_learner_uses_explicit_campaign_ppo_parameters():
    values = _shared_ppo_parameters({
        "reward_version": "explore_v1",
        "ppo_hyperparameters": {"ent_coef": 0.02, "gamma": 0.99},
    })
    assert values == {
        "ent_coef": 0.02, "gamma": 0.99, "gae_lambda": 0.95,
        "n_epochs": 4, "target_kl": 0.03, "learning_rate": 2.5e-4,
        "clip_range": 0.2,
    }
    assert _shared_ppo_parameters({"reward_version": "visual_motion_v1"}) == {}


def test_learner_waiting_for_calibration_honors_stop_file(tmp_path):
    from hk_agent.shared_learner import SharedLearnerStopped

    stop_file = tmp_path / "stop.request"
    stop_file.write_text("stop", encoding="utf-8")
    with pytest.raises(SharedLearnerStopped, match="stop requested"):
        SharedRoundStore(tmp_path).wait_for_contract(timeout=1, stop_file=stop_file)


def test_two_actor_rollouts_fill_and_train_one_learner_buffer(tmp_path):
    from types import SimpleNamespace

    store = _publish_arrays(tmp_path)
    observation_space, action_count = _spaces()
    first = store.read_rollout(
        round_number=1, slot=0, fingerprint=FINGERPRINT, n_steps=2,
        observation_space=observation_space, action_count=action_count,
    )
    second = {name: value.copy() for name, value in first.items()}
    model = SimpleNamespace(
        observation_space=observation_space,
        action_space=gym.spaces.Discrete(action_count),
        device="cpu", gae_lambda=0.95, gamma=0.99, num_timesteps=0,
        train=lambda: None,
    )
    transitions = train_shared_round(model, [first, second])
    assert transitions == 4
    assert model.num_timesteps == 4
    assert model.rollout_buffer.full
    assert model.rollout_buffer.n_envs == 2
