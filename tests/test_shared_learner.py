from __future__ import annotations

import hashlib
import json
import threading
import time

import gymnasium as gym
import numpy as np
import pytest

from hk_agent import shared_learner as shared
from hk_agent.shared_learner import (
    SharedLearnerError,
    SharedRoundStore,
    _atomic_json,
    _canonical_json,
    _restore_latest_checkpoint,
    _shared_ppo_parameters,
    _space_from_payload,
    _space_to_payload,
    actor_join_round,
    run_deterministic_evaluation,
    train_shared_round,
)

FINGERPRINT = "a" * 64


def _tiny_torch_policy():
    torch = pytest.importorskip("torch")
    return torch.nn.Linear(2, 3)


def test_shared_policy_is_immutable_and_resume_reuses_pointer(tmp_path):
    policy = _tiny_torch_policy()
    store = SharedRoundStore(tmp_path)
    published = store.publish_policy(4, policy, FINGERPRINT)
    pointer_path = store.policy_root / "round-00000004.json"
    before = pointer_path.read_bytes()
    assert "-" in published["artifact"]
    with pytest.raises(SharedLearnerError, match="already published"):
        store.publish_policy(4, policy, FINGERPRINT)
    assert store.publish_policy(4, policy, FINGERPRINT, reuse_existing=True) == published
    assert pointer_path.read_bytes() == before


def test_policy_retention_keeps_checkpoint_generation_and_cleans_pairs(tmp_path):
    policy = _tiny_torch_policy()
    store = SharedRoundStore(tmp_path)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    protected_checkpoint = checkpoints / "round-00000003.zip"
    protected_checkpoint.write_bytes(b"checkpoint")
    _atomic_json(tmp_path / "manifest.json", {"latest_checkpoint": str(protected_checkpoint)})

    for round_number in range(9):
        store.publish_policy(round_number, policy, FINGERPRINT)

    pointers = {path.stem for path in store.policy_root.glob("round-*.json")}
    assert "round-00000003" in pointers
    assert {"round-00000006", "round-00000007", "round-00000008"} <= pointers
    assert "round-00000002" not in pointers
    for round_number in (3, 6, 7, 8):
        pointer = json.loads((store.policy_root / f"round-{round_number:08d}.json").read_text("utf-8"))
        assert (store.policy_root / pointer["artifact"]).is_file()
        assert (store.policy_root / pointer["manifest"]).is_file()


def test_policy_publication_cleans_generation_after_pointer_failure(tmp_path, monkeypatch):
    policy = _tiny_torch_policy()
    store = SharedRoundStore(tmp_path)
    original_link = shared.os.link

    def fail_pointer_link(_source, _target):
        raise OSError("injected pointer publication failure")

    monkeypatch.setattr(shared.os, "link", fail_pointer_link)
    with pytest.raises(SharedLearnerError, match="new generation .* was cleaned"):
        store.publish_policy(2, policy, FINGERPRINT)
    assert list(store.policy_root.glob("round-00000002-*.safetensors")) == []
    assert list(store.policy_root.glob("round-00000002-*.policy.manifest.json")) == []
    assert not (store.policy_root / "round-00000002.json").exists()

    monkeypatch.setattr(shared.os, "link", original_link)
    published = store.publish_policy(2, policy, FINGERPRINT)
    assert (store.policy_root / published["artifact"]).is_file()
    assert (store.policy_root / "round-00000002.json").is_file()


def test_resume_uses_latest_valid_checkpoint_round_not_stale_learner_status(tmp_path):
    class FakeModel:
        num_timesteps = 12_288

    class FakeModelClass:
        @staticmethod
        def load(path, *, env, device):
            assert env == "fake-env"
            assert device == "cpu"
            assert path.endswith("round-00000015.zip")
            return FakeModel()

    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "round-00000015.zip").write_bytes(b"valid model")
    _atomic_json(tmp_path / "learner.json", {"round": 0, "completed_timesteps": 0})

    restored = _restore_latest_checkpoint(FakeModelClass, checkpoints, "fake-env")
    assert restored is not None
    model, round_number, completed = restored
    assert isinstance(model, FakeModel)
    assert round_number == 15
    assert completed == 12_288


def test_actor_imports_while_policy_is_published(tmp_path):
    policy = _tiny_torch_policy()
    store = SharedRoundStore(tmp_path)
    failures = []
    started = threading.Event()

    def publish_and_checkpoint():
        try:
            store.publish_policy(1, policy, FINGERPRINT)
            started.set()
            # Checkpointing must not republish or alter this immutable generation.
            for _ in range(25):
                time.sleep(0.002)
        except (OSError, RuntimeError, ValueError) as error:
            failures.append(error)

    thread = threading.Thread(target=publish_and_checkpoint)
    thread.start()
    assert started.wait(5)
    for _ in range(25):
        store.import_policy(1, _tiny_torch_policy(), FINGERPRINT)
    thread.join(5)
    assert not thread.is_alive()
    assert failures == []


def test_actor_liveness_grace_covers_rollout_publication_but_not_restart(tmp_path):
    store = SharedRoundStore(tmp_path)
    now = time.time_ns()
    assert store.publish_actor_status(
        slot=0, launch_id="actor-1", status="PUBLISHING", heartbeat_unix_ns=now
    )
    assert store.actor_available(0, now_unix_ns=now + 80_000_000_000) is True
    assert store.actor_available(0, now_unix_ns=now + 91_000_000_000) is False

    store.publish_actor_status(
        slot=0, launch_id="actor-1", status="RESTARTING", heartbeat_unix_ns=now
    )
    assert store.actor_available(0, now_unix_ns=now + 1_000_000_000) is False


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
    source = gym.spaces.Dict({
        "history": gym.spaces.MultiDiscrete([3, 4], start=[1, 0]),
        "pixels": gym.spaces.Box(0, 1, (3, 4, 5), dtype=np.float32),
    })
    restored = _space_from_payload(_space_to_payload(source))
    assert restored == source


def test_atomic_json_normalizes_numpy_scalars_and_arrays(tmp_path):
    path = tmp_path / "contract.json"
    _atomic_json(path, {"count": np.int64(4), "shape": np.asarray([2, 3], dtype=np.int64)})
    assert json.loads(path.read_text(encoding="utf-8")) == {"count": 4, "shape": [2, 3]}
    assert json.loads(_canonical_json({"count": np.int64(4)})) == {"count": 4}


def test_atomic_json_retries_windows_replace_sharing_error(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    original = shared.os.replace
    attempts = 0

    def flaky_replace(source, target):
        nonlocal attempts
        if attempts == 0:
            attempts += 1
            raise PermissionError("simulated Windows sharing lock")
        original(source, target)

    monkeypatch.setattr(shared.os, "replace", flaky_replace)
    _atomic_json(path, {"status": "WAITING"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "WAITING"}
    assert attempts == 1


def test_published_rollout_normalizes_maskable_buffer_masks_to_bool(tmp_path):
    from types import SimpleNamespace

    store = SharedRoundStore(tmp_path)
    buffer = SimpleNamespace(
        pos=2, buffer_size=2, full=True,
        observations={"image": np.zeros((2, 1, 3), dtype=np.float32)},
        actions=np.zeros((2, 1), dtype=np.int64),
        rewards=np.ones((2, 1), dtype=np.float32),
        episode_starts=np.zeros((2, 1), dtype=np.float32),
        values=np.zeros((2, 1), dtype=np.float32),
        log_probs=np.zeros((2, 1), dtype=np.float32),
        advantages=np.ones((2, 1), dtype=np.float32),
        returns=np.ones((2, 1), dtype=np.float32),
        action_masks=np.ones((2, 1, 2), dtype=np.float32),
    )
    store.publish_rollout(round_number=1, slot=0, launch_id="actor", buffer=buffer, fingerprint=FINGERPRINT)
    observation_space = gym.spaces.Dict({"image": gym.spaces.Box(0, 1, (3,), dtype=np.float32)})
    arrays = store.read_rollout(
        round_number=1, slot=0, fingerprint=FINGERPRINT, n_steps=2,
        observation_space=observation_space, action_count=2,
    )
    assert arrays["action_masks"].dtype == np.bool_


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


def test_wait_rollouts_rejects_rollout_from_retired_actor_launch(tmp_path):
    store = _publish_arrays(tmp_path)
    store.publish_actor_status(slot=0, launch_id="actor-current", status="COLLECTING")
    observation_space, action_count = _spaces()

    with pytest.raises(TimeoutError, match="no valid actor rollouts"):
        store.wait_rollouts(
            round_number=1, actors=1, fingerprint=FINGERPRINT, n_steps=2,
            observation_space=observation_space, action_count=action_count, timeout=0.01,
        )

    rejected = (tmp_path / "rejected.jsonl").read_text("utf-8")
    assert "retired actor launch" in rejected


def test_new_actor_starting_status_replaces_terminal_slot_owner(tmp_path):
    store = SharedRoundStore(tmp_path)
    store.publish_actor_status(slot=0, launch_id="actor-old", status="FAILED")
    assert store.publish_actor_status(slot=0, launch_id="actor-new", status="STARTING")
    assert store._read_actor_status(0)["launch_id"] == "actor-new"
    assert store.actor_available(0) is True


def test_one_actor_satisfies_quorum_after_timeout(tmp_path):
    store = _publish_arrays(tmp_path)
    observation_space, action_count = _spaces()
    rollouts, slots, metrics = store.wait_rollouts(
        round_number=1, actors=2, fingerprint=FINGERPRINT, n_steps=2,
        observation_space=observation_space, action_count=action_count, timeout=0.01,
    )
    assert slots == [0]
    assert len(rollouts) == 1
    assert metrics["received_slots"] == [0]


def test_straggler_timeout_trains_with_one_actor_rollout(tmp_path):
    store = _publish_arrays(tmp_path)
    store.publish_actor_status(slot=1, launch_id="actor-1", status="COLLECTING")
    observation_space, action_count = _spaces()
    started = __import__("time").monotonic()
    rollouts, slots, metrics = store.wait_rollouts(
        round_number=1, actors=2, fingerprint=FINGERPRINT, n_steps=2,
        observation_space=observation_space, action_count=action_count,
        timeout=2, straggler_timeout=0.05,
    )
    assert slots == [0]
    assert len(rollouts) == 1
    assert metrics["straggler_wait_ms"] < 1000
    assert __import__("time").monotonic() - started < 1


def test_restarting_actor_does_not_hold_rollout_quorum(tmp_path):
    store = _publish_arrays(tmp_path)
    store.publish_actor_status(slot=1, launch_id="actor-1", status="RESTARTING")
    observation_space, action_count = _spaces()
    rollouts, slots, metrics = store.wait_rollouts(
        round_number=1, actors=2, fingerprint=FINGERPRINT, n_steps=2,
        observation_space=observation_space, action_count=action_count,
        timeout=2, straggler_timeout=45,
    )
    assert slots == [0]
    assert len(rollouts) == 1
    assert metrics["straggler_wait_ms"] < 1000


def test_stale_rollouts_are_counted_separately_from_bug_rejections(tmp_path):
    store = SharedRoundStore(tmp_path)
    store.record_stale_rollout(round_number=2, slot=0, launch_id="restarted")
    assert store.stale_rollout_count() == 1
    assert not (tmp_path / "rejected.jsonl").exists()


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
    v2_values = _shared_ppo_parameters({"reward_version": "explore_v2"})
    assert v2_values["ent_coef"] == 0.01 and v2_values["gamma"] == 0.995


def test_deterministic_evaluation_does_not_submit_training_rollout():
    class Env:
        def __init__(self):
            self.steps = 0
            self.evaluation = False

        def env_method(self, name, *args):
            if name == "set_evaluation_mode":
                self.evaluation = bool(args[0])
                return [None]
            if name == "action_masks":
                return [np.ones(2, dtype=np.bool_)]
            raise AssertionError(name)

        def get_wrapper_attr(self, name):
            assert name == "action_masks"
            return lambda: np.ones(2, dtype=np.bool_)

        def reset(self):
            return np.zeros((1, 2), dtype=np.float32)

        def step(self, actions):
            self.steps += 1
            return np.zeros((1, 2), dtype=np.float32), np.array([1.0]), np.array([self.steps == 2]), [
                {"episode_progress": {"distinct_scenes": 2}}
            ]

    class Model:
        def __init__(self):
            self.deterministic_values = []
            self.collect_rollouts_called = False

        def predict(self, observation, *, deterministic, action_masks):
            self.deterministic_values.append(deterministic)
            assert action_masks.shape == (2,)
            return np.array([0]), None

    env, model = Env(), Model()
    result = run_deterministic_evaluation(model, env, round_number=4, slot=0, max_steps=10)
    assert result["completed"] is True
    assert result["rollout_submitted"] is False
    assert result["episode_progress"]["distinct_scenes"] == 2
    assert model.deterministic_values == [True, True]
    assert env.evaluation is False
    assert not model.collect_rollouts_called


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


def test_v3_to_v4_warm_start_accepts_equal_spaces_and_rejects_mismatch():
    observation = gym.spaces.Dict({"pixels": gym.spaces.Box(0, 255, (4, 10, 12, 3), dtype=np.uint8)})
    actions = gym.spaces.Discrete(30)
    shared.validate_warm_start_spaces(observation, actions, observation, actions)
    different_actions = gym.spaces.Discrete(29)
    with pytest.raises(SharedLearnerError, match="spaces differ"):
        shared.validate_warm_start_spaces(observation, actions, observation, different_actions)


def test_partial_warm_start_copies_compatible_visual_tensors_only():
    source = {
        "features_extractor.image.cnn.weight": np.ones((4, 3)),
        "action_net.weight": np.ones((8, 16)),
    }
    target = {
        "features_extractor.image.cnn.weight": np.zeros((4, 3)),
        "action_net.weight": np.zeros((10, 16)),
        "new_observation_projection.weight": np.zeros((16, 4)),
    }
    copied = shared.compatible_warm_start_state(source, target)
    assert list(copied) == ["features_extractor.image.cnn.weight"]


def test_context_demonstrations_are_version_checked_scoped_and_decayed(tmp_path, monkeypatch):
    import hashlib
    import io
    import json
    from types import SimpleNamespace

    import gymnasium as gym
    from PIL import Image

    from hk_agent.demonstrations import DemonstrationRecorder

    context = "boss:False_Knight"
    folder = tmp_path / "demonstrations" / hashlib.sha256(context.encode()).hexdigest()[:16]
    recorder = DemonstrationRecorder(folder, game_version="1.5.78", context=context)
    recorder.authorize(help_request_id="help-1")
    encoded = io.BytesIO()
    Image.new("RGB", (4, 3)).save(encoded, format="PNG")
    recorder.append(
        png_bytes=encoded.getvalue(), action=0, action_mask=np.array([True, False]),
        player_knowledge=np.zeros(2, dtype=np.float32),
    )
    artifact = recorder.finish()
    store = shared.SharedRoundStore(tmp_path)
    store.publish_actor_status(
        slot=0, launch_id="actor-0", status="COLLECTING",
        challenge_context=context, game_version="1.5.78",
    )
    store.publish_actor_status(
        slot=1, launch_id="actor-1", status="COLLECTING",
        challenge_context="room:other", game_version="1.5.78",
    )
    model = SimpleNamespace(
        observation_space=gym.spaces.Dict({
            "pixels": gym.spaces.Box(0, 255, (1, 8, 8, 3), dtype=np.uint8),
            "action_history": gym.spaces.Box(0, 1, (2,), dtype=np.int64),
            "player_knowledge": gym.spaces.Box(-1, 1, (2,), dtype=np.float32),
        }),
        action_space=gym.spaces.Discrete(2),
    )
    calls = []
    monkeypatch.setattr(shared, "train_behavioral_cloning_batch", lambda _model, batch, *, weight: calls.append(weight) or 0.5)
    manifest = {}
    rows = shared.apply_context_demonstrations(
        model, tmp_path, store, actor_slots=[0, 1], round_number=0, manifest=manifest,
        enabled=True,
    )
    assert len(rows) == 1 and rows[0]["context"] == context
    assert rows[0]["weight"] == pytest.approx(0.25)
    metadata = json.loads(artifact.with_suffix(".json").read_text())
    assert metadata["first_used_round"] == 0
    rows = shared.apply_context_demonstrations(
        model, tmp_path, store, actor_slots=[0], round_number=20, manifest=manifest,
        enabled=True,
    )
    assert rows == []
    assert calls == [pytest.approx(0.25)]
    assert shared.apply_context_demonstrations(
        model, tmp_path, store, actor_slots=[0], round_number=21, manifest=manifest,
    ) == []
    assert calls == [pytest.approx(0.25)]
