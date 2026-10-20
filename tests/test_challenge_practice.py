from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hk_agent.challenge_practice import (
    ChallengeAttempt,
    ChallengePracticeController,
    ChallengeSignalTracker,
    ChallengeSnapshot,
    challenge_reward,
    create_challenge_snapshot,
    save_challenge_memory,
)
from hk_agent.demonstrations import (
    DemonstrationRecorder,
    HelpRequestLedger,
    HelpTriggerConfig,
    augment_demonstration_batch,
    combine_ppo_and_imitation,
    demonstration_batch,
    help_request,
    imitation_schedule,
    load_demonstration,
    map_human_action,
    masked_behavioral_cloning_loss,
)


def test_challenge_snapshot_only_at_boundaries_and_not_personal_save(tmp_path) -> None:
    personal, live, snapshots = tmp_path / "personal", tmp_path / "live", tmp_path / "snapshots"
    personal.mkdir()
    live.mkdir()
    (live / "user1.dat").write_bytes(b"isolated")
    with pytest.raises(ValueError, match="approved challenge boundary"):
        create_challenge_snapshot(
            challenge_id="boss-a", kind="boss", source="arbitrary", save_root=live,
            snapshot_root=snapshots, personal_root=personal,
        )
    with pytest.raises(ValueError, match="personal save tree"):
        create_challenge_snapshot(
            challenge_id="boss-a", kind="boss", source="arena_entrance", save_root=personal,
            snapshot_root=snapshots, personal_root=personal,
        )
    with pytest.raises(ValueError, match="outside the personal save tree"):
        create_challenge_snapshot(
            challenge_id="boss-a", kind="boss", source="arena_entrance", save_root=live,
            snapshot_root=personal / "snapshots", personal_root=personal,
        )
    with pytest.raises(ValueError, match="invalid challenge snapshot settings"):
        create_challenge_snapshot(
            challenge_id="../personal", kind="boss", source="arena_entrance", save_root=live,
            snapshot_root=snapshots, personal_root=personal,
        )
    result = create_challenge_snapshot(
        challenge_id="boss-a", kind="boss", source="arena_entrance", save_root=live,
        snapshot_root=snapshots, personal_root=personal,
    )
    assert (snapshots.joinpath(Path(result.save_path).name) / "user1.dat").read_bytes() == b"isolated"


def test_challenge_practice_routes_rollouts_to_shared_learner_and_exits_after_wins() -> None:
    snapshot = ChallengeSnapshot("False_Knight", "boss", "arena_entrance", "snapshot", 1)
    submitted: list[str] = []
    controller = ChallengePracticeController(
        main_actor="actor-main", practice_actor="actor-practice", required_wins=2, recent_attempts=3
    )
    controller.start(snapshot, submitted.append)
    controller.submit_rollout("rollout-1")
    assert submitted == ["rollout-1"]
    assert not controller.record_attempt(ChallengeAttempt("False_Knight", "boss", "v1", hits=4))
    assert not controller.record_attempt(ChallengeAttempt("False_Knight", "boss", "v2", success=True, hits=6))
    assert controller.record_attempt(ChallengeAttempt("False_Knight", "boss", "v2", success=True, hits=5))
    assert controller.status()["mode"] == "main_playthrough"
    assert controller.status()["wins"] == 2
    assert controller.memory["False_Knight"].versions == {"v1", "v2"}


def test_challenge_progress_memory_and_rewards_are_bounded(tmp_path) -> None:
    from hk_agent.challenge_practice import ChallengeMemory

    memory = ChallengeMemory()
    memory.record(ChallengeAttempt("spikes-1", "traversal", "v1", 25, 0, 4.0))
    memory.record(ChallengeAttempt("spikes-1", "traversal", "v1", 40, 0, 7.0, success=True))
    assert memory.attempts == 2 and memory.wins == 1 and memory.best_progress == 7.0
    path = tmp_path / "challenge-memory.json"
    save_challenge_memory(path, {"spikes-1": memory})
    assert json.loads(path.read_text())["challenges"]["spikes-1"]["best_progress"] == 7.0
    assert challenge_reward(kind="boss", hit_delta=100, already_rewarded_hits=2) == {
        "challenge_hit": 0.05, "survival_record": 0.0, "challenge_success": 0.0
    }


def test_bridge_boss_presence_victory_and_death_become_attempt_events() -> None:
    tracker = ChallengeSignalTracker()
    started = tracker.update({"operator_telemetry": {
        "scene_name": "Room_Test", "visible_enemies": [{
            "boss_confirmed": True, "boss_key": "False_Knight", "boss_version": "normal",
            "struck_this_encounter": 0,
        }],
    }})
    assert started[0]["event"] == "challenge_started" and started[0]["snapshot_requested"] is True
    hit = tracker.update({"operator_telemetry": {
        "scene_name": "Room_Test", "visible_enemies": [{
            "boss_confirmed": True, "boss_key": "False_Knight", "boss_version": "normal",
            "struck_this_encounter": 4,
        }],
    }})
    assert hit == [{"event": "challenge_hit", "challenge_id": "False_Knight", "hit_delta": 3, "version": "normal"}]
    victory = tracker.update({"operator_telemetry": {"scene_name": "Room_Test", "visible_enemies": []},
                              "bestiary_events_since_last_frame": [{
                                  "boss_confirmed": True, "boss_key": "False_Knight", "boss_version": "normal",
                                  "hits_dealt": 4,
                              }]})
    assert len(victory) == 1 and victory[0]["success"] is True and victory[0]["hits"] == 4

    tracker.update({"operator_telemetry": {"scene_name": "Room_Test", "visible_enemies": [{
        "boss_confirmed": True, "boss_key": "False_Knight", "boss_version": "normal",
        "struck_this_encounter": 0,
    }]}})
    death = tracker.update({"operator_telemetry": {"scene_name": "Room_Test", "visible_enemies": []},
                            "supervisor": {"player_dead": True}})
    assert len(death) == 1 and death[0]["success"] is False


def test_help_request_thresholds_are_configurable_and_silent_before_threshold() -> None:
    config = HelpTriggerConfig(failed_attempts=3, stalled_hours=2.0, stuck_minutes=10)
    assert help_request(
        challenge="False Knight", attempts_without_success=2, no_progress_hours=0.0,
        stuck_minutes=1.0, config=config,
    ) is None
    assert "3 tentativas" in help_request(
        challenge="False Knight", attempts_without_success=3, no_progress_hours=0.0,
        stuck_minutes=1.0, config=config,
    )


def test_human_demo_is_context_and_version_bound_with_masked_bc_loss(tmp_path) -> None:
    root = tmp_path / "outside-repository"
    recorder = DemonstrationRecorder(root, game_version="1.5.78", context="boss:False_Knight")
    png = b"\x89PNG\r\n\x1a\nsmall-frame"
    with pytest.raises(ValueError, match="has not been requested"):
        recorder.append(
            png_bytes=png, action=0, action_mask=np.array([True, False]),
            player_knowledge=np.array([0.25, 0.75], dtype=np.float32),
        )
    recorder.authorize(help_request_id="request-001")
    recorder.append(
        png_bytes=png, action=0, action_mask=np.array([True, False]),
        player_knowledge=np.array([0.25, 0.75], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="illegal"):
        recorder.append(
            png_bytes=png, action=1, action_mask=np.array([True, False]),
            player_knowledge=np.zeros(2, dtype=np.float32),
        )
    artifact = recorder.finish()
    loaded = load_demonstration(
        artifact, expected_game_version="1.5.78", expected_context="boss:False_Knight"
    )
    assert loaded["actions"].tolist() == [0]
    manifest = json.loads(artifact.with_suffix(".json").read_text())
    assert manifest["authorization"] == "help_request:request-001"
    import io

    from PIL import Image

    encoded = io.BytesIO()
    Image.new("RGB", (4, 3), (10, 20, 30)).save(encoded, format="PNG")
    recorder2 = DemonstrationRecorder(root, game_version="1.5.78", context="boss:False_Knight")
    recorder2.authorize(operator_forced=True)
    recorder2.append(
        png_bytes=encoded.getvalue(), action=0, action_mask=np.array([True, False]),
        player_knowledge=np.array([0.25, 0.75], dtype=np.float32),
    )
    artifact2 = recorder2.finish()
    observation_space = __import__("gymnasium").spaces.Dict({
        "pixels": __import__("gymnasium").spaces.Box(0, 255, (2, 8, 8, 3), dtype=np.uint8),
        "action_history": __import__("gymnasium").spaces.Box(0, 1, (3,), dtype=np.int64),
        "player_knowledge": __import__("gymnasium").spaces.Box(-1, 1, (2,), dtype=np.float32),
    })
    batch = demonstration_batch(
        artifact2, expected_game_version="1.5.78", expected_context="boss:False_Knight",
        observation_space=observation_space, action_count=2,
    )
    assert batch["pixels"].shape == (1, 2, 8, 8, 3)
    assert batch["action_history"].shape == (1, 3)
    assert batch["player_knowledge"].tolist() == [[0.25, 0.75]]
    with pytest.raises(ValueError, match="different challenge"):
        load_demonstration(artifact, expected_game_version="1.5.78", expected_context="room:other")
    logits = __import__("torch").tensor([[2.0, 20.0]], requires_grad=True)
    masks = __import__("torch").tensor([[True, False]])
    loss = masked_behavioral_cloning_loss(logits, __import__("torch").tensor([0]), masks)
    assert loss.item() == pytest.approx(0.0)
    assert combine_ppo_and_imitation(loss, loss + 1, weight=0.5).item() == pytest.approx(0.5)
    assert [imitation_schedule(initial_weight=0.5, decay_rounds=4, round_number=i, samples_in_bank=64) for i in range(5)] == [0.5, 0.375, 0.25, 0.125, 0.0]
    assert imitation_schedule(initial_weight=0.5, decay_rounds=4, round_number=100, samples_in_bank=3) == pytest.approx(0.05)


def test_help_ledger_requests_early_same_cell_caps_per_hour_and_survives_restart(tmp_path) -> None:
    config = HelpTriggerConfig(repeated_stuck_count=2, requests_per_hour=1)
    ledger = HelpRequestLedger(tmp_path, slot=0, config=config)
    assert ledger.record_stuck(scene="King_Passage", cell=(3, 5), progress=2.0, now=1000) is None
    request = ledger.record_stuck(scene="King_Passage", cell=(3, 5), progress=2.0, now=1100)
    assert request is not None and request["status"] == "pending"
    assert "Instância 1 pede ajuda" in request["message"]
    restored = HelpRequestLedger(tmp_path, slot=0, config=config)
    assert restored.requests()[0]["request_id"] == request["request_id"]
    assert restored.resolve(request["request_id"], status="dismissed")["status"] == "dismissed"
    assert restored.record_stuck(scene="King_Passage", cell=(3, 5), progress=2.0, now=1200) is None
    assert restored.record_stuck(scene="King_Passage", cell=(8, 5), progress=2.0, now=1300) is None


def test_repeated_stuck_place_becomes_poi_and_new_neighbor_reward_decays(tmp_path) -> None:
    ledger = HelpRequestLedger(
        tmp_path / "help", slot=0,
        config=HelpTriggerConfig(repeated_stuck_count=10, requests_per_hour=3),
    )
    for index in range(3):
        ledger.record_stuck(scene="King_Passage", cell=(4, 7), progress=1.0, now=1000 + index)
    location = next(row for row in ledger.stuck_locations() if row["place"] == "King_Passage:c4r7")
    assert location["count"] == 3
    assert location["point_of_interest"] is True
    assert ledger._state["points_of_interest"][0]["label"] == "travei aqui"
    first = ledger.reward_new_neighbor(scene="King_Passage", cell=(5, 7))
    second = ledger.reward_new_neighbor(scene="King_Passage", cell=(5, 7))
    assert first == pytest.approx(0.1)
    assert second < first


def test_help_ledger_supports_no_progress_trigger(tmp_path) -> None:
    ledger = HelpRequestLedger(
        tmp_path, slot=1,
        config=HelpTriggerConfig(repeated_stuck_count=10, stuck_minutes=1),
    )
    ledger.observe_progress(scene="Room", progress=5, now=1000)
    assert ledger.record_stuck(scene="Room", cell=(1, 1), progress=5, now=1059) is None
    request = ledger.record_stuck(scene="Room", cell=(1, 1), progress=5, now=1061)
    assert request is not None and request["reason"] == "no_room_progress"


def test_help_request_is_campaign_wide_and_same_place_increments_open_request(tmp_path) -> None:
    config = HelpTriggerConfig(repeated_stuck_count=1, requests_per_hour=3)
    first_actor = HelpRequestLedger(tmp_path, slot=0, config=config)
    second_actor = HelpRequestLedger(tmp_path, slot=1, config=config)
    request = first_actor.record_stuck(scene="Room", cell=(2, 4), progress=3, now=1000)
    assert request is not None
    updated = second_actor.record_stuck(scene="Room", cell=(2, 4), progress=3, now=1010)
    assert updated is not None and updated["request_id"] == request["request_id"]
    assert updated["stuck_count"] == 2
    another = second_actor.record_stuck(scene="Room", cell=(8, 4), progress=3, now=1020)
    assert another is not None and another["request_id"] != request["request_id"]


def test_help_request_snooze_stays_open_and_progress_resolves_solo(tmp_path) -> None:
    ledger = HelpRequestLedger(
        tmp_path, slot=0, config=HelpTriggerConfig(repeated_stuck_count=1),
    )
    request = ledger.record_stuck(scene="Room", cell=(2, 4), progress=5, now=1000)
    assert request is not None
    snoozed = ledger.snooze(request["request_id"], now=1100)
    assert snoozed["status"] == "pending"
    assert snoozed["snoozed_until_unix"] == 2900
    ledger.observe_progress(scene="Room", progress=10, now=1200)
    assert ledger.requests()[0]["status"] == "resolved_solo"


def test_help_request_closes_when_actor_leaves_scene(tmp_path) -> None:
    ledger = HelpRequestLedger(
        tmp_path, slot=0, config=HelpTriggerConfig(repeated_stuck_count=1),
    )
    request = ledger.record_stuck(scene="RoomA", cell=(2, 4), progress=5, now=1000)
    assert request is not None
    ledger.observe_progress(scene="RoomB", progress=1, now=1200)
    assert ledger.requests()[0]["status"] == "resolved_solo"


def test_human_actions_map_with_duration_and_horizontal_augmentation() -> None:
    from hk_agent.temporal_actions import TEMPORAL_COMBAT_ACTIONS

    action, covered = map_human_action(["right", "jump"], 50, TEMPORAL_COMBAT_ACTIONS)
    assert TEMPORAL_COMBAT_ACTIONS[action].name == "RIGHT+JUMP_TAP"
    assert covered is True
    _action, covered = map_human_action(["right", "jump"], 600, TEMPORAL_COMBAT_ACTIONS)
    assert covered is False
    pixels = np.zeros((2, 1, 3, 4, 3), dtype=np.uint8)
    pixels[0, :, :, 1, :] = 20
    batch = {
        "pixels": pixels,
        "actions": np.asarray([1, 2], dtype=np.int64),
        "action_masks": np.ones((2, len(TEMPORAL_COMBAT_ACTIONS)), dtype=np.bool_),
        "action_history": np.asarray([[1, 2], [2, 1]], dtype=np.int64),
        "player_knowledge": np.zeros((2, 3), dtype=np.float32),
    }
    augmented = augment_demonstration_batch(batch, TEMPORAL_COMBAT_ACTIONS, brightness_delta=0)
    assert augmented["pixels"].shape[0] == 4
    assert augmented["actions"][2:].tolist() == [2, 1]
    assert augmented["pixels"][2, 0, 0, -1, 0] == 20


def test_manual_recorder_reports_unrepresentable_human_durations(tmp_path) -> None:
    from hk_agent.temporal_actions import TEMPORAL_COMBAT_ACTIONS

    recorder = DemonstrationRecorder(tmp_path, game_version="1.5.78", context="traversal:Room:cell-2-3")
    recorder.authorize(help_request_id="help-0-123")
    png = b"\x89PNG\r\n\x1a\nframe"
    mask = np.ones(len(TEMPORAL_COMBAT_ACTIONS), dtype=np.bool_)
    knowledge = np.zeros(4, dtype=np.float32)
    assert recorder.append_human(
        png_bytes=png, buttons=["RIGHT", "JUMP"], duration_ms=50,
        action_templates=TEMPORAL_COMBAT_ACTIONS, action_mask=mask,
        player_knowledge=knowledge,
    ) == (TEMPORAL_COMBAT_ACTIONS.index(next(a for a in TEMPORAL_COMBAT_ACTIONS if a.name == "RIGHT+JUMP_TAP")), True)
    action, representable = recorder.append_human(
        png_bytes=png, buttons=["RIGHT", "JUMP"], duration_ms=600,
        action_templates=TEMPORAL_COMBAT_ACTIONS, action_mask=mask,
        player_knowledge=knowledge,
    )
    assert action is not None and not representable
    artifact = recorder.finish()
    metadata = json.loads(artifact.with_suffix(".json").read_text())
    assert metadata["frames_seen"] == 2
    assert metadata["representable_frames"] == 1
    assert metadata["representable_percent"] == pytest.approx(50.0)
    assert metadata["unrepresentable_inputs"][0]["duration_ms"] == 600

