from __future__ import annotations

from voidgym_shared.learner_contract import learner_heartbeat_timestamp, learner_monitor_state


def test_monitor_contract_publishes_both_equal_heartbeat_names_and_durations(monkeypatch) -> None:
    monkeypatch.setattr("voidgym_shared.learner_contract.time.time_ns", lambda: 123456789)
    state = learner_monitor_state(
        status="RUNNING", round_id=7, completed_steps=4096,
        resources={"ram_available_bytes": 123},
        waiting_memory=True,
        last_wait_rollouts_ms=1500.0, last_train_ms=2100.0,
        last_straggler_wait_ms=700.0,
    )
    assert state["heartbeat_unix_ns"] == state["updated_unix_ns"] == 123456789
    assert state["last_wait_rollouts_ms"] == 1500.0
    assert state["last_train_ms"] == 2100.0
    assert state["last_straggler_wait_ms"] == 700.0
    assert state["waiting_memory"] is True
    assert learner_heartbeat_timestamp(state) == 123456789


def test_monitor_contract_defaults_waiting_memory_to_false(monkeypatch) -> None:
    monkeypatch.setattr("voidgym_shared.learner_contract.time.time_ns", lambda: 1)
    state = learner_monitor_state(
        status="RUNNING", round_id=0, completed_steps=0,
        resources={},
    )
    assert state["waiting_memory"] is False


def test_control_plane_accepts_legacy_updated_timestamp_and_prefers_canonical() -> None:
    assert learner_heartbeat_timestamp({"updated_unix_ns": 10}) == 10
    assert learner_heartbeat_timestamp({"heartbeat_unix_ns": 20, "updated_unix_ns": 10}) == 20
    assert learner_heartbeat_timestamp({"heartbeat_unix_ns": True, "updated_unix_ns": False}) is None
