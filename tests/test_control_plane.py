from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from hk_agent import control_plane as control_plane_module
from hk_agent import worker_cycle as worker_cycle_module
from hk_agent.control_plane import CampaignSpec, ControlPlane
from hk_agent.control_plane_server import ControlPlaneHTTPServer
from hk_agent.runtime_paths import prepare_instance_saves


def _project(tmp_path: Path) -> Path:
    for name in (
        "run_first_training.py",
        "run_scene_change_probe.py",
        "run_scene_change_training.py",
    ):
        path = tmp_path / "tools" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("print('dry run')\n", encoding="utf-8")
    return tmp_path


def _payload(campaign_type: str = "scene_change_probe") -> dict[str, object]:
    return {
        "name": "Kings Pass exploration",
        "campaign_type": campaign_type,
        "seed": 23,
        "timesteps": 4096,
        "checkpoint_interval": 1024,
        "episodes": 3,
        "horizon": 512,
        "action_repeat": 3,
        "lease_ms": 80,
        "weights": {
            "time_cost": -0.002,
            "novelty_shaping_budget": 0.1,
            "success_reward": 10.0,
        },
    }


def test_launch_commands_use_project_virtual_environment(tmp_path: Path) -> None:
    root = _project(tmp_path)
    python = root / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"test executable marker")
    payload = _payload("exploration_training")
    payload["weights"] = {}
    spec = CampaignSpec.from_payload(payload)

    command = ControlPlane(root).build_command(spec, execute=False)

    assert command[0] == str(python)


def test_campaign_specs_are_validated_and_versioned(tmp_path: Path) -> None:
    control = ControlPlane(_project(tmp_path))
    first = control.create_campaign(_payload())
    second = control.create_campaign(_payload())
    assert first["version"] == 1
    assert second["version"] == 2
    assert first["campaign_id"] == "kings-pass-exploration"
    assert len(control.list_campaigns()) == 2
    assert first["schema_version"] == 11
    assert first["save_source"] == "existing_save"
    assert first["save_slot"] == 1


def test_continue_campaign_resumes_latest_checkpoint_with_long_straggler_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    shared = runtime / "shared" / "jogada-1"
    checkpoints = shared / "checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "round-00000074.zip"
    checkpoint.write_bytes(b"checkpoint")
    control = ControlPlane(_project(tmp_path), runtime_root=runtime)
    monkeypatch.setattr(control, "list_campaigns", lambda: [{
        "campaign_id": "jogada-1", "version": 18, "name": "jogada-1 v4",
        "campaign_type": "exploration_training", "reward_version": "explore_v4",
        "shared_learner": True, "seed": 23, "straggler_timeout_seconds": 45.0,
    }])
    created: list[dict[str, object]] = []
    monkeypatch.setattr(control, "create_campaign", lambda payload: created.append(dict(payload)) or {
        "campaign_id": "jogada-1", "version": 19,
    })
    monkeypatch.setattr(control, "launch", lambda *args, **kwargs: {"status": "RUNNING"})

    result = control.continue_operator_campaign()

    assert result["checkpoint"] == str(checkpoint)
    assert created[0]["straggler_timeout_seconds"] == 300.0
    assert "warm_start_checkpoint" not in created[0]


def test_failed_shared_learner_stops_its_actor_slots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    control = ControlPlane(_project(tmp_path), runtime_root=tmp_path / "runtime")
    shared = tmp_path / "runtime" / "shared" / "campaign"
    shared.mkdir(parents=True)
    (shared / "learner.json").write_text(json.dumps({
        "status": "FAILED", "failure": "synthetic learner failure",
    }), encoding="utf-8")
    control._launches = {
        "learner-1": {"role": "learner", "campaign_id": "campaign",
                       "shared_group": "campaign-v0001", "shared_root": str(shared),
                       "status": "FAILED", "failure": "synthetic learner failure"},
        "actor-1": {"role": "actor", "shared_group": "campaign-v0001", "status": "RUNNING"},
    }
    monkeypatch.setattr(control, "_refresh_locked", lambda: None)
    stopped: list[str] = []
    monkeypatch.setattr(control, "stop", lambda launch_id, **kwargs: stopped.append(launch_id))

    control._supervise_failed_shared_learners()

    assert stopped == ["actor-1"]


def test_shared_learner_ignores_prelaunch_heartbeat_during_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = ControlPlane(_project(tmp_path), runtime_root=tmp_path / "runtime")
    shared = tmp_path / "runtime" / "shared" / "campaign"
    shared.mkdir(parents=True)
    now = time.time_ns()
    (shared / "learner.json").write_text(json.dumps({
        "status": "TRAINING", "heartbeat_unix_ns": now - 3_600_000_000_000,
        "last_wait_rollouts_ms": 1000,
    }), encoding="utf-8")
    control._launches = {
        "learner-new": {"role": "learner", "campaign_id": "campaign",
                        "shared_group": "campaign-v0001", "shared_root": str(shared),
                        "started_unix_ns": now, "status": "RUNNING"},
        "actor-new": {"role": "actor", "shared_group": "campaign-v0001", "status": "RUNNING"},
    }
    monkeypatch.setattr(control, "_refresh_locked", lambda: None)
    stopped: list[str] = []
    monkeypatch.setattr(control, "stop", lambda launch_id, **kwargs: stopped.append(launch_id))

    control._supervise_failed_shared_learners()

    assert control._launches["learner-new"]["status"] == "RUNNING"
    assert stopped == []


def test_shared_learner_startup_without_heartbeat_expires_after_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = ControlPlane(_project(tmp_path), runtime_root=tmp_path / "runtime")
    shared = tmp_path / "runtime" / "shared" / "campaign"
    shared.mkdir(parents=True)
    now = time.time_ns()
    (shared / "learner.json").write_text(json.dumps({
        "status": "TRAINING", "heartbeat_unix_ns": now - 3_600_000_000_000,
    }), encoding="utf-8")
    control._launches = {
        "learner-new": {"role": "learner", "campaign_id": "campaign",
                        "shared_group": "campaign-v0001", "shared_root": str(shared),
                        "started_unix_ns": now - 301_000_000_000, "status": "RUNNING"},
    }
    monkeypatch.setattr(control, "_refresh_locked", lambda: None)
    monkeypatch.setattr(control, "stop", lambda *args, **kwargs: None)

    control._supervise_failed_shared_learners()

    assert control._launches["learner-new"]["status"] == "FAILED"
    assert "heartbeat expirou" in str(control._launches["learner-new"]["failure"])


def test_map_matching_prefers_capture_with_scene_bounds() -> None:
    from hk_agent.control_plane import _preferred_map_capture

    old = (Path("old.png"), {}, 100)
    newest_without_bounds = (Path("newest.png"), {"camera_x": 5.0}, 300)
    bounded = (Path("bounded.png"), {"scene_bounds": [0.0, 0.0, 20.0, 15.0]}, 200)
    assert _preferred_map_capture([old, newest_without_bounds, bounded]) == bounded


def test_operator_snapshot_surfaces_persisted_actor_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = ControlPlane(_project(tmp_path), runtime_root=tmp_path / "runtime")
    shared = tmp_path / "runtime" / "shared" / "campaign"
    (shared / "actors").mkdir(parents=True)
    now = time.time_ns()
    (shared / "manifest.json").write_text(json.dumps({"status": "RUNNING"}), "utf-8")
    (shared / "actors" / "slot-1.json").write_text(
        json.dumps({"slot": 1, "status": "STARTING", "restart_times_unix_ns": []}),
        "utf-8",
    )
    (shared / "actor-restarts.jsonl").write_text(
        json.dumps({
            "time_unix_ns": now,
            "shared_group": "campaign-v0001",
            "old_launch_id": "actor-old",
            "new_launch_id": "actor-new",
            "restart_number_in_hour": 1,
        }) + "\n",
        "utf-8",
    )
    active = [
        {"campaign_id": "campaign", "role": "learner", "shared_group": "campaign-v0001",
         "shared_root": str(shared), "status": "RUNNING", "started_unix_ns": now},
        {"campaign_id": "campaign", "role": "actor", "shared_group": "campaign-v0001",
         "resources": {"slot_id": 1}, "status": "RUNNING", "started_unix_ns": now},
    ]
    monkeypatch.setattr(control, "status", lambda: {"active_launches": active, "capacity": 2})
    monkeypatch.setattr(control, "_ensure_auto_reports", lambda _status: None)
    monkeypatch.setattr(control, "list_campaigns", lambda: [{
        "campaign_id": "campaign", "campaign_type": "exploration_training",
        "reward_version": "explore_v4", "shared_learner": True,
    }])
    monkeypatch.setattr(control, "_campaign_scene_discoveries", lambda _campaign: [])
    monkeypatch.setattr(control, "_campaign_launch_records", lambda _campaign: [
        {"campaign_id": "campaign", "launch_id": "actor-old", "resources": {"slot_id": 1}},
        {"campaign_id": "campaign", "launch_id": "actor-new", "resources": {"slot_id": 1}},
        {"campaign_id": "campaign", "launch_id": "learner-old", "role": "learner",
         "shared_group": "campaign-v0000", "status": "FAILED", "failure": "old failure"},
    ])
    monkeypatch.setattr(control, "map_art_status", lambda: {"cached": False})

    snapshot = control.operator_snapshot()
    assert not any(alert.startswith("learner caiu:") for alert in snapshot["alerts"])

    assert "Instância 1 foi relançada automaticamente (1/3 nesta hora)" in snapshot["alerts"]
    assert snapshot["instance_restarts_last_hour"] == 1
    assert snapshot["instances"][0]["restarts_last_hour"] == 1


def test_exploration_shared_campaign_defaults_and_runner_parameters(tmp_path: Path) -> None:
    payload = _payload("exploration_training")
    payload["reward_version"] = "explore_v2"
    payload["shared_learner"] = True
    payload["rollout_steps"] = 512
    payload["batch_size"] = 256
    payload["horizon"] = 3000
    payload["checkpoint_interval"] = 1024
    payload["straggler_timeout_seconds"] = 30.0
    payload["tutorial_hint_level"] = 1
    payload["weights"] = {
        "time_cost": -0.001, "novelty_reward": 0.05, "novelty_budget": 20.0,
        "scene_change_reward": 5.0, "enemy_kill_reward": 1.0,
        "enemy_kill_reward_budget": 5.0, "shade_recovery_reward": 1.0,
        "damage_penalty": -0.5, "geo_gain_reward": 0.01,
        "geo_spend_reward": 0.02, "death_penalty": -5.0,
        "run_failed_penalty": -5.0,
    }
    spec = CampaignSpec.from_payload(payload)
    command = ControlPlane(_project(tmp_path)).build_command(spec, execute=False)
    assert spec.shared_learner is True
    assert spec.reward_version == "explore_v2"
    assert spec.ent_coef == 0.01 and spec.gamma == 0.995
    assert spec.gae_lambda == 0.95 and spec.n_epochs == 4
    assert spec.target_kl == 0.03 and spec.learning_rate == 0.00025
    assert spec.clip_range == 0.2
    assert spec.straggler_timeout_seconds == 30.0
    assert spec.challenge_practice_enabled is False
    assert spec.help_requests_enabled is False
    assert spec.tutorial_hint_level == 1
    assert "--task" in command and command[command.index("--task") + 1] == "exploration"
    assert command[command.index("--rollout-steps") + 1] == "512"
    assert command[command.index("--batch-size") + 1] == "256"
    assert command[command.index("--gamma") + 1] == "0.995"
    assert command[command.index("--reward-version") + 1] == "explore_v2"
    assert command[command.index("--tutorial-hint-level") + 1] == "1"
    assert command[command.index("--geo-gain-reward") + 1] == "0.0"
    assert command[command.index("--shade-recovery-reward") + 1] == "4.0"
    assert "--lifelong-memory-root" in command


def test_challenge_and_help_features_are_explicitly_opt_in_without_contract_arguments(tmp_path: Path):
    payload = _payload("exploration_training")
    payload.update({
        "reward_version": "explore_v4", "challenge_practice_enabled": True,
        "help_requests_enabled": True,
        "weights": {
            "time_cost": -0.001, "novelty_reward": 0.05, "novelty_budget": 20.0,
            "scene_change_reward": 5.0, "enemy_kill_reward": 1.0,
            "enemy_kill_reward_budget": 5.0, "shade_recovery_reward": 4.0,
            "damage_penalty": -0.5, "geo_gain_reward": 0.0,
            "geo_spend_reward": 0.0, "death_penalty": -5.0,
            "run_failed_penalty": -5.0,
        },
    })
    spec = CampaignSpec.from_payload(payload)
    command = ControlPlane(_project(tmp_path)).build_command(spec, execute=False)
    assert spec.challenge_practice_enabled is True
    assert spec.help_requests_enabled is True
    assert "challenge_practice_enabled" not in command
    assert "help_requests_enabled" not in command
    payload["challenge_practice_enabled"] = "true"
    with pytest.raises(TypeError, match="must be booleans"):
        CampaignSpec.from_payload(payload)


def test_completed_challenge_practice_restarts_actor_in_main_playthrough(tmp_path: Path):
    plane = ControlPlane(tmp_path)
    shared_root = plane.runtime_root / "shared" / "campaign-a"
    actor_root = shared_root / "actors"
    actor_root.mkdir(parents=True)
    (actor_root / "slot-1.json").write_text(json.dumps({
        "challenge_practice_completed": True,
        "challenge_practice": {"finished": True, "finish_reason": "attempt_limit"},
    }), encoding="utf-8")
    practice = {
        "role": "actor", "status": "CANCELLED", "shared_root": str(shared_root),
        "resources": {"slot_id": 1}, "challenge_practice": {"context": "traversal:test"},
        "launch_id": "practice-launch",
    }
    main = {
        "role": "actor", "status": "RUNNING", "shared_root": str(shared_root),
        "resources": {"slot_id": 0}, "launch_id": "main-launch",
    }
    plane._launches.update({"practice-launch": practice, "main-launch": main})
    with (patch.object(plane, "_refresh_locked"),
          patch.object(plane, "restart_shared_actor", return_value={"status": "RUNNING"}) as restart):
        plane._resume_finished_challenge_practices()
    restart.assert_called_once_with(
        "practice-launch", reason="challenge practice reached exit criteria; return to main playthrough"
    )
    assert practice["challenge_return_started"] is True


def test_challenge_practice_endpoint_is_blocked_by_default(tmp_path: Path):
    payload = _payload("exploration_training")
    payload.update({
        "reward_version": "explore_v4", "save_source": "continuous_playthrough",
        "weights": {
            "time_cost": -0.001, "novelty_reward": 0.05, "novelty_budget": 20.0,
            "scene_change_reward": 5.0, "enemy_kill_reward": 1.0,
            "enemy_kill_reward_budget": 5.0, "shade_recovery_reward": 4.0,
            "damage_penalty": -0.5, "geo_gain_reward": 0.0,
            "geo_spend_reward": 0.0, "death_penalty": -5.0,
            "run_failed_penalty": -5.0,
        },
    })
    control = ControlPlane(_project(tmp_path))
    campaign = control.create_campaign(payload)
    with pytest.raises(ValueError, match="challenge practice is disabled"):
        control.start_challenge_practice({
            "campaign_id": campaign["campaign_id"], "challenge_id": "False_Knight", "slot": 1,
        })


def test_legacy_exploration_campaign_resolves_as_explore_v1(tmp_path: Path) -> None:
    control = ControlPlane(_project(tmp_path))
    payload = _payload("exploration_training")
    payload["campaign_id"] = "legacy-run"
    payload["name"] = "Legacy run"
    payload["weights"] = {}
    payload["schema_version"] = 7
    spec_path = control.control_root / "campaigns" / "legacy-run" / "v0001.json"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    spec, _ = control._resolve_campaign("legacy-run", 1)
    assert spec.reward_version == "explore_v1"


def test_new_exploration_campaign_defaults_to_explore_v3(tmp_path: Path) -> None:
    payload = _payload("exploration_training")
    payload["weights"] = {}
    spec = CampaignSpec.from_payload(payload)
    assert spec.reward_version == "explore_v3"


@pytest.mark.parametrize(
    ("field", "value"),
    [("seed", -1), ("horizon", 2), ("lease_ms", 5), ("action_repeat", 0)],
)
def test_campaign_rejects_out_of_range_values(field: str, value: int) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(ValueError, match=field):
        CampaignSpec.from_payload(payload)


def test_shared_campaign_rejects_straggler_timeout_above_supported_bound() -> None:
    payload = _payload("exploration_training")
    payload["shared_learner"] = True
    payload["straggler_timeout_seconds"] = 301
    payload["weights"] = {}
    with pytest.raises(ValueError, match="straggler_timeout_seconds"):
        CampaignSpec.from_payload(payload)


def test_campaign_rejects_unknown_type_and_weight() -> None:
    payload = _payload("arbitrary_script")
    with pytest.raises(ValueError, match="allowlisted"):
        CampaignSpec.from_payload(payload)


@pytest.mark.parametrize(
    ("source", "slot"),
    [("existing_save", 0), ("existing_save", 5), ("unknown", 1)],
)
def test_campaign_rejects_invalid_save_origin(source: str, slot: int) -> None:
    payload = _payload()
    payload["save_source"] = source
    payload["save_slot"] = slot
    with pytest.raises((TypeError, ValueError)):
        CampaignSpec.from_payload(payload)
    payload = _payload()
    payload["weights"] = {"route_hint": 99}
    with pytest.raises(ValueError, match="unknown reward weights"):
        CampaignSpec.from_payload(payload)


def test_positive_probe_alone_does_not_open_scene_gate(tmp_path: Path) -> None:
    root = _project(tmp_path)
    manifest = root / "artifacts" / "probes" / "reach-scene-change-v1" / "run-1" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "artifact_type": "scene_change_probe",
                "status": "COMPLETED",
                "gate_passed": True,
                "positive_events": 2,
                "episodes": [{}, {}],
                "started_unix_ns": 10,
            }
        ),
        encoding="utf-8",
    )
    control = ControlPlane(root)
    runs = control.list_runs()
    assert runs[0]["metrics"] == {"positive_events": 2}
    assert runs[0]["episodes"] == 2
    assert control.scene_change_gate() == {"passed": False, "evidence": None}


def test_hashed_manual_review_opens_scene_gate(tmp_path: Path) -> None:
    root = _project(tmp_path)
    manifest = root / "artifacts" / "probes" / "reach-scene-change-v1" / "run-1" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "artifact_type": "scene_change_probe",
                "status": "COMPLETED",
                "positive_events": 1,
            }
        ),
        encoding="utf-8",
    )
    evidence_relative = manifest.relative_to(root).as_posix()
    gate = root / "artifacts" / "gates" / "reach-scene-change-v1" / "review-1.json"
    gate.parent.mkdir(parents=True)
    gate.write_text(
        json.dumps(
            {
                "artifact_type": "scene_change_gate_review",
                "status": "APPROVED",
                "gate_passed": True,
                "manual_reviewed": True,
                "evidence_probe": evidence_relative,
                "evidence_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    control = ControlPlane(root)
    assert control.scene_change_gate() == {
        "passed": True,
        "evidence": evidence_relative,
    }


def test_scene_change_training_is_blocked_until_probe_gate_passes(tmp_path: Path) -> None:
    control = ControlPlane(_project(tmp_path), allow_execute=True)
    payload = _payload("scene_change_training")
    campaign = control.create_campaign(payload)
    with pytest.raises(PermissionError, match="manual gate review"):
        control.launch(str(campaign["campaign_id"]), version=1, execute=True)


def test_real_launch_requires_explicit_server_flag(tmp_path: Path) -> None:
    control = ControlPlane(_project(tmp_path), allow_execute=False)
    campaign = control.create_campaign(_payload())
    with pytest.raises(PermissionError, match="--allow-execute"):
        control.launch(str(campaign["campaign_id"]), version=1, execute=True)


def test_execute_command_refuses_missing_managed_copy(tmp_path: Path) -> None:
    spec = CampaignSpec.from_payload(_real_campaign_payload())
    control = ControlPlane(_project(tmp_path), allow_execute=True)
    with pytest.raises(ValueError, match="validated managed installation"):
        control.build_command(spec, execute=True)


def test_command_is_allowlisted_argument_vector_with_no_shell_text(tmp_path: Path) -> None:
    control = ControlPlane(_project(tmp_path))
    spec = CampaignSpec.from_payload(_payload())
    command = control.build_command(spec, execute=False)
    assert command[1].endswith("run_scene_change_probe.py")
    assert "--execute" not in command
    assert command[command.index("--save-source") + 1] == "existing_save"
    assert command[command.index("--save-slot") + 1] == "1"
    assert command[command.index("--window-width") + 1] == "960"
    assert command[command.index("--window-height") + 1] == "540"
    assert all(";" not in item and "&&" not in item for item in command)
    stop_file = tmp_path / "artifacts" / "control-plane" / "launches" / "x" / "stop.request"
    command = control.build_command(spec, execute=False, stop_file=stop_file)
    assert command[-2:] == ["--stop-file", str(stop_file)]


def test_parallel_smoke_campaign_uses_controlled_probe_without_training(
    tmp_path: Path,
) -> None:
    payload = _payload("parallel_smoke")
    payload["weights"] = {}
    payload["horizon"] = 10_000
    spec = CampaignSpec.from_payload(payload)
    command = ControlPlane(_project(tmp_path)).build_command(spec, execute=False)

    assert command[1].endswith("run_scene_change_probe.py")
    assert command[command.index("--bridge-smoke-seconds") + 1] == "90"
    assert command[command.index("--horizon") + 1] == "10000"
    assert command[command.index("--action-repeat") + 1] == "1"
    assert "--training-device" not in command
    assert "--execute" not in command


def test_training_worker_session_resolves_only_validated_preflight_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload("visual_motion_training")
    payload["worker_session_id"] = "pc-sala-run-01"
    payload["weights"] = {
        "time_cost": -0.001,
        "visual_change_scale": 0.1,
        "success_reward": 1.0,
    }
    spec = CampaignSpec.from_payload(payload)
    policy = tmp_path / "cache" / "policy.safetensors"
    sidecar = tmp_path / "cache" / "policy.manifest.json"
    monkeypatch.setattr("hk_agent.worker_cycle.load_worker_config", lambda _: object())
    monkeypatch.setattr(
        "hk_agent.worker_cycle.warm_start_from_preflight",
        lambda config, session: worker_cycle_module.FrozenWarmStart(
            policy=policy,
            manifest=sidecar,
            policy_sha256="a" * 64,
            manifest_sha256="b" * 64,
            manifest_id="c" * 64,
        ),
    )

    command = ControlPlane(_project(tmp_path)).build_command(spec, execute=False)

    assert command[command.index("--warm-start-policy") + 1] == str(policy)
    assert command[command.index("--warm-start-policy-manifest") + 1] == str(sidecar)
    assert command[command.index("--warm-start-policy-sha256") + 1] == "a" * 64


def test_probe_rejects_worker_session() -> None:
    payload = _payload()
    payload["worker_session_id"] = "pc-sala-run-01"
    with pytest.raises(ValueError, match="cannot consume"):
        CampaignSpec.from_payload(payload)


def test_geo_weights_are_validated_and_forwarded_to_runner(tmp_path: Path) -> None:
    payload = _payload()
    weights = payload["weights"]
    assert isinstance(weights, dict)
    weights.update({"geo_gain_reward": 0.01, "geo_spend_reward": 0.02})
    spec = CampaignSpec.from_payload(payload)
    command = ControlPlane(_project(tmp_path)).build_command(spec, execute=False)
    assert command[command.index("--geo-gain-reward") + 1] == "0.01"
    assert command[command.index("--geo-spend-reward") + 1] == "0.02"

    weights["geo_gain_reward"] = -0.01
    with pytest.raises(ValueError, match="geo_gain_reward"):
        CampaignSpec.from_payload(payload)


def test_continuous_training_is_explicit_and_probe_cannot_fake_infinity(
    tmp_path: Path,
) -> None:
    payload = _payload("visual_motion_training")
    payload["duration_mode"] = "continuous"
    payload["max_checkpoints"] = 5
    payload["weights"] = {
        "time_cost": -0.001,
        "visual_change_scale": 0.1,
        "success_reward": 1.0,
    }
    spec = CampaignSpec.from_payload(payload)
    command = ControlPlane(_project(tmp_path)).build_command(spec, execute=False)
    assert "--continuous" in command
    assert command[command.index("--max-checkpoints") + 1] == "5"

    probe = _payload()
    probe["duration_mode"] = "continuous"
    with pytest.raises(ValueError, match="do not support continuous"):
        CampaignSpec.from_payload(probe)


def test_restart_reattaches_live_launch_and_stop_remains_cooperative(
    tmp_path: Path, monkeypatch: Any
) -> None:
    root = _project(tmp_path)
    launch_root = root / "artifacts" / "control-plane" / "launches" / "launch-1"
    launch_root.mkdir(parents=True)
    launch = {
        "launch_id": "launch-1",
        "campaign_id": "continuous-test",
        "pid": 31337,
        "runner_process": {"pid": 31337, "creation_time": 100, "executable": "runner.exe"},
        "launch_nonce": "launch-1",
        "status": "RUNNING",
        "stop_file": (
            "artifacts/control-plane/launches/launch-1/stop.request"
        ),
        "log_path": "artifacts/control-plane/launches/launch-1/process.log",
    }
    (launch_root / "launch.json").write_text(json.dumps(launch), encoding="utf-8")
    monkeypatch.setattr(ControlPlane, "_pid_is_alive", staticmethod(lambda pid: pid == 31337))
    monkeypatch.setattr("hk_agent.control_plane.process_identity", lambda pid: {"pid": pid, "creation_time": 100, "executable": "runner.exe"})

    control = ControlPlane(root)
    active = control.status()["active"]
    assert isinstance(active, dict)
    assert active["reattached"] is True
    stopped = control.stop()
    assert stopped == {"status": "STOP_REQUESTED", "pid": 31337}
    assert (launch_root / "stop.request").is_file()


def test_windows_launch_is_suspended_assigned_then_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    control = ControlPlane(root)
    campaign = control.create_campaign(_payload())
    events: list[object] = []

    class FakeJob:
        @classmethod
        def create(cls):
            events.append("create-job")
            return cls()

        def assign_pid(self, pid: int) -> None:
            events.append(("assign", pid))

        def close(self) -> None:
            events.append("close")

        def process_ids(self) -> list[int]:
            return []

    class RecordingProcess(_FakeProcess):
        def __init__(self, command, **kwargs):
            events.append(("popen", kwargs["creationflags"]))
            super().__init__(command, **kwargs)

    monkeypatch.setattr("hk_agent.control_plane._windows_job_enabled", lambda: True)
    monkeypatch.setattr("hk_agent.control_plane.WindowsJob", FakeJob)
    monkeypatch.setattr("hk_agent.control_plane._REAL_POPEN", RecordingProcess)
    monkeypatch.setattr("hk_agent.control_plane.resume_process", lambda pid: events.append(("resume", pid)))
    monkeypatch.setattr(ControlPlane, "_discover_runner_process", staticmethod(lambda *_args: None))
    with patch("hk_agent.control_plane.subprocess.Popen", RecordingProcess):
        launched = control.launch(str(campaign["campaign_id"]), version=1, execute=False)
    assert [event if isinstance(event, str) else event[0] for event in events[:4]] == [
        "create-job", "popen", "assign", "resume"
    ]
    popen_flags = events[1][1]  # type: ignore[index]
    assert popen_flags & getattr(subprocess, "CREATE_SUSPENDED", 0x00000004)
    _finish_launch(control, str(launched["launch_id"]))


def test_recovery_reattaches_runner_sidecar_when_redirector_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    launch_root = root / "artifacts" / "control-plane" / "launches" / "child-runner"
    launch_root.mkdir(parents=True)
    record = {
        "launch_id": "child-runner", "mode": "dry-run", "pid": 4240,
        "runner_process": {"pid": 4240, "creation_time": 10, "executable": "redirector.exe"},
        "launch_nonce": "child-runner", "status": "RUNNING",
        "resources": {"slot_id": None},
    }
    (launch_root / "launch.json").write_text(json.dumps(record), encoding="utf-8")
    (launch_root / "runner-process.json").write_text(json.dumps({
        "pid": 4241, "creation_time": 20, "executable": "base-python.exe",
        "parent_pid": 4240, "launch_nonce": "child-runner",
    }), encoding="utf-8")
    monkeypatch.setattr(ControlPlane, "_pid_is_alive", staticmethod(lambda pid: pid == 4241))
    monkeypatch.setattr("hk_agent.control_plane.process_identity", lambda pid: {
        "pid": pid, "creation_time": 20, "executable": "base-python.exe",
    })
    control = ControlPlane(root)
    active = control.status()["active_launches"]
    assert isinstance(active, list) and active[0]["reattached"] is True
    assert active[0]["runner_process"]["pid"] == 4241  # type: ignore[index]


def test_job_descendants_keep_slot_reserved_until_job_is_empty(tmp_path: Path) -> None:
    root = _project(tmp_path)
    control = ControlPlane(root)
    launch_id = "draining-job"
    launch_root = control.control_root / "launches" / launch_id
    launch_root.mkdir(parents=True)
    record: dict[str, object] = {
        "launch_id": launch_id,
        "mode": "execute",
        "status": "RUNNING",
        "resources": {"slot_id": 0},
    }
    process = _FakeProcess([])
    process._return_code = 0
    process._done.set()
    release = threading.Event()
    terminated = threading.Event()

    class DrainingJob:
        def process_ids(self) -> list[int]:
            if not terminated.is_set() or not release.is_set():
                return [999]
            return []

        def terminate(self) -> None:
            terminated.set()

        def close(self) -> None:
            return None

    control._launches[launch_id] = record
    control._processes[launch_id] = process
    control._jobs[launch_id] = DrainingJob()  # type: ignore[assignment]
    control._slot_owners[0] = launch_id
    control._slot_fences[0] = "fence"
    log_handle = (launch_root / "process.log").open("wb")
    waiter = threading.Thread(
        target=control._wait_for_process,
        args=(launch_id, process, log_handle, launch_root),
        daemon=True,
    )
    waiter.start()
    assert terminated.wait(2)
    assert control._slot_owners[0] == launch_id
    assert control.status()["active_count"] == 1
    release.set()
    waiter.join(2)
    assert not waiter.is_alive()
    assert launch_id not in control._slot_owners.values()
    assert record["status"] == "FAILED"
    assert record["draining_pids"] == []


def test_runner_sidecar_with_reused_pid_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    launch_root = root / "artifacts" / "control-plane" / "launches" / "runner-pid-reuse"
    launch_root.mkdir(parents=True)
    (launch_root / "launch.json").write_text(json.dumps({
        "launch_id": "runner-pid-reuse", "mode": "dry-run", "pid": 4240,
        "runner_process": {"pid": 4240, "creation_time": 10, "executable": "redirector.exe"},
        "launch_nonce": "runner-pid-reuse", "status": "RUNNING",
        "resources": {"slot_id": None},
    }), encoding="utf-8")
    (launch_root / "runner-process.json").write_text(json.dumps({
        "pid": 4241, "creation_time": 20, "executable": "base-python.exe",
        "parent_pid": 4240, "launch_nonce": "runner-pid-reuse",
    }), encoding="utf-8")
    monkeypatch.setattr(ControlPlane, "_pid_is_alive", staticmethod(lambda pid: pid == 4241))
    monkeypatch.setattr("hk_agent.control_plane.process_identity", lambda pid: {
        "pid": pid, "creation_time": 21, "executable": "base-python.exe",
    })
    control = ControlPlane(root)
    assert control.status()["active_count"] == 0
    saved = json.loads((launch_root / "launch.json").read_text("utf-8"))
    assert saved["status"] == "FAILED"


def test_operator_browser_form_hides_fixed_reward_weights() -> None:
    html = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "hk_agent"
        / "control_plane_web"
        / "index.html"
    ).read_text("utf-8")
    assert 'name="success_reward"' not in html
    assert 'name="enemy_kill_reward"' not in html
    assert 'name="death_penalty"' not in html
    assert 'class="panel advanced"' in html
    assert 'id="start-training"' in html
    assert 'id="continue-training"' in html
    assert 'id="repeat-training"' in html


class _FakeProcess:
    next_pid = 31337

    def __init__(self, command: list[str], **kwargs: Any) -> None:
        self.command = command
        self.kwargs = kwargs
        self.pid = self.next_pid
        type(self).next_pid += 1
        self._done = threading.Event()
        self._return_code: int | None = None

    def wait(self, timeout: float | None = None) -> int:
        self._done.wait(timeout=timeout or 2)
        return 0 if self._return_code is None else self._return_code

    def poll(self) -> int | None:
        return self._return_code

    def send_signal(self, _: int) -> None:
        self._return_code = 130
        self._done.set()

    def terminate(self) -> None:
        self.send_signal(0)

    def kill(self) -> None:
        self.send_signal(0)


def _real_campaign_payload() -> dict[str, object]:
    payload = _payload("visual_motion_training")
    payload["weights"] = {
        "time_cost": -0.001,
        "visual_change_scale": 0.1,
        "success_reward": 1.0,
    }
    return payload


def _patch_execution_config(
    monkeypatch: pytest.MonkeyPatch, root: Path, runtime_root: Path
) -> Path:
    baseline = root / "baseline"
    baseline.mkdir(parents=True, exist_ok=True)
    (baseline / "user1.dat").write_bytes(b"baseline-save")
    personal = root / "personal"
    backup = root / "backup"
    install = root / "game"
    for directory in (personal, backup, install):
        directory.mkdir(parents=True, exist_ok=True)
    config = SimpleNamespace(
        personal_save_root=personal.resolve(),
        experimental_save_root=baseline.resolve(),
        backup_root=backup.resolve(),
        game_executable=(install / "hollow_knight.exe").resolve(),
        runtime_root=runtime_root.resolve(),
        install_roots=(root / "hk-ai-1", root / "hk-ai-2"),
        instance_data_roots=(root / "data-1", root / "data-2"),
        instance_product_names=("Hollow Knight AI 1", "Hollow Knight AI 2"),
    )
    for index, install_root in enumerate(config.install_roots, start=1):
        install_root.mkdir(parents=True, exist_ok=True)
        config.instance_data_roots[index - 1].mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "hk_agent.instance_marker.load_instance_marker",
        lambda install_root, *, expected_index: SimpleNamespace(
            persistent_data_path_expected=str(config.instance_data_roots[expected_index - 1]),
            product_name=f"Hollow Knight AI {expected_index}",
        ),
    )
    monkeypatch.setattr("hk_agent.config.load_config", lambda *args, **kwargs: config)
    return baseline


def _finish_launch(control: ControlPlane, launch_id: str, return_code: int = 0) -> None:
    process = control._processes[launch_id]
    assert isinstance(process, _FakeProcess)
    process._return_code = return_code
    process._done.set()
    deadline = time.monotonic() + 2
    while launch_id in control._processes and time.monotonic() < deadline:
        time.sleep(0.01)
    assert launch_id not in control._processes


def test_real_slots_capacity_and_dry_runs_are_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    _patch_execution_config(monkeypatch, root, runtime)
    control = ControlPlane(
        root,
        allow_execute=True,
        max_active_launches=2,
        parallel_enabled=True,
        runtime_root=runtime,
    )
    campaign = control.create_campaign(_real_campaign_payload())
    campaign_id = str(campaign["campaign_id"])
    with patch("hk_agent.control_plane.subprocess.Popen", _FakeProcess):
        first = control.launch(campaign_id, version=1, execute=True)
        dry_run = control.launch(campaign_id, version=1, execute=False)
        second = control.launch(campaign_id, version=1, execute=True)
        assert first["resources"]["slot_id"] == 0  # type: ignore[index]
        assert dry_run["resources"]["slot_id"] is None  # type: ignore[index]
        assert second["resources"]["slot_id"] == 1  # type: ignore[index]
        for launch, index in ((first, 1), (second, 2)):
            resources = launch["resources"]
            assert resources["game_install"] == str(root / f"hk-ai-{index}")  # type: ignore[index]
            assert resources["instance_data_root"] == str(root / f"data-{index}")  # type: ignore[index]
            assert resources["product_name"] == f"Hollow Knight AI {index}"  # type: ignore[index]
            assert "--game-install" in launch["command"]  # type: ignore[operator]
            assert str(root / f"hk-ai-{index}") in launch["command"]  # type: ignore[operator]
            assert "--instance-data-root" in launch["command"]  # type: ignore[operator]
            assert str(resources["log_file"]).endswith("Player.log")  # type: ignore[index]
        with pytest.raises(RuntimeError, match="capacity reached"):
            control.launch(campaign_id, version=1, execute=True)
        for launch in (first, dry_run, second):
            _finish_launch(control, str(launch["launch_id"]))
        for launch in (first, second):
            runtime_launch = runtime / "launches" / str(launch["launch_id"])
            assert (runtime_launch / "previous-instance-state").is_dir()
            assert (runtime_launch / "final-save" / "user1.dat").read_bytes() == b"baseline-save"


def test_parallel_disabled_blocks_second_real_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    _patch_execution_config(monkeypatch, root, runtime)
    control = ControlPlane(root, allow_execute=True, max_active_launches=2, runtime_root=runtime)
    campaign = control.create_campaign(_real_campaign_payload())
    with patch("hk_agent.control_plane.subprocess.Popen", _FakeProcess):
        first = control.launch(str(campaign["campaign_id"]), version=1, execute=True)
        with pytest.raises(RuntimeError, match="parallel execution is disabled"):
            control.launch(str(campaign["campaign_id"]), version=1, execute=True)
        _finish_launch(control, str(first["launch_id"]))


@pytest.mark.parametrize(
    ("stop_first", "return_code", "expected_status"),
    [(False, 1, "FAILED"), (True, 0, "CANCELLED")],
)
def test_final_instance_saves_are_captured_on_failure_and_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop_first: bool,
    return_code: int,
    expected_status: str,
) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    _patch_execution_config(monkeypatch, root, runtime)
    control = ControlPlane(root, allow_execute=True, runtime_root=runtime)
    campaign = control.create_campaign(_real_campaign_payload())
    with patch("hk_agent.control_plane.subprocess.Popen", _FakeProcess):
        launch = control.launch(str(campaign["campaign_id"]), version=1, execute=True)
        resources = launch["resources"]
        data_root = Path(str(resources["instance_data_root"]))  # type: ignore[index]
        (data_root / "user1.dat").write_bytes(b"final-instance-save")
        if stop_first:
            assert control.stop(str(launch["launch_id"]))["status"] == "STOP_REQUESTED"
        _finish_launch(control, str(launch["launch_id"]), return_code=return_code)

    launch_root = runtime / "launches" / str(launch["launch_id"])
    record_path = (
        root
        / "artifacts"
        / "control-plane"
        / "launches"
        / str(launch["launch_id"])
        / "launch.json"
    )
    record = json.loads(record_path.read_text("utf-8"))
    assert record["status"] == expected_status
    assert (launch_root / "final-save" / "user1.dat").read_bytes() == b"final-instance-save"


def test_slots_and_seeds_are_reproducible_after_finish_and_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    _patch_execution_config(monkeypatch, root, runtime)
    control = ControlPlane(root, allow_execute=True, runtime_root=runtime)
    campaign = control.create_campaign(_real_campaign_payload())
    campaign_id = str(campaign["campaign_id"])
    with patch("hk_agent.control_plane.subprocess.Popen", _FakeProcess):
        first = control.launch(campaign_id, version=1, execute=True)
        assert first["resources"]["effective_seed"] == 23  # type: ignore[index]
        _finish_launch(control, str(first["launch_id"]))
        second = control.launch(campaign_id, version=1, execute=True)
        control.stop(str(second["launch_id"]))
        _finish_launch(control, str(second["launch_id"]))
        third = control.launch(campaign_id, version=1, execute=True)
        assert third["resources"]["slot_id"] == 0  # type: ignore[index]
        assert third["resources"]["effective_seed"] == 23  # type: ignore[index]
        _finish_launch(control, str(third["launch_id"]))


def test_reattached_slot_is_reserved_without_seed_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    _patch_execution_config(monkeypatch, root, runtime)
    launch_root = root / "artifacts" / "control-plane" / "launches" / "reattached"
    launch_root.mkdir(parents=True)
    (launch_root / "launch.json").write_text(
        json.dumps(
            {
                "launch_id": "reattached",
                "mode": "execute",
                "pid": 4242,
                "runner_process": {"pid": 4242, "creation_time": 100, "executable": "runner.exe"},
                "launch_nonce": "reattached",
                "status": "RUNNING",
                "started_unix_ns": 1,
                "stop_file": "artifacts/control-plane/launches/reattached/stop.request",
                "log_path": "artifacts/control-plane/launches/reattached/process.log",
                "campaign_origin": {"isolated_save_root": "relative-save-root"},
                "resources": {"slot_id": 1, "effective_seed": 24},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ControlPlane, "_pid_is_alive", staticmethod(lambda pid: pid == 4242))
    monkeypatch.setattr("hk_agent.control_plane.process_identity", lambda pid: {"pid": pid, "creation_time": 100, "executable": "runner.exe"})
    control = ControlPlane(
        root,
        allow_execute=True,
        max_active_launches=2,
        parallel_enabled=True,
        runtime_root=runtime,
    )
    campaign = control.create_campaign(_real_campaign_payload())
    with patch("hk_agent.control_plane.subprocess.Popen", _FakeProcess):
        launch = control.launch(str(campaign["campaign_id"]), version=1, execute=True)
        assert launch["resources"]["slot_id"] == 0  # type: ignore[index]
        assert launch["resources"]["effective_seed"] == 23  # type: ignore[index]
        _finish_launch(control, str(launch["launch_id"]))


def test_child_environment_contains_launch_slot_and_thread_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    _patch_execution_config(monkeypatch, root, runtime)
    control = ControlPlane(
        root, allow_execute=True, runtime_root=runtime, torch_threads=4, torch_interop_threads=2
    )
    campaign = control.create_campaign(_real_campaign_payload())
    with patch("hk_agent.control_plane.subprocess.Popen", _FakeProcess):
        launch = control.launch(str(campaign["campaign_id"]), version=1, execute=True)
        process = control._processes[str(launch["launch_id"])]
        environment = process.kwargs["env"]  # type: ignore[attr-defined]
        assert environment["HK_AGENT_LAUNCH_ID"] == launch["launch_id"]
        assert environment["HK_AGENT_SLOT_ID"] == "0"
        assert environment["OMP_NUM_THREADS"] == "4"
        assert environment["MKL_NUM_THREADS"] == "4"
        assert environment["OPENBLAS_NUM_THREADS"] == "4"
        _finish_launch(control, str(launch["launch_id"]))


def test_scene_training_uses_per_slot_route_memory_and_legacy_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    _patch_execution_config(monkeypatch, root, runtime)
    legacy = (
        root
        / "artifacts"
        / "training"
        / "reach-scene-change-v1"
        / "route-memory-existing_save-slot-1.json"
    )
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"legacy": true}', encoding="utf-8")
    control = ControlPlane(
        root,
        allow_execute=True,
        runtime_root=runtime,
        training_device="cpu",
        torch_threads=3,
        torch_interop_threads=1,
    )
    monkeypatch.setattr(control, "scene_change_gate", lambda: {"passed": True})
    campaign = control.create_campaign(_payload("scene_change_training"))
    with patch("hk_agent.control_plane.subprocess.Popen", _FakeProcess):
        launch = control.launch(str(campaign["campaign_id"]), version=1, execute=True)
        command = launch["command"]
        assert isinstance(command, list)
        route_memory = Path(command[command.index("--route-memory") + 1])
        assert route_memory == (
            runtime / "route-memory" / "existing_save-slot-1" / "instance-0.json"
        ).resolve()
        assert route_memory.read_text("utf-8") == '{"legacy": true}'
        assert command[command.index("--training-device") + 1] == "cpu"
        assert command[command.index("--torch-threads") + 1] == "3"
        assert command[command.index("--torch-interop-threads") + 1] == "1"
        _finish_launch(control, str(launch["launch_id"]))


def test_execute_clones_save_inside_runtime_without_mutating_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    baseline = _patch_execution_config(monkeypatch, root, runtime)
    before = hashlib.sha256((baseline / "user1.dat").read_bytes()).hexdigest()
    control = ControlPlane(root, allow_execute=True, runtime_root=runtime)
    campaign = control.create_campaign(_real_campaign_payload())
    with patch("hk_agent.control_plane.subprocess.Popen", _FakeProcess):
        launch = control.launch(str(campaign["campaign_id"]), version=1, execute=True)
        clone = Path(str(launch["campaign_origin"]["isolated_save_root"]))  # type: ignore[index]
        clone.relative_to(runtime.resolve())
        assert (clone / "user1.dat").read_bytes() == b"baseline-save"
        after = hashlib.sha256((baseline / "user1.dat").read_bytes()).hexdigest()
        assert after == before
        _finish_launch(control, str(launch["launch_id"]))


def test_copy_failure_releases_slot_and_removes_only_launch_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    baseline = _patch_execution_config(monkeypatch, root, runtime)
    control = ControlPlane(root, allow_execute=True, runtime_root=runtime)
    campaign = control.create_campaign(_real_campaign_payload())
    original_copytree = shutil.copytree

    def fail_copytree(*args: Any, **kwargs: Any) -> None:
        raise OSError("copy failed")

    monkeypatch.setattr("hk_agent.control_plane.shutil.copytree", fail_copytree)
    with pytest.raises(OSError, match="copy failed"):
        control.launch(str(campaign["campaign_id"]), version=1, execute=True)
    assert control.status()["active_count"] == 0
    assert (baseline / "user1.dat").is_file()
    assert any((runtime / "launches").glob("*/previous-instance-state"))
    failed = json.loads((root / "artifacts" / "control-plane" / "launches" / next(
        path.name for path in (root / "artifacts" / "control-plane" / "launches").iterdir()
    ) / "launch.json").read_text("utf-8"))
    assert failed["status"] == "FAILED"
    assert failed["failure"] == "copy failed"
    failed_runtime = runtime / "launches" / str(failed["launch_id"])
    assert (failed_runtime / "final-save" / "user1.dat").read_bytes() == b"baseline-save"

    monkeypatch.setattr("hk_agent.control_plane.shutil.copytree", original_copytree)
    with patch("hk_agent.control_plane.subprocess.Popen", _FakeProcess):
        launch = control.launch(str(campaign["campaign_id"]), version=1, execute=True)
        assert launch["resources"]["slot_id"] == 0  # type: ignore[index]
        _finish_launch(control, str(launch["launch_id"]))


def test_runtime_inside_baseline_is_rejected_at_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    runtime = root / "baseline" / "runtime"
    _patch_execution_config(monkeypatch, root, runtime)
    control = ControlPlane(root, allow_execute=True, runtime_root=runtime)
    campaign = control.create_campaign(_real_campaign_payload())
    with pytest.raises(ValueError, match="experimental saves"):
        control.launch(str(campaign["campaign_id"]), version=1, execute=True)
    assert control.status()["active_count"] == 0


def test_status_does_not_wait_for_save_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    _patch_execution_config(monkeypatch, root, runtime)
    control = ControlPlane(root, allow_execute=True, runtime_root=runtime)
    campaign = control.create_campaign(_real_campaign_payload())
    copy_started = threading.Event()
    release_copy = threading.Event()
    original_copytree = shutil.copytree

    def blocking_copytree(*args: Any, **kwargs: Any) -> Any:
        copy_started.set()
        assert release_copy.wait(timeout=2)
        return original_copytree(*args, **kwargs)

    monkeypatch.setattr("hk_agent.control_plane.shutil.copytree", blocking_copytree)
    result: dict[str, object] = {}

    def run_launch() -> None:
        result.update(control.launch(str(campaign["campaign_id"]), version=1, execute=True))

    with patch("hk_agent.control_plane.subprocess.Popen", _FakeProcess):
        thread = threading.Thread(target=run_launch)
        thread.start()
        assert copy_started.wait(timeout=2)
        started = time.monotonic()
        status = control.status()
        elapsed = time.monotonic() - started
        assert elapsed < 0.5
        assert status["active_launches"][0]["status"] == "PREPARING"  # type: ignore[index]
        release_copy.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        _finish_launch(control, str(result["launch_id"]))


def test_legacy_running_without_identity_is_failed_and_does_not_block_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    _patch_execution_config(monkeypatch, root, runtime)
    legacy_root = root / "artifacts" / "control-plane" / "launches" / "old-launch"
    legacy_root.mkdir(parents=True)
    (legacy_root / "launch.json").write_text(json.dumps({
        "launch_id": "old-launch", "pid": 8001, "status": "RUNNING",
        "stop_file": "artifacts/control-plane/launches/old-launch/stop.request",
    }), encoding="utf-8")
    monkeypatch.setattr(ControlPlane, "_pid_is_alive", staticmethod(lambda _pid: True))
    control = ControlPlane(root, allow_execute=True, runtime_root=runtime)
    failed_record = json.loads((legacy_root / "launch.json").read_text("utf-8"))
    assert failed_record["status"] == "FAILED"
    assert failed_record["failure"] == "processo não pertence mais a este launch"
    campaign = control.create_campaign(_real_campaign_payload())
    with patch("hk_agent.control_plane.subprocess.Popen", _FakeProcess):
        launched = control.launch(str(campaign["campaign_id"]), version=1, execute=True)
        assert launched["resources"]["slot_id"] == 0  # type: ignore[index]
        _finish_launch(control, str(launched["launch_id"]))


def test_preparing_record_is_persisted_and_stop_cancels_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    _patch_execution_config(monkeypatch, root, runtime)
    control = ControlPlane(root, allow_execute=True, runtime_root=runtime)
    campaign = control.create_campaign(_real_campaign_payload())
    entered, release = threading.Event(), threading.Event()
    original_copytree = shutil.copytree
    def blocked_copy(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(3)
        return original_copytree(*args, **kwargs)
    monkeypatch.setattr("hk_agent.control_plane.shutil.copytree", blocked_copy)
    errors: list[InterruptedError] = []
    def launch() -> None:
        try:
            control.launch(str(campaign["campaign_id"]), version=1, execute=True)
        except InterruptedError as error:
            errors.append(error)
    thread = threading.Thread(target=launch)
    thread.start()
    assert entered.wait(2)
    launch_id = next(iter(control._preparing_ids))
    launch_json = root / "artifacts" / "control-plane" / "launches" / launch_id / "launch.json"
    assert json.loads(launch_json.read_text("utf-8"))["status"] == "PREPARING"
    assert control.stop(launch_id) == {"status": "CANCEL_REQUESTED"}
    release.set()
    thread.join(3)
    assert not thread.is_alive()
    record = json.loads(launch_json.read_text("utf-8"))
    assert record["status"] == "CANCELLED"
    assert control.status()["active_count"] == 0
    assert (runtime / "launches" / launch_id / "previous-instance-state").is_dir()
    assert errors and isinstance(errors[0], InterruptedError)


def test_preparation_timeout_fails_and_releases_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    _patch_execution_config(monkeypatch, root, runtime)
    control = ControlPlane(root, allow_execute=True, runtime_root=runtime, preparation_timeout_seconds=0)
    campaign = control.create_campaign(_real_campaign_payload())
    with pytest.raises(TimeoutError, match="timed out"):
        control.launch(str(campaign["campaign_id"]), version=1, execute=True)
    record_path = next((root / "artifacts" / "control-plane" / "launches").glob("*/launch.json"))
    assert json.loads(record_path.read_text("utf-8"))["status"] == "FAILED"
    assert control.status()["active_count"] == 0


def test_blocked_copy_timeout_releases_slot_and_cleans_after_copy_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    _patch_execution_config(monkeypatch, root, runtime)
    control = ControlPlane(root, allow_execute=True, runtime_root=runtime, preparation_timeout_seconds=0.5)
    campaign = control.create_campaign(_real_campaign_payload())
    entered, release = threading.Event(), threading.Event()
    original_copy = shutil.copytree

    def blocked_copy(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(3)
        return original_copy(*args, **kwargs)

    monkeypatch.setattr("hk_agent.control_plane.shutil.copytree", blocked_copy)
    errors: list[Exception] = []
    thread = threading.Thread(target=lambda: _capture_launch_error(control, campaign, errors))
    with patch("hk_agent.control_plane.subprocess.Popen", _FakeProcess):
        thread.start()
        assert entered.wait(2)
        launch_json = next((root / "artifacts" / "control-plane" / "launches").glob("*/launch.json"))
        for _ in range(100):
            saved = json.loads(launch_json.read_text("utf-8"))
            if saved["status"] == "FAILED":
                break
            threading.Event().wait(0.01)
        assert saved["status"] == "FAILED"
        draining = control.status()
        assert draining["active_count"] == 1
        assert draining["active"]["slot_state"] == "draining"  # type: ignore[index]
        assert thread.is_alive()
        release.set()
        thread.join(3)
        assert not thread.is_alive()
        assert errors and isinstance(errors[0], TimeoutError)
        assert (runtime / "launches" / launch_json.parent.name / "previous-instance-state").is_dir()
        assert control._processes == {}


def _capture_launch_error(control: ControlPlane, campaign: Mapping[str, object], errors: list[Exception]) -> None:
    try:
        control.launch(str(campaign["campaign_id"]), version=1, execute=True)
    except Exception as error:  # noqa: BLE001
        errors.append(error)


def test_stop_during_blocked_copy_cancels_immediately_and_releases_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    _patch_execution_config(monkeypatch, root, runtime)
    control = ControlPlane(root, allow_execute=True, runtime_root=runtime)
    campaign = control.create_campaign(_real_campaign_payload())
    entered, release = threading.Event(), threading.Event()
    original_copy = shutil.copytree

    def blocked_copy(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        assert release.wait(3)
        return original_copy(*args, **kwargs)

    monkeypatch.setattr("hk_agent.control_plane.shutil.copytree", blocked_copy)
    errors: list[Exception] = []
    thread = threading.Thread(target=lambda: _capture_launch_error(control, campaign, errors))
    with patch("hk_agent.control_plane.subprocess.Popen", _FakeProcess):
        thread.start()
        assert entered.wait(2)
        launch_id = next(iter(control._preparing_ids))
        assert control.stop(launch_id) == {"status": "CANCEL_REQUESTED"}
        draining = control.status()
        assert draining["active_count"] == 1
        assert draining["active"]["slot_state"] == "draining"  # type: ignore[index]
        assert thread.is_alive()
        release.set()
        thread.join(3)
        assert not thread.is_alive()
        assert errors and isinstance(errors[0], InterruptedError)
        assert (runtime / "launches" / launch_id / "previous-instance-state").is_dir()
        assert control._processes == {}


def test_restart_reattaches_live_dry_run_without_blocking_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    _patch_execution_config(monkeypatch, root, runtime)
    launch_root = root / "artifacts" / "control-plane" / "launches" / "live-dry-run"
    launch_root.mkdir(parents=True)
    (launch_root / "launch.json").write_text(json.dumps({
        "launch_id": "live-dry-run", "mode": "dry-run", "pid": 4243,
        "runner_process": {"pid": 4243, "creation_time": 100, "executable": "runner.exe"},
        "launch_nonce": "live-dry-run",
        "status": "RUNNING", "resources": {"slot_id": None},
        "campaign_origin": {"isolated_save_root": None},
        "stop_file": "artifacts/control-plane/launches/live-dry-run/stop.request",
    }), encoding="utf-8")
    monkeypatch.setattr(ControlPlane, "_pid_is_alive", staticmethod(lambda pid: pid == 4243))
    monkeypatch.setattr("hk_agent.control_plane.process_identity", lambda pid: {"pid": pid, "creation_time": 100, "executable": "runner.exe"})
    control = ControlPlane(root, allow_execute=True, runtime_root=runtime)
    campaign = control.create_campaign(_real_campaign_payload())
    active = control.status()["active_launches"]
    assert isinstance(active, list)
    assert active[0]["mode"] == "dry-run"
    with patch("hk_agent.control_plane.subprocess.Popen", _FakeProcess):
        launched = control.launch(str(campaign["campaign_id"]), version=1, execute=True)
        assert launched["resources"]["slot_id"] == 0  # type: ignore[index]


def test_recovery_rejects_reused_pid_with_different_creation_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    record = {
        "launch_id": "reused-pid", "mode": "dry-run", "pid": 4246,
        "runner_process": {"pid": 4246, "creation_time": 100, "executable": "runner.exe"},
        "launch_nonce": "reused-pid", "status": "RUNNING", "resources": {"slot_id": None},
    }
    launch = root / "artifacts" / "control-plane" / "launches" / "reused-pid"
    launch.mkdir(parents=True)
    (launch / "launch.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(ControlPlane, "_pid_is_alive", staticmethod(lambda _pid: True))
    monkeypatch.setattr("hk_agent.control_plane.process_identity", lambda pid: {"pid": pid, "creation_time": 200, "executable": "runner.exe"})
    control = ControlPlane(root)
    assert control.status()["active_count"] == 0
    saved = json.loads((launch / "launch.json").read_text("utf-8"))
    assert saved["status"] == "FAILED"
    assert saved["failure"] == "processo não pertence mais a este launch"


@pytest.mark.parametrize("cancel_with_stop", [False, True])
def test_blocked_save_preparation_drains_slot_and_stale_worker_cannot_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_with_stop: bool
) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    _patch_execution_config(monkeypatch, root, runtime)
    instance = root / "data-1"
    instance.mkdir(parents=True, exist_ok=True)
    (instance / "user1.dat").write_bytes(b"original")
    control = ControlPlane(
        root, allow_execute=True, runtime_root=runtime, parallel_enabled=True,
        preparation_timeout_seconds=0.5,
    )
    campaign = control.create_campaign(_real_campaign_payload())
    entered, release = threading.Event(), threading.Event()
    failed = threading.Event()
    from hk_agent.control_plane import atomic_json as original_atomic_json

    def observe_terminal(path: Path, payload: Mapping[str, object]) -> None:
        original_atomic_json(path, payload)
        if payload.get("status") == "FAILED":
            failed.set()

    monkeypatch.setattr("hk_agent.control_plane.atomic_json", observe_terminal)
    original_prepare = prepare_instance_saves
    first_call = True

    def blocked_prepare(*args: Any, **kwargs: Any) -> Any:
        nonlocal first_call
        if first_call:
            first_call = False
            entered.set()
            assert release.wait(3)
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr("hk_agent.control_plane.prepare_instance_saves", blocked_prepare)
    errors: list[Exception] = []
    thread = threading.Thread(target=lambda: _capture_launch_error(control, campaign, errors))
    with patch("hk_agent.control_plane.subprocess.Popen", _FakeProcess):
        thread.start()
        assert entered.wait(2)
        launch_id = next(iter(control._preparing_ids))
        launch_json = root / "artifacts" / "control-plane" / "launches" / launch_id / "launch.json"
        if cancel_with_stop:
            assert control.stop(launch_id) == {"status": "CANCEL_REQUESTED"}
            expected_status = "CANCELLED"
        else:
            assert failed.wait(2)
            expected_status = "FAILED"
        record = json.loads(launch_json.read_text("utf-8"))
        assert record["status"] == expected_status
        assert record["slot_state"] == "draining"
        control.preparation_timeout_seconds = 5
        other_slot = control.launch(str(campaign["campaign_id"]), version=1, execute=True)
        assert other_slot["resources"]["slot_id"] == 1  # type: ignore[index]
        with pytest.raises(RuntimeError, match="slot 0 is draining"):
            control.launch(str(campaign["campaign_id"]), version=1, execute=True)
        release.set()
        thread.join(3)
        assert not thread.is_alive()
        assert errors
        assert (instance / "user1.dat").read_bytes() == b"original"
        assert control.status()["active_count"] == 1
        launched = control.launch(str(campaign["campaign_id"]), version=1, execute=True)
        assert launched["resources"]["slot_id"] == 0  # type: ignore[index]
        _finish_launch(control, str(launched["launch_id"]))
        _finish_launch(control, str(other_slot["launch_id"]))


def test_dead_reattached_execute_captures_before_releasing_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    instance = root / "instance-0"
    instance.mkdir(parents=True)
    (instance / "user1.dat").write_bytes(b"final")
    launch_root = root / "artifacts" / "control-plane" / "launches" / "reattached-execute"
    launch_root.mkdir(parents=True)
    (runtime / "launches" / "reattached-execute").mkdir(parents=True)
    (launch_root / "launch.json").write_text(json.dumps({
        "launch_id": "reattached-execute", "mode": "execute", "pid": 4244,
        "status": "RUNNING", "resources": {
            "slot_id": 0, "instance_data_root": str(instance),
        }, "campaign_origin": {"isolated_save_root": "runtime/save-root"},
        "stop_file": "artifacts/control-plane/launches/reattached-execute/stop.request",
    }), encoding="utf-8")
    monkeypatch.setattr(ControlPlane, "_pid_is_alive", staticmethod(lambda pid: False))
    control = ControlPlane(root, allow_execute=True, runtime_root=runtime)
    result = control.status()
    saved = json.loads((launch_root / "launch.json").read_text("utf-8"))
    assert saved["final_save_status"] == "captured"
    assert (runtime / "launches" / "reattached-execute" / "final-save" / "user1.dat").read_bytes() == b"final"
    assert saved["status"] == "FAILED"
    assert result["active_count"] == 0


def test_reattached_capture_failure_is_persisted_before_slot_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    launch_root = root / "artifacts" / "control-plane" / "launches" / "reattached-failure"
    launch_root.mkdir(parents=True)
    (runtime / "launches" / "reattached-failure").mkdir(parents=True)
    (launch_root / "launch.json").write_text(json.dumps({
        "launch_id": "reattached-failure", "mode": "execute", "pid": 4245,
        "status": "RUNNING", "resources": {
            "slot_id": 0, "instance_data_root": str(root / "instance-0"),
        }, "campaign_origin": {"isolated_save_root": "runtime/save-root"},
        "stop_file": "artifacts/control-plane/launches/reattached-failure/stop.request",
    }), encoding="utf-8")
    monkeypatch.setattr(ControlPlane, "_pid_is_alive", staticmethod(lambda pid: False))
    monkeypatch.setattr("hk_agent.control_plane.capture_instance_saves", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk failure")))
    control = ControlPlane(root, allow_execute=True, runtime_root=runtime)
    control.status()
    saved = json.loads((launch_root / "launch.json").read_text("utf-8"))
    assert saved["final_save_status"] == "failed"
    assert "disk failure" in saved["final_save_error"]
    assert saved["status"] == "FAILED"
    assert control._slot_owners == {}


def test_stop_after_preparation_before_popen_prevents_process_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    _patch_execution_config(monkeypatch, root, runtime)
    control = ControlPlane(root, allow_execute=True, runtime_root=runtime)
    campaign = control.create_campaign(_real_campaign_payload())
    opening_log, release_log = threading.Event(), threading.Event()
    original_open = Path.open

    def gated_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.name == "process.log":
            opening_log.set()
            assert release_log.wait(3)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", gated_open)
    errors: list[Exception] = []
    thread = threading.Thread(target=lambda: _capture_launch_error(control, campaign, errors))
    popen_calls: list[object] = []

    def recording_popen(*args: Any, **kwargs: Any) -> _FakeProcess:
        popen_calls.append((args, kwargs))
        return _FakeProcess(*args, **kwargs)

    with patch("hk_agent.control_plane.subprocess.Popen", recording_popen):
        thread.start()
        assert opening_log.wait(2)
        launch_id = next(iter(control._preparing_ids))
        assert control.stop(launch_id) == {"status": "CANCEL_REQUESTED"}
        release_log.set()
        thread.join(3)
        assert not thread.is_alive()
        assert not popen_calls
        assert errors and isinstance(errors[0], InterruptedError)
        assert json.loads((root / "artifacts" / "control-plane" / "launches" / launch_id / "launch.json").read_text("utf-8"))["status"] == "CANCELLED"


def test_restart_quarantines_orphaned_preparing_clone(tmp_path: Path) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    launch_id = "orphan"
    control_launch = root / "artifacts" / "control-plane" / "launches" / launch_id
    control_launch.mkdir(parents=True)
    (control_launch / "launch.json").write_text(json.dumps({"launch_id": launch_id, "status": "PREPARING"}), encoding="utf-8")
    clone = runtime / "launches" / launch_id
    clone.mkdir(parents=True)
    (clone / "partial").write_text("keep", encoding="utf-8")
    ControlPlane(root, runtime_root=runtime)
    saved = json.loads((control_launch / "launch.json").read_text("utf-8"))
    assert saved["status"] == "FAILED"
    assert (runtime / "quarantine" / launch_id / "partial").read_text("utf-8") == "keep"
    assert not clone.exists()


def test_baseline_link_is_rejected_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    runtime = root / "runtime"
    baseline = _patch_execution_config(monkeypatch, root, runtime)
    target = root / "outside-target"
    target.mkdir()
    link = baseline / "linked-directory"
    created = False
    try:
        link.symlink_to(target, target_is_directory=True)
        created = True
    except OSError:
        pass
    if not created and os.name == "nt":
        result = subprocess.run(
            ["cmd.exe", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        created = result.returncode == 0
    if not created:
        pytest.skip("SO não permitiu criar symlink nem junction")
    control = ControlPlane(root, allow_execute=True, runtime_root=runtime)
    campaign = control.create_campaign(_real_campaign_payload())
    with pytest.raises(ValueError, match="baseline contains a link/reparse point"):
        control.launch(str(campaign["campaign_id"]), version=1, execute=True)
    launch_json = next((root / "artifacts" / "control-plane" / "launches").glob("*/launch.json"))
    assert json.loads(launch_json.read_text("utf-8"))["status"] == "FAILED"


def test_two_campaign_processes_can_be_controlled_independently(tmp_path: Path) -> None:
    control = ControlPlane(_project(tmp_path))
    campaign = control.create_campaign(_payload())
    with patch("hk_agent.control_plane.subprocess.Popen", _FakeProcess):
        first = control.launch(str(campaign["campaign_id"]), version=1, execute=False)
        second = control.launch(str(campaign["campaign_id"]), version=1, execute=False)
        assert first["mode"] == "dry-run"
        command = first["command"]
        assert isinstance(command, list)
        assert "--stop-file" in command
        with pytest.raises(RuntimeError, match="capacity reached"):
            control.launch(str(campaign["campaign_id"]), version=1, execute=False)
        status = control.status()
        assert status["capacity"] == 2
        assert status["active_count"] == 2
        assert len(status["active_launches"]) == 2  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="launch_id is required"):
            control.stop()
        stopped = control.stop(str(first["launch_id"]))
        assert stopped["status"] == "STOP_REQUESTED"
        active = control.status()["active_launches"]
        assert isinstance(active, list)
        records = {str(item["launch_id"]): item for item in active}
        assert records[str(first["launch_id"])]["status"] == "STOP_REQUESTED"
        assert records[str(second["launch_id"])]["status"] == "RUNNING"
        stop_file = tmp_path / str(records[str(first["launch_id"])]["stop_file"])
        assert stop_file.is_file()
        assert stop_file.read_text("utf-8").startswith("{")


def test_log_can_be_selected_by_launch_id(tmp_path: Path) -> None:
    root = _project(tmp_path)
    launches = root / "artifacts" / "control-plane" / "launches"
    for launch_id, text in (("launch-a", "alpha"), ("launch-b", "beta")):
        launch_root = launches / launch_id
        launch_root.mkdir(parents=True)
        (launch_root / "process.log").write_text(text, encoding="utf-8")
        (launch_root / "launch.json").write_text(
            json.dumps(
                {
                    "launch_id": launch_id,
                    "pid": 1,
                    "status": "COMPLETED",
                    "log_path": (
                        f"artifacts/control-plane/launches/{launch_id}/process.log"
                    ),
                }
            ),
            encoding="utf-8",
        )
    control = ControlPlane(root)
    assert control.read_log(launch_id="launch-a")["text"] == "alpha"
    assert control.read_log(launch_id="launch-b")["text"] == "beta"


def test_http_api_is_local_and_requires_token_for_mutations(tmp_path: Path) -> None:
    root = _project(tmp_path)
    static_root = Path(__file__).resolve().parents[1] / "src" / "hk_agent" / "control_plane_web"
    server = ControlPlaneHTTPServer(("127.0.0.1", 0), ControlPlane(root), static_root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        connection.request("GET", "/api/session")
        response = connection.getresponse()
        token = json.loads(response.read())["csrf_token"]
        assert response.status == 200

        payload = _payload("exploration_training")
        payload.update({
            "reward_version": "explore_v4",
            "shared_learner": True,
            "actors": 1,
            "rollout_steps": 256,
            "batch_size": 256,
            "checkpoint_interval": 256,
            "duration_mode": "continuous",
            "duration_limit_seconds": 0,
            "weights": {},
            "warm_start_checkpoint": "C:/checkpoints/v3.zip",
            "warm_start_source_campaign": "prior-v3",
        })
        body = json.dumps(payload)
        connection.request("POST", "/api/campaigns", body, {"Content-Type": "application/json"})
        denied = connection.getresponse()
        denied.read()
        assert denied.status == 403

        connection.request(
            "POST",
            "/api/campaigns",
            body,
            {
                "Content-Type": "application/json",
                "X-Control-Token": token,
                "Origin": f"http://127.0.0.1:{server.server_port}",
            },
        )
        created = connection.getresponse()
        result = json.loads(created.read())
        assert created.status == 201
        assert result["version"] == 1
        assert result["warm_start_source_campaign"] == "prior-v3"
        assert result["actors"] == 1
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_panel_adopts_game_identity_sidecar_only_for_its_own_launch(tmp_path: Path) -> None:
    control = ControlPlane(_project(tmp_path))
    launch_root = control.control_root / "launches" / "L1"
    launch_root.mkdir(parents=True)
    control._launches["L1"] = {"launch_id": "L1", "status": "STOP_REQUESTED"}
    sidecar = launch_root / "game-process.json"

    sidecar.write_text(json.dumps({"pid": 5, "launch_nonce": "OTHER"}), "utf-8")
    control._merge_game_identity_locked("L1")
    assert "game_process" not in control._launches["L1"]

    sidecar.write_text(json.dumps({"pid": 5, "launch_nonce": "L1"}), "utf-8")
    control._merge_game_identity_locked("L1")
    assert control._launches["L1"]["game_process"] == {"pid": 5, "launch_nonce": "L1"}
    # The panel's own STOP_REQUESTED state is preserved.
    assert control._launches["L1"]["status"] == "STOP_REQUESTED"


@pytest.mark.parametrize("campaign_type", ["scene_change_probe", "parallel_smoke"])
def test_probe_runner_campaigns_reject_horizon_the_runner_would_reject(campaign_type: str) -> None:
    from hk_agent.control_plane import PROBE_MAX_HORIZON
    from hk_agent.scene_change_probe import SceneChangeProbeConfig

    payload = _payload(campaign_type)
    if campaign_type == "parallel_smoke":
        payload["weights"] = {}
    payload["horizon"] = PROBE_MAX_HORIZON + 1
    with pytest.raises(ValueError, match="horizon must be at most"):
        CampaignSpec.from_payload(payload)
    payload["horizon"] = PROBE_MAX_HORIZON
    assert CampaignSpec.from_payload(payload).horizon == PROBE_MAX_HORIZON
    # The panel limit is exactly the runner's own limit.
    SceneChangeProbeConfig(horizon_steps=PROBE_MAX_HORIZON).validate()
    with pytest.raises(ValueError):
        SceneChangeProbeConfig(horizon_steps=PROBE_MAX_HORIZON + 1).validate()


@pytest.mark.parametrize("campaign_type", ["scene_change_probe", "parallel_smoke"])
def test_concurrent_probe_launches_never_share_route_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, campaign_type: str
) -> None:
    """Regression: two probe/smoke instances used the same default route-memory
    file, and the second writer was (correctly) rejected mid-run."""
    root = _project(tmp_path)
    runtime = root / "runtime"
    _patch_execution_config(monkeypatch, root, runtime)
    legacy = (
        root / "artifacts" / "training" / "reach-scene-change-v1"
        / "route-memory-existing_save-slot-1.json"
    )
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"legacy": true}', encoding="utf-8")
    control = ControlPlane(
        root, allow_execute=True, parallel_enabled=True, runtime_root=runtime,
        training_device="cpu", torch_threads=3, torch_interop_threads=1,
    )
    payload = _payload(campaign_type)
    if campaign_type == "parallel_smoke":
        payload["weights"] = {}
        payload["horizon"] = 10_000
    campaign = control.create_campaign(payload)
    with patch("hk_agent.control_plane.subprocess.Popen", _FakeProcess):
        first = control.launch(str(campaign["campaign_id"]), version=1, execute=True)
        second = control.launch(str(campaign["campaign_id"]), version=1, execute=True)
        paths = []
        for launch in (first, second):
            command = launch["command"]
            assert isinstance(command, list)
            paths.append(Path(command[command.index("--route-memory") + 1]))
        assert paths[0] != paths[1]
        training_root = (runtime / "route-memory").resolve()
        for path in paths:
            # Diagnostics never write into, or get seeded from, training memory.
            assert training_root not in path.resolve().parents
            assert not path.exists() or path.read_text("utf-8") != '{"legacy": true}'
        if campaign_type == "parallel_smoke":
            for launch, path in zip((first, second), paths, strict=True):
                assert path.parent == (runtime / "launches" / str(launch["launch_id"])).resolve()
        _finish_launch(control, str(first["launch_id"]))
        _finish_launch(control, str(second["launch_id"]))


@pytest.mark.parametrize(
    ("initial_status", "return_code", "expected"),
    [("STOP_REQUESTED", 0, "CANCELLED"), ("STOP_REQUESTED", 1, "CANCELLED"), ("RUNNING", 0, "FAILED")],
)
def test_requested_stop_with_job_draining_stays_cancelled(
    tmp_path: Path, initial_status: str, return_code: int, expected: str
) -> None:
    """Regression (review P2): draining the job after a requested stop must not
    turn a cooperative stop into FAILED; an unrequested drain still fails."""
    root = _project(tmp_path)
    control = ControlPlane(root)
    launch_id = "drain-after-stop"
    launch_root = control.control_root / "launches" / launch_id
    launch_root.mkdir(parents=True)
    record: dict[str, object] = {
        "launch_id": launch_id,
        "mode": "execute",
        "status": initial_status,
        "resources": {"slot_id": 0},
    }
    process = _FakeProcess([])
    process._return_code = return_code
    process._done.set()
    calls = {"n": 0}

    class DrainOnceJob:
        def process_ids(self) -> list[int]:
            calls["n"] += 1
            return [777] if calls["n"] == 1 else []

        def terminate(self) -> None:
            return None

        def close(self) -> None:
            return None

    control._launches[launch_id] = record
    control._processes[launch_id] = process
    control._jobs[launch_id] = DrainOnceJob()  # type: ignore[assignment]
    control._slot_owners[0] = launch_id
    control._slot_fences[0] = "fence"
    log_handle = (launch_root / "process.log").open("wb")
    control._wait_for_process(launch_id, process, log_handle, launch_root)  # type: ignore[arg-type]
    assert record["status"] == expected
    assert record["drain_reason"] == "launcher exited while job processes remained"
    assert record["draining_pids"] == []
    assert launch_id not in control._slot_owners.values()
    persisted = json.loads((launch_root / "launch.json").read_text("utf-8"))
    assert persisted["status"] == expected
    if expected == "CANCELLED":
        assert "failure" not in persisted


def test_shared_learner_terminal_transition_skips_game_job_and_save_drain(tmp_path: Path) -> None:
    root = _project(tmp_path)
    control = ControlPlane(root)
    launch_id = "shared-learner"
    launch_root = control.control_root / "launches" / launch_id
    launch_root.mkdir(parents=True)
    shared_root = tmp_path / "shared"
    shared_root.mkdir()
    (shared_root / "learner.json").write_text(
        json.dumps({"status": "CANCELLED", "round": 4}), encoding="utf-8"
    )
    record: dict[str, object] = {
        "launch_id": launch_id, "role": "learner", "shared_group": "group",
        "shared_root": str(shared_root), "status": "STOP_REQUESTED",
        "resources": {"slot_id": None},
    }
    process = _FakeProcess([])
    process._return_code = 0
    process._done.set()
    calls = {"query": 0, "close": 0}

    class LearnerJob:
        def process_ids(self) -> list[int]:
            calls["query"] += 1
            raise AssertionError("learner terminal path must not wait on game-process drain")

        def close(self) -> None:
            calls["close"] += 1

    control._launches[launch_id] = record
    control._processes[launch_id] = process
    control._jobs[launch_id] = LearnerJob()  # type: ignore[assignment]
    log_handle = (launch_root / "process.log").open("wb")
    control._wait_for_process(launch_id, process, log_handle, launch_root)  # type: ignore[arg-type]
    assert record["status"] == "CANCELLED"
    assert record["shared_learner_status"] == "CANCELLED"
    assert record["final_save_status"] == "not_applicable"
    assert calls == {"query": 0, "close": 1}


def test_shared_learner_launch_starts_terminal_status_watcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    runtime = tmp_path / "runtime"
    control = ControlPlane(root, runtime_root=runtime)
    shared_root = tmp_path / "shared"
    shared_root.mkdir()
    (shared_root / "learner.json").write_text(
        json.dumps({"status": "CANCELLED", "round": 1}), encoding="utf-8"
    )
    payload = _payload("exploration_training")
    checkpoint = runtime / "shared" / "v3" / "checkpoints" / "round-00000002.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"ppo")
    payload.update({
        "shared_learner": True, "actors": 2, "rollout_steps": 512, "batch_size": 256,
        "warm_start_checkpoint": str(checkpoint), "warm_start_source_campaign": "explore_v3",
    })
    payload["weights"] = {}
    spec = CampaignSpec.from_payload(payload)
    actor_command = control.build_command(spec, execute=False)
    assert "--warm-start-checkpoint" not in actor_command
    assert "--warm-start-source-campaign" not in actor_command
    process = _FakeProcess([])
    process._return_code = 0
    process._done.set()

    class ImmediateThread:
        def __init__(self, *, target: Any, args: tuple[Any, ...], daemon: bool) -> None:
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self) -> None:
            self.target(*self.args)

    monkeypatch.setattr(control_plane_module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(control_plane_module, "process_identity", lambda _pid: {"pid": process.pid})
    monkeypatch.setattr(control_plane_module.threading, "Thread", ImmediateThread)
    record = control._launch_shared_learner(
        spec, version=1, group_id="group", shared_root=shared_root
    )

    assert str(checkpoint.resolve()) in record["command"]
    assert "--warm-start-checkpoint" in record["command"]
    assert "--warm-start-source-campaign" in record["command"]

    assert record["status"] == "CANCELLED"
    assert record["shared_learner_status"] == "CANCELLED"
    assert record["final_save_status"] == "not_applicable"
    assert record["launch_id"] not in control._processes
    persisted = json.loads(
        (control.control_root / "launches" / str(record["launch_id"]) / "launch.json").read_text()
    )
    assert persisted["status"] == "CANCELLED"


@pytest.mark.parametrize("learner_finishes_last", [True, False])
def test_shared_group_requires_terminal_learner_and_all_actors(
    tmp_path: Path, learner_finishes_last: bool
) -> None:
    control = ControlPlane.__new__(ControlPlane)
    control._lock = threading.RLock()
    control.status = lambda: {"active_count": 0}  # type: ignore[method-assign]
    control._launches = {
        "learner": {"shared_group": "g", "role": "learner", "status": "RUNNING"},
        "actor-0": {"shared_group": "g", "role": "actor", "status": "RUNNING"},
        "actor-1": {"shared_group": "g", "role": "actor", "status": "RUNNING"},
    }
    if learner_finishes_last:
        control._launches["actor-0"]["status"] = "CANCELLED"
        control._launches["actor-1"]["status"] = "CANCELLED"
        assert not control.shared_group_stopped("g")
        control._launches["learner"]["status"] = "CANCELLED"
    else:
        control._launches["learner"]["status"] = "CANCELLED"
        assert not control.shared_group_stopped("g")
        control._launches["actor-0"]["status"] = "CANCELLED"
        control._launches["actor-1"]["status"] = "COMPLETED"
    assert control.shared_group_stopped("g")


@pytest.mark.parametrize("actors_finish_first", [True, False])
def test_shared_group_stop_waits_for_fake_processes_in_either_order(
    tmp_path: Path, actors_finish_first: bool
) -> None:
    root = _project(tmp_path)
    control = ControlPlane(root)
    group = "shared-stop-order"
    shared_root = tmp_path / "shared"
    shared_root.mkdir()
    (shared_root / "learner.json").write_text(
        json.dumps({"status": "CANCELLED", "round": 1}), encoding="utf-8"
    )
    launches: dict[str, tuple[dict[str, object], _FakeProcess, Path]] = {}
    for launch_id, role in (("learner", "learner"), ("actor-0", "actor"), ("actor-1", "actor")):
        launch_root = control.control_root / "launches" / launch_id
        launch_root.mkdir(parents=True)
        process = _FakeProcess([])
        process._return_code = 0
        process._done.set()
        record: dict[str, object] = {
            "launch_id": launch_id, "mode": "execute", "role": role,
            "shared_group": group, "shared_root": str(shared_root),
            "status": "RUNNING", "pid": process.pid,
            "resources": {"slot_id": None},
            "stop_file": (launch_root / "stop.request").relative_to(root).as_posix(),
        }
        (launch_root / "launch.json").write_text(json.dumps(record), encoding="utf-8")
        control._launches[launch_id] = record
        control._processes[launch_id] = process
        launches[launch_id] = (record, process, launch_root)

    result = control.stop("learner")
    assert result["status"] == "STOP_REQUESTED"
    assert all(row[0]["status"] == "STOP_REQUESTED" for row in launches.values())
    order = ["actor-0", "actor-1", "learner"] if actors_finish_first else ["learner", "actor-0", "actor-1"]
    for launch_id in order:
        record, process, launch_root = launches[launch_id]
        with (launch_root / "process.log").open("wb") as log_handle:
            control._wait_for_process(launch_id, process, log_handle, launch_root)  # type: ignore[arg-type]
        if launch_id != order[-1]:
            assert not control.shared_group_stopped(group)
    assert control.shared_group_stopped(group)
    assert all(row[0]["status"] == "CANCELLED" for row in launches.values())


def _foreign_owner(control_root: Path, pid: int = 424242, creation_time: object = 99) -> Path:
    control_root.mkdir(parents=True, exist_ok=True)
    lock = control_root / ".control-plane.owner.json"
    lock.write_text(json.dumps({"pid": pid, "creation_time": creation_time}), encoding="utf-8")
    return lock


def test_second_process_cannot_take_over_a_live_control_plane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression (soak): a status query from another process built a full
    ControlPlane, ran restart recovery on live launches and marked them FAILED."""
    from hk_agent.control_plane import ControlPlaneBusyError

    root = _project(tmp_path)
    control_root = root / "artifacts" / "control-plane"
    _foreign_owner(control_root)
    live = control_root / "launches" / "L-live" / "launch.json"
    live.parent.mkdir(parents=True)
    record = {"launch_id": "L-live", "mode": "execute", "status": "RUNNING", "pid": 5151,
              "resources": {"slot_id": 0}}
    live.write_text(json.dumps(record), encoding="utf-8")
    before = live.read_bytes()

    def identity(pid: int) -> dict[str, object]:
        if pid == 424242:
            return {"pid": pid, "creation_time": 99, "executable": "python.exe"}
        return {"pid": pid, "creation_time": 1, "executable": "python.exe"}

    monkeypatch.setattr("hk_agent.control_plane.process_identity", identity)
    with pytest.raises(ControlPlaneBusyError, match="observer=True"):
        ControlPlane(root)
    assert live.read_bytes() == before  # nothing recovered or rewritten

    observer = ControlPlane(root, observer=True)
    status = observer.status()
    assert status["observer"] is True
    assert status["active_count"] == 1
    assert live.read_bytes() == before
    with pytest.raises(PermissionError, match="observer"):
        observer.stop("L-live")
    with pytest.raises(PermissionError, match="observer"):
        observer.create_campaign(_payload())
    assert live.read_bytes() == before


def test_stale_owner_lock_from_dead_process_is_taken_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    lock = _foreign_owner(root / "artifacts" / "control-plane")

    def identity(pid: int) -> dict[str, object]:
        if pid == 424242:
            raise OSError("no such process")
        return {"pid": pid, "creation_time": 7, "executable": "python.exe"}

    monkeypatch.setattr("hk_agent.control_plane.process_identity", identity)
    control = ControlPlane(root)
    assert json.loads(lock.read_text("utf-8"))["pid"] == os.getpid()
    control.close()
    assert not lock.exists()


def test_reused_owner_pid_with_different_creation_time_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    _foreign_owner(root / "artifacts" / "control-plane", creation_time=99)
    monkeypatch.setattr(
        "hk_agent.control_plane.process_identity",
        lambda pid: {"pid": pid, "creation_time": 12345, "executable": "python.exe"},
    )
    ControlPlane(root)  # PID reused by an unrelated process: not a live owner


def test_status_supervises_failed_shared_actor_while_learner_runs(tmp_path: Path) -> None:
    from hk_agent.shared_learner import _atomic_json

    control = ControlPlane.__new__(ControlPlane)
    control.observer = False
    control._lock = threading.RLock()
    control._refresh_locked = lambda: None
    shared_root = tmp_path / "shared"
    (shared_root / "actors").mkdir(parents=True)
    control._launches = {
        "learner-live": {"role": "learner", "shared_group": "campaign-v0001", "status": "RUNNING"},
        "actor-lost": {
            "role": "actor", "shared_group": "campaign-v0001", "shared_root": str(shared_root),
            "shared_actor_count": 2, "campaign_id": "campaign", "campaign_version": 1,
            "status": "FAILED", "resources": {"slot_id": 1},
        },
    }
    control._shared_actor_restarts = {}
    control._shared_actor_supervision = set()
    control.control_root = tmp_path / "control"
    _atomic_json(shared_root / "actors" / "slot-1.json", {"slot": 1, "launch_id": "actor-lost", "status": "FAILED"})
    started = []

    def launch(campaign_id: str, **kwargs: object) -> dict[str, object]:
        started.append((campaign_id, kwargs))
        launch_id = "actor-replacement"
        launch_dir = control.control_root / "launches" / launch_id
        launch_dir.mkdir(parents=True)
        return {"launch_id": launch_id, "status": "RUNNING"}

    control._launch_single = launch
    control._supervise_failed_shared_actors()
    control._supervise_failed_shared_actors()

    assert len(started) == 1
    assert started[0][0] == "campaign"
    assert started[0][1]["shared_restart_of"] == "actor-lost"
    assert control._launches["actor-lost"]["auto_replacement_launch_id"] == "actor-replacement"


def test_shared_actor_restart_is_bounded_and_recorded(tmp_path: Path) -> None:
    control = ControlPlane.__new__(ControlPlane)
    control.observer = False
    control._lock = threading.RLock()
    control._refresh_locked = lambda: None
    control._launches = {
        "actor-old": {
            "role": "actor", "shared_group": "campaign-v0001", "shared_root": str(tmp_path / "shared"),
            "shared_actor_count": 2, "campaign_id": "campaign", "campaign_version": 1,
            "status": "CANCELLED",
        }
    }
    control._shared_actor_restarts = {}
    control.control_root = tmp_path / "control"
    launch_number = 0

    def launch(campaign_id: str, **kwargs: object) -> dict[str, object]:
        nonlocal launch_number
        launch_number += 1
        launch_id = f"actor-new-{launch_number}"
        launch_dir = control.control_root / "launches" / launch_id
        launch_dir.mkdir(parents=True)
        return {"launch_id": launch_id, "status": "RUNNING"}

    control._launch_single = launch
    for _ in range(3):
        control.restart_shared_actor("actor-old", reason="soak test")
    with pytest.raises(RuntimeError, match="3 per hour"):
        control.restart_shared_actor("actor-old", reason="limit test")
    rows = (tmp_path / "shared" / "actor-restarts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3
    assert json.loads(rows[-1])["restart_number_in_hour"] == 3
