from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hk_agent.control_plane import ControlPlane
from hk_agent.control_plane_server import ControlPlaneHTTPServer
from hk_agent.operator_reports import build_report


def test_operator_report_has_synthetic_metrics_in_json_and_standalone_html() -> None:
    payload, document = build_report({
        "campaign_id": "synthetic-run", "duration_seconds": 3600, "instances": 2,
        "training_steps": 1200, "learner_updates": 4, "deaths": 2,
        "shade_recoveries": 1, "shade_losses": 1, "instance_restarts": 3,
        "save_safety_checks": "passed", "end_reason": "time_limit",
        "scenes_discovered": [{"scene_name": "Crossroads", "first_seen": "10:30"}],
        "episode_progress": [{"episode": 1, "steps": 20, "distinct_scenes": 2, "deaths": 1,
                              "shade_recoveries": 1, "enemy_kills": 3, "termination_reason": "death", "return": 2.5}],
        "evaluations": [{"round": 1, "steps": 10, "return": 1.2, "completed": True, "rollout_submitted": False}],
        "alerts": ["synthetic alert"], "checkpoints_saved": ["round-0004.zip"],
    })
    assert payload["campaign_id"] == "synthetic-run"
    for section in ("Resumo", "Salas descobertas", "Evolução por episódio", "Avaliações periódicas",
                    "Alertas e reinícios", "Checkpoints e segurança dos saves", "Motivo do fim"):
        assert section in document
    for evidence in ("Crossroads", "10:30", "synthetic alert", "round-0004.zip", "passed", "time_limit"):
        assert evidence in document
    assert "<script" not in document


def test_deadline_writes_cooperative_stop_request_for_launch_group(tmp_path: Path) -> None:
    plane = ControlPlane(tmp_path)
    launch_dir = plane.control_root / "launches" / "actor-1"
    launch_dir.mkdir(parents=True)
    stop_file = launch_dir / "stop.request"
    record: dict[str, object] = {
        "launch_id": "actor-1", "status": "RUNNING", "deadline_unix_ns": 1,
        "shared_group": "campaign-a", "stop_file": str(stop_file),
    }
    plane._launches["actor-1"] = record
    plane._processes["actor-1"] = object()  # type: ignore[assignment]
    plane._enforce_deadlines_locked()
    assert record["status"] == "STOP_REQUESTED"
    assert record["stop_reason"] == "time_limit"
    assert json.loads(stop_file.read_text("utf-8"))["reason"] == "time_limit"


def test_continue_warm_starts_v4_from_v3_and_repeat_uses_new_seed(tmp_path: Path) -> None:
    plane = ControlPlane(tmp_path, allow_execute=True)
    campaign_id = "shared-campaign"
    checkpoint_dir = plane.runtime_root / "shared" / campaign_id / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "round-0002.zip").write_bytes(b"synthetic checkpoint")
    old = {"campaign_id": campaign_id, "name": "Night run", "campaign_type": "exploration_training",
           "reward_version": "explore_v3", "shared_learner": True, "seed": 7, "version": 2,
           "created_at": "old", "spec_path": "old.json"}
    with (patch.object(plane, "list_campaigns", return_value=[old]),
          patch.object(plane, "create_campaign", side_effect=lambda data: {**data, "version": 3}),
          patch.object(plane, "launch", return_value={"launch_id": "next"}) as launch):
        resumed = plane.continue_operator_campaign()
        assert resumed["checkpoint"].endswith("round-0002.zip")
        campaign = resumed["campaign"]
        assert campaign["campaign_id"] != campaign_id
        assert campaign["reward_version"] == "explore_v4"
        assert campaign["warm_start_checkpoint"].endswith("round-0002.zip")
        assert campaign["warm_start_source_campaign"] == campaign_id
        assert "conhecimento foi mantido" in resumed["notice"]
        launch.assert_called_once_with(campaign["campaign_id"], version=3, execute=True)
    with (patch.object(plane, "list_campaigns", return_value=[old]),
          patch.object(plane, "create_campaign", side_effect=lambda data: {**data, "version": 1}),
          patch.object(plane, "launch", return_value={"launch_id": "fresh"}) as launch):
        repeated = plane.repeat_operator_campaign()
        assert repeated["campaign"]["campaign_id"] != campaign_id
        assert repeated["campaign"]["seed"] != old["seed"]
        launch.assert_called_once()


def test_operator_start_uses_latest_compatible_policy_for_jogada1(tmp_path: Path) -> None:
    plane = ControlPlane(tmp_path, allow_execute=True)
    recurrent = {"campaign_id": "newer-recurrent", "campaign_type": "exploration_training",
                 "reward_version": "explore_v4", "shared_learner": True}
    feedforward = {"campaign_id": "latest-compatible", "campaign_type": "exploration_training",
                   "reward_version": "explore_v4", "shared_learner": True}
    for record, algorithm in ((recurrent, "MaskableRecurrentPPO"), (feedforward, "MaskablePPO")):
        root = plane.runtime_root / "shared" / str(record["campaign_id"])
        (root / "checkpoints").mkdir(parents=True)
        (root / "manifest.json").write_text(json.dumps({"algorithm": algorithm}), encoding="utf-8")
        (root / "checkpoints" / "round-0001.zip").write_bytes(b"checkpoint")
    with (patch("hk_agent.config.load_config", return_value=SimpleNamespace(instance_count=2)),
          patch.object(plane, "list_campaigns", return_value=[recurrent, feedforward]),
          patch.object(plane, "launch", return_value={"status": "RUNNING"}) as launch):
        result = plane.start_operator_campaign(
            instances=2, duration_kind="until_stopped", duration_value=None,
            campaign_name="jogada-1",
        )
    campaign = result["campaign"]
    assert campaign["campaign_id"] == "jogada-1"
    assert campaign["name"] == "jogada-1"
    assert campaign["actors"] == 2 and campaign["duration_mode"] == "continuous"
    assert campaign["duration_limit_seconds"] == 0.0
    assert campaign["challenge_practice_enabled"] is True
    assert campaign["help_requests_enabled"] is True
    assert campaign["warm_start_source_campaign"] == "latest-compatible"
    assert campaign["warm_start_checkpoint"].endswith("round-0001.zip")
    launch.assert_called_once_with("jogada-1", version=1, execute=True)


def test_operator_start_does_not_warm_start_into_existing_campaign_checkpoints(tmp_path: Path) -> None:
    plane = ControlPlane(tmp_path, allow_execute=True)
    target = plane.runtime_root / "shared" / "jogada-1" / "checkpoints"
    target.mkdir(parents=True)
    (target / "round-0010.zip").write_bytes(b"existing policy")
    prior = {"campaign_id": "compatible", "campaign_type": "exploration_training",
             "reward_version": "explore_v4", "shared_learner": True}
    source = plane.runtime_root / "shared" / "compatible"
    (source / "checkpoints").mkdir(parents=True)
    (source / "manifest.json").write_text(json.dumps({"algorithm": "MaskablePPO"}), encoding="utf-8")
    (source / "checkpoints" / "round-0011.zip").write_bytes(b"source policy")
    with (patch("hk_agent.config.load_config", return_value=SimpleNamespace(instance_count=2)),
          patch.object(plane, "list_campaigns", return_value=[prior]),
          patch.object(plane, "launch", return_value={"status": "RUNNING"})):
        result = plane.start_operator_campaign(
            instances=2, duration_kind="until_stopped", duration_value=None,
            campaign_name="jogada-1", help_stuck_count=8, help_stuck_minutes=90,
        )
    campaign = result["campaign"]
    assert campaign["campaign_id"] == "jogada-1"
    assert campaign.get("warm_start_checkpoint") is None
    assert campaign.get("warm_start_source_campaign") is None
    assert campaign["help_stuck_count"] == 8 and campaign["help_stuck_minutes"] == 90


def test_failure_reason_and_log_tail_are_available_for_operator_banner(tmp_path: Path) -> None:
    plane = ControlPlane(tmp_path)
    log = plane.control_root / "launches" / "learner-1" / "process.log"
    log.parent.mkdir(parents=True)
    log.write_text("Starting learner\nTraceback (most recent call last):\nValueError: warm-start campaigns must use a new shared checkpoint directory\n", encoding="utf-8")
    record = {"log_path": log.relative_to(tmp_path).as_posix()}
    assert plane._operator_reason("SharedLearnerError: warm-start campaigns must use a new shared checkpoint directory") == (
        "A pasta de checkpoints já está em uso; continue a campanha existente ou use outra pasta."
    )
    assert plane._launch_error_detail(record) == "ValueError: warm-start campaigns must use a new shared checkpoint directory"


def test_operator_snapshot_explains_failed_group_and_cancelled_actor(tmp_path: Path) -> None:
    plane = ControlPlane(tmp_path)
    campaign = {"campaign_id": "campaign-a", "campaign_type": "exploration_training",
                "reward_version": "explore_v4", "shared_learner": True}
    shared_root = plane.runtime_root / "shared" / "campaign-a"
    (shared_root / "actors").mkdir(parents=True)
    (shared_root / "actors" / "slot-0.json").write_text(
        json.dumps({"slot": 0, "status": "CANCELLED"}), encoding="utf-8")
    learner_log = plane.control_root / "launches" / "learner-a" / "process.log"
    learner_log.parent.mkdir(parents=True)
    learner_log.write_text("Traceback (most recent call last):\nValueError: warm-start campaigns must use a new shared checkpoint directory\n", encoding="utf-8")
    records = [
        {"launch_id": "learner-a", "role": "learner", "status": "FAILED", "shared_group": "group-a",
         "campaign_id": "campaign-a", "failure": "learner heartbeat expirou", "finished_unix_ns": 20,
         "log_path": learner_log.relative_to(tmp_path).as_posix()},
        {"launch_id": "actor-a", "role": "actor", "status": "CANCELLED", "shared_group": "group-a",
         "campaign_id": "campaign-a", "stop_reason": "stop requested", "finished_unix_ns": 19,
         "resources": {"slot_id": 0}},
    ]
    with (patch.object(plane, "status", return_value={"active_count": 0, "active_launches": [], "capacity": 2}),
          patch.object(plane, "_ensure_auto_reports"),
          patch.object(plane, "list_campaigns", return_value=[campaign]),
          patch.object(plane, "_campaign_launch_records", return_value=records),
          patch.object(plane, "_campaign_scene_discoveries", return_value=[]),
          patch.object(plane, "_operator_playthroughs", return_value=[]),
          patch.object(plane, "_operator_system_resources", return_value={}),
          patch.object(plane, "map_art_status", return_value={}),
          patch.object(plane, "list_reports", return_value=[])):
        snapshot = plane.operator_snapshot()
    assert snapshot["status_banner"]["kind"] == "error"
    assert "pasta de checkpoints" in snapshot["status_banner"]["message"]
    assert snapshot["status_banner"]["technical_detail"].startswith("ValueError:")
    assert "parada solicitada" in snapshot["instances"][0]["reason"]
    assert plane._operator_reason("learner failed: actor cancelled") == "cancelada porque o learner caiu"


def test_operator_snapshot_promotes_repeated_stuck_place_to_warning_banner(tmp_path: Path) -> None:
    plane = ControlPlane(tmp_path)
    campaign = {"campaign_id": "campaign-stuck", "campaign_type": "exploration_training",
                "reward_version": "explore_v4", "shared_learner": True}
    ledger = plane.runtime_root / "shared" / "campaign-stuck" / "help-requests" / "slot-0.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({"stuck_details": {
        "Tutorial_01 (5, 9)": {"count": 5, "last_stuck_unix": 1_800_000_000.0},
    }}), encoding="utf-8")
    with (patch.object(plane, "status", return_value={"active_count": 0, "active_launches": [], "capacity": 2}),
          patch.object(plane, "_ensure_auto_reports"),
          patch.object(plane, "list_campaigns", return_value=[campaign]),
          patch.object(plane, "_campaign_launch_records", return_value=[]),
          patch.object(plane, "_campaign_scene_discoveries", return_value=[]),
          patch.object(plane, "_operator_playthroughs", return_value=[]),
          patch.object(plane, "_operator_system_resources", return_value={}),
          patch.object(plane, "map_art_status", return_value={}),
          patch.object(plane, "list_reports", return_value=[])):
        snapshot = plane.operator_snapshot()
    assert snapshot["status_banner"]["kind"] == "warning"
    assert "travada há" in snapshot["status_banner"]["message"]


def test_server_binds_loopback_and_rejects_panel_weights(tmp_path: Path) -> None:
    plane = ControlPlane(tmp_path, allow_execute=False)
    server = ControlPlaneHTTPServer(("127.0.0.1", 0), plane, tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert server.server_address[0] == "127.0.0.1"
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        body = json.dumps({"instances": 1, "duration_kind": "until_stopped", "duration_value": None,
                           "weights": {"progress": 99}})
        connection.request("POST", "/api/operator/start", body=body,
                           headers={"Content-Type": "application/json", "X-Control-Token": plane.csrf_token})
        response = connection.getresponse()
        result = json.loads(response.read())
        assert response.status == 400
        assert "unknown fields: weights" in result["error"]
        connection.close()

        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        body = json.dumps({"instances": 1, "duration_kind": "until_stopped", "duration_value": None,
                           "campaign_name": "smoke", "help_stuck_count": 9, "help_stuck_minutes": 180})
        connection.request("POST", "/api/operator/start", body=body,
                           headers={"Content-Type": "application/json", "X-Control-Token": plane.csrf_token})
        response = connection.getresponse()
        result = json.loads(response.read())
        assert response.status == 403
        assert "--allow-execute" in result["error"]
        assert "unknown fields" not in result["error"]
        connection.close()

        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        connection.request("GET", "/api/operator")
        response = connection.getresponse()
        snapshot = json.loads(response.read())
        assert response.status == 200
        assert snapshot["status_banner"]["kind"] == "idle"
        assert "Treino parado" in snapshot["status_banner"]["message"]
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_automatic_report_refreshes_after_a_later_campaign_completion(tmp_path: Path) -> None:
    plane = ControlPlane(tmp_path)
    campaign = {"campaign_id": "shared-campaign", "shared_learner": True, "reward_version": "explore_v3"}
    terminal = [{"status": "CANCELLED", "finished_unix_ns": 200, "campaign_id": "shared-campaign"}]
    with (patch.object(plane, "list_campaigns", return_value=[campaign]),
          patch.object(plane, "list_reports", return_value=[{"campaign_id": "shared-campaign", "last_finished_unix_ns": 100}]),
          patch.object(plane, "_campaign_launch_records", return_value=terminal),
          patch.object(plane, "generate_run_report") as generate):
        plane._ensure_auto_reports({"active_launches": []})
        generate.assert_called_once_with("shared-campaign")


def test_journey_summary_wraps_and_map_art_controls_are_operator_only() -> None:
    web = Path(__file__).parents[1] / "src" / "hk_agent" / "control_plane_web"
    html = (web / "index.html").read_text("utf-8")
    app = (web / "app.js").read_text("utf-8")
    css = (web / "app.css").read_text("utf-8")
    assert 'id="map-download"' in html
    assert 'id="map-import"' in html
    assert 'id="map-progress"' in html
    assert 'id="map-view-toggle"' in html
    assert '/api/journey/map/tile?' in app
    assert 'worldTile' in app
    assert 'class="journey-run-summary"' in app
    assert ".journey-run-summary{flex:1 1 100%" in css
    assert "appearance:none" in css and "data:image/svg+xml" in css
    assert 'id="status-banner"' in html and 'id="status-technical-detail"' in html
    assert ".status-banner.error" in css and ".status-banner.warning" in css
    assert "setStatusBanner(\"error\"" in app and "friendlyFailure" in app
