from __future__ import annotations

import http.client
import json
import threading
import time
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
    (checkpoint_dir.parent / "manifest.json").write_text(json.dumps({"algorithm": "MaskablePPO"}), encoding="utf-8")
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
        assert "round-0002.zip" in resumed["notice"]
        launch.assert_called_once_with(campaign["campaign_id"], version=3, execute=True)
    with (patch.object(plane, "list_campaigns", return_value=[old]),
          patch.object(plane, "create_campaign", side_effect=lambda data: {**data, "version": 1}),
          patch.object(plane, "launch", return_value={"launch_id": "fresh"}) as launch):
        repeated = plane.repeat_operator_campaign()
        assert repeated["campaign"]["campaign_id"] != campaign_id
        assert repeated["campaign"]["seed"] != old["seed"]
        launch.assert_called_once()


def test_continue_operator_campaign_uses_requested_actor_count(tmp_path: Path) -> None:
    plane = ControlPlane(tmp_path, allow_execute=True)
    campaign_id = "shared-campaign"
    shared_root = plane.runtime_root / "shared" / campaign_id
    (shared_root / "checkpoints").mkdir(parents=True)
    checkpoint = shared_root / "checkpoints" / "round-0003.zip"
    checkpoint.write_bytes(b"synthetic checkpoint")
    prior = {"campaign_id": campaign_id, "name": "Night run", "campaign_type": "exploration_training",
             "reward_version": "explore_v4", "shared_learner": True, "seed": 7,
             "actors": 1, "version": 2, "created_at": "old", "spec_path": "old.json"}
    with (patch("hk_agent.config.load_config", return_value=SimpleNamespace(instance_count=2)),
          patch.object(plane, "_operator_checkpoint_candidate", return_value={
              "campaign_id": campaign_id, "campaign": prior, "checkpoint": str(checkpoint),
              "round": 3, "algorithm": "MaskablePPO",
          }),
          patch.object(plane, "create_campaign", side_effect=lambda data: {**data, "version": 3}) as create,
          patch.object(plane, "launch", return_value={"launch_id": "next"})):
        plane.continue_operator_campaign(campaign_id=campaign_id, instances=2)
    assert create.call_args.args[0]["actors"] == 2
    assert create.call_args.args[0]["checkpoint_interval"] == 1024


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
    assert campaign["help_stuck_minutes"] == 60.0
    assert campaign["warm_start_source_campaign"] == "newer-recurrent"
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
    assert campaign["campaign_id"].startswith("jogada-1-new-")
    assert campaign["warm_start_source_campaign"] == "compatible"
    assert campaign["warm_start_checkpoint"].endswith("round-0011.zip")
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


def test_operator_snapshot_keeps_dashboard_and_explains_instance_dll_marker_mismatch(tmp_path: Path) -> None:
    from hk_agent.config import ConfigError

    plane = ControlPlane(tmp_path)
    with (patch.object(plane, "status", return_value={"active_count": 0, "active_launches": [], "capacity": 1}),
          patch.object(plane, "_ensure_auto_reports"),
          patch.object(plane, "list_campaigns", return_value=[]),
          patch.object(plane, "_campaign_scene_discoveries", return_value=[]),
          patch.object(plane, "_operator_playthroughs",
                       side_effect=ConfigError("managed instance marker is missing or invalid: hk-ai-1")),
          patch.object(plane, "_operator_system_resources", return_value={}),
          patch.object(plane, "map_art_status", return_value={}),
          patch.object(plane, "list_reports", return_value=[])):
        snapshot = plane.operator_snapshot()
    phrase = "A instância 1 está com a DLL diferente do registro — rode a implantação da DLL"
    assert snapshot["status_banner"]["kind"] == "error"
    assert snapshot["status_banner"]["message"] == phrase
    assert phrase in snapshot["alerts"]
    assert snapshot["playthroughs"] == []
    assert "reports" in snapshot and "journey" in snapshot


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


def test_active_campaign_hides_old_failure_from_banner_but_keeps_event_history(tmp_path: Path) -> None:
    plane = ControlPlane(tmp_path)
    campaign = {"campaign_id": "campaign-live", "campaign_type": "exploration_training",
                "reward_version": "explore_v4", "shared_learner": True}
    active = {"launch_id": "learner-live", "role": "learner", "status": "RUNNING",
              "campaign_id": "campaign-live", "shared_group": "group-live",
              "shared_root": str(plane.runtime_root / "shared" / "campaign-live"),
              "started_unix_ns": 30}
    failed = {"launch_id": "learner-old", "role": "learner", "status": "FAILED",
              "campaign_id": "campaign-live", "shared_group": "group-old",
              "failure": "old crash", "finished_unix_ns": 20}
    with (patch.object(plane, "status", return_value={"active_count": 1, "active_launches": [active], "capacity": 1}),
          patch.object(plane, "_ensure_auto_reports"),
          patch.object(plane, "list_campaigns", return_value=[campaign]),
          patch.object(plane, "_campaign_launch_records", return_value=[failed, active]),
          patch.object(plane, "_campaign_scene_discoveries", return_value=[]),
          patch.object(plane, "_operator_playthroughs", return_value=[]),
          patch.object(plane, "_operator_system_resources", return_value={}),
          patch.object(plane, "map_art_status", return_value={}),
          patch.object(plane, "list_reports", return_value=[])):
        snapshot = plane.operator_snapshot()
    assert snapshot["status_banner"]["kind"] in {"running", "warning"}
    assert "old crash" not in snapshot["status_banner"]["message"]
    assert snapshot["recent_events"][0]["message"].startswith("Falha:")


def test_high_system_memory_adds_operator_warning(tmp_path: Path) -> None:
    plane = ControlPlane(tmp_path)
    with (patch.object(plane, "status", return_value={"active_count": 1, "active_launches": [], "capacity": 1}),
          patch.object(plane, "_ensure_auto_reports"),
          patch.object(plane, "list_campaigns", return_value=[]),
          patch.object(plane, "_campaign_scene_discoveries", return_value=[]),
          patch.object(plane, "_operator_system_resources", return_value={"memory_percent": 90}),
          patch.object(plane, "map_art_status", return_value={}),
          patch.object(plane, "list_reports", return_value=[])):
        snapshot = plane.operator_snapshot()
    assert any("Memória do sistema acima de 85%" in item for item in snapshot["alerts"])


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


def test_help_request_api_starts_and_finishes_live_help_without_launching_game(tmp_path: Path) -> None:
    plane = ControlPlane(tmp_path, allow_execute=False)
    campaign_id = "campaign-help"
    help_root = plane.runtime_root / "shared" / campaign_id / "help-requests"
    help_root.mkdir(parents=True)
    request = {
        "request_id": "help-0-test", "slot": 0, "scene": "Tutorial_01",
        "place": "Tutorial_01:c9r1", "context": "traversal:Tutorial_01:cell-9-1",
        "status": "pending", "created_unix": 10.0,
    }
    (help_root / "slot-0.json").write_text(json.dumps({"schema_version": 1, "requests": [request]}), encoding="utf-8")
    campaign = {"campaign_id": campaign_id, "help_requests_enabled": True}
    with patch.object(plane, "list_campaigns", return_value=[campaign]):
        started = plane.resolve_help_request({"campaign_id": campaign_id, "request_id": request["request_id"], "action": "help_now"})
        assert started["status"] == "helping"
        assert started["expires_unix"] > time.time()
        help_snapshot = plane.help_request_snapshot(campaign_id)
        assert help_snapshot["help_requests"][0]["status"] == "helping"
        completed = plane.resolve_help_request({"campaign_id": campaign_id, "request_id": request["request_id"], "action": "complete"})
    assert completed["status"] == "completion_requested"


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

        with patch.object(plane, "_operator_system_resources", return_value={}):
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


def test_map_calibration_api_supports_bounds_captures_and_removal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    plane = ControlPlane(tmp_path)
    digest = "a" * 64
    monkeypatch.setattr(plane, "map_art_status", lambda: {
        "sha256": digest, "tiles_ready": True, "tiles": {"width": 500, "height": 400},
    })
    server = ControlPlaneHTTPServer(("127.0.0.1", 0), plane, tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    headers = {"Content-Type": "application/json", "X-Control-Token": plane.csrf_token}
    try:
        for mode, anchors in (("bounds", [
            {"world_x": -20, "world_y": 10, "map_x": 100, "map_y": 200},
            {"world_x": 180, "world_y": 10, "map_x": 300, "map_y": 200},
        ]), ("captures", [
            {"world_x": 0, "world_y": 0, "map_x": 100, "map_y": 100},
            {"world_x": 100, "world_y": 0, "map_x": 200, "map_y": 100},
            {"world_x": 0, "world_y": 100, "map_x": 100, "map_y": 0},
        ])):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            connection.request("POST", "/api/journey/map/calibration", body=json.dumps({
                "campaign_id": "jogada-1", "scene": "Tutorial_01", "mode": mode, "anchors": anchors,
            }), headers=headers)
            response = connection.getresponse()
            result = json.loads(response.read())
            assert response.status == 201
            assert result["mode"] == mode and result["positioned"] is True
            assert result["rmse"] < 0.01
            connection.close()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        connection.request("POST", "/api/journey/map/calibration/remove", body=json.dumps({
            "scene": "Tutorial_01",
        }), headers=headers)
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["removed"] is True
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
    assert 'id="calibration-open"' in html and 'id="calibration-wizard"' in html
    assert 'id="theme-toggle"' in html and 'value="60"' in html
    assert 'Começar campanha nova' in html and 'id="supervisor-restart"' in html
    assert 'localStorage.setItem("voidgym-theme"' in app
    assert 'window.matchMedia("(prefers-color-scheme: dark)")' in app
    assert 'fetch("/api/session",{cache:"no-store"})' in app
    assert 'data.resumable_campaign' in app and 'new AbortController()' in app and 'setInterval(refresh,2500)' not in app
    assert 'id="retry-status"' in html and 'document.hidden' in app
    assert 'style="stroke:${color}"' not in app and 'style="fill:${color}"' not in app
    assert 'data-theme="dark"' in css and "prefers-color-scheme:dark" in css
    assert 'resolved_solo:"Resolvido por ela"' in app
    assert 'value="bounds"' in html and 'value="captures"' in html
    assert '/api/journey/map/calibration/remove' in app
    assert 'rgba(14,32,43,.78)' not in app and 'destination-out' not in app
    assert '/api/journey/map/tile?' in app
    assert 'worldTile' in app
    assert 'class="journey-run-summary"' in app
    assert ".journey-run-summary{flex:1 1 100%" in css
    assert "appearance:none" in css and "data:image/svg+xml" in css
    assert 'id="status-banner"' in html and 'id="status-technical-detail"' in html
    assert ".status-banner.error" in css and ".status-banner.warning" in css
    assert "setStatusBanner(\"error\"" in app and "friendlyFailure" in app



def test_resume_candidate_prefers_recurrent_checkpoint_and_marks_empty_failure(tmp_path: Path) -> None:
    plane = ControlPlane(tmp_path)
    campaigns_root = plane.control_root / "campaigns"
    for campaign_id, created, algorithm, with_checkpoint in (
        ("older-recurrent", "2026-01-01T00:00:00Z", "MaskableRecurrentPPO", True),
        ("newer-feedforward", "2026-01-02T00:00:00Z", "MaskablePPO", True),
        ("newest-recurrent-short", "2026-01-04T00:00:00Z", "MaskableRecurrentPPO", True),
        ("empty-failure", "2026-01-03T00:00:00Z", "MaskableRecurrentPPO", False),
    ):
        directory = campaigns_root / campaign_id
        directory.mkdir(parents=True)
        (directory / "v0001.json").write_text(json.dumps({
            "campaign_id": campaign_id, "name": campaign_id,
            "campaign_type": "exploration_training", "reward_version": "explore_v4",
            "shared_learner": True, "created_at": created,
        }), encoding="utf-8")
        shared = plane.runtime_root / "shared" / campaign_id
        shared.mkdir(parents=True)
        (shared / "manifest.json").write_text(json.dumps({"algorithm": algorithm}), encoding="utf-8")
        if with_checkpoint:
            checkpoints = shared / "checkpoints"
            checkpoints.mkdir()
            round_number = "00000002" if campaign_id == "newest-recurrent-short" else "00000108"
            (checkpoints / f"round-{round_number}.zip").write_bytes(b"checkpoint")
    failed = plane.control_root / "launches" / "failed-empty"
    failed.mkdir(parents=True)
    (failed / "launch.json").write_text(json.dumps({
        "campaign_id": "empty-failure", "status": "FAILED",
        "finished_unix_ns": 10, "failure": "learner failed before first round",
    }), encoding="utf-8")

    candidate = plane._operator_checkpoint_candidate()
    assert candidate is not None
    assert candidate["campaign_id"] == "older-recurrent"
    assert candidate["round"] == 108
    empty = next(row for row in plane.list_campaigns() if row["campaign_id"] == "empty-failure")
    assert empty["operator_state"] == "not_started"
    assert empty["operator_failure"] == "learner failed before first round"
