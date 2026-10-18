from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path
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
