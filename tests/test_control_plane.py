from __future__ import annotations

import hashlib
import http.client
import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from hk_agent import worker_cycle as worker_cycle_module
from hk_agent.control_plane import CampaignSpec, ControlPlane
from hk_agent.control_plane_server import ControlPlaneHTTPServer


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


def test_campaign_specs_are_validated_and_versioned(tmp_path: Path) -> None:
    control = ControlPlane(_project(tmp_path))
    first = control.create_campaign(_payload())
    second = control.create_campaign(_payload())
    assert first["version"] == 1
    assert second["version"] == 2
    assert first["campaign_id"] == "kings-pass-exploration"
    assert len(control.list_campaigns()) == 2
    assert first["schema_version"] == 5
    assert first["save_source"] == "existing_save"
    assert first["save_slot"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [("seed", -1), ("horizon", 2), ("lease_ms", 5), ("action_repeat", 0)],
)
def test_campaign_rejects_out_of_range_values(field: str, value: int) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(ValueError, match=field):
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
        "status": "RUNNING",
        "stop_file": (
            "artifacts/control-plane/launches/launch-1/stop.request"
        ),
        "log_path": "artifacts/control-plane/launches/launch-1/process.log",
    }
    (launch_root / "launch.json").write_text(json.dumps(launch), encoding="utf-8")
    monkeypatch.setattr(ControlPlane, "_pid_is_alive", staticmethod(lambda pid: pid == 31337))

    control = ControlPlane(root)
    active = control.status()["active"]
    assert isinstance(active, dict)
    assert active["reattached"] is True
    stopped = control.stop()
    assert stopped == {"status": "STOP_REQUESTED", "pid": 31337}
    assert (launch_root / "stop.request").is_file()


def test_default_reward_values_are_valid_in_the_browser_form() -> None:
    html = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "hk_agent"
        / "control_plane_web"
        / "index.html"
    ).read_text("utf-8")
    assert 'name="success_reward"' in html
    assert 'name="visual_change_scale"' in html
    assert 'name="enemy_kill_reward"' in html
    assert 'name="enemy_kill_reward_budget"' in html
    assert 'name="shade_recovery_reward"' in html
    assert 'name="damage_penalty"' in html
    assert 'name="geo_gain_reward"' in html
    assert 'name="geo_spend_reward"' in html
    assert 'name="death_penalty"' in html
    assert 'name="run_failed_penalty"' in html
    assert html.count('step="any"') == 10


class _FakeProcess:
    def __init__(self, command: list[str], **kwargs: Any) -> None:
        self.command = command
        self.kwargs = kwargs
        self.pid = 31337
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

        body = json.dumps(_payload())
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
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
