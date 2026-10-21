from __future__ import annotations

import http.client
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from hk_agent import control_plane_server as server_module
from hk_agent.config import ConfigError
from hk_agent.control_plane import ControlPlane
from hk_agent.control_plane_server import (
    ControlPlaneHTTPServer,
    _last_supervisor_log_line,
    build_control_plane,
    main,
)


def test_dry_run_factory_does_not_require_local_config(tmp_path: Path) -> None:
    control_plane = build_control_plane(tmp_path, allow_execute=False)

    assert not control_plane.allow_execute
    assert control_plane.max_active_launches == 1
    assert not control_plane.parallel_enabled
    assert control_plane.runtime_root == (tmp_path / "artifacts" / "runtime").resolve()
    assert control_plane.training_device == "cpu"
    assert control_plane.torch_threads == 3
    assert control_plane.torch_interop_threads == 1


def test_allow_execute_without_valid_config_exits_with_short_message() -> None:
    error = ConfigError("Configuration not found: config.training.local.toml")
    with (
        patch.object(sys, "argv", ["control-plane", "--allow-execute"]),
        patch("hk_agent.control_plane_server.build_control_plane", side_effect=error),
        pytest.raises(SystemExit, match="Configuration not found"),
    ):
        main()


def test_dashboard_client_relays_authenticated_api_calls_to_supervisor(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    class UpstreamHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            content = b'{"csrf_token":"campaign-secret"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self) -> None:
            seen["token"] = self.headers.get("X-Control-Token")
            seen["origin"] = self.headers.get("Origin")
            seen["body"] = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            seen["path"] = self.path
            content = b'{"status":"STOP_REQUESTED"}'
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(content)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    plane = ControlPlane(tmp_path, observer=True)
    client = ControlPlaneHTTPServer(("127.0.0.1", 0), plane, tmp_path, upstream_port=upstream.server_port)
    client_thread = threading.Thread(target=client.serve_forever, daemon=True)
    client_thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", client.server_port, timeout=3)
        connection.request("GET", "/api/session")
        session_response = connection.getresponse()
        session = json.loads(session_response.read())
        assert session == {"csrf_token": "campaign-secret"}
        connection.request("POST", "/api/operator/stop", body=b"{}", headers={
            "Content-Type": "application/json", "Content-Length": "2",
            "X-Control-Token": session["csrf_token"],
            "Origin": f"http://127.0.0.1:{client.server_port}",
        })
        response = connection.getresponse()
        assert response.status == 202
        assert response.read() == b'{"status":"STOP_REQUESTED"}'
        assert {key: value for key, value in seen.items() if key != "path"} == {"token": "campaign-secret", "origin": None, "body": b"{}"}
        assert seen["path"] == "/api/operator/stop"
        connection.request("POST", "/api/operator/abort", body=b"{}", headers={
            "Content-Type": "application/json", "Content-Length": "2",
            "X-Control-Token": session["csrf_token"],
            "Origin": f"http://127.0.0.1:{client.server_port}",
        })
        response = connection.getresponse()
        assert response.status == 202
        assert seen["path"] == "/api/operator/abort"
        assert seen["body"] == b"{}"
        connection.close()
    finally:
        client.shutdown()
        client.server_close()
        client_thread.join(timeout=3)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=3)



def test_supervisor_diagnostic_reads_latest_nonempty_log_line(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "control-plane"
    root.mkdir(parents=True)
    (root / "supervisor-20261020-120000-1.log").write_text("older failure\n", encoding="utf-8")
    latest = root / "supervisor-20261020-120001-1.log"
    latest.write_text("Starting supervisor\nRuntimeError: owner lock busy\n", encoding="utf-8")
    assert _last_supervisor_log_line(tmp_path) == "RuntimeError: owner lock busy"



def test_detached_supervisor_writes_output_and_reports_last_line(tmp_path: Path, monkeypatch) -> None:
    ticks = iter((0.0, 16.0))
    monkeypatch.setattr(server_module, "_supervisor_available", lambda _port: False)
    monkeypatch.setattr(server_module.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(server_module.time, "sleep", lambda _seconds: None)

    def fake_popen(_command, **kwargs):
        kwargs["stdout"].write(b"starting supervisor\nRuntimeError: lock unavailable\n")
        return object()

    monkeypatch.setattr(server_module.subprocess, "Popen", fake_popen)
    with pytest.raises(RuntimeError, match="RuntimeError: lock unavailable"):
        server_module._start_detached_supervisor(tmp_path, port=18766)
    logs = list((tmp_path / "artifacts" / "control-plane").glob("supervisor-*.log"))
    assert len(logs) == 1
    assert "RuntimeError: lock unavailable" in logs[0].read_text("utf-8")


def test_detached_supervisor_prefers_project_virtual_environment(tmp_path: Path, monkeypatch) -> None:
    python = tmp_path / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"test executable marker")
    checks = iter((False, True))
    monkeypatch.setattr(server_module, "_supervisor_available", lambda _port: next(checks))
    commands: list[list[str]] = []
    monkeypatch.setattr(server_module.subprocess, "Popen", lambda command, **_kwargs: commands.append(command))

    server_module._start_detached_supervisor(tmp_path, port=18766)

    assert commands[0][0] == str(python)


def test_panel_manual_supervisor_restart_accepts_local_token(tmp_path: Path, monkeypatch) -> None:
    called: list[tuple[Path, int]] = []
    monkeypatch.setattr(server_module, "_start_detached_supervisor",
                        lambda root, *, port: called.append((root, port)))
    plane = ControlPlane(tmp_path, observer=True)
    client = ControlPlaneHTTPServer(("127.0.0.1", 0), plane, tmp_path,
                                    upstream_port=18766, project_root=tmp_path)
    thread = threading.Thread(target=client.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", client.server_port, timeout=3)
        body = b"{}"
        connection.request("POST", "/api/operator/supervisor/restart", body=body, headers={
            "Content-Type": "application/json", "Content-Length": "2",
            "X-Control-Token": plane.csrf_token,
            "Origin": f"http://127.0.0.1:{client.server_port}",
        })
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == {"started": True, "port": 18766}
        assert called == [(tmp_path, 18766)]
        connection.close()
    finally:
        client.shutdown()
        client.server_close()
        thread.join(timeout=3)


def test_operator_get_auto_restarts_supervisor_only_when_no_live_launches(tmp_path: Path, monkeypatch) -> None:
    upstream_port = 0
    monkeypatch.setattr(server_module, "_supervisor_available", lambda _port: False)
    monkeypatch.setattr(server_module, "_active_launch_process_exists", lambda _root: False)
    restarted: list[int] = []
    monkeypatch.setattr(server_module, "_start_detached_supervisor",
                        lambda _root, *, port: restarted.append(port))
    plane = ControlPlane(tmp_path, observer=True)
    client = ControlPlaneHTTPServer(("127.0.0.1", 0), plane, tmp_path,
                                    upstream_port=upstream_port, project_root=tmp_path)
    thread = threading.Thread(target=client.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", client.server_port, timeout=3)
        connection.request("GET", "/api/operator")
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 503
        assert payload["error"].startswith("O supervisor do treino não está rodando.")
        assert restarted == [upstream_port]
        connection.close()
    finally:
        client.shutdown()
        client.server_close()
        thread.join(timeout=3)
