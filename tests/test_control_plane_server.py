from __future__ import annotations

import http.client
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from hk_agent.config import ConfigError
from hk_agent.control_plane import ControlPlane
from hk_agent.control_plane_server import ControlPlaneHTTPServer, build_control_plane, main


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
        def do_POST(self) -> None:
            seen["token"] = self.headers.get("X-Control-Token")
            seen["origin"] = self.headers.get("Origin")
            seen["body"] = self.rfile.read(int(self.headers.get("Content-Length", "0")))
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
        connection.request("POST", "/api/operator/stop", body=b"{}", headers={
            "Content-Type": "application/json", "Content-Length": "2",
            "X-Control-Token": "campaign-secret",
            "Origin": f"http://127.0.0.1:{client.server_port}",
        })
        response = connection.getresponse()
        assert response.status == 202
        assert response.read() == b'{"status":"STOP_REQUESTED"}'
        assert seen == {"token": "campaign-secret", "origin": None, "body": b"{}"}
        connection.close()
    finally:
        client.shutdown()
        client.server_close()
        client_thread.join(timeout=3)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=3)
