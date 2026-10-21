from __future__ import annotations

import argparse
import http.client
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen

from hk_agent.config import ConfigError, load_config
from hk_agent.control_plane import (
    ControlPlane,
    ControlPlaneBusyError,
    parse_json_body,
    require_keys,
)


class ControlPlaneHandler(BaseHTTPRequestHandler):
    server: ControlPlaneHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        super().log_message(format, *args)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/operator":
            with self.server.client_lock(self.client_address[0]):
                self._do_get()
            return
        self._do_get()

    def _do_get(self) -> None:
        path = urlsplit(self.path).path
        if self.server.upstream_port is not None and path == "/api/session" and not _supervisor_available(self.server.upstream_port):
            self._json(HTTPStatus.OK, {"csrf_token": self.server.control_plane.csrf_token})
        elif self.server.upstream_port is not None and path.startswith("/api/"):
            self._relay("GET")
            return
        elif path == "/api/status":
            self._json(HTTPStatus.OK, self.server.control_plane.status())
        elif path == "/api/operator":
            self._json(HTTPStatus.OK, self.server.control_plane.operator_snapshot())
        elif path == "/api/journey/run":
            query = parse_qs(urlsplit(self.path).query)
            try:
                detail = self.server.control_plane.journey_run(
                    query.get("campaign", [""])[0], query.get("run", [""])[0]
                )
            except (OSError, ValueError, TypeError) as error:
                self._error(HTTPStatus.NOT_FOUND, str(error))
                return
            self._json(HTTPStatus.OK, detail)
        elif path == "/api/reports":
            self._json(HTTPStatus.OK, {"reports": self.server.control_plane.list_reports()})
        elif path == "/api/report":
            query = parse_qs(urlsplit(self.path).query)
            relative = query.get("path", [""])[0]
            try:
                content, content_type = self.server.control_plane.read_report(relative)
            except (OSError, ValueError) as error:
                self._error(HTTPStatus.NOT_FOUND, str(error))
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)
        elif path == "/api/runs":
            self._json(HTTPStatus.OK, {"runs": self.server.control_plane.list_runs()})
        elif path == "/api/campaigns":
            self._json(HTTPStatus.OK, {"campaigns": self.server.control_plane.list_campaigns()})
        elif path == "/api/log":
            query = parse_qs(urlsplit(self.path).query)
            try:
                offset = int(query.get("offset", ["0"])[0])
            except ValueError:
                self._error(HTTPStatus.BAD_REQUEST, "offset must be an integer")
                return
            launch_id = query.get("launch_id", [None])[0]
            self._json(
                HTTPStatus.OK,
                self.server.control_plane.read_log(offset, launch_id=launch_id),
            )
        elif path == "/api/session":
            self._json(HTTPStatus.OK, {"csrf_token": self.server.control_plane.csrf_token})
        elif path == "/api/journey/capture":
            query = parse_qs(urlsplit(self.path).query)
            campaign = query.get("campaign", [""])[0]
            capture = query.get("path", [""])[0]
            try:
                content = self.server.control_plane.read_journey_capture(campaign, capture)
            except (OSError, ValueError) as error:
                self._error(HTTPStatus.NOT_FOUND, str(error))
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)
        elif path == "/api/help-thumbnail":
            query = parse_qs(urlsplit(self.path).query)
            try:
                content = self.server.control_plane.read_help_thumbnail(
                    query.get("campaign_id", [""])[0], query.get("request_id", [""])[0]
                )
            except (OSError, ValueError) as error:
                self._error(HTTPStatus.NOT_FOUND, str(error))
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)
        elif path in {"/api/journey/map/tile", "/api/journey/map/thumbnail"}:
            query = parse_qs(urlsplit(self.path).query)
            try:
                digest = query.get("sha256", [""])[0]
                if path.endswith("/thumbnail"):
                    content = self.server.control_plane.read_map_thumbnail(digest)
                    content_type = "image/jpeg"
                else:
                    level = int(query.get("level", [""])[0])
                    column = int(query.get("column", [""])[0])
                    row = int(query.get("row", [""])[0])
                    content = self.server.control_plane.read_map_tile(digest, level, column, row)
                    content_type = "image/jpeg"
            except (OSError, ValueError, TypeError) as error:
                self._error(HTTPStatus.NOT_FOUND, str(error))
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "private, max-age=31536000, immutable")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)
        else:
            self._static(path)

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/operator/supervisor/restart":
            allowed_tokens = {self.server.control_plane.csrf_token}
            if self.server.upstream_port is not None and _supervisor_available(self.server.upstream_port):
                try:
                    with urlopen(f"http://127.0.0.1:{self.server.upstream_port}/api/session", timeout=0.5) as response:
                        session = json.loads(response.read().decode("utf-8"))
                    if isinstance(session, dict) and isinstance(session.get("csrf_token"), str):
                        allowed_tokens.add(session["csrf_token"])
                except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                    pass
            if not self._mutation_is_allowed(allowed_tokens):
                return
            try:
                payload = self._payload()
                if payload:
                    raise ValueError("supervisor restart takes no settings")
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
                return
            if self.server.project_root is None or self.server.supervisor_port is None:
                self._error(HTTPStatus.CONFLICT, "supervisor relaunch is unavailable in this panel mode")
                return
            try:
                _start_detached_supervisor(self.server.project_root, port=self.server.supervisor_port)
                self._json(HTTPStatus.OK, {"started": True, "port": self.server.supervisor_port})
            except (OSError, RuntimeError) as error:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(error))
            return
        if self.server.upstream_port is not None and path.startswith("/api/"):
            self._relay("POST")
            return
        if not self._mutation_is_allowed():
            return
        path = urlsplit(self.path).path
        if path == "/api/journey/map/download":
            try:
                payload = self._payload()
                if payload:
                    raise ValueError("map download takes no settings")
                self._json(HTTPStatus.OK, self.server.control_plane.download_operator_map_art())
            except (OSError, RuntimeError, ValueError) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
            return
        if path == "/api/journey/map/calibration":
            try:
                result = self.server.control_plane.save_map_scene_calibration(self._payload())
                self._json(HTTPStatus.CREATED, result)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
            return
        if path == "/api/journey/map/calibration/remove":
            try:
                result = self.server.control_plane.remove_map_scene_calibration(self._payload())
                self._json(HTTPStatus.OK, result)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
            return
        if path == "/api/journey/map/import":
            try:
                if self.headers.get("Content-Type", "").split(";", 1)[0].strip() != "image/png":
                    raise ValueError("Content-Type must be image/png")
                length = int(self.headers.get("Content-Length", "0"))
                self._json(HTTPStatus.CREATED, self.server.control_plane.import_operator_map_art(self.rfile, length))
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                self._error(HTTPStatus.BAD_REQUEST, str(error))
            return
        try:
            payload = self._payload()
            if path == "/api/operator/start":
                require_keys(
                    payload,
                    ("instances", "duration_kind", "duration_value", "campaign_name", "help_stuck_count", "help_stuck_minutes"),
                )
                if set(payload) not in (
                    {"instances", "duration_kind", "duration_value"},
                    {"instances", "duration_kind", "duration_value", "campaign_name"},
                    {"instances", "duration_kind", "duration_value", "campaign_name", "help_stuck_count"},
                    {"instances", "duration_kind", "duration_value", "campaign_name", "help_stuck_count", "help_stuck_minutes"},
                ):
                    raise ValueError("start accepts instances, duration, campaign_name, help_stuck_count, and help_stuck_minutes")
                if not self.server.control_plane.allow_execute:
                    raise PermissionError("Start the panel with --allow-execute to open game windows")
                result = self.server.control_plane.start_operator_campaign(
                    instances=payload.get("instances"), duration_kind=payload.get("duration_kind"),
                    duration_value=payload.get("duration_value"),
                    campaign_name=payload.get("campaign_name"),
                    help_stuck_count=payload.get("help_stuck_count", 10),
                    help_stuck_minutes=payload.get("help_stuck_minutes", 60.0),
                )
                self._json(HTTPStatus.ACCEPTED, result)
            elif path == "/api/operator/continue":
                if set(payload) not in (
                    {}, {"help_stuck_count"}, {"help_stuck_count", "help_stuck_minutes"},
                    {"campaign_id"}, {"campaign_id", "help_stuck_count"},
                    {"campaign_id", "help_stuck_count", "help_stuck_minutes"},
                ):
                    raise ValueError("continue only accepts campaign id and help trigger settings")
                if not self.server.control_plane.allow_execute:
                    raise PermissionError("Start the panel with --allow-execute to open game windows")
                self._json(HTTPStatus.ACCEPTED, self.server.control_plane.continue_operator_campaign(
                    help_stuck_count=payload.get("help_stuck_count", 10),
                    help_stuck_minutes=payload.get("help_stuck_minutes", 60.0),
                    campaign_id=payload.get("campaign_id"),
                ))
            elif path == "/api/operator/repeat":
                if payload:
                    raise ValueError("repeat does not accept reward settings")
                if not self.server.control_plane.allow_execute:
                    raise PermissionError("Start the panel with --allow-execute to open game windows")
                self._json(HTTPStatus.ACCEPTED, self.server.control_plane.repeat_operator_campaign())
            elif path == "/api/operator/stop":
                if payload:
                    raise ValueError("safe stop does not accept settings")
                current = self.server.control_plane.status()
                active = current.get("active_launches", [])
                if not isinstance(active, list) or not active:
                    raise RuntimeError("No training is currently running")
                actor = next((row for row in active if isinstance(row, Mapping) and row.get("role") == "actor"), active[0])
                launch_id = actor.get("launch_id") if isinstance(actor, Mapping) else None
                self._json(HTTPStatus.ACCEPTED, self.server.control_plane.stop(str(launch_id)))
            elif path in {"/api/playthrough/new", "/api/playthrough/restore"}:
                require_keys(payload, ("campaign_id", "slot"))
                if set(payload) != {"campaign_id", "slot"}:
                    raise ValueError("playthrough action accepts campaign_id and slot")
                if path.endswith("/new"):
                    result = self.server.control_plane.start_new_playthrough(payload)
                else:
                    result = self.server.control_plane.restore_playthrough_backup(payload)
                self._json(HTTPStatus.OK, result)
            elif path == "/api/challenges/practice":
                require_keys(payload, ("campaign_id", "slot", "challenge_id"))
                if set(payload) != {"campaign_id", "slot", "challenge_id"}:
                    raise ValueError("challenge practice accepts campaign_id, slot, and challenge_id")
                if not self.server.control_plane.allow_execute:
                    raise PermissionError("Start the panel with --allow-execute to launch a challenge actor")
                result = self.server.control_plane.start_challenge_practice(payload)
                self._json(HTTPStatus.ACCEPTED, result)
            elif path == "/api/help-requests/resolve":
                require_keys(payload, ("campaign_id", "request_id", "action"))
                if set(payload) != {"campaign_id", "request_id", "action"}:
                    raise ValueError("help request resolution accepts campaign_id, request_id, and action")
                self._json(HTTPStatus.OK, self.server.control_plane.resolve_help_request(payload))
            elif path == "/api/reports/generate":
                require_keys(payload, ("campaign_id",))
                campaign_id = payload.get("campaign_id")
                if not isinstance(campaign_id, str):
                    raise ValueError("campaign_id must be a string")
                self._json(HTTPStatus.CREATED, self.server.control_plane.generate_run_report(campaign_id))
            elif path == "/api/campaigns":
                require_keys(
                    payload,
                    (
                        "campaign_id",
                        "name",
                        "campaign_type",
                        "seed",
                        "timesteps",
                        "checkpoint_interval",
                        "episodes",
                        "horizon",
                        "action_repeat",
                        "lease_ms",
                        "window_width",
                        "window_height",
                        "duration_mode",
                        "max_checkpoints",
                        "save_source",
                        "save_slot",
                        "weights",
                        "shared_learner",
                        "actors",
                        "rollout_steps",
                        "batch_size",
                        "ent_coef",
                        "gamma",
                        "gae_lambda",
                        "n_epochs",
                        "target_kl",
                        "learning_rate",
                        "clip_range",
                        "duration_limit_seconds",
                        "reward_version",
                        "warm_start_checkpoint",
                        "warm_start_source_campaign",
                        "stuck_seconds",
                        "evaluation_interval",
                        "tutorial_hint_level",
                        "straggler_timeout_seconds",
                        "challenge_practice_enabled",
                        "help_requests_enabled",
                    ),
                )
                result = self.server.control_plane.create_campaign(payload)
                self._json(HTTPStatus.CREATED, result)
            elif path == "/api/launch":
                require_keys(payload, ("campaign_id", "version", "execute"))
                campaign_id = str(payload.get("campaign_id", ""))
                raw_version = payload.get("version")
                if raw_version is not None and (
                    isinstance(raw_version, bool) or not isinstance(raw_version, int)
                ):
                    raise TypeError("version must be an integer")
                version = raw_version
                execute = payload.get("execute", False)
                if not isinstance(execute, bool):
                    raise ValueError("execute must be a boolean")
                result = self.server.control_plane.launch(
                    campaign_id, version=version, execute=execute
                )
                self._json(HTTPStatus.ACCEPTED, result)
            elif path == "/api/stop":
                require_keys(payload, ("launch_id",))
                raw_launch_id = payload.get("launch_id")
                if raw_launch_id is not None and not isinstance(raw_launch_id, str):
                    raise TypeError("launch_id must be a string")
                self._json(
                    HTTPStatus.ACCEPTED,
                    self.server.control_plane.stop(raw_launch_id),
                )
            else:
                self._error(HTTPStatus.NOT_FOUND, "endpoint not found")
        except PermissionError as error:
            self._error(HTTPStatus.FORBIDDEN, str(error))
        except RuntimeError as error:
            self._error(HTTPStatus.CONFLICT, str(error))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))

    def _payload(self) -> Mapping[str, object]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
        if content_type != "application/json":
            raise ValueError("Content-Type must be application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        return parse_json_body(self.rfile.read(length))

    def _relay(self, method: str) -> None:
        path = urlsplit(self.path).path
        body = self.rfile.read(int(self.headers.get("Content-Length", "0"))) if method == "POST" else None
        headers = {name: value for name, value in self.headers.items()
                   if name.lower() not in {"host", "origin", "connection", "content-length"}}
        if body is not None:
            headers["Content-Length"] = str(len(body))
        try:
            timeout = 120 if method == "GET" and path == "/api/operator" else 15
            connection = http.client.HTTPConnection("127.0.0.1", self.server.upstream_port, timeout=timeout)
            connection.request(method, self.path, body=body, headers=headers)
            response = connection.getresponse()
            content = response.read()
            self.send_response(response.status)
            for name, value in response.getheaders():
                if name.lower() in {"content-type", "cache-control", "x-content-type-options", "content-disposition"}:
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            connection.close()
        except OSError as error:
            if self.server.supervisor_port is not None and _supervisor_available(self.server.supervisor_port):
                self._error(
                    HTTPStatus.GATEWAY_TIMEOUT,
                    f"O supervisor está online, mas a consulta {path} excedeu o tempo de resposta: {error}",
                )
                return
            if (method == "GET" and self.server.project_root is not None
                    and self.server.supervisor_port is not None
                    and path in {"/api/status", "/api/operator", "/api/session"}
                    and not getattr(self, "_supervisor_restart_attempted", False)
                    and not _active_launch_process_exists(self.server.project_root)):
                self._supervisor_restart_attempted = True
                try:
                    _start_detached_supervisor(self.server.project_root, port=self.server.supervisor_port)
                except (OSError, RuntimeError):
                    pass
                else:
                    return self._relay(method)
            last_line = _last_supervisor_log_line(self.server.project_root) if self.server.project_root else None
            message = "O supervisor do treino não está rodando."
            if last_line:
                message += f" Última linha do log: {last_line}"
            message += f" ({error})"
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, message)

    def _mutation_is_allowed(self, expected_tokens: set[str] | None = None) -> bool:
        allowed = expected_tokens or {self.server.control_plane.csrf_token}
        if self.headers.get("X-Control-Token") not in allowed:
            self._error(HTTPStatus.FORBIDDEN, "missing or invalid control token")
            return False
        origin = self.headers.get("Origin")
        if origin is not None:
            expected_origins = {
                f"http://127.0.0.1:{self.server.server_port}",
                f"http://localhost:{self.server.server_port}",
            }
            if origin not in expected_origins:
                self._error(HTTPStatus.FORBIDDEN, "cross-origin mutation rejected")
                return False
        return True

    def _static(self, path: str) -> None:
        name = "index.html" if path in ("", "/") else path.lstrip("/")
        if name not in {"index.html", "app.css", "app.js"}:
            self._error(HTTPStatus.NOT_FOUND, "resource not found")
            return
        target = self.server.static_root / name
        try:
            content = target.read_bytes()
        except OSError:
            self._error(HTTPStatus.NOT_FOUND, "resource not found")
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(content)

    def _json(self, status: HTTPStatus, payload: Mapping[str, object]) -> None:
        content = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"error": message})


class ControlPlaneHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        control_plane: ControlPlane,
        static_root: Path,
        upstream_port: int | None = None,
        project_root: Path | None = None,
    ) -> None:
        super().__init__(address, ControlPlaneHandler)
        self.control_plane = control_plane
        self.static_root = static_root
        self.upstream_port = upstream_port
        self.supervisor_port = upstream_port
        self.project_root = project_root
        self._client_locks_guard = threading.Lock()
        self._client_locks: dict[str, threading.Lock] = {}

    def client_lock(self, client: str) -> threading.Lock:
        with self._client_locks_guard:
            if len(self._client_locks) > 256:
                self._client_locks.clear()
            return self._client_locks.setdefault(client, threading.Lock())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the local Hollow Knight training panel")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allow-execute",
        action="store_true",
        help="permit explicit real campaign launches; omitted means dry-run only",
    )
    parser.add_argument("--supervisor-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-port", type=int, help=argparse.SUPPRESS)
    return parser


def build_control_plane(project_root: Path, *, allow_execute: bool, observer: bool = False) -> ControlPlane:
    if not allow_execute:
        return ControlPlane(
            project_root,
            allow_execute=False,
            max_active_launches=1,
            parallel_enabled=False,
            runtime_root=project_root / "artifacts" / "runtime",
            training_device="cpu",
            torch_threads=3,
            torch_interop_threads=1,
            observer=observer,
        )
    project_config = load_config(
        project_root / "config.training.local.toml", allow_training=True
    )
    return ControlPlane(
        project_root,
        allow_execute=True,
        max_active_launches=project_config.instance_count,
        parallel_enabled=project_config.parallel_enabled,
        runtime_root=project_config.runtime_root,
        training_device=project_config.training_device,
        torch_threads=project_config.torch_threads,
        torch_interop_threads=project_config.torch_interop_threads,
        observer=observer,
    )


def _supervisor_available(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/session", timeout=0.3):
            return True
    except OSError:
        return False


def _last_supervisor_log_line(project_root: Path) -> str | None:
    log_root = project_root / "artifacts" / "control-plane"
    try:
        logs = sorted(log_root.glob("supervisor-*.log"), key=lambda path: path.stat().st_mtime_ns)
    except OSError:
        return None
    for path in reversed(logs):
        try:
            lines = path.read_text("utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            if line.strip():
                return line.strip()[-1000:]
    return None


def _active_launch_process_exists(project_root: Path) -> bool:
    launch_root = project_root / "artifacts" / "control-plane" / "launches"
    try:
        paths = launch_root.glob("*/launch.json")
        for path in paths:
            try:
                record = json.loads(path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict) or record.get("status") not in {"PREPARING", "RUNNING", "STOP_REQUESTED"}:
                continue
            if record.get("status") == "PREPARING":
                try:
                    if time.time_ns() - path.stat().st_mtime_ns < 15 * 60 * 1_000_000_000:
                        return True
                except OSError:
                    continue
                continue
            pid = record.get("pid")
            if isinstance(pid, int) and not isinstance(pid, bool) and ControlPlane._launch_process_matches(record, pid):
                return True
    except OSError:
        return True
    return False


def _start_detached_supervisor(project_root: Path, *, port: int) -> None:
    if _supervisor_available(port):
        return
    log_root = project_root / "artifacts" / "control-plane"
    log_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    log_path = log_root / f"supervisor-{stamp}-{os.getpid()}.log"
    launcher = project_root / "tools" / "serve_training_dashboard.py"
    environment_python = project_root / ".venv" / "Scripts" / "python.exe"
    if not environment_python.is_file():
        environment_python = project_root / ".venv" / "bin" / "python"
    python = str(environment_python) if environment_python.is_file() else sys.executable
    command = [python, str(launcher), "--allow-execute", "--supervisor-mode", "--port", str(port)]
    flags = (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
             | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
             | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000)) if os.name == "nt" else 0
    with log_path.open("ab") as log_stream:
        if os.name == "nt":
            subprocess.Popen(
                command, cwd=project_root, stdin=subprocess.DEVNULL, stdout=log_stream,
                stderr=subprocess.STDOUT, close_fds=True, creationflags=flags,
            )
        else:
            subprocess.Popen(
                command, cwd=project_root, stdin=subprocess.DEVNULL, stdout=log_stream,
                stderr=subprocess.STDOUT, close_fds=True, start_new_session=True,
            )
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if _supervisor_available(port):
            return
        time.sleep(0.1)
    last_line = _last_supervisor_log_line(project_root)
    detail = f" Last log line: {last_line}" if last_line else " The supervisor log is empty."
    raise RuntimeError(f"campaign supervisor did not start on 127.0.0.1:{port}.{detail} Log: {log_path}")


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.port <= 65_535:
        raise SystemExit("port must be in [1, 65535]")
    project_root = Path(__file__).resolve().parents[2]
    static_root = Path(__file__).resolve().parent / "control_plane_web"
    try:
        upstream_port: int | None = None
        if args.supervisor_mode:
            control_plane = build_control_plane(project_root, allow_execute=True)
        elif args.allow_execute:
            supervisor_port = args.supervisor_port or args.port + 1
            if not 1 <= supervisor_port <= 65_535 or supervisor_port == args.port:
                raise SystemExit("supervisor port must be valid and distinct from panel port")
            # Validate training configuration without taking the owner lock before
            # starting the detached process. The returned observer is the panel's
            # read-only view; all operator API calls are relayed to the supervisor.
            control_plane = build_control_plane(
                project_root, allow_execute=True, observer=True
            )
            upstream_port = supervisor_port
            try:
                _start_detached_supervisor(project_root, port=supervisor_port)
            except (OSError, RuntimeError):
                # Keep the operator panel alive so it can show the log and offer recovery.
                pass
        else:
            control_plane = build_control_plane(project_root, allow_execute=False)
    except (ConfigError, ControlPlaneBusyError) as error:
        raise SystemExit(str(error)) from error
    server = ControlPlaneHTTPServer(("127.0.0.1", args.port), control_plane, static_root,
                                    upstream_port=upstream_port, project_root=project_root)
    if args.supervisor_mode:
        # Warm the mtime-backed operator caches before serving the first browser
        # request, so initial page render does not pay the cold filesystem scan.
        control_plane.operator_snapshot()
    print(
        f"HK Training Control Plane: http://127.0.0.1:{args.port} "
        f"({'real launches enabled' if args.allow_execute else 'dry-run only'})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
