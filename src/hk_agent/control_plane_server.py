from __future__ import annotations

import argparse
import json
import mimetypes
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

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
        if path == "/api/status":
            self._json(HTTPStatus.OK, self.server.control_plane.status())
        elif path == "/api/operator":
            self._json(HTTPStatus.OK, self.server.control_plane.operator_snapshot())
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
                require_keys(payload, ("instances", "duration_kind", "duration_value"))
                if set(payload) != {"instances", "duration_kind", "duration_value"}:
                    raise ValueError("start accepts only number of instances and duration")
                if not self.server.control_plane.allow_execute:
                    raise PermissionError("Start the panel with --allow-execute to open game windows")
                result = self.server.control_plane.start_operator_campaign(
                    instances=payload.get("instances"), duration_kind=payload.get("duration_kind"),
                    duration_value=payload.get("duration_value"),
                )
                self._json(HTTPStatus.ACCEPTED, result)
            elif path == "/api/operator/continue":
                if payload:
                    raise ValueError("continue does not accept settings")
                if not self.server.control_plane.allow_execute:
                    raise PermissionError("Start the panel with --allow-execute to open game windows")
                self._json(HTTPStatus.ACCEPTED, self.server.control_plane.continue_operator_campaign())
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
                        "straggler_timeout_seconds",
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

    def _mutation_is_allowed(self) -> bool:
        expected = self.server.control_plane.csrf_token
        if self.headers.get("X-Control-Token") != expected:
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
    ) -> None:
        super().__init__(address, ControlPlaneHandler)
        self.control_plane = control_plane
        self.static_root = static_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the local Hollow Knight training panel")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allow-execute",
        action="store_true",
        help="permit explicit real campaign launches; omitted means dry-run only",
    )
    return parser


def build_control_plane(project_root: Path, *, allow_execute: bool) -> ControlPlane:
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
    )


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.port <= 65_535:
        raise SystemExit("port must be in [1, 65535]")
    project_root = Path(__file__).resolve().parents[2]
    static_root = Path(__file__).resolve().parent / "control_plane_web"
    try:
        control_plane = build_control_plane(project_root, allow_execute=args.allow_execute)
    except (ConfigError, ControlPlaneBusyError) as error:
        raise SystemExit(str(error)) from error
    server = ControlPlaneHTTPServer(("127.0.0.1", args.port), control_plane, static_root)
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
