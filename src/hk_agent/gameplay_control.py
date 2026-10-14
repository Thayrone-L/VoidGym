from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hk_agent.protocol import ProtocolError
from hk_agent.supervisor import AcceptedBridge, CapturedFrame, SupervisorServer

GAMEPLAY_BUTTONS = frozenset(
    {"LEFT", "RIGHT", "UP", "DOWN", "JUMP", "ATTACK", "DASH", "CAST", "SUPER_DASH", "DREAM_NAIL"}
)
MAXIMUM_GAMEPLAY_LEASE_MILLISECONDS = 2_000
MAXIMUM_CONTROL_MESSAGE_BYTES = 8 * 1024


class ControlRequestError(ValueError):
    """Raised when a local control request violates the closed command schema."""


@dataclass(frozen=True, slots=True)
class ControlAddress:
    host: str
    port: int
    token: str


class GameplayControlServer:
    """Authenticated localhost-only adapter for a single live game bridge."""

    def __init__(
        self,
        supervisor: SupervisorServer,
        bridge: AcceptedBridge,
        *,
        output_dir: Path,
        idle_timeout_seconds: float,
        settle_seconds: float,
        command_timeout_seconds: float,
    ) -> None:
        self._supervisor = supervisor
        self._bridge = bridge
        self._output_dir = output_dir
        self._idle_timeout_seconds = idle_timeout_seconds
        self._settle_seconds = settle_seconds
        self._command_timeout_seconds = command_timeout_seconds
        self._token = secrets.token_urlsafe(32)
        self._server: asyncio.Server | None = None
        self._command_lock = asyncio.Lock()
        self._handlers: set[asyncio.Task[None]] = set()
        self._close_event = asyncio.Event()
        self._activity_event = asyncio.Event()
        self._last_activity = time.monotonic()
        self._capture_index = 0
        self._action_id = 0
        self.close_reason = "not_started"
        self.commands_completed = 0
        self.captures: list[dict[str, object]] = []

    @property
    def address(self) -> ControlAddress:
        if self._server is None or not self._server.sockets:
            raise RuntimeError("Gameplay control server is not listening")
        host, port = self._server.sockets[0].getsockname()[:2]
        return ControlAddress(str(host), int(port), self._token)

    async def start(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._server = await asyncio.start_server(
            self._handle_client,
            "127.0.0.1",
            0,
            limit=MAXIMUM_CONTROL_MESSAGE_BYTES + 1,
        )
        self.close_reason = "running"

    async def wait_until_closed(self) -> str:
        while not self._close_event.is_set():
            remaining = self._idle_timeout_seconds - (time.monotonic() - self._last_activity)
            if remaining <= 0:
                self._signal_close("idle_timeout")
                break
            self._activity_event.clear()
            try:
                await asyncio.wait_for(self._activity_event.wait(), timeout=remaining)
            except TimeoutError:
                async with self._command_lock:
                    if time.monotonic() - self._last_activity >= self._idle_timeout_seconds:
                        self._signal_close("idle_timeout")
        await self._stop_accepting_and_drain()
        return self.close_reason

    async def close(self) -> None:
        self._signal_close(self.close_reason)
        await self._stop_accepting_and_drain()

    def _signal_close(self, reason: str) -> None:
        self.close_reason = reason
        self._close_event.set()
        # Wake wait_until_closed even when it consumed the activity signal
        # emitted at the beginning of a still-running CLOSE command.
        self._activity_event.set()

    async def _stop_accepting_and_drain(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        async with self._command_lock:
            pass
        current = asyncio.current_task()
        pending = tuple(task for task in self._handlers if task is not current)
        if pending:
            # Once the command lock has drained, handlers can only be finishing
            # response I/O.  Do not let a peer that never completes its TCP
            # close keep the isolated save transaction alive indefinitely.
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        current = asyncio.current_task()
        if current is not None:
            self._handlers.add(current)
        response: dict[str, object]
        try:
            raw = await asyncio.wait_for(
                reader.readline(), timeout=min(5.0, self._command_timeout_seconds)
            )
            if not raw or len(raw) > MAXIMUM_CONTROL_MESSAGE_BYTES or not raw.endswith(b"\n"):
                raise ControlRequestError("request must be one bounded newline-delimited JSON object")
            try:
                request = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ControlRequestError("request is not valid UTF-8 JSON") from error
            response = await self._dispatch(request)
        except (
            asyncio.LimitOverrunError,
            ControlRequestError,
            ConnectionError,
            OSError,
            ProtocolError,
            RuntimeError,
            TimeoutError,
            ValueError,
        ) as error:
            response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        try:
            writer.write(json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n")
            await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            if current is not None:
                self._handlers.discard(current)

    async def _dispatch(self, request: Any) -> dict[str, object]:
        if not isinstance(request, dict):
            raise ControlRequestError("request must be a JSON object")
        token = request.get("token")
        if not isinstance(token, str) or not secrets.compare_digest(token, self._token):
            raise ControlRequestError("authentication failed")
        command = request.get("command")
        if not isinstance(command, str):
            raise ControlRequestError("command must be a string")
        if command not in {"capture", "action", "release", "close"}:
            raise ControlRequestError("command must be capture, action, release, or close")
        expected_fields = (
            {"token", "command", "buttons", "lease_ms"}
            if command == "action"
            else {"token", "command"}
        )
        if set(request) != expected_fields:
            raise ControlRequestError("request fields do not match the command schema")

        async with self._command_lock:
            self._last_activity = time.monotonic()
            self._activity_event.set()
            if command == "capture":
                result = {"ok": True, "command": command, "capture": await self._capture("capture")}
            elif command == "action":
                result = await self._action(request)
            elif command == "release":
                await self._require_release()
                result = {"ok": True, "command": command, "released": True}
            else:
                await self._require_release()
                self._signal_close("client_close")
                result = {"ok": True, "command": command, "closing": True}
            self.commands_completed += 1
            return result

    async def _action(self, request: dict[str, Any]) -> dict[str, object]:
        buttons = request.get("buttons")
        lease_ms = request.get("lease_ms")
        if (
            not isinstance(buttons, list)
            or not buttons
            or len(buttons) > len(GAMEPLAY_BUTTONS)
            or not all(isinstance(button, str) and button in GAMEPLAY_BUTTONS for button in buttons)
            or len(set(buttons)) != len(buttons)
        ):
            raise ControlRequestError("buttons must be a non-empty unique gameplay-button list")
        if (
            isinstance(lease_ms, bool)
            or not isinstance(lease_ms, int)
            or lease_ms < 1
            or lease_ms > MAXIMUM_GAMEPLAY_LEASE_MILLISECONDS
        ):
            raise ControlRequestError("lease_ms must be an integer between 1 and 2000")

        self._action_id += 1
        released = False
        try:
            response = await self._supervisor.send_command(
                "APPLY_ACTION",
                {"action_id": self._action_id, "buttons": buttons, "lease_ms": lease_ms},
                timeout=self._command_timeout_seconds,
                bridge=self._bridge,
            )
            if response.message_type == "ERROR":
                raise ProtocolError(
                    f"Gameplay action failed: {response.payload.get('error_code', 'unknown')}"
                )
            if response.message_type != "ACK":
                raise ProtocolError(
                    f"Expected ACK for APPLY_ACTION, received {response.message_type}"
                )
            await asyncio.sleep((lease_ms / 1000.0) + self._settle_seconds)
        finally:
            released = await self._release()
            if not released:
                self._signal_close("release_failed")
                raise RuntimeError("RELEASE_ALL failed after gameplay action attempt")
        capture = await self._capture(f"action-{self._action_id:04d}")
        return {
            "ok": True,
            "command": "action",
            "action_id": self._action_id,
            "buttons": buttons,
            "lease_ms": lease_ms,
            "released": True,
            "capture": capture,
        }

    async def _release(self) -> bool:
        try:
            response = await self._supervisor.send_command(
                "RELEASE_ALL", {}, timeout=2.0, bridge=self._bridge
            )
            return response.message_type == "ACK"
        except (ConnectionError, OSError, ProtocolError, RuntimeError, TimeoutError):
            return False

    async def _require_release(self) -> None:
        if not await self._release():
            self._signal_close("release_failed")
            raise RuntimeError("RELEASE_ALL failed")

    async def _capture(self, label: str) -> dict[str, object]:
        frame = await self._supervisor.capture_once(
            timeout=self._command_timeout_seconds, bridge=self._bridge
        )
        self._capture_index += 1
        path = self._output_dir / f"{self._capture_index:04d}-{label}.png"
        _write_frame_atomic(path, frame)
        record: dict[str, object] = {
            "index": self._capture_index,
            "path": str(path),
            "frame_id": frame.frame_id,
            "capture_monotonic_ns": frame.capture_monotonic_ns,
            "width": frame.width,
            "height": frame.height,
            "bytes": len(frame.data),
            "scene_transition": frame.scene_transition,
            "duplicate_of": frame.duplicate_of,
            "last_applied_action_id": frame.last_applied_action_id,
        }
        self.captures.append(record)
        return record


def _write_frame_atomic(path: Path, frame: CapturedFrame) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    temporary.write_bytes(frame.data)
    temporary.replace(path)
