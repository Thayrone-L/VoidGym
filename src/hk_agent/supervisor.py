from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import math
import os
import secrets
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from hk_agent.action_masking import ActionAvailability, parse_action_availability
from hk_agent.protocol import PROTOCOL_VERSION, Envelope, ProtocolError, read_message, write_message
from hk_agent.session import SessionError, SessionMachine, SessionState


@dataclass(slots=True)
class AcceptedBridge:
    instance_id: str
    session_id: str
    game_profile: str
    game_version: str
    effective_save_root: str | None
    writer: asyncio.StreamWriter = field(repr=False)
    persistent_data_path: str | None = None
    product_name: str | None = None
    heartbeats_received: int = 0
    last_heartbeat_ns: int | None = None
    machine: SessionMachine = field(default_factory=SessionMachine)
    outbound_seq: int = 1
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    pending: dict[str, asyncio.Future[Envelope]] = field(default_factory=dict, repr=False)
    capabilities: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    episode_id: str | None
    frame_id: int
    capture_monotonic_ns: int
    width: int
    height: int
    channels: int
    pixel_format: str
    encoding: str
    data: bytes
    scene_transition: bool
    duplicate_of: int | None
    last_applied_action_id: int | None
    player_dead: bool = False
    death_persisted: bool = False
    shade_pending: bool = False
    shade_recovered: bool = False
    run_failed: bool = False
    enemy_kills_since_last_frame: int = 0
    hp_lost_since_last_frame: int = 0
    geo_gained_since_last_frame: int = 0
    geo_spent_since_last_frame: int = 0
    # Optional diagnostics from the bridge (name, value); never used as observation.
    bridge_timing: tuple[tuple[str, float], ...] = ()
    action_availability: ActionAvailability | None = None


def require_png_frame(frame: CapturedFrame) -> None:
    """Reject raw observations at code paths that persist .png artifacts."""
    if frame.encoding != "png":
        raise ProtocolError("PNG artifact writer requires FRAME encoding=png")


# Optional, diagnostic-only FRAME fields emitted by the bridge's capture backend.
BRIDGE_TIMING_FIELDS = (
    "queued_to_start_ms",
    "capture_ms",
    "encode_ms",
    "png_bytes",
    "payload_bytes",
    "log_ms",
    "unity_fps",
    "target_frame_rate",
    "vsync_count",
)
def parse_bridge_timing(payload: dict[str, object]) -> tuple[tuple[str, float], ...]:
    """Keep only well-formed optional metrics; malformed values are dropped, not fatal.

    Every value is non-negative, because the environment rejects negative timings.
    Unity reports target_frame_rate = -1 for "platform default"; that becomes the
    flag target_frame_rate_default = 1.
    """
    metrics: list[tuple[str, float]] = []
    for name in BRIDGE_TIMING_FIELDS:
        value = payload.get(name)
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if not math.isfinite(number):
            continue
        if name == "target_frame_rate" and number < 0:
            metrics.append(("target_frame_rate_default", 1.0))
            continue
        if number < 0:
            continue
        metrics.append((name, number))
    return tuple(metrics)


class SupervisorServer:
    def __init__(
        self,
        *,
        token: str | None = None,
        game_profile: str = "development",
        heartbeat_timeout_seconds: float = 2.0,
        expected_save_root: Path | None = None,
        expected_persistent_data_path: Path | None = None,
        forbidden_personal_root: Path | None = None,
        expected_product_name: str | None = None,
        forbidden_instance_data_roots: tuple[Path, ...] = (),
    ) -> None:
        if heartbeat_timeout_seconds <= 0:
            raise ValueError("heartbeat_timeout_seconds must be positive")
        self.token = token or secrets.token_urlsafe(32)
        self.game_profile = game_profile
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self.expected_save_root = (
            _canonical_path(expected_save_root) if expected_save_root is not None else None
        )
        self.expected_persistent_data_path = (
            _canonical_path(expected_persistent_data_path or Path(os.environ["HK_AGENT_INSTANCE_DATA_ROOT"]))
            if expected_persistent_data_path is not None or os.environ.get("HK_AGENT_INSTANCE_DATA_ROOT") else None
        )
        self.forbidden_personal_root = _canonical_path(forbidden_personal_root or Path(os.environ["HK_AGENT_PERSONAL_ROOT"])) if forbidden_personal_root is not None or os.environ.get("HK_AGENT_PERSONAL_ROOT") else None
        self.expected_product_name = expected_product_name or os.environ.get("HK_AGENT_PRODUCT_NAME")
        raw_forbidden = os.environ.get("HK_AGENT_OTHER_INSTANCE_DATA_ROOTS", "")
        self.forbidden_instance_data_roots = tuple(_canonical_path(path) for path in forbidden_instance_data_roots) or tuple(_canonical_path(Path(path)) for path in raw_forbidden.split(os.pathsep) if path)
        self.forbidden_personal_root = (
            _canonical_path(forbidden_personal_root) if forbidden_personal_root is not None else None
        )
        self.run_id = str(uuid.uuid4())
        self.port: int | None = None
        self.accepted: list[AcceptedBridge] = []
        self._server: asyncio.Server | None = None
        self._accepted_event = asyncio.Event()
        self._heartbeat_event = asyncio.Event()
        self._connections: set[asyncio.StreamWriter] = set()

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._server = await asyncio.start_server(self._handle_client, host, port)
        socket = self._server.sockets[0]
        self.port = int(socket.getsockname()[1])

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()
        connections = tuple(self._connections)
        for writer in connections:
            writer.close()
        for writer in connections:
            with contextlib.suppress(Exception):
                # A peer may never finish its TCP half-close (notably with the
                # Windows proactor).  The listener is already closed and no
                # command remains valid, so teardown must stay bounded.
                await asyncio.wait_for(writer.wait_closed(), timeout=0.5)
        # On Python 3.12 wait_closed() can wait for active client handlers, so
        # close their transports before awaiting the listener shutdown.
        await asyncio.wait_for(self._server.wait_closed(), timeout=0.5)
        self._server = None

    async def wait_for_bridge(self, timeout: float) -> AcceptedBridge:
        await asyncio.wait_for(self._accepted_event.wait(), timeout=timeout)
        return self.accepted[-1]

    async def wait_for_new_bridge(
        self, *, after_session_id: str | None, timeout: float
    ) -> AcceptedBridge:
        """Wait for a live bridge whose session differs from ``after_session_id``."""

        async def wait() -> AcceptedBridge:
            while True:
                if self.accepted:
                    candidate = self.accepted[-1]
                    if (
                        candidate.session_id != after_session_id
                        and not candidate.writer.is_closing()
                    ):
                        return candidate
                await asyncio.sleep(0.01)

        return await asyncio.wait_for(wait(), timeout=timeout)

    async def wait_for_heartbeat(self, timeout: float) -> AcceptedBridge:
        await asyncio.wait_for(self._heartbeat_event.wait(), timeout=timeout)
        return self.accepted[-1]

    async def send_command(
        self,
        message_type: str,
        payload: dict[str, object],
        *,
        timeout: float = 2.0,
        bridge: AcceptedBridge | None = None,
    ) -> Envelope:
        target = bridge or (self.accepted[-1] if self.accepted else None)
        if target is None or target.writer.is_closing():
            raise ConnectionError("No connected bridge")

        trace_id = str(uuid.uuid4())
        command = Envelope(
            protocol_version=PROTOCOL_VERSION,
            message_type=message_type,
            run_id=self.run_id,
            instance_id=target.instance_id,
            session_id=target.session_id,
            seq=target.outbound_seq,
            trace_id=trace_id,
            sent_monotonic_ns=time.monotonic_ns(),
            payload=payload,
        )
        target.outbound_seq += 1
        response = asyncio.get_running_loop().create_future()
        target.pending[trace_id] = response
        try:
            async with target.write_lock:
                await write_message(target.writer, command)
            return await asyncio.wait_for(response, timeout)
        finally:
            target.pending.pop(trace_id, None)

    async def begin_episode(
        self,
        episode_id: str,
        *,
        timeout: float = 2.0,
        bridge: AcceptedBridge | None = None,
    ) -> None:
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        response = await self.send_command(
            "BEGIN_EPISODE",
            {"episode_id": episode_id},
            timeout=timeout,
            bridge=bridge,
        )
        if response.message_type == "ERROR":
            raise ProtocolError(
                f"BEGIN_EPISODE failed: {response.payload.get('error_code', 'unknown')}"
            )
        if response.message_type != "ACK" or response.payload.get("episode_id") != episode_id:
            raise ProtocolError("Invalid BEGIN_EPISODE acknowledgement")

    async def capture_once(
        self,
        *,
        timeout: float = 5.0,
        bridge: AcceptedBridge | None = None,
        episode_id: str | None = None,
        encoding: str = "png",
    ) -> CapturedFrame:
        if encoding not in {"png", "raw_rgb8"}:
            raise ValueError("encoding must be png or raw_rgb8")
        effective_encoding = encoding
        if effective_encoding == "raw_rgb8" and (
            bridge is None or "frame_raw_rgb8" not in bridge.capabilities
        ):
            effective_encoding = "png"
        payload: dict[str, object] = {}
        if episode_id is not None:
            payload["episode_id"] = episode_id
        if effective_encoding == "raw_rgb8":
            payload["encoding"] = effective_encoding
        response = await self.send_command(
            "CAPTURE_ONCE", payload, timeout=timeout, bridge=bridge
        )
        if response.message_type == "ERROR":
            raise ProtocolError(f"Capture failed: {response.payload.get('error_code', 'unknown')}")
        if response.message_type != "FRAME":
            raise ProtocolError(f"Expected FRAME, received {response.message_type}")

        payload = response.payload
        frame_id = payload.get("frame_id")
        capture_monotonic_ns = payload.get("capture_monotonic_ns")
        width = payload.get("width")
        height = payload.get("height")
        channels = payload.get("channels")
        for name, value in (
            ("frame_id", frame_id),
            ("capture_monotonic_ns", capture_monotonic_ns),
            ("width", width),
            ("height", height),
            ("channels", channels),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ProtocolError(f"Invalid FRAME field: {name}")
        assert isinstance(frame_id, int)
        assert isinstance(capture_monotonic_ns, int)
        assert isinstance(width, int)
        assert isinstance(height, int)
        assert isinstance(channels, int)
        if width == 0 or height == 0:
            raise ProtocolError("FRAME dimensions must be positive")
        pixel_format = payload.get("pixel_format")
        frame_encoding = payload.get("encoding")
        if channels != 3 or pixel_format != "RGB8":
            raise ProtocolError("FRAME must contain RGB8 pixels")
        if not isinstance(frame_encoding, str) or frame_encoding not in {"png", "raw_rgb8"}:
            raise ProtocolError("FRAME encoding must be png or raw_rgb8")
        scene_transition = payload.get("scene_transition")
        if not isinstance(scene_transition, bool):
            raise ProtocolError("Invalid FRAME field: scene_transition")
        player_dead = payload.get("player_dead", False)
        if not isinstance(player_dead, bool):
            raise ProtocolError("Invalid FRAME field: player_dead")
        death_persisted = payload.get("death_persisted", player_dead)
        if not isinstance(death_persisted, bool):
            raise ProtocolError("Invalid FRAME field: death_persisted")
        if player_dead and not death_persisted:
            raise ProtocolError("player_dead requires the post-save persistence barrier")
        shade_pending = payload.get("shade_pending", False)
        shade_recovered = payload.get("shade_recovered", False)
        run_failed = payload.get("run_failed", False)
        for name, value in (
            ("shade_pending", shade_pending),
            ("shade_recovered", shade_recovered),
            ("run_failed", run_failed),
        ):
            if not isinstance(value, bool):
                raise ProtocolError(f"Invalid FRAME field: {name}")
        if run_failed and not player_dead:
            raise ProtocolError("run_failed requires player_dead")
        enemy_kills_since_last_frame = payload.get(
            "enemy_kills_since_last_frame", 0
        )
        hp_lost_since_last_frame = payload.get("hp_lost_since_last_frame", 0)
        geo_gained_since_last_frame = payload.get("geo_gained_since_last_frame", 0)
        geo_spent_since_last_frame = payload.get("geo_spent_since_last_frame", 0)
        for name, value in (
            ("enemy_kills_since_last_frame", enemy_kills_since_last_frame),
            ("hp_lost_since_last_frame", hp_lost_since_last_frame),
            ("geo_gained_since_last_frame", geo_gained_since_last_frame),
            ("geo_spent_since_last_frame", geo_spent_since_last_frame),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProtocolError(f"Invalid FRAME field: {name}")
        assert isinstance(shade_pending, bool)
        assert isinstance(shade_recovered, bool)
        assert isinstance(run_failed, bool)
        assert isinstance(enemy_kills_since_last_frame, int)
        assert isinstance(hp_lost_since_last_frame, int)
        assert isinstance(geo_gained_since_last_frame, int)
        assert isinstance(geo_spent_since_last_frame, int)
        assert isinstance(pixel_format, str)
        assert isinstance(encoding, str)

        encoded = payload.get("data_base64")
        if not isinstance(encoded, str):
            raise ProtocolError("FRAME data_base64 must be a string")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ProtocolError("FRAME data_base64 is invalid") from error
        payload_bytes = payload.get("payload_bytes")
        if frame_encoding == "png":
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ProtocolError("FRAME payload is not a PNG")
            if payload_bytes is not None and (
                not isinstance(payload_bytes, int)
                or isinstance(payload_bytes, bool)
                or payload_bytes != len(data)
            ):
                raise ProtocolError("FRAME payload_bytes does not match PNG data")
        else:
            if width > 600 or height > 338:
                raise ProtocolError("raw RGB8 FRAME dimensions exceed 600x338")
            expected_bytes = width * height * 3
            if (
                not isinstance(payload_bytes, int)
                or isinstance(payload_bytes, bool)
                or payload_bytes != expected_bytes
                or len(data) != expected_bytes
            ):
                raise ProtocolError("raw RGB8 FRAME payload size is invalid")

        duplicate_of = payload.get("duplicate_of")
        last_action = payload.get("last_applied_action_id")
        frame_episode_id = payload.get("episode_id")
        if frame_episode_id is not None and not isinstance(frame_episode_id, str):
            raise ProtocolError("Invalid FRAME field: episode_id")
        if episode_id is not None and frame_episode_id != episode_id:
            raise ProtocolError("FRAME episode_id does not match the requested episode")
        for name, value in (
            ("duplicate_of", duplicate_of),
            ("last_applied_action_id", last_action),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ProtocolError(f"Invalid FRAME field: {name}")
        assert duplicate_of is None or isinstance(duplicate_of, int)
        assert last_action is None or isinstance(last_action, int)

        return CapturedFrame(
            episode_id=frame_episode_id,
            frame_id=frame_id,
            capture_monotonic_ns=capture_monotonic_ns,
            width=width,
            height=height,
            channels=channels,
            pixel_format=pixel_format,
            encoding=frame_encoding,
            data=data,
            scene_transition=scene_transition,
            duplicate_of=duplicate_of,
            last_applied_action_id=last_action,
            player_dead=player_dead,
            death_persisted=death_persisted,
            shade_pending=shade_pending,
            shade_recovered=shade_recovered,
            run_failed=run_failed,
            enemy_kills_since_last_frame=enemy_kills_since_last_frame,
            hp_lost_since_last_frame=hp_lost_since_last_frame,
            geo_gained_since_last_frame=geo_gained_since_last_frame,
            geo_spent_since_last_frame=geo_spent_since_last_frame,
            bridge_timing=parse_bridge_timing(payload),
            action_availability=parse_action_availability(payload.get("action_availability")),
        )

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        machine = SessionMachine()
        accepted: AcceptedBridge | None = None
        self._connections.add(writer)
        machine.transition(SessionState.CONNECTING)
        machine.transition(SessionState.HANDSHAKING)
        try:
            hello = await asyncio.wait_for(read_message(reader), timeout=5.0)
            machine.accept_sequence(hello.seq)
            if hello.message_type != "HELLO":
                raise ProtocolError("First message must be HELLO")
            if hello.run_id != self.run_id:
                raise ProtocolError("Run identifier mismatch")
            if hello.payload.get("token") != self.token:
                raise ProtocolError("Authentication failed")
            if hello.payload.get("game_profile") != self.game_profile:
                raise ProtocolError("Game profile mismatch")
            game_version = hello.payload.get("game_version")
            if not isinstance(game_version, str) or not game_version:
                raise ProtocolError("HELLO requires game_version")
            effective_save_root = hello.payload.get("effective_save_root")
            if effective_save_root is not None and not isinstance(effective_save_root, str):
                raise ProtocolError("HELLO effective_save_root must be a string")
            if self.expected_save_root is not None:
                if not effective_save_root:
                    raise ProtocolError("HELLO requires effective_save_root")
                if _canonical_path(Path(effective_save_root)) != self.expected_save_root:
                    raise ProtocolError("Effective save root mismatch")
            if self.expected_persistent_data_path is not None:
                persistent = hello.payload.get("persistent_data_path")
                if not isinstance(persistent, str):
                    raise ProtocolError("HELLO requires persistent_data_path")
                canonical_persistent = _canonical_path(Path(persistent))
                if canonical_persistent != self.expected_persistent_data_path:
                    raise ProtocolError("Persistent data path mismatch")
                if self.forbidden_personal_root is not None and canonical_persistent == self.forbidden_personal_root:
                    raise ProtocolError("Persistent data path resolves to personal saves")
                if canonical_persistent in self.forbidden_instance_data_roots:
                    raise ProtocolError("Persistent data path resolves to another instance")
            if self.expected_product_name is not None and hello.payload.get("product_name") != self.expected_product_name:
                raise ProtocolError("Product name mismatch")
            raw_capabilities = hello.payload.get("capabilities", [])
            if not isinstance(raw_capabilities, list) or any(
                not isinstance(item, str) for item in raw_capabilities
            ):
                raise ProtocolError("HELLO capabilities must be an array of strings")

            accepted = AcceptedBridge(
                instance_id=hello.instance_id,
                session_id=hello.session_id,
                game_profile=self.game_profile,
                game_version=game_version,
                effective_save_root=effective_save_root,
                writer=writer,
                persistent_data_path=(
                    hello.payload.get("persistent_data_path")
                    if isinstance(hello.payload.get("persistent_data_path"), str)
                    else None
                ),
                product_name=(
                    hello.payload.get("product_name")
                    if isinstance(hello.payload.get("product_name"), str)
                    else None
                ),
                capabilities=frozenset(raw_capabilities),
                machine=machine,
            )
            machine.transition(SessionState.READY)
            self.accepted.append(accepted)

            welcome = Envelope(
                protocol_version=PROTOCOL_VERSION,
                message_type="WELCOME",
                run_id=self.run_id,
                instance_id=hello.instance_id,
                session_id=hello.session_id,
                seq=0,
                trace_id=hello.trace_id,
                sent_monotonic_ns=time.monotonic_ns(),
                payload={
                    "heartbeat_interval_ms": 500,
                    "heartbeat_timeout_ms": int(self.heartbeat_timeout_seconds * 1000),
                },
            )
            await write_message(writer, welcome)
            self._accepted_event.set()

            while True:
                message = await asyncio.wait_for(
                    read_message(reader), timeout=self.heartbeat_timeout_seconds
                )
                machine.accept_sequence(message.seq)
                if message.run_id != self.run_id:
                    raise ProtocolError("Run identifier mismatch")
                if message.instance_id != accepted.instance_id:
                    raise ProtocolError("Instance identifier mismatch")
                if message.session_id != accepted.session_id:
                    raise ProtocolError("Session identifier mismatch")
                if message.message_type == "HEARTBEAT":
                    accepted.heartbeats_received += 1
                    accepted.last_heartbeat_ns = time.monotonic_ns()
                    self._heartbeat_event.set()
                    continue
                if message.message_type in {"ACK", "ERROR", "FRAME"}:
                    pending = accepted.pending.get(message.trace_id)
                    if pending is None:
                        raise ProtocolError("Response has no matching command")
                    if not pending.done():
                        pending.set_result(message)
                    continue
                raise ProtocolError(f"Unsupported bridge message: {message.message_type}")
        except (ProtocolError, SessionError, asyncio.IncompleteReadError, TimeoutError):
            with contextlib.suppress(Exception):
                machine.transition(SessionState.FAULTED)
        finally:
            for pending in tuple(accepted.pending.values()) if accepted is not None else ():
                if not pending.done():
                    pending.set_exception(ConnectionError("Bridge disconnected"))
            self._connections.discard(writer)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()


def _canonical_path(path: Path) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))
