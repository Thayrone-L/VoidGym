from __future__ import annotations

import asyncio
import concurrent.futures
import struct
import threading
import time
import uuid
import zlib
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from hk_agent.gym_env import AppliedAction, BackendStep, PixelFrame, ResetResult
from hk_agent.menu_bootstrap import (
    MenuBootstrapPlan,
    ReadinessVerifier,
    build_readiness_verifier,
    run_menu_bootstrap,
)
from hk_agent.protocol import ProtocolError
from hk_agent.supervisor import AcceptedBridge, CapturedFrame, SupervisorServer

_T = TypeVar("_T")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True, slots=True)
class SupervisorEndpoint:
    host: str
    port: int
    run_id: str
    token: str
    game_profile: str


class ProcessLaunchCancelled(RuntimeError):
    """Raised when a timed-out reset attempts a late process launch."""


class ProcessLaunchCancellation:
    """Non-blocking cooperative cancellation for a reset generation."""

    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._handler_lock = threading.Lock()
        self._cleanup: Callable[[], None] | None = None

    def cancel(self) -> None:
        self._cancelled.set()
        with self._handler_lock:
            cleanup = self._cleanup
        if cleanup is not None:
            self._start_cleanup(cleanup)

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise ProcessLaunchCancelled("process launch generation was cancelled")

    def register_cleanup(self, cleanup: Callable[[], None]) -> None:
        """Register non-blocking teardown for cancellation racing a launch."""

        with self._handler_lock:
            if self._cleanup is not None:
                raise RuntimeError("process launch cleanup is already registered")
            self._cleanup = cleanup
            cancelled = self.is_cancelled
        if cancelled:
            self._start_cleanup(cleanup)

    @staticmethod
    def _start_cleanup(cleanup: Callable[[], None]) -> None:
        def run() -> None:
            try:
                cleanup()
            except Exception:  # noqa: BLE001 - daemon cannot surface teardown errors
                # The owner preserves its transaction state on cleanup failure;
                # reset has already failed closed and a later close can retry.
                return

        threading.Thread(
            target=run,
            name="hk-cancelled-launch-cleanup",
            daemon=True,
        ).start()


ResetProcess = Callable[[SupervisorEndpoint, str, ProcessLaunchCancellation], None]
MenuBootstrapPlanProvider = Callable[[], MenuBootstrapPlan]


class SupervisorEnvironmentBackend:
    """Thread-safe synchronous adapter around one asynchronous supervisor.

    The adapter never navigates menus or encodes a route.  A caller that wants
    process-level resets must provide ``reset_process``; the callback receives
    connection metadata, the freshly generated episode identifier, and a
    cooperative cancellation guard that must cover the final process launch.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        game_profile: str = "development",
        reset_process: ResetProcess | None = None,
        shutdown_process: Callable[[], None] | None = None,
        bridge_timeout_seconds: float = 15.0,
        command_timeout_seconds: float = 8.0,
        process_timeout_seconds: float = 30.0,
        settle_seconds: float = 0.0,
        menu_bootstrap_plan: MenuBootstrapPlan | None = None,
        menu_bootstrap_plan_provider: MenuBootstrapPlanProvider | None = None,
        menu_bootstrap_artifact_root: Path | None = None,
        menu_readiness_verifier: ReadinessVerifier | None = None,
        expected_save_root: Path | None = None,
    ) -> None:
        if (
            bridge_timeout_seconds <= 0
            or command_timeout_seconds <= 0
            or process_timeout_seconds <= 0
        ):
            raise ValueError("timeouts must be positive")
        if settle_seconds < 0:
            raise ValueError("settle_seconds must be non-negative")
        if menu_bootstrap_plan is not None and menu_bootstrap_plan_provider is not None:
            raise ValueError("provide either a menu bootstrap plan or a provider")
        has_bootstrap = menu_bootstrap_plan is not None or menu_bootstrap_plan_provider is not None
        if has_bootstrap != (menu_bootstrap_artifact_root is not None):
            raise ValueError("menu bootstrap plan and artifact root must be provided together")
        if expected_save_root is None and reset_process is not None:
            owner = getattr(reset_process, "__self__", None)
            candidate = getattr(owner, "expected_save_root", None)
            if isinstance(candidate, Path):
                expected_save_root = candidate
        self._supervisor = SupervisorServer(
            token=token,
            game_profile=game_profile,
            expected_save_root=expected_save_root,
        )
        self._reset_process = reset_process
        self._shutdown_process = shutdown_process
        self._bridge_timeout = bridge_timeout_seconds
        self._command_timeout = command_timeout_seconds
        self._process_timeout = process_timeout_seconds
        self._settle_seconds = settle_seconds
        self._menu_bootstrap_plan = menu_bootstrap_plan
        self._menu_bootstrap_plan_provider = menu_bootstrap_plan_provider
        self._menu_bootstrap_artifact_root = menu_bootstrap_artifact_root
        if menu_readiness_verifier is not None and menu_bootstrap_plan is None:
            raise ValueError("menu readiness verifier requires a bootstrap plan")
        self._menu_readiness_verifier: ReadinessVerifier | None = menu_readiness_verifier
        if menu_bootstrap_plan is not None and self._menu_readiness_verifier is None:
            self._menu_readiness_verifier = build_readiness_verifier(menu_bootstrap_plan)
        self._operation_lock = threading.RLock()
        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="hk-supervisor-event-loop",
            daemon=True,
        )
        self._thread.start()
        if not self._loop_ready.wait(timeout=2.0):
            raise RuntimeError("supervisor event loop did not start")
        self._bridge: AcceptedBridge | None = None
        self._episode_id: str | None = None
        self._closed = False
        try:
            self._submit(self._supervisor.start(), timeout=self._command_timeout)
        except Exception:
            self._stop_loop()
            raise

    @property
    def run_id(self) -> str:
        return self._supervisor.run_id

    @property
    def instance_id(self) -> str:
        with self._operation_lock:
            return self._bridge.instance_id if self._bridge is not None else "unconnected"

    @property
    def endpoint(self) -> SupervisorEndpoint:
        port = self._supervisor.port
        if port is None:
            raise RuntimeError("supervisor is not listening")
        return SupervisorEndpoint(
            host="127.0.0.1",
            port=port,
            run_id=self._supervisor.run_id,
            token=self._supervisor.token,
            game_profile=self._supervisor.game_profile,
        )

    def reset(
        self, *, seed: int | None, options: Mapping[str, Any] | None
    ) -> ResetResult:
        del seed, options
        with self._operation_lock:
            self._ensure_open()
            previous_session = self._bridge.session_id if self._bridge is not None else None
            had_bridge = self._bridge is not None
            if self._bridge is not None:
                self._release_locked(best_effort=True)
            # A failed reset must never leave the previous episode usable.
            self._bridge = None
            self._episode_id = None

            episode_id = str(uuid.uuid4())
            if self._reset_process is not None:
                try:
                    self._invoke_process_callback(
                        self._reset_process,
                        self.endpoint,
                        episode_id,
                        label="reset_process",
                        cancellation=ProcessLaunchCancellation(),
                    )
                except TimeoutError:
                    # The timed-out daemon thread cannot be forcibly stopped.
                    # Permanently close this backend so a late process cannot
                    # connect and be mistaken for a later reset generation.
                    self._fault_close_listener()
                    raise
                bridge = self._submit(
                    self._supervisor.wait_for_new_bridge(
                        after_session_id=previous_session,
                        timeout=self._bridge_timeout,
                    ),
                    timeout=self._bridge_timeout + 0.5,
                )
            elif not had_bridge:
                bridge = self._submit(
                    self._supervisor.wait_for_bridge(self._bridge_timeout),
                    timeout=self._bridge_timeout + 0.5,
                )
            else:
                raise RuntimeError("a reset_process callback is required for another episode")

            try:
                menu_plan = (
                    self._menu_bootstrap_plan_provider()
                    if self._menu_bootstrap_plan_provider is not None
                    else self._menu_bootstrap_plan
                )
                if menu_plan is not None:
                    assert self._menu_bootstrap_artifact_root is not None
                    readiness_verifier = (
                        self._menu_readiness_verifier
                        if self._menu_bootstrap_plan_provider is None
                        else build_readiness_verifier(menu_plan)
                    )
                    self._submit(
                        run_menu_bootstrap(
                            menu_plan,
                            supervisor=self._supervisor,
                            bridge=bridge,
                            episode_id=episode_id,
                            artifact_root=self._menu_bootstrap_artifact_root,
                            command_timeout_seconds=self._command_timeout,
                            readiness_verifier=readiness_verifier,
                        ),
                        timeout=menu_plan.timeout_seconds + 0.5,
                    )
                self._submit(
                    self._supervisor.begin_episode(
                        episode_id,
                        timeout=self._command_timeout,
                        bridge=bridge,
                    ),
                    timeout=self._command_timeout + 0.5,
                )
                frame = self._submit(
                    self._supervisor.capture_once(
                        timeout=self._command_timeout,
                        bridge=bridge,
                        episode_id=episode_id,
                        encoding="raw_rgb8",
                    ),
                    timeout=self._command_timeout + 0.5,
                )
                if frame.last_applied_action_id is not None:
                    raise ProtocolError("initial FRAME retained a previous action")
                captured = _to_pixel_frame(frame, episode_id)
            except Exception as error:
                self._bridge = bridge
                self._release_locked(best_effort=True)
                self._bridge = None
                if self._shutdown_process is not None:
                    try:
                        self._invoke_process_callback(
                            self._shutdown_process,
                            label="failed_reset_shutdown_process",
                        )
                    except Exception as shutdown_error:  # noqa: BLE001
                        error.add_note(
                            "failed reset teardown also failed: "
                            f"{type(shutdown_error).__name__}: {shutdown_error}"
                        )
                raise
            self._bridge = bridge
            self._episode_id = episode_id
            return ResetResult(episode_id=episode_id, frame=captured)

    def step(self, *, action_id: int, buttons: tuple[str, ...], lease_ms: int) -> BackendStep:
        if action_id < 0 or not 1 <= lease_ms <= 2_000:
            raise ValueError("invalid action_id or lease_ms")
        with self._operation_lock:
            self._ensure_open()
            bridge, episode_id = self._require_episode()
            started = time.monotonic_ns()
            timings: dict[str, float] = {}
            response = None
            try:
                apply_started = time.monotonic_ns()
                response = self._submit(
                    self._supervisor.send_command(
                        "APPLY_ACTION",
                        {
                            "episode_id": episode_id,
                            "action_id": action_id,
                            "buttons": list(buttons),
                            "lease_ms": lease_ms,
                        },
                        timeout=self._command_timeout,
                        bridge=bridge,
                    ),
                    timeout=self._command_timeout + 0.5,
                )
                if response.message_type == "ERROR":
                    raise ProtocolError(
                        f"APPLY_ACTION failed: {response.payload.get('error_code', 'unknown')}"
                    )
                if response.message_type != "ACK":
                    raise ProtocolError(f"Expected ACK, received {response.message_type}")
                self._validate_action_ack(
                    response.payload, episode_id, action_id, buttons, lease_ms
                )
                timings["apply_rtt"] = (time.monotonic_ns() - apply_started) / 1_000_000
                sleep_started = time.monotonic_ns()
                time.sleep((lease_ms / 1000.0) + self._settle_seconds)
                timings["lease_sleep"] = (time.monotonic_ns() - sleep_started) / 1_000_000
            finally:
                release_started = time.monotonic_ns()
                self._release_locked(best_effort=response is None)
                timings["release_rtt"] = (time.monotonic_ns() - release_started) / 1_000_000

            capture_started = time.monotonic_ns()
            frame = self._capture_locked(expected_action_id=action_id)
            timings["capture_rtt"] = (time.monotonic_ns() - capture_started) / 1_000_000
            timings["decode"] = frame.decode_ms
            for name, value in frame.bridge_timing:
                timings[f"bridge_{name}"] = value
            timings["backend"] = (time.monotonic_ns() - started) / 1_000_000
            return BackendStep(
                frame=frame,
                applied_action=AppliedAction(action_id, buttons, buttons, lease_ms),
                terminated=frame.player_dead,
                termination_reason="player_death" if frame.player_dead else None,
                timing_ms=timings,
            )

    def release_all(self) -> None:
        with self._operation_lock:
            self._ensure_open()
            self._release_locked(best_effort=False)

    def close(self) -> None:
        with self._operation_lock:
            if self._closed:
                return
            self._release_locked(best_effort=True)
            try:
                if self._shutdown_process is not None:
                    self._invoke_process_callback(
                        self._shutdown_process,
                        label="shutdown_process",
                    )
            finally:
                try:
                    self._submit(self._supervisor.close(), timeout=self._command_timeout + 0.5)
                finally:
                    self._closed = True
                    self._stop_loop()

    def _capture_locked(self, *, expected_action_id: int | None) -> PixelFrame:
        bridge, episode_id = self._require_episode()
        frame = self._submit(
            self._supervisor.capture_once(
                timeout=self._command_timeout,
                bridge=bridge,
                episode_id=episode_id,
                encoding="raw_rgb8",
            ),
            timeout=self._command_timeout + 0.5,
        )
        if frame.last_applied_action_id != expected_action_id:
            raise ProtocolError("FRAME is not correlated with the requested action")
        return _to_pixel_frame(frame, episode_id)

    def _release_locked(self, *, best_effort: bool) -> None:
        bridge = self._bridge
        if bridge is None or bridge.writer.is_closing():
            return
        try:
            response = self._submit(
                self._supervisor.send_command(
                    "RELEASE_ALL", {}, timeout=self._command_timeout, bridge=bridge
                ),
                timeout=self._command_timeout + 0.5,
            )
            if response.message_type != "ACK":
                raise ProtocolError("RELEASE_ALL was not acknowledged")
        except Exception:
            if not best_effort:
                raise

    @staticmethod
    def _validate_action_ack(
        payload: dict[str, Any],
        episode_id: str,
        action_id: int,
        buttons: tuple[str, ...],
        lease_ms: int,
    ) -> None:
        if payload.get("episode_id") != episode_id or payload.get("action_id") != action_id:
            raise ProtocolError("APPLY_ACTION acknowledgement identity mismatch")
        if payload.get("buttons") != list(buttons) or payload.get("lease_ms") != lease_ms:
            raise ProtocolError("APPLY_ACTION acknowledgement differs from the request")

    def _require_episode(self) -> tuple[AcceptedBridge, str]:
        if self._bridge is None or self._episode_id is None:
            raise RuntimeError("reset() is required before this operation")
        if self._bridge.writer.is_closing():
            raise ConnectionError("bridge is disconnected")
        return self._bridge, self._episode_id

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("backend is closed")

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        self._loop.run_forever()
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self._loop.close()

    def _submit(self, coroutine: Coroutine[Any, Any, _T], *, timeout: float) -> _T:
        if threading.current_thread() is self._thread:
            coroutine.close()
            raise RuntimeError("blocking backend operation called from its asyncio thread")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as error:
            future.cancel()
            raise TimeoutError("backend operation timed out") from error

    def _invoke_process_callback(
        self,
        callback: Callable[..., None],
        *args: object,
        label: str,
        cancellation: ProcessLaunchCancellation | None = None,
    ) -> None:
        """Run trusted process plumbing without allowing it to block this adapter."""

        future: concurrent.futures.Future[None] = concurrent.futures.Future()

        def invoke() -> None:
            try:
                if cancellation is None:
                    callback(*args)
                else:
                    callback(*args, cancellation)
            except BaseException as error:  # noqa: BLE001 - forwarded to the caller
                future.set_exception(error)
            else:
                future.set_result(None)

        threading.Thread(
            target=invoke,
            name=f"hk-{label}",
            daemon=True,
        ).start()
        try:
            future.result(timeout=self._process_timeout)
        except concurrent.futures.TimeoutError as error:
            if cancellation is not None:
                cancellation.cancel()
            raise TimeoutError(f"{label} timed out") from error

    def _fault_close_listener(self) -> None:
        try:
            self._submit(self._supervisor.close(), timeout=self._command_timeout + 0.5)
        finally:
            self._closed = True
            self._stop_loop()

    def _stop_loop(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError("supervisor event-loop thread did not stop")


def _to_pixel_frame(frame: CapturedFrame, episode_id: str) -> PixelFrame:
    if frame.episode_id != episode_id:
        raise ProtocolError("FRAME belongs to another episode")
    decode_started = time.monotonic_ns()
    if frame.encoding == "raw_rgb8":
        expected_bytes = frame.width * frame.height * 3
        if (
            frame.channels != 3
            or frame.width > 600
            or frame.height > 338
            or len(frame.data) != expected_bytes
        ):
            raise ProtocolError("raw RGB8 FRAME dimensions or payload are invalid")
        pixels = np.frombuffer(frame.data, dtype=np.uint8).reshape(
            frame.height, frame.width, 3
        ).copy()
    elif frame.encoding == "png":
        pixels = decode_rgb8_png(
            frame.data, expected_width=frame.width, expected_height=frame.height
        )
    else:
        raise ProtocolError(f"unsupported FRAME encoding: {frame.encoding}")
    if pixels.shape != (frame.height, frame.width, 3):
        raise ProtocolError("decoded PNG dimensions differ from FRAME metadata")
    return PixelFrame(
        episode_id=episode_id,
        frame_id=frame.frame_id,
        capture_monotonic_ns=frame.capture_monotonic_ns,
        pixels=pixels,
        scene_transition=frame.scene_transition,
        duplicate_of=frame.duplicate_of,
        last_applied_action_id=frame.last_applied_action_id,
        player_dead=frame.player_dead,
        death_persisted=frame.death_persisted,
        shade_pending=frame.shade_pending,
        shade_recovered=frame.shade_recovered,
        run_failed=frame.run_failed,
        enemy_kills_since_last_frame=frame.enemy_kills_since_last_frame,
        hp_lost_since_last_frame=frame.hp_lost_since_last_frame,
        geo_gained_since_last_frame=frame.geo_gained_since_last_frame,
        geo_spent_since_last_frame=frame.geo_spent_since_last_frame,
        hp_restored_since_last_frame=frame.hp_restored_since_last_frame,
        shade_hits_since_last_frame=frame.shade_hits_since_last_frame,
        shade_distance=frame.shade_distance,
        scene_name=frame.scene_name,
        decode_ms=(time.monotonic_ns() - decode_started) / 1_000_000,
        bridge_timing=frame.bridge_timing,
        action_availability=frame.action_availability,
    )


def decode_rgb8_png(
    data: bytes,
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
    maximum_pixels: int = 600 * 338,
) -> NDArray[np.uint8]:
    """Decode the bridge's strict, non-interlaced RGB8 PNG contract."""

    if not data.startswith(_PNG_SIGNATURE):
        raise ProtocolError("invalid PNG signature")
    offset = len(_PNG_SIGNATURE)
    width = height = 0
    compressed = bytearray()
    saw_header = saw_end = False
    while offset < len(data):
        if offset + 12 > len(data):
            raise ProtocolError("truncated PNG chunk")
        length = struct.unpack_from(">I", data, offset)[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_end = offset + 12 + length
        if chunk_end > len(data):
            raise ProtocolError("truncated PNG payload")
        payload = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack_from(">I", data, offset + 8 + length)[0]
        if zlib.crc32(chunk_type + payload) & 0xFFFFFFFF != expected_crc:
            raise ProtocolError("PNG chunk CRC mismatch")
        if chunk_type == b"IHDR":
            if saw_header or length != 13:
                raise ProtocolError("invalid PNG header")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if (
                width <= 0
                or height <= 0
                or width * height > maximum_pixels
                or depth != 8
                or color != 2
                or compression != 0
                or filtering != 0
                or interlace != 0
            ):
                raise ProtocolError("PNG must be non-interlaced RGB8")
            if expected_width is not None and width != expected_width:
                raise ProtocolError("PNG width differs from FRAME metadata")
            if expected_height is not None and height != expected_height:
                raise ProtocolError("PNG height differs from FRAME metadata")
            saw_header = True
        elif chunk_type == b"IDAT":
            if not saw_header:
                raise ProtocolError("PNG IDAT precedes IHDR")
            compressed.extend(payload)
        elif chunk_type == b"IEND":
            saw_end = True
            break
        offset = chunk_end
    if not saw_header or not saw_end or not compressed:
        raise ProtocolError("incomplete PNG")
    expected_size = height * ((width * 3) + 1)
    try:
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(bytes(compressed), expected_size + 1)
    except zlib.error as error:
        raise ProtocolError("invalid PNG compressed data") from error
    if (
        len(raw) != expected_size
        or decompressor.unconsumed_tail
        or decompressor.unused_data
        or not decompressor.eof
    ):
        raise ProtocolError("PNG decompressed data does not match its declared dimensions")
    stride = width * 3
    output = bytearray(height * stride)
    previous = bytearray(stride)
    for row_index in range(height):
        start = row_index * (stride + 1)
        filter_type = raw[start]
        current = bytearray(raw[start + 1 : start + 1 + stride])
        for index in range(stride):
            left = current[index - 3] if index >= 3 else 0
            above = previous[index]
            upper_left = previous[index - 3] if index >= 3 else 0
            if filter_type == 1:
                current[index] = (current[index] + left) & 0xFF
            elif filter_type == 2:
                current[index] = (current[index] + above) & 0xFF
            elif filter_type == 3:
                current[index] = (current[index] + ((left + above) // 2)) & 0xFF
            elif filter_type == 4:
                current[index] = (
                    current[index] + _paeth_predictor(left, above, upper_left)
                ) & 0xFF
            elif filter_type != 0:
                raise ProtocolError("unsupported PNG filter")
        output[row_index * stride : (row_index + 1) * stride] = current
        previous = current
    return np.frombuffer(output, dtype=np.uint8).reshape((height, width, 3)).copy()


def _paeth_predictor(left: int, above: int, upper_left: int) -> int:
    candidate = left + above - upper_left
    left_distance = abs(candidate - left)
    above_distance = abs(candidate - above)
    diagonal_distance = abs(candidate - upper_left)
    if left_distance <= above_distance and left_distance <= diagonal_distance:
        return left
    if above_distance <= diagonal_distance:
        return above
    return upper_left
