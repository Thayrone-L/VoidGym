from __future__ import annotations

import asyncio
import base64
import json
import struct
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from hk_agent.menu_bootstrap import (
    MenuBootstrapPlan,
    MenuBootstrapStep,
    ReadinessVerdict,
)
from hk_agent.protocol import PROTOCOL_VERSION, Envelope, ProtocolError, read_message, write_message
from hk_agent.supervisor_backend import (
    ProcessLaunchCancellation,
    SupervisorEndpoint,
    SupervisorEnvironmentBackend,
    decode_rgb8_png,
)


def _png_rgb8(pixels: np.ndarray) -> bytes:
    height, width, channels = pixels.shape
    assert channels == 3 and pixels.dtype == np.uint8

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    rows = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(
        b"IDAT", zlib.compress(rows)
    ) + chunk(b"IEND", b"")


@dataclass
class FakeBridge:
    endpoint: SupervisorEndpoint
    expected_episode: str
    commands: list[str] = field(default_factory=list)
    failure: BaseException | None = None
    player_dead_after_action: bool = False
    shade_pending_after_action: bool = False
    run_failed_after_action: bool = False
    capture_encodings: list[str | None] = field(default_factory=list)
    allow_episode_changes: bool = False

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        return thread

    def _run(self) -> None:
        try:
            asyncio.run(self._scenario())
        except BaseException as error:  # noqa: BLE001 - surfaced to the test thread
            self.failure = error

    async def _scenario(self) -> None:
        reader, writer = await asyncio.open_connection(self.endpoint.host, self.endpoint.port)
        session_id = str(uuid.uuid4())
        outbound_seq = 0
        hello = Envelope(
            protocol_version=PROTOCOL_VERSION,
            message_type="HELLO",
            run_id=self.endpoint.run_id,
            instance_id="hk-fake",
            session_id=session_id,
            seq=outbound_seq,
            trace_id=str(uuid.uuid4()),
            sent_monotonic_ns=time.monotonic_ns(),
            payload={
                "token": self.endpoint.token,
                "game_profile": self.endpoint.game_profile,
                "game_version": "fake",
                "capabilities": ["frame_png_rgb8", "frame_raw_rgb8"],
            },
        )
        await write_message(writer, hello)
        welcome = await read_message(reader)
        assert welcome.message_type == "WELCOME"
        active_episode: str | None = None
        last_action: int | None = None
        frame_id = 0
        pixels = np.array(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            dtype=np.uint8,
        )
        try:
            while True:
                command = await read_message(reader)
                self.commands.append(command.message_type)
                response_type = "ACK"
                if command.message_type == "BEGIN_EPISODE":
                    active_episode = str(command.payload["episode_id"])
                    if not self.allow_episode_changes:
                        assert active_episode == self.expected_episode
                    last_action = None
                    payload: dict[str, object] = {
                        "command_type": "BEGIN_EPISODE",
                        "status": "accepted",
                        "episode_id": active_episode,
                    }
                elif command.message_type == "APPLY_ACTION":
                    assert command.payload["episode_id"] == active_episode
                    last_action = int(command.payload["action_id"])
                    payload = {
                        "command_type": "APPLY_ACTION",
                        "status": "accepted",
                        "episode_id": active_episode or "",
                        "action_id": last_action,
                        "buttons": command.payload["buttons"],
                        "lease_ms": command.payload["lease_ms"],
                    }
                elif command.message_type == "APPLY_MENU_ACTION":
                    payload = {
                        "command_type": "APPLY_MENU_ACTION",
                        "status": "accepted",
                        "buttons": command.payload["buttons"],
                        "lease_ms": command.payload["lease_ms"],
                    }
                elif command.message_type == "RELEASE_ALL":
                    payload = {"command_type": "RELEASE_ALL", "status": "released"}
                elif command.message_type == "CAPTURE_ONCE":
                    requested_episode = command.payload.get("episode_id")
                    assert requested_episode is None or requested_episode == active_episode
                    capture_encoding = command.payload.get("encoding")
                    self.capture_encodings.append(
                        capture_encoding if isinstance(capture_encoding, str) else None
                    )
                    frame_id += 1
                    frame_pixels = (pixels.astype(np.uint16) + frame_id - 1).astype(np.uint8)
                    frame_data = (
                        frame_pixels.tobytes()
                        if capture_encoding == "raw_rgb8"
                        else _png_rgb8(frame_pixels)
                    )
                    response_type = "FRAME"
                    payload = {
                        "episode_id": active_episode or "",
                        "frame_id": frame_id,
                        "capture_monotonic_ns": time.monotonic_ns(),
                        "width": 2,
                        "height": 2,
                        "channels": 3,
                        "pixel_format": "RGB8",
                        "encoding": capture_encoding or "png",
                        "payload_bytes": len(frame_data),
                        "data_base64": base64.b64encode(frame_data).decode("ascii"),
                        "scene_transition": False,
                        "player_dead": self.player_dead_after_action and last_action is not None,
                        "death_persisted": self.player_dead_after_action
                        and last_action is not None,
                        "shade_pending": self.shade_pending_after_action
                        and last_action is not None,
                        "shade_recovered": False,
                        "run_failed": self.run_failed_after_action and last_action is not None,
                        "duplicate_of": None,
                        "last_applied_action_id": last_action,
                    }
                    if capture_encoding != "raw_rgb8":
                        payload["png_bytes"] = len(frame_data)
                else:
                    raise AssertionError(f"unexpected command: {command.message_type}")
                outbound_seq += 1
                await write_message(
                    writer,
                    Envelope(
                        protocol_version=PROTOCOL_VERSION,
                        message_type=response_type,
                        run_id=self.endpoint.run_id,
                        instance_id="hk-fake",
                        session_id=session_id,
                        seq=outbound_seq,
                        trace_id=command.trace_id,
                        sent_monotonic_ns=time.monotonic_ns(),
                        payload=payload,
                    ),
                )
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()


def test_real_backend_action_to_correlated_rgb_frame() -> None:
    fake: FakeBridge | None = None
    fake_thread: threading.Thread | None = None

    def reset_process(
        endpoint: SupervisorEndpoint,
        episode_id: str,
        cancellation: ProcessLaunchCancellation,
    ) -> None:
        del cancellation
        nonlocal fake, fake_thread
        fake = FakeBridge(endpoint, episode_id)
        fake_thread = fake.start()

    backend = SupervisorEnvironmentBackend(
        token="backend-test-token",
        reset_process=reset_process,
        bridge_timeout_seconds=2.0,
        command_timeout_seconds=1.0,
    )
    try:
        reset = backend.reset(seed=7, options=None)
        assert reset.frame.episode_id == reset.episode_id
        assert reset.frame.last_applied_action_id is None
        assert reset.frame.pixels.tolist() == [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ]

        step = backend.step(action_id=41, buttons=("RIGHT", "JUMP"), lease_ms=1)
        assert step.applied_action.action_id == 41
        assert step.frame.episode_id == reset.episode_id
        assert step.frame.last_applied_action_id == 41
        assert step.frame.frame_id > reset.frame.frame_id
        assert step.frame.pixels.dtype == np.uint8
        assert fake is not None
        assert fake.capture_encodings == ["raw_rgb8", "raw_rgb8"]
        assert fake.commands == [
            "BEGIN_EPISODE",
            "CAPTURE_ONCE",
            "APPLY_ACTION",
            "RELEASE_ALL",
            "CAPTURE_ONCE",
        ]
    finally:
        backend.close()
        if fake_thread is not None:
            fake_thread.join(timeout=2.0)
        assert fake is None or fake.failure is None


def test_continuous_playthrough_reset_reuses_existing_game_bridge() -> None:
    fake: FakeBridge | None = None
    fake_thread: threading.Thread | None = None
    launches = 0

    def reset_process(
        endpoint: SupervisorEndpoint,
        episode_id: str,
        cancellation: ProcessLaunchCancellation,
    ) -> None:
        nonlocal fake, fake_thread, launches
        del cancellation
        launches += 1
        if launches == 1:
            fake = FakeBridge(endpoint, episode_id, allow_episode_changes=True)
            fake_thread = fake.start()

    backend = SupervisorEnvironmentBackend(
        token="continuous-playthrough-test",
        reset_process=reset_process,
        reuse_game_on_reset=True,
        bridge_timeout_seconds=2.0,
        command_timeout_seconds=1.0,
    )
    try:
        first = backend.reset(seed=1, options=None)
        second = backend.reset(seed=2, options=None)
        assert first.episode_id != second.episode_id
        assert launches == 1
        assert fake is not None
        assert fake.commands.count("BEGIN_EPISODE") == 2
        assert fake.commands.count("CAPTURE_ONCE") == 2
    finally:
        backend.close()
        if fake_thread is not None:
            fake_thread.join(timeout=2.0)
        assert fake is None or fake.failure is None


def test_checkpoint_reload_uses_validated_menu_plan_on_existing_bridge(
    tmp_path: Path,
) -> None:
    fake: FakeBridge | None = None
    fake_thread: threading.Thread | None = None
    launches = 0
    plan = MenuBootstrapPlan(
        plan_id="checkpoint-reload-test",
        steps=(
            MenuBootstrapStep("CANCEL", 50, 0, "open-pause"),
            MenuBootstrapStep("SUBMIT", 50, 0, "save-and-quit"),
        ),
        initial_wait_seconds=0,
        final_wait_seconds=0,
        timeout_seconds=2,
        minimum_distinct_captures=2,
    )

    def reset_process(
        endpoint: SupervisorEndpoint,
        episode_id: str,
        cancellation: ProcessLaunchCancellation,
    ) -> None:
        del cancellation
        nonlocal fake, fake_thread, launches
        launches += 1
        if launches == 1:
            fake = FakeBridge(endpoint, episode_id, allow_episode_changes=True)
            fake_thread = fake.start()

    backend = SupervisorEnvironmentBackend(
        token="checkpoint-menu-success",
        reset_process=reset_process,
        reuse_game_on_reset=True,
        checkpoint_reload_plan_provider=lambda: plan,
        checkpoint_reload_readiness_verifier=lambda _: ReadinessVerdict(
            True, "synthetic-checkpoint-v1", "gameplay checkpoint", {"matched": 1}
        ),
        menu_bootstrap_artifact_root=tmp_path,
        bridge_timeout_seconds=2.0,
        command_timeout_seconds=1.0,
    )
    try:
        first = backend.reset(seed=1, options=None)
        second = backend.reset(seed=2, options={"reload_checkpoint": True})
        assert first.episode_id != second.episode_id
        assert launches == 1
        assert fake is not None
        assert fake.commands.count("BEGIN_EPISODE") == 2
        assert fake.commands.count("APPLY_MENU_ACTION") == 2
        assert fake.commands.count("CAPTURE_ONCE") == 6
        assert backend.last_reset_recovery == {
            "requested": True,
            "method": "menu",
            "plan_id": "checkpoint-reload-test",
            "status": "READY",
        }
        report_root = tmp_path / "checkpoint-reloads" / second.episode_id / "menu-bootstrap"
        report = json.loads((report_root / "report.json").read_text(encoding="utf-8"))
        assert report["status"] == "READY"
        assert len(report["captures"]) == 4
        assert all(Path(row["path"]).is_file() for row in report["captures"])
    finally:
        backend.close()
        if fake_thread is not None:
            fake_thread.join(timeout=2.0)
        assert fake is None or fake.failure is None


def test_checkpoint_menu_failure_reopens_save_through_continue_plan(
    tmp_path: Path,
) -> None:
    fakes: list[FakeBridge] = []
    threads: list[threading.Thread] = []
    launches = 0
    shutdowns = 0
    menu_plan = MenuBootstrapPlan(
        plan_id="continue-fallback-test",
        steps=(MenuBootstrapStep("SUBMIT", 50, 0, "continue"),),
        initial_wait_seconds=0,
        final_wait_seconds=0,
        timeout_seconds=2,
        minimum_distinct_captures=2,
    )
    reload_plan = MenuBootstrapPlan(
        plan_id="checkpoint-reload-fail-test",
        steps=(MenuBootstrapStep("CANCEL", 50, 0, "open-pause"),),
        initial_wait_seconds=0,
        final_wait_seconds=0,
        timeout_seconds=2,
        minimum_distinct_captures=2,
    )

    def reset_process(
        endpoint: SupervisorEndpoint,
        episode_id: str,
        cancellation: ProcessLaunchCancellation,
    ) -> None:
        del cancellation
        nonlocal launches
        launches += 1
        fake = FakeBridge(endpoint, episode_id, allow_episode_changes=True)
        fakes.append(fake)
        threads.append(fake.start())

    def shutdown_process() -> None:
        nonlocal shutdowns
        shutdowns += 1

    backend = SupervisorEnvironmentBackend(
        token="checkpoint-menu-fallback",
        reset_process=reset_process,
        shutdown_process=shutdown_process,
        reuse_game_on_reset=True,
        menu_bootstrap_plan=menu_plan,
        menu_bootstrap_artifact_root=tmp_path,
        menu_readiness_verifier=lambda _: ReadinessVerdict(
            True, "synthetic-continue-v1", "continued gameplay", {"matched": 1}
        ),
        checkpoint_reload_plan_provider=lambda: reload_plan,
        checkpoint_reload_readiness_verifier=lambda _: ReadinessVerdict(
            False, "synthetic-checkpoint-v1", "wrong screen", {"matched": 0}
        ),
        bridge_timeout_seconds=2.0,
        command_timeout_seconds=1.0,
    )
    try:
        backend.reset(seed=1, options=None)
        reset = backend.reset(seed=2, options={"reload_checkpoint": True})
        assert launches == 2
        assert shutdowns == 1
        assert reset.frame.episode_id == reset.episode_id
        assert backend.last_reset_recovery["method"] == "save-reopen-fallback"
        assert "readiness verifier" in str(backend.last_reset_recovery["error"])
        report_path = (
            tmp_path / "checkpoint-reloads" / reset.episode_id / "menu-bootstrap" / "report.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["status"] == "FAILED"
        assert fakes[1].commands.count("BEGIN_EPISODE") == 1
        assert "APPLY_MENU_ACTION" in fakes[1].commands
        assert fakes[1].commands[-2:] == ["BEGIN_EPISODE", "CAPTURE_ONCE"]
    finally:
        backend.close()
        for thread in threads:
            thread.join(timeout=2.0)
        assert all(fake.failure is None for fake in fakes)


def test_real_backend_reports_player_death_as_episode_termination() -> None:
    fake: FakeBridge | None = None
    fake_thread: threading.Thread | None = None

    def reset_process(
        endpoint: SupervisorEndpoint,
        episode_id: str,
        cancellation: ProcessLaunchCancellation,
    ) -> None:
        del cancellation
        nonlocal fake, fake_thread
        fake = FakeBridge(
            endpoint,
            episode_id,
            player_dead_after_action=True,
            shade_pending_after_action=True,
            run_failed_after_action=True,
        )
        fake_thread = fake.start()

    backend = SupervisorEnvironmentBackend(
        token="backend-death-test-token",
        reset_process=reset_process,
        bridge_timeout_seconds=2.0,
        command_timeout_seconds=1.0,
    )
    try:
        backend.reset(seed=3, options=None)
        step = backend.step(action_id=1, buttons=("RIGHT",), lease_ms=1)
        assert step.terminated is True
        assert step.truncated is False
        assert step.termination_reason == "player_death"
        assert step.frame.player_dead is True
        assert step.frame.death_persisted is True
        assert step.frame.shade_pending is True
        assert step.frame.run_failed is True
    finally:
        backend.close()
        if fake_thread is not None:
            fake_thread.join(timeout=2.0)
        assert fake is None or fake.failure is None


def test_backend_bootstraps_before_episode_and_keeps_menu_out_of_experience(
    tmp_path: Path,
) -> None:
    fake: FakeBridge | None = None
    fake_thread: threading.Thread | None = None

    def reset_process(
        endpoint: SupervisorEndpoint,
        episode_id: str,
        cancellation: ProcessLaunchCancellation,
    ) -> None:
        del cancellation
        nonlocal fake, fake_thread
        fake = FakeBridge(endpoint, episode_id)
        fake_thread = fake.start()

    plan = MenuBootstrapPlan(
        plan_id="test",
        steps=(MenuBootstrapStep("SUBMIT", 50, 0, "submit"),),
        initial_wait_seconds=0,
        final_wait_seconds=0,
        timeout_seconds=2,
        minimum_distinct_captures=2,
    )
    backend = SupervisorEnvironmentBackend(
        token="backend-test-token",
        reset_process=reset_process,
        menu_bootstrap_plan=plan,
        menu_bootstrap_artifact_root=tmp_path,
        menu_readiness_verifier=lambda _: ReadinessVerdict(
            True, "test-v1", "gameplay", {"score": 1}
        ),
        bridge_timeout_seconds=2.0,
        command_timeout_seconds=1.0,
    )
    try:
        reset = backend.reset(seed=1, options=None)
        assert reset.frame.last_applied_action_id is None
        assert fake is not None
        assert fake.capture_encodings == [None, None, None, "raw_rgb8"]
        assert fake.commands == [
            "CAPTURE_ONCE",
            "APPLY_MENU_ACTION",
            "RELEASE_ALL",
            "CAPTURE_ONCE",
            "CAPTURE_ONCE",
            "RELEASE_ALL",
            "BEGIN_EPISODE",
            "CAPTURE_ONCE",
        ]
        report = json.loads(
            (
                tmp_path
                / reset.episode_id
                / "menu-bootstrap"
                / "report.json"
            ).read_text(encoding="utf-8")
        )
        assert report["status"] == "READY"
    finally:
        backend.close()
        if fake_thread is not None:
            fake_thread.join(timeout=2)
        assert fake is None or fake.failure is None


def test_backend_rejects_second_reset_without_process_callback() -> None:
    backend = SupervisorEnvironmentBackend(
        token="backend-test-token",
        bridge_timeout_seconds=0.1,
        command_timeout_seconds=0.1,
    )
    try:
        # There is deliberately no process callback or bridge.  This proves
        # reset is process-boundary plumbing, not hidden menu automation.
        try:
            backend.reset(seed=None, options=None)
        except TimeoutError:
            pass
        else:
            raise AssertionError("reset unexpectedly invented a game reset path")
    finally:
        backend.close()


def test_png_decoder_rejects_compressed_expansion_beyond_dimensions() -> None:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    bomb = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00" * 5_000_000))
        + chunk(b"IEND", b"")
    )
    with pytest.raises(ProtocolError, match="declared dimensions"):
        decode_rgb8_png(bomb, expected_width=1, expected_height=1)


def test_process_callbacks_have_enforced_timeouts() -> None:
    reset_blocker = threading.Event()

    def blocked_reset(
        _: SupervisorEndpoint, __: str, cancellation: ProcessLaunchCancellation
    ) -> None:
        del cancellation
        reset_blocker.wait()

    backend = SupervisorEnvironmentBackend(
        reset_process=blocked_reset,
        bridge_timeout_seconds=0.1,
        command_timeout_seconds=0.1,
        process_timeout_seconds=0.05,
    )
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="reset_process timed out"):
            backend.reset(seed=None, options=None)
        assert time.monotonic() - started < 1.0
        with pytest.raises(RuntimeError, match="backend is closed"):
            backend.reset(seed=None, options=None)
    finally:
        reset_blocker.set()
        backend.close()

    shutdown_blocker = threading.Event()

    def blocked_shutdown() -> None:
        shutdown_blocker.wait()

    closing_backend = SupervisorEnvironmentBackend(
        shutdown_process=blocked_shutdown,
        command_timeout_seconds=0.1,
        process_timeout_seconds=0.05,
    )
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="shutdown_process timed out"):
            closing_backend.close()
        assert time.monotonic() - started < 1.0
    finally:
        shutdown_blocker.set()
