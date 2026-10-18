from __future__ import annotations

import asyncio
import base64
import time
import unittest
import uuid
from pathlib import Path

from hk_agent.protocol import PROTOCOL_VERSION, Envelope, ProtocolError, read_message, write_message
from hk_agent.session import SessionState
from hk_agent.supervisor import SupervisorServer


class SupervisorTests(unittest.TestCase):
    def test_rejects_instance_hello_for_personal_other_slot_and_wrong_product(self) -> None:
        async def scenario(persistent: Path, product: str) -> None:
            expected = Path.cwd() / "slot-1"
            personal = Path.cwd() / "personal"
            other = Path.cwd() / "slot-2"
            supervisor = SupervisorServer(
                token="test-token",
                expected_persistent_data_path=expected,
                forbidden_personal_root=personal,
                expected_product_name="Hollow Knight AI 1",
                forbidden_instance_data_roots=(other,),
            )
            await supervisor.start()
            assert supervisor.port is not None
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", supervisor.port)
                hello = Envelope(
                    protocol_version=PROTOCOL_VERSION,
                    message_type="HELLO",
                    run_id=supervisor.run_id,
                    instance_id="hk-invalid",
                    session_id=str(uuid.uuid4()),
                    seq=0,
                    trace_id=str(uuid.uuid4()),
                    sent_monotonic_ns=time.monotonic_ns(),
                    payload={
                        "token": "test-token",
                        "game_profile": "development",
                        "game_version": "bridge-harness",
                        "persistent_data_path": str(persistent),
                        "product_name": product,
                    },
                )
                await write_message(writer, hello)
                with self.assertRaises(asyncio.IncompleteReadError):
                    await read_message(reader)
                self.assertEqual(supervisor.accepted, [])
                writer.close()
                await writer.wait_closed()
            finally:
                await supervisor.close()

        asyncio.run(scenario(Path.cwd() / "personal", "Hollow Knight AI 1"))
        asyncio.run(scenario(Path.cwd() / "slot-2", "Hollow Knight AI 1"))
        asyncio.run(scenario(Path.cwd() / "slot-1", "Hollow Knight AI 2"))

    def test_rejects_bridge_with_wrong_effective_save_root(self) -> None:
        async def scenario() -> None:
            expected = Path.cwd() / "expected-save-root"
            supervisor = SupervisorServer(
                token="test-token", expected_save_root=expected
            )
            await supervisor.start()
            assert supervisor.port is not None
            try:
                reader, writer = await asyncio.open_connection(
                    "127.0.0.1", supervisor.port
                )
                hello = Envelope(
                    protocol_version=PROTOCOL_VERSION,
                    message_type="HELLO",
                    run_id=supervisor.run_id,
                    instance_id="hk-wrong-root",
                    session_id=str(uuid.uuid4()),
                    seq=0,
                    trace_id=str(uuid.uuid4()),
                    sent_monotonic_ns=time.monotonic_ns(),
                    payload={
                        "token": "test-token",
                        "game_profile": "development",
                        "game_version": "bridge-harness",
                        "effective_save_root": str(expected.parent / "other-root"),
                    },
                )
                await write_message(writer, hello)
                with self.assertRaises(asyncio.IncompleteReadError):
                    await read_message(reader)
                self.assertEqual(supervisor.accepted, [])
                writer.close()
                await writer.wait_closed()
            finally:
                await supervisor.close()

        asyncio.run(scenario())

    def test_accepts_expected_effective_save_root(self) -> None:
        async def scenario() -> None:
            expected = Path.cwd() / "expected-save-root"
            supervisor = SupervisorServer(
                token="test-token", expected_save_root=expected
            )
            await supervisor.start()
            assert supervisor.port is not None
            try:
                reader, writer = await asyncio.open_connection(
                    "127.0.0.1", supervisor.port
                )
                hello = Envelope(
                    protocol_version=PROTOCOL_VERSION,
                    message_type="HELLO",
                    run_id=supervisor.run_id,
                    instance_id="hk-right-root",
                    session_id=str(uuid.uuid4()),
                    seq=0,
                    trace_id=str(uuid.uuid4()),
                    sent_monotonic_ns=time.monotonic_ns(),
                    payload={
                        "token": "test-token",
                        "game_profile": "development",
                        "game_version": "bridge-harness",
                        "effective_save_root": str(expected),
                    },
                )
                await write_message(writer, hello)
                welcome = await read_message(reader)
                accepted = await supervisor.wait_for_bridge(1.0)
                self.assertEqual(welcome.message_type, "WELCOME")
                self.assertEqual(accepted.effective_save_root, str(expected))
                writer.close()
                await writer.wait_closed()
            finally:
                await supervisor.close()

        asyncio.run(scenario())

    def test_accepts_authenticated_bridge(self) -> None:
        async def scenario() -> None:
            supervisor = SupervisorServer(token="test-token")
            await supervisor.start()
            assert supervisor.port is not None
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", supervisor.port)
                session_id = str(uuid.uuid4())
                hello = Envelope(
                    protocol_version=PROTOCOL_VERSION,
                    message_type="HELLO",
                    run_id=supervisor.run_id,
                    instance_id="hk-00",
                    session_id=session_id,
                    seq=0,
                    trace_id=str(uuid.uuid4()),
                    sent_monotonic_ns=time.monotonic_ns(),
                    payload={
                        "token": "test-token",
                        "game_profile": "development",
                        "game_version": "bridge-harness",
                    },
                )
                await write_message(writer, hello)
                welcome = await read_message(reader)
                accepted = await supervisor.wait_for_bridge(1.0)

                self.assertEqual(welcome.message_type, "WELCOME")
                self.assertEqual(accepted.session_id, session_id)
                self.assertEqual(accepted.machine.state, SessionState.READY)
                self.assertEqual(accepted.capabilities, frozenset())

                heartbeat = Envelope(
                    protocol_version=PROTOCOL_VERSION,
                    message_type="HEARTBEAT",
                    run_id=supervisor.run_id,
                    instance_id="hk-00",
                    session_id=session_id,
                    seq=1,
                    trace_id=str(uuid.uuid4()),
                    sent_monotonic_ns=time.monotonic_ns(),
                    payload={"state": "READY"},
                )
                await write_message(writer, heartbeat)
                alive = await supervisor.wait_for_heartbeat(1.0)
                self.assertEqual(alive.heartbeats_received, 1)
                self.assertIsNotNone(alive.last_heartbeat_ns)

                ping_task = asyncio.create_task(
                    supervisor.send_command("PING", {}, bridge=accepted)
                )
                ping = await read_message(reader)
                self.assertEqual(ping.message_type, "PING")
                pong = Envelope(
                    protocol_version=PROTOCOL_VERSION,
                    message_type="ACK",
                    run_id=supervisor.run_id,
                    instance_id="hk-00",
                    session_id=session_id,
                    seq=2,
                    trace_id=ping.trace_id,
                    sent_monotonic_ns=time.monotonic_ns(),
                    payload={"command_type": "PING", "status": "ok"},
                )
                await write_message(writer, pong)
                response = await ping_task
                self.assertEqual(response.payload["command_type"], "PING")

                capture_task = asyncio.create_task(supervisor.capture_once(bridge=accepted))
                capture = await read_message(reader)
                self.assertEqual(capture.message_type, "CAPTURE_ONCE")
                self.assertNotIn("encoding", capture.payload)
                frame = Envelope(
                    protocol_version=PROTOCOL_VERSION,
                    message_type="FRAME",
                    run_id=supervisor.run_id,
                    instance_id="hk-00",
                    session_id=session_id,
                    seq=3,
                    trace_id=capture.trace_id,
                    sent_monotonic_ns=time.monotonic_ns(),
                    payload={
                        "frame_id": 1,
                        "capture_monotonic_ns": time.monotonic_ns(),
                        "width": 1,
                        "height": 1,
                        "channels": 3,
                        "pixel_format": "RGB8",
                        "encoding": "png",
                        "data_base64": base64.b64encode(b"\x89PNG\r\n\x1a\n").decode(),
                        "scene_transition": False,
                        "player_dead": True,
                        "death_persisted": True,
                        "shade_pending": True,
                        "shade_recovered": False,
                        "run_failed": True,
                        "enemy_kills_since_last_frame": 2,
                        "hp_lost_since_last_frame": 1,
                        "geo_gained_since_last_frame": 17,
                        "geo_spent_since_last_frame": 4,
                        "hp_restored_since_last_frame": 1,
                        "shade_hits_since_last_frame": 2,
                        "shade_distance": 1.25,
                        "scene_name": "Room_Test",
                        "enemy_events_since_last_frame": [
                            {"scene_name": "Room_Test", "enemy_type": "Crawler"}
                        ],
                        "duplicate_of": None,
                        "last_applied_action_id": None,
                    },
                )
                await write_message(writer, frame)
                captured = await capture_task
                self.assertEqual(captured.width, 1)
                self.assertTrue(captured.data.startswith(b"\x89PNG"))
                self.assertTrue(captured.player_dead)
                self.assertTrue(captured.death_persisted)
                self.assertTrue(captured.shade_pending)
                self.assertFalse(captured.shade_recovered)
                self.assertTrue(captured.run_failed)
                self.assertEqual(captured.enemy_kills_since_last_frame, 2)
                self.assertEqual(captured.hp_lost_since_last_frame, 1)
                self.assertEqual(captured.geo_gained_since_last_frame, 17)
                self.assertEqual(captured.geo_spent_since_last_frame, 4)
                self.assertEqual(captured.hp_restored_since_last_frame, 1)
                self.assertEqual(captured.shade_hits_since_last_frame, 2)
                self.assertEqual(captured.shade_distance, 1.25)
                self.assertEqual(captured.scene_name, "Room_Test")
                self.assertEqual(captured.enemy_events_since_last_frame, (("Room_Test", "Crawler"),))

                fallback_task = asyncio.create_task(
                    supervisor.capture_once(bridge=accepted, encoding="raw_rgb8")
                )
                fallback_command = await read_message(reader)
                self.assertEqual(fallback_command.message_type, "CAPTURE_ONCE")
                self.assertNotIn("encoding", fallback_command.payload)
                fallback_data = b"\x89PNG\r\n\x1a\n"
                await write_message(
                    writer,
                    Envelope(
                        protocol_version=PROTOCOL_VERSION,
                        message_type="FRAME",
                        run_id=supervisor.run_id,
                        instance_id="hk-00",
                        session_id=session_id,
                        seq=4,
                        trace_id=fallback_command.trace_id,
                        sent_monotonic_ns=time.monotonic_ns(),
                        payload={
                            "frame_id": 2,
                            "capture_monotonic_ns": time.monotonic_ns(),
                            "width": 1,
                            "height": 1,
                            "channels": 3,
                            "pixel_format": "RGB8",
                            "encoding": "png",
                            "payload_bytes": len(fallback_data),
                            "data_base64": base64.b64encode(fallback_data).decode(),
                            "scene_transition": False,
                        },
                    ),
                )
                self.assertEqual((await fallback_task).encoding, "png")
                writer.close()
                await writer.wait_closed()
            finally:
                await supervisor.close()

        asyncio.run(scenario())

    def test_raw_frame_validation(self) -> None:
        async def scenario(frame_encoding: str, encoded: str, size: int, width: int = 2) -> None:
            supervisor = SupervisorServer(token="raw-test-token")
            await supervisor.start()
            assert supervisor.port is not None
            writer: asyncio.StreamWriter | None = None
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", supervisor.port)
                session_id = str(uuid.uuid4())
                await write_message(
                    writer,
                    Envelope(
                        protocol_version=PROTOCOL_VERSION,
                        message_type="HELLO",
                        run_id=supervisor.run_id,
                        instance_id="hk-raw",
                        session_id=session_id,
                        seq=0,
                        trace_id=str(uuid.uuid4()),
                        sent_monotonic_ns=time.monotonic_ns(),
                        payload={
                            "token": "raw-test-token",
                            "game_profile": "development",
                            "game_version": "fake",
                            "capabilities": ["frame_png_rgb8", "frame_raw_rgb8"],
                        },
                    ),
                )
                await read_message(reader)
                accepted = await supervisor.wait_for_bridge(1.0)
                self.assertEqual(accepted.capabilities, frozenset({"frame_png_rgb8", "frame_raw_rgb8"}))
                capture_task = asyncio.create_task(
                    supervisor.capture_once(bridge=accepted, encoding="raw_rgb8")
                )
                command = await read_message(reader)
                self.assertEqual(command.payload.get("encoding"), "raw_rgb8")
                raw_data = b"\x01\x02\x03" * size
                await write_message(
                    writer,
                    Envelope(
                        protocol_version=PROTOCOL_VERSION,
                        message_type="FRAME",
                        run_id=supervisor.run_id,
                        instance_id="hk-raw",
                        session_id=session_id,
                        seq=1,
                        trace_id=command.trace_id,
                        sent_monotonic_ns=time.monotonic_ns(),
                        payload={
                            "frame_id": 1,
                            "capture_monotonic_ns": time.monotonic_ns(),
                            "width": width,
                            "height": 1,
                            "channels": 3,
                            "pixel_format": "RGB8",
                            "encoding": frame_encoding,
                            "payload_bytes": len(raw_data),
                            "data_base64": encoded if encoded != "__VALID__" else base64.b64encode(raw_data).decode(),
                            "scene_transition": False,
                        },
                    ),
                )
                captured = await capture_task
                self.assertEqual(captured.encoding, "raw_rgb8")
                self.assertEqual(captured.data, raw_data)
            finally:
                if writer is not None:
                    writer.close()
                    await writer.wait_closed()
                await supervisor.close()

        asyncio.run(scenario("raw_rgb8", "__VALID__", 2))
        for encoding, encoded, size, width in (
            ("raw_rgb8", "__VALID__", 1, 2),
            ("raw_rgb8", "%%%", 2, 2),
            ("bitmap", "__VALID__", 2, 2),
        ):
            with self.assertRaises(ProtocolError):
                asyncio.run(scenario(encoding, encoded, size, width))
        with self.assertRaises(ProtocolError):
            asyncio.run(scenario("raw_rgb8", "__VALID__", 601, 601))

    def test_unknown_capture_request_encoding_is_rejected_locally(self) -> None:
        with self.assertRaisesRegex(ValueError, "encoding must be png or raw_rgb8"):
            asyncio.run(SupervisorServer().capture_once(encoding="jpeg"))

    def test_rejects_wrong_token(self) -> None:
        async def scenario() -> None:
            supervisor = SupervisorServer(token="correct")
            await supervisor.start()
            assert supervisor.port is not None
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", supervisor.port)
                hello = Envelope(
                    protocol_version=PROTOCOL_VERSION,
                    message_type="HELLO",
                    run_id=supervisor.run_id,
                    instance_id="hk-00",
                    session_id=str(uuid.uuid4()),
                    seq=0,
                    trace_id=str(uuid.uuid4()),
                    sent_monotonic_ns=time.monotonic_ns(),
                    payload={
                        "token": "wrong",
                        "game_profile": "development",
                        "game_version": "bridge-harness",
                    },
                )
                await write_message(writer, hello)
                with self.assertRaises(asyncio.IncompleteReadError):
                    await read_message(reader)
                self.assertEqual(supervisor.accepted, [])
                writer.close()
                await writer.wait_closed()
            finally:
                await supervisor.close()

        asyncio.run(scenario())

    def test_faults_bridge_when_heartbeat_stops(self) -> None:
        async def scenario() -> None:
            supervisor = SupervisorServer(token="test-token", heartbeat_timeout_seconds=0.05)
            await supervisor.start()
            assert supervisor.port is not None
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", supervisor.port)
                hello = Envelope(
                    protocol_version=PROTOCOL_VERSION,
                    message_type="HELLO",
                    run_id=supervisor.run_id,
                    instance_id="hk-stalled",
                    session_id=str(uuid.uuid4()),
                    seq=0,
                    trace_id=str(uuid.uuid4()),
                    sent_monotonic_ns=time.monotonic_ns(),
                    payload={
                        "token": "test-token",
                        "game_profile": "development",
                        "game_version": "bridge-harness",
                    },
                )
                await write_message(writer, hello)
                welcome = await read_message(reader)
                accepted = await supervisor.wait_for_bridge(1.0)
                self.assertEqual(welcome.payload["heartbeat_timeout_ms"], 50)

                await asyncio.sleep(0.1)
                self.assertEqual(accepted.machine.state, SessionState.FAULTED)
                with self.assertRaises(asyncio.IncompleteReadError):
                    await read_message(reader)
                writer.close()
                await writer.wait_closed()
            finally:
                await supervisor.close()

        asyncio.run(scenario())
