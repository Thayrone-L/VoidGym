from __future__ import annotations

import asyncio
import base64
import struct
import time
import unittest
import uuid
import zlib

from hk_agent.protocol import PROTOCOL_VERSION, Envelope, ProtocolError, read_message, write_message
from hk_agent.supervisor import SupervisorServer
from hk_agent.supervisor_backend import _PNG_SIGNATURE, decode_rgb8_png


def _chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _valid_rgb8_png() -> bytes:
    header = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
    pixels = bytes((0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 0))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(pixels))
        + _chunk(b"IEND", b"")
    )


class PngSignatureTests(unittest.TestCase):
    def test_decoder_accepts_valid_rgb8_png_and_signature_is_eight_bytes(self) -> None:
        png = _valid_rgb8_png()
        self.assertEqual(len(_PNG_SIGNATURE), 8)
        self.assertEqual(_PNG_SIGNATURE, b"\x89PNG\r\n\x1a\n")
        decoded = decode_rgb8_png(png, expected_width=2, expected_height=2)
        self.assertEqual(decoded.shape, (2, 2, 3))
        self.assertEqual(decoded[0, 0].tolist(), [255, 0, 0])
        self.assertEqual(decoded[1, 1].tolist(), [255, 255, 0])

    def test_supervisor_frame_parser_accepts_valid_png_signature(self) -> None:
        asyncio.run(self._parse_frame(_valid_rgb8_png(), expect_valid=True))

    def test_supervisor_frame_parser_rejects_corrupted_png_signature(self) -> None:
        png = _valid_rgb8_png()
        corrupted = b"\x89PNG\n\x1a\n" + png[len(b"\x89PNG\r\n\x1a\n") :]
        asyncio.run(self._parse_frame(corrupted, expect_valid=False))

    async def _parse_frame(self, png: bytes, *, expect_valid: bool) -> None:
        supervisor = SupervisorServer(token="png-signature-test-token")
        await supervisor.start()
        assert supervisor.port is not None
        writer = None
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", supervisor.port)
            session_id = str(uuid.uuid4())
            await write_message(
                writer,
                Envelope(
                    protocol_version=PROTOCOL_VERSION,
                    message_type="HELLO",
                    run_id=supervisor.run_id,
                    instance_id="png-test-instance",
                    session_id=session_id,
                    seq=0,
                    trace_id=str(uuid.uuid4()),
                    sent_monotonic_ns=time.monotonic_ns(),
                    payload={
                        "token": "png-signature-test-token",
                        "game_profile": "development",
                        "game_version": "png-signature-test",
                    },
                ),
            )
            self.assertEqual((await read_message(reader)).message_type, "WELCOME")
            bridge = await supervisor.wait_for_bridge(1.0)
            capture_task = asyncio.create_task(supervisor.capture_once(bridge=bridge))
            command = await read_message(reader)
            self.assertEqual(command.message_type, "CAPTURE_ONCE")
            await write_message(
                writer,
                Envelope(
                    protocol_version=PROTOCOL_VERSION,
                    message_type="FRAME",
                    run_id=supervisor.run_id,
                    instance_id="png-test-instance",
                    session_id=session_id,
                    seq=1,
                    trace_id=command.trace_id,
                    sent_monotonic_ns=time.monotonic_ns(),
                    payload={
                        "frame_id": 1,
                        "capture_monotonic_ns": time.monotonic_ns(),
                        "width": 2,
                        "height": 2,
                        "channels": 3,
                        "pixel_format": "RGB8",
                        "encoding": "png",
                        "payload_bytes": len(png),
                        "data_base64": base64.b64encode(png).decode("ascii"),
                        "scene_transition": False,
                    },
                ),
            )
            if expect_valid:
                frame = await capture_task
                self.assertEqual(frame.encoding, "png")
                self.assertEqual(frame.data, png)
            else:
                with self.assertRaisesRegex(ProtocolError, "FRAME payload is not a PNG"):
                    await capture_task
        finally:
            if writer is not None:
                writer.close()
                await writer.wait_closed()
            await supervisor.close()
