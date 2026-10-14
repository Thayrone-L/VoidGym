from __future__ import annotations

import asyncio
import struct
import time
import unittest
import uuid

from hk_agent.protocol import (
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    Envelope,
    ProtocolError,
    encode_message,
    read_message,
)


def _envelope(sequence: int = 0) -> Envelope:
    return Envelope(
        protocol_version=PROTOCOL_VERSION,
        message_type="PING",
        run_id=str(uuid.uuid4()),
        instance_id="hk-00",
        session_id=str(uuid.uuid4()),
        seq=sequence,
        trace_id=str(uuid.uuid4()),
        sent_monotonic_ns=time.monotonic_ns(),
        payload={"text": "olá"},
    )


class ProtocolTests(unittest.TestCase):
    def test_round_trip_supports_fragmented_input(self) -> None:
        async def scenario() -> None:
            expected = _envelope(7)
            encoded = encode_message(expected)
            reader = asyncio.StreamReader()
            reader.feed_data(encoded[:2])
            await asyncio.sleep(0)
            reader.feed_data(encoded[2:])
            reader.feed_eof()
            actual = await read_message(reader)
            self.assertEqual(actual, expected)

        asyncio.run(scenario())

    def test_rejects_oversized_length_before_reading_body(self) -> None:
        async def scenario() -> None:
            reader = asyncio.StreamReader()
            reader.feed_data(struct.pack(">I", MAX_MESSAGE_BYTES + 1))
            reader.feed_eof()
            with self.assertRaisesRegex(ProtocolError, "Invalid message length"):
                await read_message(reader)

        asyncio.run(scenario())

    def test_rejects_wrong_protocol_version(self) -> None:
        value = _envelope().to_mapping()
        value["protocol_version"] = 999
        with self.assertRaisesRegex(ProtocolError, "Unsupported protocol"):
            Envelope.from_mapping(value)
