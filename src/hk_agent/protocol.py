from __future__ import annotations

import asyncio
import json
import struct
from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 1_048_576
_LENGTH = struct.Struct(">I")


class ProtocolError(RuntimeError):
    """Raised when a peer violates the framed protocol."""


@dataclass(frozen=True, slots=True)
class Envelope:
    protocol_version: int
    message_type: str
    run_id: str
    instance_id: str
    session_id: str
    seq: int
    trace_id: str
    sent_monotonic_ns: int
    payload: dict[str, Any]

    @classmethod
    def from_mapping(cls, value: Any) -> Envelope:
        if not isinstance(value, dict):
            raise ProtocolError("Envelope must be a JSON object")

        required = {
            "protocol_version": int,
            "message_type": str,
            "run_id": str,
            "instance_id": str,
            "session_id": str,
            "seq": int,
            "trace_id": str,
            "sent_monotonic_ns": int,
            "payload": dict,
        }
        for key, expected_type in required.items():
            field = value.get(key)
            if not isinstance(field, expected_type) or isinstance(field, bool):
                raise ProtocolError(f"Invalid or missing envelope field: {key}")

        if value["protocol_version"] != PROTOCOL_VERSION:
            raise ProtocolError(f"Unsupported protocol version: {value['protocol_version']}")
        if value["seq"] < 0:
            raise ProtocolError("Sequence must be non-negative")
        if not value["message_type"]:
            raise ProtocolError("Message type must not be empty")

        return cls(**{key: value[key] for key in required})

    def to_mapping(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "message_type": self.message_type,
            "run_id": self.run_id,
            "instance_id": self.instance_id,
            "session_id": self.session_id,
            "seq": self.seq,
            "trace_id": self.trace_id,
            "sent_monotonic_ns": self.sent_monotonic_ns,
            "payload": self.payload,
        }


def encode_message(envelope: Envelope) -> bytes:
    body = json.dumps(envelope.to_mapping(), separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    if not body or len(body) > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"Invalid encoded message size: {len(body)}")
    return _LENGTH.pack(len(body)) + body


async def read_message(reader: asyncio.StreamReader) -> Envelope:
    header = await reader.readexactly(_LENGTH.size)
    (length,) = _LENGTH.unpack(header)
    if length == 0 or length > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"Invalid message length: {length}")

    body = await reader.readexactly(length)
    try:
        value = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ProtocolError("Message body is not valid UTF-8 JSON") from error
    return Envelope.from_mapping(value)


async def write_message(writer: asyncio.StreamWriter, envelope: Envelope) -> None:
    writer.write(encode_message(envelope))
    await writer.drain()
