from __future__ import annotations

import math
import struct
import zlib

from hk_agent.supervisor import CapturedFrame, parse_bridge_timing
from hk_agent.supervisor_backend import _to_pixel_frame


def _rgb_png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    rows = b"".join(b"\x00" + b"\x10\x20\x30" * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def test_bridge_timing_keeps_valid_metrics_and_drops_malformed_ones() -> None:
    payload: dict[str, object] = {
        "capture_ms": 12.5,
        "encode_ms": 30,
        "unity_fps": 58.9,
        "target_frame_rate": -1,  # Unity's "platform default" becomes a flag
        "png_bytes": True,  # bool is not a number here
        "log_ms": -3.0,  # negative duration is rejected
        "queued_to_start_ms": math.inf,  # non-finite is rejected
        "vsync_count": "1",  # strings are rejected
        "unrelated": 99,
    }
    assert dict(parse_bridge_timing(payload)) == {
        "capture_ms": 12.5,
        "encode_ms": 30.0,
        "unity_fps": 58.9,
        "target_frame_rate_default": 1.0,
    }
    # Every value is non-negative: the environment rejects negative timings.
    assert all(value >= 0 for _, value in parse_bridge_timing(payload))


def test_old_bridge_without_metrics_still_parses() -> None:
    assert parse_bridge_timing({}) == ()


def test_bridge_timing_reaches_the_pixel_frame() -> None:
    frame = CapturedFrame(
        episode_id="ep",
        frame_id=1,
        capture_monotonic_ns=1,
        width=4,
        height=2,
        channels=3,
        pixel_format="RGB8",
        encoding="png",
        data=_rgb_png(4, 2),
        scene_transition=False,
        duplicate_of=None,
        last_applied_action_id=7,
        bridge_timing=(("capture_ms", 11.0), ("encode_ms", 22.0)),
    )
    pixel = _to_pixel_frame(frame, "ep")
    assert dict(pixel.bridge_timing) == {"capture_ms": 11.0, "encode_ms": 22.0}
    assert pixel.decode_ms >= 0.0
