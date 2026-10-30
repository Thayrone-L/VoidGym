from __future__ import annotations

import pytest

from tools.capture_hk_moving_frames import _capture_with_retries


def test_capture_retries_transient_bridge_failure_and_recovers() -> None:
    calls = 0
    events: list[dict[str, object]] = []

    def request(command: str) -> dict[str, object]:
        nonlocal calls
        assert command == "capture"
        calls += 1
        if calls < 3:
            raise ConnectionError("bridge temporarily unavailable")
        return {"capture": {"frame_id": 7}}

    assert _capture_with_retries(request, retry_events=events) == {"frame_id": 7}
    assert calls == 3
    assert [event["attempt"] for event in events] == [1, 2]


def test_capture_stops_after_three_retries() -> None:
    calls = 0
    events: list[dict[str, object]] = []

    def request(_command: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise TimeoutError("capture timed out")

    with pytest.raises(TimeoutError, match="capture timed out"):
        _capture_with_retries(request, retry_events=events)
    assert calls == 4  # initial attempt plus three reconnect retries
    assert [event["attempt"] for event in events] == [1, 2, 3, 4]
