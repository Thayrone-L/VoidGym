from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hk_agent.gameplay_control import GameplayControlServer, _write_frame_atomic
from hk_agent.protocol import ProtocolError
from tools.smoke_game_bridge import _write_png_capture


class FakeSupervisor:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def send_command(
        self, message_type: str, payload: dict[str, object], **_: Any
    ) -> Any:
        if message_type == "APPLY_ACTION":
            self.last_action_payload = dict(payload)
            self.events.append(f"action:{','.join(payload['buttons'])}")  # type: ignore[arg-type]
        elif message_type == "RELEASE_ALL":
            self.events.append("release")
        else:
            raise AssertionError(message_type)
        return SimpleNamespace(message_type="ACK", payload={})

    async def capture_once(self, **_: Any) -> Any:
        self.events.append("capture")
        return SimpleNamespace(
            encoding="png",
            data=b"\x89PNG\r\n\x1a\nframe",
            frame_id=7,
            capture_monotonic_ns=11,
            width=2,
            height=1,
            scene_transition=False,
            duplicate_of=None,
            last_applied_action_id=1,
        )


async def _request(server: GameplayControlServer, value: dict[str, object]) -> dict[str, Any]:
    address = server.address
    reader, writer = await asyncio.open_connection(address.host, address.port)
    writer.write(json.dumps(value).encode("utf-8") + b"\n")
    await writer.drain()
    response = json.loads(await reader.readline())
    writer.close()
    await writer.wait_closed()
    if not isinstance(response, dict):
        raise TypeError("gameplay response must be an object")
    return response


def test_action_is_bounded_released_and_captured(tmp_path: Path) -> None:
    async def scenario() -> None:
        supervisor = FakeSupervisor()
        server = GameplayControlServer(
            supervisor,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            output_dir=tmp_path,
            idle_timeout_seconds=10,
            settle_seconds=0,
            command_timeout_seconds=1,
            episode_id="test-episode",
        )
        await server.start()
        try:
            response = await _request(
                server,
                {
                    "token": server.address.token,
                    "command": "action",
                    "buttons": ["RIGHT", "JUMP"],
                    "lease_ms": 25,
                },
            )
            assert response["ok"] is True
            assert response["released"] is True
            assert supervisor.last_action_payload["episode_id"] == "test-episode"
            assert supervisor.events == ["action:RIGHT,JUMP", "release", "capture"]
            assert Path(response["capture"]["path"]).is_file()
        finally:
            await server.close()

    asyncio.run(scenario())


def test_gameplay_control_png_writer_refuses_raw_frame(tmp_path: Path) -> None:
    with pytest.raises(ProtocolError, match="requires FRAME encoding=png"):
        _write_frame_atomic(tmp_path / "raw.png", SimpleNamespace(encoding="raw_rgb8"))  # type: ignore[arg-type]
    assert not (tmp_path / "raw.png").exists()


def test_smoke_capture_writer_refuses_raw_frame(tmp_path: Path) -> None:
    with pytest.raises(ProtocolError, match="requires FRAME encoding=png"):
        _write_png_capture(tmp_path / "raw.png", SimpleNamespace(encoding="raw_rgb8"))  # type: ignore[arg-type]
    assert not (tmp_path / "raw.png").exists()


def test_rejects_wrong_token_and_unsafe_action(tmp_path: Path) -> None:
    async def scenario() -> None:
        supervisor = FakeSupervisor()
        server = GameplayControlServer(
            supervisor,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            output_dir=tmp_path,
            idle_timeout_seconds=10,
            settle_seconds=0,
            command_timeout_seconds=1,
            episode_id="test-episode",
        )
        await server.start()
        try:
            unauthenticated = await _request(
                server, {"token": "wrong", "command": "capture"}
            )
            unsafe = await _request(
                server,
                {
                    "token": server.address.token,
                    "command": "action",
                    "buttons": ["SUBMIT"],
                    "lease_ms": 25,
                },
            )
            too_long = await _request(
                server,
                {
                    "token": server.address.token,
                    "command": "action",
                    "buttons": ["RIGHT"],
                    "lease_ms": 2001,
                },
            )
            assert unauthenticated["ok"] is False
            assert unsafe["ok"] is False
            assert too_long["ok"] is False
            assert supervisor.events == []
        finally:
            await server.close()

    asyncio.run(scenario())


def test_close_releases_and_finishes_waiter(tmp_path: Path) -> None:
    async def scenario() -> None:
        supervisor = FakeSupervisor()
        server = GameplayControlServer(
            supervisor,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            output_dir=tmp_path,
            idle_timeout_seconds=10,
            settle_seconds=0,
            command_timeout_seconds=1,
            episode_id="test-episode",
        )
        await server.start()
        waiter = asyncio.create_task(server.wait_until_closed())
        response = await _request(
            server, {"token": server.address.token, "command": "close"}
        )
        assert response == {"ok": True, "command": "close", "closing": True}
        assert await waiter == "client_close"
        assert supervisor.events == ["release"]
        await server.close()

    asyncio.run(scenario())


def test_close_wakes_waiter_after_delayed_release(tmp_path: Path) -> None:
    class DelayedReleaseSupervisor(FakeSupervisor):
        async def send_command(
            self, message_type: str, payload: dict[str, object], **kwargs: Any
        ) -> Any:
            if message_type == "RELEASE_ALL":
                await asyncio.sleep(0.05)
            return await super().send_command(message_type, payload, **kwargs)

    async def scenario() -> None:
        supervisor = DelayedReleaseSupervisor()
        server = GameplayControlServer(
            supervisor,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            output_dir=tmp_path,
            idle_timeout_seconds=2,
            settle_seconds=0,
            command_timeout_seconds=1,
            episode_id="test-episode",
        )
        await server.start()
        waiter = asyncio.create_task(server.wait_until_closed())
        response = await asyncio.wait_for(
            _request(server, {"token": server.address.token, "command": "close"}),
            timeout=1,
        )
        assert response["ok"] is True
        assert await asyncio.wait_for(waiter, timeout=1) == "client_close"
        await server.close()

    asyncio.run(scenario())
