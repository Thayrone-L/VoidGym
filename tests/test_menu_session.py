from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hk_agent.isolated_session import (
    MenuStep,
    SessionRequest,
    SessionRequestError,
    SessionResult,
    _run_menu_sequence,
)


def test_menu_step_accepts_one_allowlisted_short_pulse() -> None:
    assert MenuStep.from_mapping({"button": "SUBMIT", "lease_ms": 50}) == MenuStep(
        button="SUBMIT", lease_ms=50
    )


@pytest.mark.parametrize(
    "value",
    [
        {"button": "QUIT", "lease_ms": 50},
        {"button": "UP", "lease_ms": 0},
        {"button": "UP", "lease_ms": 101},
        {"button": ["UP", "SUBMIT"], "lease_ms": 50},
    ],
)
def test_menu_step_rejects_unsafe_values(value: object) -> None:
    with pytest.raises(SessionRequestError):
        MenuStep.from_mapping(value)


def test_menu_mode_allows_empty_sequence_for_observation(tmp_path: Path) -> None:
    executable = tmp_path / "game.exe"
    executable.write_bytes(b"fake")
    request = SessionRequest.from_mapping(
        {
            "mode": "menu",
            "game_executable": str(executable),
            "game_profile": "test",
            "instance_id": "menu-observe",
            "result_path": str(tmp_path / "result.json"),
            "capture_output": None,
            "personal_root": str(tmp_path / "personal"),
            "experimental_root": str(tmp_path / "experimental"),
            "timeout_seconds": 10,
            "close_timeout_seconds": 2,
            "menu_steps": [],
            "menu_initial_wait_seconds": 5,
            "menu_step_wait_seconds": 0.15,
            "menu_output_dir": str(tmp_path / "frames"),
            "menu_report_output": str(tmp_path / "report.json"),
        }
    )
    assert request.menu_steps == ()
    assert request.mode == "menu"


def test_menu_sequence_orders_capture_action_release_capture(tmp_path: Path) -> None:
    executable = tmp_path / "game.exe"
    executable.write_bytes(b"fake")
    request = SessionRequest(
        mode="menu",
        game_executable=executable,
        game_arguments=(),
        game_profile="test",
        instance_id="menu-order",
        result_path=tmp_path / "result.json",
        capture_output=None,
        personal_root=tmp_path / "personal",
        experimental_root=tmp_path / "experimental",
        timeout_seconds=1,
        close_timeout_seconds=1,
        menu_steps=(MenuStep("SUBMIT", 50),),
        menu_initial_wait_seconds=0,
        menu_step_wait_seconds=0,
        menu_final_wait_seconds=0,
        menu_output_dir=tmp_path / "frames",
        menu_report_output=tmp_path / "report.json",
    )
    events: list[str] = []

    class FakeSupervisor:
        async def capture_once(self, **_: Any) -> Any:
            events.append("capture")
            return SimpleNamespace(
                data=b"\x89PNG\r\n\x1a\nframe",
                encoding="png",
                frame_id=len(events),
                capture_monotonic_ns=1,
                width=1,
                height=1,
                scene_transition=False,
                duplicate_of=None,
            )

        async def send_command(
            self, message_type: str, payload: dict[str, object], **_: Any
        ) -> Any:
            assert message_type != "APPLY_ACTION"
            if message_type == "APPLY_MENU_ACTION":
                assert payload == {"buttons": ["SUBMIT"], "lease_ms": 50}
                events.append("menu:SUBMIT")
            elif message_type == "RELEASE_ALL":
                events.append("release")
            else:
                raise AssertionError(f"unexpected command: {message_type}")
            return SimpleNamespace(message_type="ACK", payload={})

    result = SessionResult(mode="menu")
    asyncio.run(
        _run_menu_sequence(
            request,
            result,
            FakeSupervisor(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
        )
    )

    assert events == ["capture", "menu:SUBMIT", "release", "capture", "release"]
    assert result.menu_steps_completed == 1
    assert result.release_all_ok is True
