from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hk_agent.menu_bootstrap import (
    MenuBootstrapError,
    MenuBootstrapPlan,
    MenuBootstrapStep,
    ReadinessVerdict,
    run_menu_bootstrap,
)


class FakeSupervisor:
    def __init__(self, *, distinct: bool = True) -> None:
        self.events: list[str] = []
        self.capture_index = 0
        self.distinct = distinct

    async def capture_once(self, **_: Any) -> Any:
        self.events.append("capture")
        self.capture_index += 1
        suffix = bytes([self.capture_index]) if self.distinct else b"same"
        return SimpleNamespace(
            data=b"\x89PNG\r\n\x1a\n" + suffix,
            frame_id=self.capture_index,
            capture_monotonic_ns=self.capture_index,
            width=600,
            height=338,
            scene_transition=False,
            duplicate_of=None,
        )

    async def send_command(
        self, message_type: str, payload: dict[str, object], **_: Any
    ) -> Any:
        if message_type == "APPLY_MENU_ACTION":
            button = payload["buttons"]
            self.events.append(f"menu:{button[0]}")  # type: ignore[index]
            return SimpleNamespace(message_type="ACK", payload=payload)
        assert message_type == "RELEASE_ALL"
        self.events.append("release")
        return SimpleNamespace(message_type="ACK", payload={})


def _plan(*, minimum_distinct: int = 2) -> MenuBootstrapPlan:
    return MenuBootstrapPlan(
        plan_id="test-plan",
        steps=(MenuBootstrapStep("SUBMIT", 50, 0, "submit"),),
        initial_wait_seconds=0,
        final_wait_seconds=0,
        timeout_seconds=1,
        minimum_distinct_captures=minimum_distinct,
    )


def _verdict(ready: bool) -> ReadinessVerdict:
    return ReadinessVerdict(
        ready=ready,
        verifier_id="test-readiness-v1",
        detail="gameplay" if ready else "animated menu",
        metrics={"score": 1 if ready else 0},
    )


def test_plan_loader_rejects_gameplay_buttons_and_unbounded_plan(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "plan_id": "unsafe",
                "timeout_seconds": 1,
                "steps": [{"button": "JUMP", "settle_seconds": 0}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="allowed menu button"):
        MenuBootstrapPlan.load(path)


def test_bootstrap_is_pre_episode_menu_only_and_persists_evidence(tmp_path: Path) -> None:
    supervisor = FakeSupervisor()
    report = asyncio.run(
        run_menu_bootstrap(
            _plan(),
            supervisor=supervisor,  # type: ignore[arg-type]
            bridge=object(),  # type: ignore[arg-type]
            episode_id="episode-1",
            artifact_root=tmp_path,
            command_timeout_seconds=0.5,
            readiness_verifier=lambda _: _verdict(True),
        )
    )

    assert supervisor.events == ["capture", "menu:SUBMIT", "release", "capture", "capture", "release"]
    assert report.status == "READY"
    assert report.steps_completed == 1
    saved = json.loads(
        (tmp_path / "episode-1" / "menu-bootstrap" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["status"] == "READY"
    assert len(saved["captures"]) == 3


def test_bootstrap_fails_closed_when_readiness_evidence_is_not_distinct(tmp_path: Path) -> None:
    supervisor = FakeSupervisor(distinct=False)
    with pytest.raises(MenuBootstrapError, match="readiness evidence"):
        asyncio.run(
            run_menu_bootstrap(
                _plan(),
                supervisor=supervisor,  # type: ignore[arg-type]
                bridge=object(),  # type: ignore[arg-type]
                episode_id="episode-2",
                artifact_root=tmp_path,
                command_timeout_seconds=0.5,
                readiness_verifier=lambda _: _verdict(True),
            )
        )
    saved = json.loads(
        (tmp_path / "episode-2" / "menu-bootstrap" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["status"] == "FAILED"
    assert supervisor.events[-1] == "release"


def test_animated_distinct_menu_is_rejected_by_explicit_readiness_verifier(
    tmp_path: Path,
) -> None:
    supervisor = FakeSupervisor(distinct=True)
    with pytest.raises(MenuBootstrapError, match="animated menu"):
        asyncio.run(
            run_menu_bootstrap(
                _plan(),
                supervisor=supervisor,  # type: ignore[arg-type]
                bridge=object(),  # type: ignore[arg-type]
                episode_id="episode-menu-animation",
                artifact_root=tmp_path,
                command_timeout_seconds=0.5,
                readiness_verifier=lambda _: _verdict(False),
            )
        )
    saved = json.loads(
        (
            tmp_path
            / "episode-menu-animation"
            / "menu-bootstrap"
            / "report.json"
        ).read_text(encoding="utf-8")
    )
    assert saved["status"] == "FAILED"
    assert saved["readiness"]["ready"] is False


def _rgb_png(width: int, height: int) -> bytes:
    import struct
    import zlib

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


def test_readiness_verifier_explains_window_size_change(tmp_path: Path) -> None:
    from dataclasses import replace

    from hk_agent.menu_bootstrap import ReferenceReadinessSpec, build_readiness_verifier
    from hk_agent.protocol import ProtocolError

    reference = tmp_path / "reference.png"
    reference.write_bytes(_rgb_png(4, 2))
    plan = replace(
        _plan(),
        readiness=ReferenceReadinessSpec(
            verifier_id="reference-rgb-mae-v1",
            reference_image=reference,
            maximum_mean_absolute_error=1.0,
        ),
    )
    verifier = build_readiness_verifier(plan)
    same_size = SimpleNamespace(data=_rgb_png(4, 2), width=4, height=2)
    assert verifier(same_size).ready is True  # type: ignore[arg-type]
    resized = SimpleNamespace(data=_rgb_png(4, 3), width=4, height=3)
    with pytest.raises(ProtocolError, match="client area changed size"):
        verifier(resized)  # type: ignore[arg-type]
