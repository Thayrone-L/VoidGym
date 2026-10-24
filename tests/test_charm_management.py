from __future__ import annotations

import asyncio
import struct
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hk_agent.charm_management import CharmMenuState, build_charm_menu_plan
from hk_agent.menu_bootstrap import build_readiness_verifier, run_menu_bootstrap


def _png(color: tuple[int, int, int]) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(
        b"IDAT", zlib.compress(b"\x00" + bytes(color))
    ) + chunk(b"IEND", b"")


class FakeMenuBridge:
    def __init__(self, frames: tuple[bytes, ...]) -> None:
        self.frames = frames
        self.index = 0
        self.actions: list[str] = []

    async def capture_once(self, **_: Any) -> Any:
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return SimpleNamespace(encoding="png", data=frame, frame_id=self.index,
                               capture_monotonic_ns=self.index, width=1, height=1,
                               scene_transition=False, duplicate_of=None)

    async def send_command(self, message_type: str, payload: dict[str, object], **_: Any) -> Any:
        if message_type == "APPLY_MENU_ACTION":
            self.actions.append(str(payload["buttons"][0]))  # type: ignore[index]
            return SimpleNamespace(message_type="ACK", payload=payload)
        assert message_type == "RELEASE_ALL"
        return SimpleNamespace(message_type="ACK", payload={})


def test_macro_requires_bench_menu_owned_charms_capacity_and_calibrated_frames(tmp_path: Path) -> None:
    state = CharmMenuState(True, True, True, 0, frozenset({"gotCharm_1"}), 2)
    costs = {"gotCharm_1": 1, "gotCharm_2": 2}
    images = []
    for index in range(5):
        image = tmp_path / f"step-{index}.png"
        image.write_bytes(_png((index * 40, 0, 0)))
        images.append(image)
    with pytest.raises(ValueError, match="seated player"):
        build_charm_menu_plan(plan_id="x", state=CharmMenuState(False, True, True, 0, state.equipped, 2),
                              target=("gotCharm_2",), acquired_notch_costs=costs,
                              step_references=images[:4], gameplay_reference=images[4])
    with pytest.raises(ValueError, match="not been acquired"):
        build_charm_menu_plan(plan_id="x", state=state, target=("gotCharm_3",),
                              acquired_notch_costs=costs, step_references=images[:4],
                              gameplay_reference=images[4])
    with pytest.raises(ValueError, match="exceeds observed notch"):
        build_charm_menu_plan(plan_id="x", state=state, target=("gotCharm_1", "gotCharm_2"),
                              acquired_notch_costs=costs, step_references=images[:4],
                              gameplay_reference=images[4])
    plan = build_charm_menu_plan(plan_id="equip-two", state=state, target=("gotCharm_2",),
                                 acquired_notch_costs=costs, step_references=images[:4],
                                 gameplay_reference=images[4])
    assert [step.button for step in plan.steps] == ["SUBMIT", "RIGHT", "SUBMIT", "CANCEL"]


def test_charm_macro_executes_through_fake_bridge_with_png_guard_for_each_step(tmp_path: Path) -> None:
    colors = ((0, 0, 0), (40, 0, 0), (80, 0, 0), (120, 0, 0), (160, 0, 0), (200, 0, 0))
    paths = []
    frames = []
    for index, color in enumerate(colors):
        path = tmp_path / f"frame-{index}.png"
        path.write_bytes(_png(color))
        paths.append(path)
        frames.append(path.read_bytes())
    state = CharmMenuState(True, True, True, 0, frozenset({"gotCharm_1"}), 2)
    plan = build_charm_menu_plan(plan_id="equip-two", state=state, target=("gotCharm_2",),
                                 acquired_notch_costs={"gotCharm_1": 1, "gotCharm_2": 2},
                                 step_references=paths[1:5], gameplay_reference=paths[5])
    bridge = FakeMenuBridge(tuple(frames))
    report = asyncio.run(run_menu_bootstrap(
        plan, supervisor=bridge, bridge=object(), episode_id="fake-charm-menu",
        artifact_root=tmp_path / "artifacts", command_timeout_seconds=0.5,
        readiness_verifier=build_readiness_verifier(plan),
    ))
    assert report.status == "READY" and report.steps_completed == 4
    assert bridge.actions == ["SUBMIT", "RIGHT", "SUBMIT", "CANCEL"]
