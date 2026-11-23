from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools import dump_game_map_layout as dump_tool


class _Process:
    pid = 987654
    returncode = 0

    def __init__(self, *, running: bool) -> None:
        self.running = running

    def poll(self) -> int | None:
        return None if self.running else self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.running = False
        return self.returncode

    def terminate(self) -> None:
        self.running = False

    def kill(self) -> None:
        self.running = False


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    instances = tmp_path / "instances"
    install = instances / "hk-ai-2"
    install.mkdir(parents=True)
    (install / "Hollow Knight.exe").write_bytes(b"fake executable")
    save_root = tmp_path / "LocalLow" / "Team Cherry" / "Hollow Knight AI 2"
    save_root.mkdir(parents=True)
    (save_root / "user1.dat").write_bytes(b"original save bytes")
    marker = SimpleNamespace(
        bridge_dll_sha256="a" * 64,
        persistent_data_path_expected=str(save_root),
    )
    monkeypatch.setattr(dump_tool, "load_instance_marker", lambda *_args, **_kwargs: marker)
    return instances, install, save_root


def test_dump_refuses_active_campaign_before_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    instances, _install, _saves = _fixture(tmp_path, monkeypatch)
    called = False

    def launch(*_args: Any, **_kwargs: Any) -> _Process:
        nonlocal called
        called = True
        return _Process(running=True)

    with pytest.raises(RuntimeError, match="campanha ativa"):
        dump_tool.dump_layout(
            2, tmp_path / "runtime", instances_root=instances,
            status_reader=lambda: {"active_count": 1, "active_launches": [
                {"instance_index": 2, "status": "RUNNING"},
            ]},
            process_reader=list, process_factory=launch,
        )
    assert not called
    assert not (tmp_path / "runtime").exists()


@pytest.mark.parametrize("running_path", ["instance", "personal"])
def test_dump_refuses_open_game_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, running_path: str
) -> None:
    instances, install, _saves = _fixture(tmp_path, monkeypatch)
    executable = install / "Hollow Knight.exe" if running_path == "instance" else (
        dump_tool.ORIGINAL_INSTALL / "Hollow Knight.exe"
    )
    with pytest.raises(RuntimeError, match="está aberto"):
        dump_tool.dump_layout(
            2, tmp_path / "runtime", instances_root=instances,
            status_reader=lambda: {"active_count": 0, "active_launches": []},
            process_reader=lambda: [(1, executable)],
        )


@pytest.mark.parametrize("outcome", ["timeout", "error"])
def test_saves_are_restored_byte_for_byte_after_dump_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    instances, _install, save_root = _fixture(tmp_path, monkeypatch)
    runtime = tmp_path / "runtime"

    def launch(_command: list[str], *, cwd: Path, env: dict[str, str]) -> _Process:
        (save_root / "user1.dat").write_bytes(b"changed by game")
        (save_root / "user2.dat").write_bytes(b"new save")
        if outcome == "error":
            request = Path(env["HK_AGENT_MAP_LAYOUT_DUMP_REQUEST"])
            request.with_suffix(request.suffix + ".error.txt").write_text("dump failed", encoding="utf-8")
            return _Process(running=False)
        return _Process(running=True)

    with pytest.raises(TimeoutError if outcome == "timeout" else RuntimeError):
        dump_tool.dump_layout(
            2, runtime, timeout=0.01, instances_root=instances,
            status_reader=lambda: {"active_count": 0, "active_launches": []},
            process_reader=list, process_factory=launch,
        )
    assert (save_root / "user1.dat").read_bytes() == b"original save bytes"
    assert not (save_root / "user2.dat").exists()
    backup = next((runtime / "map-layout" / "save-backups").glob("hk-ai-2-*"), None)
    assert backup is not None
    manifest = json.loads((backup / "sha256.json").read_text(encoding="utf-8"))
    assert manifest["user1.dat"]
