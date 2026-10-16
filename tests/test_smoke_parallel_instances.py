from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar

import pytest

from hk_agent.control_plane import CampaignSpec, ControlPlane
from tools.smoke_parallel_instances import (
    _campaign_payload,
    _is_save_file,
    _resource_summary,
    hash_tree,
    tree_changes,
    verify_distinct,
    verify_no_orphans,
    verify_root_isolation,
)


def test_tree_snapshot_detects_add_remove_and_change(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "same.txt").write_text("before", encoding="utf-8")
    (root / "removed.txt").write_text("gone", encoding="utf-8")
    before = hash_tree(root)
    (root / "same.txt").write_text("after", encoding="utf-8")
    (root / "removed.txt").unlink()
    (root / "added.txt").write_text("new", encoding="utf-8")

    assert tree_changes(before, hash_tree(root)) == {
        "added": ["added.txt"], "removed": ["removed.txt"], "changed": ["same.txt"]
    }


def test_tree_snapshot_requires_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        hash_tree(tmp_path / "missing")


def test_distinct_ids_and_ports_are_required() -> None:
    verify_distinct([10, 11], "ports")
    with pytest.raises(AssertionError, match="distinct"):
        verify_distinct([10, 10], "ports")


def test_root_isolation_allows_each_roots_own_changes(tmp_path: Path) -> None:
    before = {"slot0": {"user1.dat": "a"}, "slot1": {"user1.dat": "b"}}
    after = {"slot0": {"user1.dat": "changed-a"}, "slot1": {"user1.dat": "changed-b"}}
    verify_root_isolation(before, after, {"slot0": "slot0", "slot1": "slot1"})


def test_root_isolation_rejects_crossed_or_unowned_change() -> None:
    before = {"slot0": {"user1.dat": "a"}, "slot1": {"user1.dat": "b"}}
    after = {"slot0": {"user1.dat": "a"}, "slot1": {"user1.dat": "changed-b"}}
    with pytest.raises(AssertionError, match="outside its owning launch"):
        verify_root_isolation(before, after, {"slot0": "slot0", "slot1": "slot0"})


def test_root_isolation_rejects_unexpected_files() -> None:
    before = {"slot0": {"settings.json": "a"}}
    after = {"slot0": {"settings.json": "changed"}}
    with pytest.raises(AssertionError, match="non-save"):
        verify_root_isolation(before, after, {"slot0": "slot0"})


def test_orphan_detection_uses_fake_process_records() -> None:
    records = [
        {"pid": 101, "alive": False},
        {"pid": 202, "alive": True},
        {"pid": 303, "alive": True},
    ]
    verify_no_orphans(records, {101})
    with pytest.raises(AssertionError, match="orphan"):
        verify_no_orphans(records, {101, 202, 303})


def test_parallel_smoke_campaign_is_non_training_and_uses_fixed_probe_runner(tmp_path: Path) -> None:
    campaign = CampaignSpec.from_payload(_campaign_payload("parallel-smoke-test"))
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "run_scene_change_probe.py").write_text("", encoding="utf-8")
    command = ControlPlane(tmp_path).build_command(campaign, execute=False)

    assert campaign.campaign_type == "parallel_smoke"
    assert command[1].endswith("run_scene_change_probe.py")
    assert command[command.index("--bridge-smoke-seconds") + 1] == "90"
    assert command[command.index("--horizon") + 1] == "10000"
    assert "--continuous" not in command
    assert "--training-device" not in command


def test_only_expected_save_files_may_change_in_instance_data_root() -> None:
    assert _is_save_file("user1.dat")
    assert _is_save_file("user1.modded.json.bak")
    assert not _is_save_file("settings.json")


def test_resource_summary_reports_average_and_peak() -> None:
    samples = [
        {"instances": {"0": {"cpu_percent": 10, "ram_bytes": 100, "vram_mib": 5}},
         "total": {"cpu_percent": 10, "ram_bytes": 100, "vram_mib": 5}},
        {"instances": {"0": {"cpu_percent": 20, "ram_bytes": 200, "vram_mib": 9}},
         "total": {"cpu_percent": 20, "ram_bytes": 200, "vram_mib": 9}},
    ]

    summary = _resource_summary(samples)

    assert summary["0"]["cpu_percent"] == {"average": 15, "peak": 20}
    assert summary["total"]["vram_mib"] == {"average": 7.0, "peak": 9}


def test_smoke_fails_fast_when_runner_exits_before_hello(tmp_path: Path) -> None:
    import tools.smoke_parallel_instances as smoke

    log = tmp_path / "artifacts" / "process.log"
    log.parent.mkdir(parents=True)
    log.write_text("ValueError: horizon_steps must be in [1, 10000]\n", encoding="utf-8")

    class FakeControl:
        project_root = tmp_path
        _launches: ClassVar[dict[str, dict[str, object]]] = {
            "L1": {"status": "FAILED", "return_code": 1, "log_path": "artifacts/process.log"}
        }

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="horizon_steps must be in"):
        smoke._wait_for_evidence(tmp_path / "missing.json", 120, control=FakeControl(), launch_id="L1")
    assert time.monotonic() - started < 5


def test_smoke_campaign_horizon_is_accepted_by_the_probe_runner() -> None:
    from hk_agent.scene_change_probe import SceneChangeProbeConfig

    spec = CampaignSpec.from_payload(_campaign_payload("parallel-smoke-test"))
    SceneChangeProbeConfig(horizon_steps=spec.horizon).validate()


def _reg(sections: dict[str, str]) -> bytes:
    body = "Windows Registry Editor Version 5.00\r\n\r\n" + "".join(
        f"[{name}]\r\n{values}\r\n\r\n" for name, values in sections.items()
    )
    return "\ufeff".encode("utf-16-le") + body.encode("utf-16-le")


def test_registry_digest_ignores_instance_keys_but_detects_personal_changes() -> None:
    from tools.smoke_parallel_instances import protected_registry_digest

    base = r"HKEY_CURRENT_USER\Software\Team Cherry"
    before = _reg({base + r"\Hollow Knight": '"a"=dword:00000001',
                   base + r"\Hollow Knight AI 1": '"w"=dword:000003c0'})
    instance_changed = _reg({base + r"\Hollow Knight": '"a"=dword:00000001',
                             base + r"\Hollow Knight AI 1": '"w"=dword:000003b0'})
    personal_changed = _reg({base + r"\Hollow Knight": '"a"=dword:00000002',
                             base + r"\Hollow Knight AI 1": '"w"=dword:000003c0'})
    assert protected_registry_digest(before) == protected_registry_digest(instance_changed)
    assert protected_registry_digest(before) != protected_registry_digest(personal_changed)


def test_instance_runtime_files_are_allowed_only_in_their_own_root() -> None:
    before = {"slot0": {"user1.dat": "a"}, "slot1": {"user1.dat": "b"}}
    after = {
        "slot0": {"user1.dat": "a2", "ModLog.txt": "x", "Old ModLogs/ModLog 1.txt": "y",
                  "ModdingApi.GlobalSettings.json": "z"},
        "slot1": {"user1.dat": "b"},
    }
    verify_root_isolation(before, after, {"slot0": "slot0", "slot1": "slot1"})
    after["slot0"]["unexpected.exe"] = "w"
    with pytest.raises(AssertionError, match="unexpected non-save changes"):
        verify_root_isolation(before, after, {"slot0": "slot0", "slot1": "slot1"})


def test_runner_identity_accepts_direct_and_venv_redirector_child_only() -> None:
    from tools.smoke_parallel_instances import verify_runner_identity

    venv = r"C:\repo\.venv\Scripts\python.exe"
    base = r"C:\Python312\python.exe"
    launched = {"pid": 100, "creation_time": 5000, "executable": venv}
    assert verify_runner_identity(launched, dict(launched)) == "direct"
    child = {"pid": 200, "creation_time": 5001, "executable": base, "parent_pid": 100}
    assert verify_runner_identity(launched, child) == "venv-redirector"
    for bad in (
        {**child, "parent_pid": 999},  # not our child
        {**child, "creation_time": 4000},  # older than the launcher: PID reuse
        {**child, "executable": r"C:\evil\tool.exe"},  # not an interpreter
        {**child, "pid": 100},  # same PID but different identity
    ):
        with pytest.raises(AssertionError, match="runner identity differs"):
            verify_runner_identity(launched, bad)
    direct_python = {"pid": 100, "creation_time": 5000, "executable": base}
    with pytest.raises(AssertionError):
        verify_runner_identity(direct_python, {**child, "parent_pid": 100})


def test_frame_captures_are_checked_in_two_phases(tmp_path: Path) -> None:
    from tools.smoke_parallel_instances import verify_frame_captures

    initial = tmp_path / "initial.png"
    after = tmp_path / "after-action.png"
    evidence = {"frame_paths": [str(initial), str(after)]}
    with pytest.raises(AssertionError, match="initial"):
        verify_frame_captures(evidence, require_after_action=False)
    initial.write_bytes(b"png")
    verify_frame_captures(evidence, require_after_action=False)  # HELLO time: ok
    with pytest.raises(AssertionError, match="post-action"):
        verify_frame_captures(evidence, require_after_action=True)
    after.write_bytes(b"png")
    verify_frame_captures(evidence, require_after_action=True)
    with pytest.raises(AssertionError, match="does not declare"):
        verify_frame_captures({}, require_after_action=False)


def test_exited_process_is_reported_dead_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    import tools.smoke_parallel_instances as smoke

    def gone(pid: int) -> None:
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", gone)
    assert smoke._same_process_alive(8288, 1.0) is False
