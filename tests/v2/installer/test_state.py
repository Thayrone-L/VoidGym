from __future__ import annotations

import json
from pathlib import Path

from voidgym.installer.state import InstallState, safe_uninstall_path, safe_uninstall_registry_key


def test_install_state_round_trips_records(tmp_path: Path) -> None:
    state_path = tmp_path / "VoidGym" / "install-state.json"
    state = InstallState(state_path)
    created = tmp_path / "VoidGym" / "instances" / "hk-ai-1"
    state.record_path(created)
    state.record_shortcut(tmp_path / "Startup" / "VoidGym.lnk")
    state.complete_step("hardware")

    restored = InstallState(state_path)
    assert restored.data["created_paths"] == [str(created.resolve())]
    assert restored.data["shortcuts"] == [str((tmp_path / "Startup" / "VoidGym.lnk").resolve())]
    assert restored.data["completed_steps"] == ["hardware"]
    assert json.loads(state_path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_uninstall_path_is_limited_to_owned_roots(tmp_path: Path) -> None:
    app_root = tmp_path / "VoidGym"
    local_low = tmp_path / "LocalLow" / "Team Cherry"
    assert safe_uninstall_path(app_root / "instances" / "hk-ai-1", app_root=app_root, local_low=local_low)
    assert safe_uninstall_path(local_low / "Hollow Knight AI 1", app_root=app_root, local_low=local_low)
    assert not safe_uninstall_path(tmp_path / "personal-save", app_root=app_root, local_low=local_low)
    assert not safe_uninstall_path(local_low / "Hollow Knight", app_root=app_root, local_low=local_low)
    assert not safe_uninstall_path(app_root, app_root=app_root, local_low=local_low)


def test_registry_uninstall_only_accepts_instance_keys() -> None:
    assert safe_uninstall_registry_key(r"HKCU\Software\Team Cherry\Hollow Knight AI 1")
    assert safe_uninstall_registry_key(r"Software\Team Cherry\Hollow Knight AI 2\Settings")
    assert not safe_uninstall_registry_key(r"HKCU\Software\Team Cherry\Hollow Knight")
    assert not safe_uninstall_registry_key(r"HKCU\Software\Other\Hollow Knight AI 1")
    assert not safe_uninstall_registry_key(r"HKCU\Software\Team Cherry\Hollow Knight AI 1\..\Hollow Knight")
