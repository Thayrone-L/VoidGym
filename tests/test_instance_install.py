from __future__ import annotations

import json
import os
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from hk_agent import instance_install
from hk_agent.instance_install import (
    COMPANY_NAME,
    MARKER_NAME,
    InstanceInstallError,
    provision_instance,
    validate_destination,
    validate_source_tree,
)
from hk_agent.instance_marker import load_instance_marker
from tools.verify_instance_isolation import (
    _registry_section,
    pids_for_window_records,
    snapshot_changes,
)


class FakeProductNameEditor:
    def read(self, path: Path) -> tuple[str, str]:
        record = json.loads(path.read_text(encoding="utf-8"))
        return record["productName"], record["companyName"]

    def write(self, path: Path, product_name: str) -> None:
        record = json.loads(path.read_text(encoding="utf-8"))
        record["productName"] = product_name
        path.write_text(json.dumps(record), encoding="utf-8")


@pytest.fixture(autouse=True)
def avoid_host_onedrive_overlap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OneDrive", str(tmp_path.parent / "not-onedrive"))


def make_install(root: Path, *, api: bool = True) -> Path:
    install = root / "original"
    (install / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge").mkdir(
        parents=True
    )
    (install / "Hollow Knight.exe").write_bytes(b"game executable")
    (install / "Hollow Knight_Data" / "globalgamemanagers").write_text(
        json.dumps({"productName": "Hollow Knight", "companyName": COMPANY_NAME}),
        encoding="utf-8",
    )
    mods = install / "Hollow Knight_Data" / "Managed" / "Mods"
    (mods / "HkAgentBridge" / "HkAgentBridge.dll").write_bytes(b"bridge")
    if api:
        managed = install / "Hollow Knight_Data" / "Managed"
        for name in ("MMHOOK_Assembly-CSharp.dll", "MMHOOK_PlayMaker.dll", "MonoMod.RuntimeDetour.dll"):
            (managed / name).write_bytes(b"api")
    return install


def make_api_archive(root: Path) -> Path:
    archive = root / "moddingapi.v77.windows.zip"
    with zipfile.ZipFile(archive, "w") as package:
        for name in (
            "MMHOOK_Assembly-CSharp.dll",
            "MMHOOK_PlayMaker.dll",
            "MonoMod.RuntimeDetour.dll",
            "MonoMod.Utils.dll",
        ):
            package.writestr(name, b"api-v77")
    return archive


def provision(root: Path, *, index: int = 1, **kwargs: object) -> dict[str, object]:
    source = root / "original"
    if not source.exists():
        source = make_install(root)
    instances = root / "safe" / "instances"
    return provision_instance(
        index,
        source=source,
        instances_root=instances,
        workspace=root / "workspace",
        editor=FakeProductNameEditor(),
        free_bytes=lambda _: 20 * 1024**3,
        api_archive=make_api_archive(root),
        **kwargs,
    )


@pytest.mark.parametrize("protected_name", ["OneDrive", "runtime", "saves"])
def test_refuses_destination_inside_protected_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protected_name: str
) -> None:
    import hk_agent.instance_install as module

    protected = tmp_path / protected_name
    instances = protected / "instances"
    if protected_name == "OneDrive":
        monkeypatch.setenv("OneDrive", str(protected))
    elif protected_name == "runtime":
        monkeypatch.setattr(module, "RUNTIME_ROOT", protected)
    else:
        monkeypatch.setattr(module, "SAVES_ROOT", protected)
    with pytest.raises(InstanceInstallError, match=protected_name):
        validate_destination(
            instances / "hk-ai-1", workspace=tmp_path / "workspace", instances_root=instances
        )


def test_refuses_unmanaged_destination(tmp_path: Path) -> None:
    source = make_install(tmp_path)
    instances = tmp_path / "safe" / "instances"
    destination = instances / "hk-ai-1"
    destination.mkdir(parents=True)
    (destination / "user.dat").write_bytes(b"keep")
    with pytest.raises(InstanceInstallError, match="unmanaged"):
        provision_instance(
            1, source=source, instances_root=instances, workspace=tmp_path / "workspace",
            editor=FakeProductNameEditor(), free_bytes=lambda _: 20 * 1024**3,
        )


def test_rejects_source_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = make_install(tmp_path)
    target = source / "linked-file"
    target.write_bytes(b"pretend symbolic link")
    real_lstat = os.lstat

    def fake_lstat(path: os.PathLike[str] | str) -> os.stat_result:
        if Path(path) == target:
            return SimpleNamespace(
                st_mode=stat.S_IFLNK | 0o777, st_file_attributes=0
            )  # type: ignore[return-value]
        return real_lstat(path)

    monkeypatch.setattr(instance_install.os, "lstat", fake_lstat)
    with pytest.raises(InstanceInstallError, match="symlink/reparse"):
        validate_source_tree(source)


def test_writes_complete_marker_and_renames_synthetic_settings(tmp_path: Path) -> None:
    marker = provision(tmp_path)
    destination = tmp_path / "safe" / "instances" / "hk-ai-1"
    assert marker["product_name"] == "Hollow Knight AI 1"
    assert marker["company_name"] == COMPANY_NAME
    assert marker["schema_version"] == 1
    assert marker["persistent_data_path_expected"].endswith("Hollow Knight AI 1")
    assert marker["registry_key_expected"].endswith("Hollow Knight AI 1")
    assert json.loads((destination / MARKER_NAME).read_text(encoding="utf-8"))["index"] == 1
    assert (destination / "Hollow Knight_Data" / "globalgamemanagers.orig").exists()


def test_resync_requires_valid_marker(tmp_path: Path) -> None:
    provision(tmp_path)
    source = tmp_path / "original"
    instances = tmp_path / "safe" / "instances"
    destination = instances / "hk-ai-1"
    (destination / MARKER_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(InstanceInstallError, match="marker is invalid"):
        provision_instance(
            1, source=source, instances_root=instances, workspace=tmp_path / "workspace",
            resync=True, editor=FakeProductNameEditor(), free_bytes=lambda _: 20 * 1024**3,
        )


def test_resync_requires_flag(tmp_path: Path) -> None:
    provision(tmp_path)
    with pytest.raises(InstanceInstallError, match="--resync"):
        provision(tmp_path)


def test_snapshot_comparison_detects_added_changed_removed() -> None:
    result = snapshot_changes({"a": "one", "b": "old"}, {"b": "new", "c": "three"})
    assert result == {"added": ["c"], "removed": ["a"], "changed": ["b"]}


def test_window_selection_filters_by_launched_pid() -> None:
    assert pids_for_window_records([(10, 5), (11, 8), (12, 5)], 5) == [10, 12]


def test_api_installs_into_managed_from_local_package(tmp_path: Path) -> None:
    source = make_install(tmp_path, api=False)
    instances = tmp_path / "safe" / "instances"
    marker = provision_instance(
        1, source=source, instances_root=instances, workspace=tmp_path / "workspace",
        sync_mods=True, bridge_dll=source / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge" / "HkAgentBridge.dll",
        api_archive=make_api_archive(tmp_path), editor=FakeProductNameEditor(),
        free_bytes=lambda _: 20 * 1024**3,
    )
    assert marker["modding_api_version"] == "1.5.78.11833-77"
    assert (instances / "hk-ai-1" / "Hollow Knight_Data" / "Managed" / "MMHOOK_Assembly-CSharp.dll").is_file()
    assert (instances / "hk-ai-1" / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge" / "HkAgentBridge.dll").is_file()


def test_sync_mods_requires_explicit_bridge_build(tmp_path: Path) -> None:
    with pytest.raises(InstanceInstallError, match="requires --bridge-dll"):
        provision(tmp_path, sync_mods=True)


def test_deploy_bridge_backups_copies_updates_marker_and_validates(tmp_path: Path) -> None:
    from tools.deploy_bridge import deploy

    source = make_install(tmp_path)
    source_bridge = source / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge" / "HkAgentBridge.dll"
    provision(tmp_path, sync_mods=True, bridge_dll=source_bridge)
    build = tmp_path / "release" / "HkAgentBridge.dll"
    build.parent.mkdir()
    build.write_bytes(b"release bridge")
    instances_root = tmp_path / "safe" / "instances"
    target = instances_root / "hk-ai-1" / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge" / "HkAgentBridge.dll"
    old_dll = target.read_bytes()
    old_marker = (instances_root / "hk-ai-1" / MARKER_NAME).read_bytes()

    result = deploy(build, instance_indices=[1], instances_root=instances_root,
                    backup_root=tmp_path / "backups")[0]

    assert target.read_bytes() == build.read_bytes()
    assert load_instance_marker(instances_root / "hk-ai-1", expected_index=1).bridge_dll_sha256 == result["sha256"]
    backup = Path(str(result["backup"]))
    assert (backup / target.name).read_bytes() == old_dll
    assert (backup / MARKER_NAME).read_bytes() == old_marker


def test_bridge_marker_refresh_requires_and_records_exact_installed_hash(tmp_path: Path) -> None:
    source = make_install(tmp_path)
    bridge = source / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge" / "HkAgentBridge.dll"
    provision(tmp_path, sync_mods=True, bridge_dll=bridge)
    root = tmp_path / "safe" / "instances" / "hk-ai-1"
    installed = root / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge" / "HkAgentBridge.dll"
    installed.write_bytes(b"new verified bridge build")
    from hk_agent.instance_install import refresh_instance_bridge_marker, sha256_file

    original_created_unix_ns = json.loads((root / MARKER_NAME).read_text(encoding="utf-8"))["created_unix_ns"]
    expected = sha256_file(installed)
    marker = refresh_instance_bridge_marker(root, expected_index=1, expected_bridge_hash=expected)
    assert marker["bridge_dll_sha256"] == expected
    assert load_instance_marker(root, expected_index=1).bridge_dll_sha256 == expected
    assert marker["created_unix_ns"] == original_created_unix_ns


def test_multi_instance_deployment_rolls_back_every_instance_on_later_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import psutil
    import tools.deploy_bridge as deployment

    source = make_install(tmp_path)
    source_bridge = source / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge" / "HkAgentBridge.dll"
    provision(tmp_path, index=1, sync_mods=True, bridge_dll=source_bridge)
    provision(tmp_path, index=2, sync_mods=True, bridge_dll=source_bridge)
    build = tmp_path / "release" / "HkAgentBridge.dll"
    build.parent.mkdir()
    build.write_bytes(b"transaction build")
    instances_root = tmp_path / "safe" / "instances"
    originals: dict[int, tuple[bytes, bytes]] = {}
    for index in (1, 2):
        root = instances_root / f"hk-ai-{index}"
        dll = root / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge" / "HkAgentBridge.dll"
        marker = root / MARKER_NAME
        originals[index] = (dll.read_bytes(), marker.read_bytes())

    monkeypatch.setattr(psutil, "process_iter", lambda *_args, **_kwargs: [])
    real_refresh = deployment.refresh_instance_bridge_marker

    def fail_second(root: Path, *, expected_index: int, expected_bridge_hash: str | None = None):
        if expected_index == 2:
            raise InstanceInstallError("injected instance 2 failure")
        return real_refresh(root, expected_index=expected_index, expected_bridge_hash=expected_bridge_hash)

    monkeypatch.setattr(deployment, "refresh_instance_bridge_marker", fail_second)
    with pytest.raises(InstanceInstallError, match="all touched instances were restored"):
        deployment.deploy(build, instance_indices=[1, 2], instances_root=instances_root,
                          backup_root=tmp_path / "backups")
    for index in (1, 2):
        root = instances_root / f"hk-ai-{index}"
        dll = root / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge" / "HkAgentBridge.dll"
        marker = root / MARKER_NAME
        assert (dll.read_bytes(), marker.read_bytes()) == originals[index]


def test_multi_instance_deployment_refuses_running_game_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import psutil
    from tools.deploy_bridge import deploy

    source = make_install(tmp_path)
    source_bridge = source / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge" / "HkAgentBridge.dll"
    provision(tmp_path, index=1, sync_mods=True, bridge_dll=source_bridge)
    build = tmp_path / "release" / "HkAgentBridge.dll"
    build.parent.mkdir()
    build.write_bytes(b"deployment must not happen")
    instances_root = tmp_path / "safe" / "instances"
    root = instances_root / "hk-ai-1"
    dll = root / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge" / "HkAgentBridge.dll"
    original = dll.read_bytes()
    game = root / "Hollow Knight.exe"
    monkeypatch.setattr(psutil, "process_iter", lambda *_args, **_kwargs: [
        SimpleNamespace(info={"name": "Hollow Knight.exe", "exe": str(game)})
    ])
    from tools.deploy_bridge import InstanceInstallError

    with pytest.raises(InstanceInstallError, match="game is running"):
        deploy(build, instance_indices=[1], instances_root=instances_root,
               backup_root=tmp_path / "backups")
    assert dll.read_bytes() == original


def test_bridge_marker_refresh_rejects_wrong_build_without_mutating_marker(tmp_path: Path) -> None:
    source = make_install(tmp_path)
    bridge = source / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge" / "HkAgentBridge.dll"
    provision(tmp_path, sync_mods=True, bridge_dll=bridge)
    root = tmp_path / "safe" / "instances" / "hk-ai-1"
    marker_path = root / MARKER_NAME
    before = marker_path.read_bytes()
    from hk_agent.instance_install import refresh_instance_bridge_marker

    with pytest.raises(InstanceInstallError, match="does not match the requested build"):
        refresh_instance_bridge_marker(root, expected_index=1, expected_bridge_hash="0" * 64)
    assert marker_path.read_bytes() == before


def test_marker_loader_checks_digest_and_identity(tmp_path: Path) -> None:
    provision(tmp_path, sync_mods=True, bridge_dll=make_install(tmp_path) / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge" / "HkAgentBridge.dll")
    root = tmp_path / "safe" / "instances" / "hk-ai-1"
    assert load_instance_marker(root, expected_index=1).index == 1
    marker_path = root / MARKER_NAME
    data = json.loads(marker_path.read_text(encoding="utf-8"))
    data["bridge_dll_sha256"] = "0" * 64
    marker_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="bridge DLL hash"):
        load_instance_marker(root, expected_index=1)


@pytest.mark.parametrize("mutation", ["extra", "missing", "index", "product", "exe", "ggm"])
def test_marker_loader_rejects_invalid_schema_identity_or_hash(tmp_path: Path, mutation: str) -> None:
    source = make_install(tmp_path)
    bridge = source / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge" / "HkAgentBridge.dll"
    provision(tmp_path, sync_mods=True, bridge_dll=bridge)
    root = tmp_path / "safe" / "instances" / "hk-ai-1"
    marker_path = root / MARKER_NAME
    data = json.loads(marker_path.read_text(encoding="utf-8"))
    if mutation == "extra":
        data["unexpected"] = True
    elif mutation == "missing":
        del data["company_name"]
    elif mutation == "index":
        data["index"] = 2
    elif mutation == "product":
        data["product_name"] = "Hollow Knight AI 2"
    elif mutation == "exe":
        data["exe_sha256"] = "0" * 64
    else:
        data["globalgamemanagers_sha256"] = "0" * 64
    marker_path.write_text(json.dumps(data), encoding="utf-8")
    if mutation == "index":
        from hk_agent.instance_install import InstanceMarkerMismatch

        with pytest.raises(InstanceMarkerMismatch) as error:
            load_instance_marker(root, expected_index=1)
        assert error.value.instance_index == 1
    else:
        with pytest.raises(ValueError):
            load_instance_marker(root, expected_index=1)


def test_registry_parser_normalizes_hives_and_sorts_values(tmp_path: Path) -> None:
    short = tmp_path / "short.reg"
    long = tmp_path / "long.reg"
    short.write_text('Windows Registry Editor Version 5.00\n\n[HKCU\\Software\\Team Cherry\\Hollow Knight]\n"z"="2"\n"a"="1"\n', encoding="utf-16")
    long.write_text('Windows Registry Editor Version 5.00\n\n[HKEY_CURRENT_USER\\Software\\Team Cherry\\Hollow Knight]\n"a"="1"\n"z"="2"\n', encoding="utf-16")
    assert _registry_section(short, r"HKCU\Software\Team Cherry\Hollow Knight") == _registry_section(long, r"HKEY_CURRENT_USER\Software\Team Cherry\Hollow Knight")
    long.write_text(long.read_text(encoding="utf-16").replace('"z"="2"', '"z"="changed"'), encoding="utf-16")
    assert _registry_section(short, r"HKCU\Software\Team Cherry\Hollow Knight") != _registry_section(long, r"HKCU\Software\Team Cherry\Hollow Knight")


def test_registry_parser_fails_when_section_is_absent(tmp_path: Path) -> None:
    export = tmp_path / "missing.reg"
    export.write_text("Windows Registry Editor Version 5.00\n", encoding="utf-16")
    with pytest.raises(InstanceInstallError, match="missing required section"):
        _registry_section(export, r"HKCU\Software\Team Cherry\Hollow Knight")
