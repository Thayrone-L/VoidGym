from __future__ import annotations

import json
import os
import shutil
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "tools" / "start-isolated-hollow-knight.ps1"
RECOVERY = PROJECT_ROOT / "tools" / "restore-save-isolation.ps1"
MODULE = PROJECT_ROOT / "tools" / "HkSaveIsolation.psm1"
POWERSHELL = (
    Path(os.environ.get("SystemRoot", r"C:\Windows"))
    / "System32"
    / "WindowsPowerShell"
    / "v1.0"
    / "powershell.exe"
)


def _run_script(script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(POWERSHELL),
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _inventory(root: Path) -> list[dict[str, object]]:
    return [
        {
            "relative_path": str(path.relative_to(root)).replace("/", "\\"),
            "size": path.stat().st_size,
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


@pytest.mark.skipif(os.name != "nt", reason="Windows process inspection")
def test_guard_matches_exact_executable_path_not_only_process_name(tmp_path: Path) -> None:
    system_ping = Path(os.environ["SystemRoot"]) / "System32" / "PING.EXE"
    copied_ping = tmp_path / "same-name" / "PING.EXE"
    copied_ping.parent.mkdir()
    shutil.copy2(system_ping, copied_ping)
    guard = subprocess.Popen(
        [str(system_ping), "127.0.0.1", "-n", "30"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "HK_TEST_MODULE": str(MODULE),
            "HK_TEST_RUNNING": str(system_ping),
            "HK_TEST_SAME_NAME": str(copied_ping),
        }
    )
    try:
        result = subprocess.run(
            [
                str(POWERSHELL),
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                (
                    "Import-Module $env:HK_TEST_MODULE -Force; "
                    "$exact=Test-HkExecutableProcessRunning $env:HK_TEST_RUNNING; "
                    "$same=Test-HkExecutableProcessRunning $env:HK_TEST_SAME_NAME; "
                    "if($exact -and -not $same){exit 0}else{exit 9}"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=10,
        )
    finally:
        guard.terminate()
        guard.wait(timeout=5)
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.skipif(os.name != "nt", reason="Windows process inspection")
def test_guard_fails_closed_for_same_name_process_with_unknown_path(tmp_path: Path) -> None:
    guard = tmp_path / "guard.exe"
    guard.touch()
    environment = os.environ.copy()
    environment.update({"HK_TEST_MODULE": str(MODULE), "HK_TEST_GUARD": str(guard)})
    result = subprocess.run(
        [
            str(POWERSHELL),
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                "function global:Get-CimInstance { "
                "[pscustomobject]@{Name='guard.exe';ExecutablePath=$null} }; "
                "Import-Module $env:HK_TEST_MODULE -Force; "
                "try { Test-HkExecutableProcessRunning $env:HK_TEST_GUARD | Out-Null; exit 9 } "
                "catch { if($_.Exception.Message -like '*Unable to verify*'){exit 0}else{exit 8} }"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.skipif(os.name != "nt", reason="Windows junction behavior")
def test_launcher_round_trip_preserves_personal_saves(tmp_path: Path) -> None:
    personal = tmp_path / "local-low" / "Team Cherry" / "Hollow Knight"
    experimental = tmp_path / "experimental"
    backups = tmp_path / "backups"
    personal.mkdir(parents=True)
    experimental.mkdir()
    (personal / "user1.dat").write_bytes(b"personal-save")
    (personal / "nested").mkdir()
    (personal / "nested" / "settings.json").write_text("personal", encoding="utf-8")
    (experimental / "user1.dat").write_bytes(b"experimental-save")
    before = _inventory(personal)

    result = _run_script(
        LAUNCHER,
        "-GameExecutable",
        str(Path(os.environ["SystemRoot"]) / "System32" / "where.exe"),
        "-PersonalRoot",
        str(personal),
        "-ExperimentalRoot",
        str(experimental),
        "-BackupRoot",
        str(backups),
    )

    assert result.returncode == 0, result.stderr
    assert _inventory(personal) == before
    assert not personal.is_symlink()
    assert (experimental / "user1.dat").read_bytes() == b"experimental-save"
    manifests = list(backups.glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8-sig"))
    assert manifest["status"] == "COMPLETED"
    assert _inventory(Path(manifest["validated_backup_root"])) == before
    assert not Path(manifest["preserved_personal_root"]).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction behavior")
def test_launcher_rejects_nested_roots_without_changes(tmp_path: Path) -> None:
    personal = tmp_path / "personal"
    personal.mkdir()
    save = personal / "user1.dat"
    save.write_bytes(b"untouched")

    result = _run_script(
        LAUNCHER,
        "-GameExecutable",
        str(Path(os.environ["SystemRoot"]) / "System32" / "where.exe"),
        "-PersonalRoot",
        str(personal),
        "-ExperimentalRoot",
        str(personal / "experimental"),
        "-BackupRoot",
        str(tmp_path / "backups"),
    )

    assert result.returncode != 0
    assert save.read_bytes() == b"untouched"
    assert not (tmp_path / "backups").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction behavior")
def test_launcher_rejects_experimental_junction_without_changes(tmp_path: Path) -> None:
    personal = tmp_path / "personal"
    personal.mkdir()
    save = personal / "user1.dat"
    save.write_bytes(b"untouched")
    experimental_link = tmp_path / "experimental-link"
    junction_environment = os.environ.copy()
    junction_environment["HK_TEST_LINK"] = str(experimental_link)
    junction_environment["HK_TEST_TARGET"] = str(personal)
    subprocess.run(
        [
            str(POWERSHELL),
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                "New-Item -ItemType Junction -Path $env:HK_TEST_LINK "
                "-Target $env:HK_TEST_TARGET | Out-Null"
            ),
        ],
        check=True,
        env=junction_environment,
    )

    result = _run_script(
        LAUNCHER,
        "-GameExecutable",
        str(Path(os.environ["SystemRoot"]) / "System32" / "where.exe"),
        "-PersonalRoot",
        str(personal),
        "-ExperimentalRoot",
        str(experimental_link),
        "-BackupRoot",
        str(tmp_path / "backups"),
    )

    assert result.returncode != 0
    assert "reparse point" in result.stderr
    assert save.read_bytes() == b"untouched"
    assert not (tmp_path / "backups").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction behavior")
def test_recovery_removes_only_expected_junction(tmp_path: Path) -> None:
    personal = tmp_path / "active" / "Hollow Knight"
    personal.parent.mkdir()
    experimental = tmp_path / "experimental"
    experimental.mkdir()
    backup_root = tmp_path / "backups"
    transaction = backup_root / "hk-save-test"
    validated = transaction / "validated-backup"
    preserved = personal.parent / ".hk-save-test.preserved"
    transaction.mkdir(parents=True)
    validated.mkdir()
    preserved.mkdir()
    (preserved / "user1.dat").write_bytes(b"personal-save")
    (validated / "user1.dat").write_bytes(b"personal-save")
    (experimental / "user1.dat").write_bytes(b"experiment")
    junction_environment = os.environ.copy()
    junction_environment["HK_TEST_LINK"] = str(personal)
    junction_environment["HK_TEST_TARGET"] = str(experimental)
    subprocess.run(
        [
            str(POWERSHELL),
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                "New-Item -ItemType Junction -Path $env:HK_TEST_LINK "
                "-Target $env:HK_TEST_TARGET | Out-Null"
            ),
        ],
        check=True,
        env=junction_environment,
    )
    manifest_path = transaction / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "transaction_id": "hk-save-test",
                "status": "ACTIVE",
                "game_executable": str(Path(os.environ["SystemRoot"]) / "System32" / "where.exe"),
                "personal_root": str(personal),
                "experimental_root": str(experimental),
                "backup_root": str(backup_root),
                "preserved_personal_root": str(preserved),
                "validated_backup_root": str(validated),
                "personal_root_existed": True,
                "personal_inventory": _inventory(preserved),
            }
        ),
        encoding="utf-8",
    )

    result = _run_script(
        RECOVERY,
        "-ManifestPath",
        str(manifest_path),
        "-BackupRoot",
        str(backup_root),
    )

    assert result.returncode == 0, result.stderr
    assert _inventory(personal) == _inventory(validated)
    assert not preserved.exists()
    assert (experimental / "user1.dat").read_bytes() == b"experiment"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    assert manifest["status"] == "COMPLETED"


@pytest.mark.skipif(os.name != "nt", reason="Windows junction behavior")
def test_recovery_refuses_while_separate_guard_process_is_running(tmp_path: Path) -> None:
    personal = tmp_path / "active" / "Hollow Knight"
    personal.parent.mkdir()
    experimental = tmp_path / "experimental"
    experimental.mkdir()
    backup_root = tmp_path / "backups"
    transaction = backup_root / "hk-save-guard-test"
    validated = transaction / "validated-backup"
    preserved = personal.parent / ".hk-save-guard-test.preserved"
    transaction.mkdir(parents=True)
    validated.mkdir()
    preserved.mkdir()
    (preserved / "user1.dat").write_bytes(b"personal-save")
    (validated / "user1.dat").write_bytes(b"personal-save")
    guard_executable = Path(os.environ["SystemRoot"]) / "System32" / "PING.EXE"
    junction_environment = os.environ.copy()
    junction_environment["HK_TEST_LINK"] = str(personal)
    junction_environment["HK_TEST_TARGET"] = str(experimental)
    subprocess.run(
        [
            str(POWERSHELL),
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                "New-Item -ItemType Junction -Path $env:HK_TEST_LINK "
                "-Target $env:HK_TEST_TARGET | Out-Null"
            ),
        ],
        check=True,
        env=junction_environment,
    )
    manifest_path = transaction / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "transaction_id": "hk-save-guard-test",
                "status": "ACTIVE",
                "game_executable": str(Path(os.environ["SystemRoot"]) / "System32" / "where.exe"),
                "guard_process_executable": str(guard_executable),
                "guard_process_name": "PING",
                "personal_root": str(personal),
                "experimental_root": str(experimental),
                "backup_root": str(backup_root),
                "preserved_personal_root": str(preserved),
                "validated_backup_root": str(validated),
                "personal_root_existed": True,
                "personal_inventory": _inventory(preserved),
            }
        ),
        encoding="utf-8",
    )

    guard = subprocess.Popen(
        [str(guard_executable), "127.0.0.1", "-n", "30"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        result = _run_script(
            RECOVERY,
            "-ManifestPath",
            str(manifest_path),
            "-BackupRoot",
            str(backup_root),
        )
    finally:
        guard.terminate()
        guard.wait(timeout=5)

    assert result.returncode != 0
    assert "game process is running" in result.stderr
    assert personal.is_dir()
    assert preserved.is_dir()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "ACTIVE"

    recovered = _run_script(
        RECOVERY,
        "-ManifestPath",
        str(manifest_path),
        "-BackupRoot",
        str(backup_root),
    )
    assert recovered.returncode == 0, recovered.stderr
    assert _inventory(personal) == _inventory(validated)
    assert not preserved.exists()
    completed_manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    assert completed_manifest["status"] == "COMPLETED"
