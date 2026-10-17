from __future__ import annotations

import os
from pathlib import Path

import pytest

from hk_agent.runtime_paths import (
    capture_instance_saves,
    ensure_inside_runtime,
    prepare_instance_saves,
    validate_runtime_root,
)


def _validate(runtime: Path, root: Path) -> Path:
    return validate_runtime_root(
        runtime,
        personal=root / "personal",
        experimental=root / "experimental",
        backup=root / "backup",
        game_install=root / "game",
    )


@pytest.mark.parametrize(
    "runtime_parts",
    [
        ("experimental",),
        ("experimental", "runtime"),
        ("backup", "runtime"),
        ("personal", "runtime"),
        ("game", "runtime"),
    ],
)
def test_rejects_runtime_equal_to_or_inside_protected_root(
    tmp_path: Path, runtime_parts: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        _validate(tmp_path.joinpath(*runtime_parts), tmp_path)


@pytest.mark.parametrize("protected_name", ["experimental", "personal"])
def test_rejects_runtime_parent_of_protected_root(tmp_path: Path, protected_name: str) -> None:
    protected_parent = tmp_path / "protected"
    kwargs = {
        "personal": tmp_path / "personal",
        "experimental": tmp_path / "experimental",
        "backup": tmp_path / "backup",
        "game_install": tmp_path / "game",
    }
    kwargs["experimental" if protected_name == "experimental" else "personal"] = (
        protected_parent / protected_name
    )

    with pytest.raises(ValueError, match="must not overlap"):
        validate_runtime_root(protected_parent, **kwargs)


def test_accepts_separate_runtime_and_resolves_it(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"

    assert _validate(runtime, tmp_path) == runtime.resolve()


def test_rejects_symlink_in_runtime_path(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "runtime-link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(ValueError, match="symlinks or reparse points"):
        _validate(link / "launches", tmp_path)


def test_ensure_inside_runtime_rejects_escape(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"

    assert ensure_inside_runtime(runtime / "launches" / "one", runtime) == (
        runtime / "launches" / "one"
    ).resolve()
    with pytest.raises(ValueError, match="inside runtime root"):
        ensure_inside_runtime(tmp_path / "outside", runtime)


def test_instance_saves_are_preserved_seeded_and_captured(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    launch = runtime / "launches" / "launch-1"
    launch.mkdir(parents=True)
    instance = tmp_path / "LocalLow" / "Hollow Knight AI 1"
    baseline = tmp_path / "baseline"
    instance.mkdir(parents=True)
    baseline.mkdir()
    (instance / "user1.dat").write_bytes(b"previous")
    (baseline / "user1.dat").write_bytes(b"seed")

    previous, _ = prepare_instance_saves(instance, baseline, launch)

    assert (previous / "user1.dat").read_bytes() == b"previous"
    assert (instance / "user1.dat").read_bytes() == b"seed"
    (instance / "user1.dat").write_bytes(b"final")
    evidence = capture_instance_saves(instance, launch)
    assert (evidence / "user1.dat").read_bytes() == b"final"
    (instance / "user1.dat").write_bytes(b"later")
    assert capture_instance_saves(instance, launch) == evidence
    assert (evidence / "user1.dat").read_bytes() == b"final"


def test_stale_fence_refuses_prepare_and_capture_writes(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    launch = runtime / "launches" / "launch-1"
    launch.mkdir(parents=True)
    instance = tmp_path / "LocalLow" / "Hollow Knight AI 1"
    baseline = tmp_path / "baseline"
    instance.mkdir(parents=True)
    baseline.mkdir()
    (instance / "user1.dat").write_bytes(b"prior")
    (baseline / "user1.dat").write_bytes(b"seed")

    with pytest.raises(InterruptedError, match="fence"):
        prepare_instance_saves(
            instance, baseline, launch, fence_token="old",
            is_fence_current=lambda token: False,
        )
    assert (instance / "user1.dat").read_bytes() == b"prior"
    with pytest.raises(InterruptedError, match="fence"):
        capture_instance_saves(
            instance, launch, fence_token="old",
            is_fence_current=lambda token: False,
        )
    assert not (launch / "final-save").exists()


def test_instance_save_preparation_failure_keeps_previous_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = tmp_path / "runtime"
    launch = runtime / "launches" / "launch-1"
    launch.mkdir(parents=True)
    instance = tmp_path / "LocalLow" / "Hollow Knight AI 1"
    baseline = tmp_path / "baseline"
    instance.mkdir(parents=True)
    baseline.mkdir()
    (instance / "user1.dat").write_bytes(b"previous")
    (baseline / "user1.dat").write_bytes(b"seed")
    from hk_agent import runtime_paths

    def fail_copy(*args: object, **kwargs: object) -> None:
        raise OSError("simulated seed failure")

    monkeypatch.setattr(runtime_paths.shutil, "copy2", fail_copy)
    with pytest.raises(OSError, match="simulated seed failure"):
        prepare_instance_saves(instance, baseline, launch)
    assert (instance / "user1.dat").read_bytes() == b"previous"


@pytest.mark.skipif(os.name != "nt", reason="Windows case-insensitive behavior")
def test_runtime_comparison_is_case_insensitive_on_windows(tmp_path: Path) -> None:
    personal = tmp_path / "Personal"

    with pytest.raises(ValueError, match="must not overlap"):
        validate_runtime_root(
            tmp_path / "personal" / "runtime",
            personal=personal,
            experimental=tmp_path / "experimental",
            backup=tmp_path / "backup",
            game_install=tmp_path / "game",
        )



@pytest.mark.parametrize(
    "name",
    ["user1.dat", "USER2.DAT", "user1.dat.bak", "user1.dat.bak1", "user1.dat.bak12",
     "user1_1.5.78.11833.dat", "user1.modded.json", "user1.modded.json.bak",
     "user4.modded.json.bak3"],
)
def test_save_pattern_includes_game_rotating_backups(name: str) -> None:
    """Regression (soak): the game renames user1.dat to user1.dat.bak1 when saving;
    it must be moved, seeded and captured like any other save file."""
    from hk_agent.runtime_paths import is_save_file

    assert is_save_file(name)


@pytest.mark.parametrize(
    "name",
    ["ModLog.txt", "AppConfig.ini", "user1.txt", "user.dat", "settings.json",
     "user1.dat.old", "xuser1.dat", "user1.dat.bak1.tmp"],
)
def test_save_pattern_rejects_non_save_files(name: str) -> None:
    from hk_agent.runtime_paths import is_save_file

    assert not is_save_file(name)


def test_previous_state_moves_rotating_backups(tmp_path: Path) -> None:
    from hk_agent.runtime_paths import _save_files

    root = tmp_path / "instance"
    root.mkdir()
    for name in ("user1.dat", "user1.dat.bak1", "user1_1.5.78.11833.dat", "ModLog.txt"):
        (root / name).write_text("x", encoding="utf-8")
    assert sorted(p.name for p in _save_files(root)) == [
        "user1.dat", "user1.dat.bak1", "user1_1.5.78.11833.dat"
    ]
