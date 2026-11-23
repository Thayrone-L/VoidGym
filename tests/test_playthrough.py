from __future__ import annotations

import hashlib
from pathlib import Path

from hk_agent.playthrough import (
    initialize_playthrough,
    persist_playthrough,
    playthrough_root,
    playthrough_status,
    restore_last_backup,
)


def _roots(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    runtime = tmp_path / "runtime"
    baseline = tmp_path / "development"
    personal = tmp_path / "personal"
    backups = tmp_path / "save-backups"
    install = tmp_path / "game"
    for path in (runtime, baseline, personal, backups, install):
        path.mkdir()
    return runtime, baseline, personal, backups, install


def test_playthrough_is_initialized_once_and_keeps_progress_out_of_baseline(tmp_path: Path) -> None:
    runtime, baseline, personal, backups, install = _roots(tmp_path)
    (baseline / "user1.dat").write_bytes(b"initial-save")
    root = playthrough_root(runtime, "explore-campaign", 1)
    initialize_playthrough(root, baseline, runtime_root=runtime, personal_root=personal,
                           backup_root=backups, game_install=install,
                           source="continuous_playthrough", save_slot=1)

    changed = root / "user1.dat"
    changed.write_bytes(b"campaign-progress")
    (baseline / "user1.dat").write_bytes(b"later-baseline-change")
    _, initialized_again = initialize_playthrough(
        root, baseline, runtime_root=runtime, personal_root=personal,
        backup_root=backups, game_install=install,
        source="continuous_playthrough", save_slot=1,
    )

    assert initialized_again is False
    assert changed.read_bytes() == b"campaign-progress"
    assert (baseline / "user1.dat").read_bytes() == b"later-baseline-change"
    assert not list(personal.iterdir())


def test_changed_save_gets_backup_and_last_good_can_restore_instance(tmp_path: Path) -> None:
    runtime, baseline, personal, backups, install = _roots(tmp_path)
    (baseline / "user1.dat").write_bytes(b"seed")
    root = playthrough_root(runtime, "explore-campaign", 2)
    initialize_playthrough(root, baseline, runtime_root=runtime, personal_root=personal,
                           backup_root=backups, game_install=install,
                           source="existing_save", save_slot=1)
    instance = tmp_path / "instance-data"
    instance.mkdir()
    (instance / "user1.dat").write_bytes(b"last-known-good")
    persist_playthrough(root, instance)
    (instance / "user1.dat").write_bytes(b"new-checkpoint")
    persist_playthrough(root, instance)
    assert len(list((root / "backups").glob("*/backup.json"))) == 2

    (instance / "user1.dat").write_bytes(b"corrupt")
    restored = restore_last_backup(root, instance_root=instance)
    assert restored.is_dir()
    assert (root / "user1.dat").read_bytes() == b"last-known-good"
    assert (instance / "user1.dat").read_bytes() == b"last-known-good"
    assert playthrough_status(root)["backup_count"] == 2


def test_playthrough_slot_separation_and_baseline_digest(tmp_path: Path) -> None:
    runtime, baseline, personal, backups, install = _roots(tmp_path)
    save = baseline / "user1.dat"
    save.write_bytes(b"unchanged")
    before = hashlib.sha256(save.read_bytes()).hexdigest()
    roots = [playthrough_root(runtime, "campaign", slot) for slot in (1, 2)]
    for root in roots:
        initialize_playthrough(root, baseline, runtime_root=runtime, personal_root=personal,
                               backup_root=backups, game_install=install,
                               source="continuous_playthrough", save_slot=1)
    (roots[0] / "user1.dat").write_bytes(b"slot-one")
    assert (roots[1] / "user1.dat").read_bytes() == b"unchanged"
    assert hashlib.sha256(save.read_bytes()).hexdigest() == before
