from __future__ import annotations

from pathlib import Path

import pytest

from hk_agent.provenance import fingerprint_directory


def test_directory_fingerprint_is_stable_and_changes_with_content(tmp_path: Path) -> None:
    root = tmp_path / "save"
    root.mkdir()
    (root / "user1.dat").write_bytes(b"save-a")
    (root / "settings.json").write_text("{}", encoding="utf-8")
    first = fingerprint_directory(root)
    second = fingerprint_directory(root)
    assert first == second
    assert first["file_count"] == 2
    assert len(str(first["tree_sha256"])) == 64
    (root / "user1.dat").write_bytes(b"save-b")
    assert fingerprint_directory(root)["tree_sha256"] != first["tree_sha256"]


def test_directory_fingerprint_is_bounded(tmp_path: Path) -> None:
    root = tmp_path / "save"
    root.mkdir()
    (root / "large.dat").write_bytes(b"1234")
    with pytest.raises(ValueError, match="bounded inventory"):
        fingerprint_directory(root, maximum_bytes=3)
