from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest
from typing import Self

from voidgym.installer.manifest import ManifestError
from voidgym.installer.steps import safe_download


class _Response(io.BytesIO):
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_download_verifies_hash_then_atomically_publishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"pinned asset bytes"
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: _Response(payload))
    destination = tmp_path / "cache" / "asset.zip"

    assert safe_download("https://example.org/asset.zip", hashlib.sha256(payload).hexdigest(), destination) == destination
    assert destination.read_bytes() == payload
    assert list(destination.parent.iterdir()) == [destination]


def test_download_hash_mismatch_leaves_no_partial_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: _Response(b"wrong bytes"))
    destination = tmp_path / "cache" / "asset.zip"

    with pytest.raises(ManifestError, match="SHA-256 divergente"):
        safe_download("https://example.org/asset.zip", "a" * 64, destination)
    assert not destination.exists()
    assert not list(destination.parent.iterdir())


def test_download_refuses_non_https(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="HTTPS"):
        safe_download("http://example.org/asset.zip", "a" * 64, tmp_path / "asset.zip")
