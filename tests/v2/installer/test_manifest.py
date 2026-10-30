from __future__ import annotations

import json
from pathlib import Path

import pytest

from voidgym.installer.manifest import ManifestError, load_manifest, validate_manifest


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "voidgym_version": "2.0.0.dev0",
        "python": {"version": "3.12.10", "url": "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe", "sha256": "a" * 64},
        "repo": {"url": "https://github.com/Thayrone-L/VoidGym", "ref": "v2.0.0-dev"},
        "bridge_dll": {"available": False, "url": None, "sha256": None},
        "modding_api": {"version": "77", "game_version": "1.5.78.11833", "url": "https://example.org/api.zip", "sha256": "b" * 64},
        "supported_games": [{"store": "gog", "version": "1.5.78.11833", "exe_sha256": "c" * 64, "globalgamemanagers_sha256": "d" * 64}],
        "map_art": {"url": None, "sha256": None, "credit": "Hollow Knight © Team Cherry", "requires_consent": True, "distribution_approved": False},
    }


def test_valid_manifest_allows_pending_optional_assets() -> None:
    assert validate_manifest(_manifest())["schema_version"] == 1


def test_missing_required_key_is_rejected() -> None:
    value = _manifest()
    del value["repo"]
    with pytest.raises(ManifestError, match="repo"):
        validate_manifest(value)


def test_bad_hash_is_rejected() -> None:
    value = _manifest()
    value["python"]["sha256"] = "not-a-hash"  # type: ignore[index]
    with pytest.raises(ManifestError, match="SHA-256"):
        validate_manifest(value)


def test_non_boolean_consent_flag_is_rejected() -> None:
    value = _manifest()
    value["map_art"]["distribution_approved"] = 0  # type: ignore[index]
    with pytest.raises(ManifestError, match="booleano"):
        validate_manifest(value)


def test_installable_manifest_requires_published_bridge() -> None:
    with pytest.raises(ManifestError, match="DLL do bridge"):
        validate_manifest(_manifest(), require_installable=True)


def test_load_manifest_reads_utf8_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    assert load_manifest(manifest_path)["schema_version"] == 1
