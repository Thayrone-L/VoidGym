from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hk_agent.instance_install import (
    COMPANY_NAME,
    MARKER_NAME,
    MODDING_API_VERSION,
    TOOL_VERSION,
    InstanceMarkerMismatch,
    installation_hashes,
    sha256_file,
)


@dataclass(frozen=True)
class InstanceMarker:
    schema_version: int
    index: int
    product_name: str
    company_name: str
    persistent_data_path_expected: str
    registry_key_expected: str
    source_install: str
    exe_sha256: str
    globalgamemanagers_sha256: str
    bridge_dll_sha256: str
    modding_api_version: str
    tool_version: str
    created_unix_ns: int


_FIELDS = frozenset(InstanceMarker.__dataclass_fields__)


def load_instance_marker(install_root: Path, *, expected_index: int) -> InstanceMarker:
    marker_path = install_root / MARKER_NAME
    try:
        raw: Any = json.loads(marker_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) != _FIELDS:
            raise ValueError("marker fields do not match the closed schema")
        marker = InstanceMarker(**raw)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid instance marker: {marker_path}: {error}") from error
    index = expected_index
    if marker.index != index:
        raise InstanceMarkerMismatch(index, f"Instance {index} marker records index {marker.index}")
    if marker.schema_version != 1:
        raise ValueError("Instance marker schema version or index does not match")
    if marker.product_name != f"Hollow Knight AI {index}" or marker.company_name != COMPANY_NAME:
        raise ValueError("Instance marker Unity identity does not match its index")
    if marker.tool_version != TOOL_VERSION or marker.modding_api_version != MODDING_API_VERSION:
        raise ValueError("Instance marker tool or Modding API version is unsupported")
    if not marker.persistent_data_path_expected.rstrip("\\/").lower().endswith(
        rf"team cherry\hollow knight ai {index}".lower()
    ):
        raise ValueError("Instance marker persistent data path does not match its index")
    expected_registry = rf"HKCU\Software\{COMPANY_NAME}\{marker.product_name}"
    if marker.registry_key_expected.lower() != expected_registry.lower():
        raise ValueError("Instance marker registry key does not match its Unity identity")
    hashes = installation_hashes(install_root)
    bridge_path = install_root / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge" / "HkAgentBridge.dll"
    if marker.exe_sha256 != hashes.executable:
        raise ValueError("Instance executable hash differs from marker")
    if marker.globalgamemanagers_sha256 != hashes.globalgamemanagers:
        raise ValueError("Instance globalgamemanagers hash differs from marker")
    if not bridge_path.is_file() or marker.bridge_dll_sha256 != sha256_file(bridge_path):
        raise ValueError("Instance bridge DLL hash differs from marker")
    return marker
