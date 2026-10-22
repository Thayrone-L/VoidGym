from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import stat
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

ORIGINAL_INSTALL = Path(r"C:\GOG Games\Hollow Knight")
INSTANCES_ROOT = Path(r"C:\Users\thayr\Games\HollowKnight-AI\instances")
PERSONAL_ROOT = Path(r"C:\Users\thayr\AppData\LocalLow\Team Cherry\Hollow Knight")
SAVES_ROOT = Path(r"C:\Users\thayr\Games\HollowKnight-AI\saves")
RUNTIME_ROOT = Path(r"C:\Users\thayr\Games\HollowKnight-AI\runtime")
MARKER_NAME = ".hk-agent-instance.json"
TOOL_VERSION = "1.0.0"
PROFILE = "gog-1.5.78.11833"
COMPANY_NAME = "Team Cherry"
MODDING_API_VERSION = "1.5.78.11833-77"
MODDING_API_ARCHIVE = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "modding-api"
    / MODDING_API_VERSION
    / "moddingapi.v77.windows.zip"
)


class InstanceInstallError(RuntimeError):
    """An installation cannot be provisioned safely."""


class InstanceMarkerMismatch(InstanceInstallError):
    """The marker identifies a different managed instance than the caller selected."""

    def __init__(self, instance_index: int, message: str) -> None:
        self.instance_index = instance_index
        super().__init__(message)


class ProductNameEditor(Protocol):
    def read(self, path: Path) -> tuple[str, str]: ...

    def write(self, path: Path, product_name: str) -> None: ...


class UnityPyProductNameEditor:
    """UnityPy adapter kept separate so provisioning logic can be tested without UnityPy."""

    def _load(self, path: Path) -> Any:
        try:
            import UnityPy
        except ImportError as error:
            raise InstanceInstallError(
                "UnityPy is required. Install the optional 'instances' extra; "
                "no binary string editing fallback is supported."
            ) from error
        return UnityPy.load(str(path))

    def _settings(self, environment: Any) -> Any:
        for asset in environment.objects:
            if asset.type.name == "PlayerSettings":
                return asset
        raise InstanceInstallError("PlayerSettings object not found in globalgamemanagers")

    def read(self, path: Path) -> tuple[str, str]:
        environment = self._load(path)
        try:
            settings = self._settings(environment).read()
            return str(settings.productName), str(settings.companyName)
        finally:
            environment.file.reader.stream.close()

    def write(self, path: Path, product_name: str) -> None:
        environment = self._load(path)
        settings_asset = self._settings(environment)
        settings = settings_asset.read()
        settings.productName = product_name
        settings_asset.save_typetree(settings)
        serialized = environment.file.save()
        if not isinstance(serialized, bytes):
            raise InstanceInstallError("UnityPy did not return serialized bytes")
        replacement = path.with_name(path.name + ".unitypy.tmp")
        replacement.write_bytes(serialized)
        environment.file.reader.stream.close()
        del settings, settings_asset, environment
        gc.collect()
        os.replace(replacement, path)


@dataclass(frozen=True)
class InstallationHashes:
    executable: str
    globalgamemanagers: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def installation_hashes(root: Path) -> InstallationHashes:
    return InstallationHashes(
        executable=sha256_file(root / "Hollow Knight.exe"),
        globalgamemanagers=sha256_file(root / "Hollow Knight_Data" / "globalgamemanagers"),
    )


def _is_reparse(metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def validate_source_tree(source: Path) -> int:
    """Reject links, reparse points, and special files; return regular file bytes."""
    if not source.is_dir():
        raise InstanceInstallError(f"Original installation not found: {source}")
    total = 0
    pending = [source]
    while pending:
        root_path = pending.pop()
        for entry in os.scandir(root_path):
            item = Path(entry.path)
            metadata = os.lstat(item)
            if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
                raise InstanceInstallError(f"Source contains a symlink/reparse point: {item}")
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(item)
            elif stat.S_ISREG(metadata.st_mode):
                total += metadata.st_size
            else:
                raise InstanceInstallError(f"Source contains a non-regular file: {item}")
    return total


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False)))


def _overlaps(first: Path, second: Path) -> bool:
    first_key, second_key = _path_key(first), _path_key(second)
    try:
        common = os.path.commonpath([first_key, second_key])
    except ValueError:
        return False
    return common in {first_key, second_key}


def validate_destination(
    destination: Path, *, workspace: Path, instances_root: Path = INSTANCES_ROOT
) -> None:
    protected = [
        ("OneDrive", Path(os.environ.get("OneDrive", str(Path.home() / "OneDrive")))),
        ("workspace", workspace),
        ("runtime", RUNTIME_ROOT),
        ("personal saves", PERSONAL_ROOT),
        ("saves", SAVES_ROOT),
    ]
    for label, root in protected:
        if _overlaps(destination, root):
            raise InstanceInstallError(f"Destination must not overlap {label}: {root}")
    expected_parent = instances_root.resolve(strict=False)
    if _path_key(destination.parent) != _path_key(expected_parent):
        raise InstanceInstallError(f"Destination must be directly under {expected_parent}")


def _read_marker(destination: Path) -> dict[str, Any]:
    marker = destination / MARKER_NAME
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InstanceInstallError(f"Managed destination has no valid marker: {marker}") from error
    if not isinstance(value, dict) or value.get("tool_version") != TOOL_VERSION:
        raise InstanceInstallError(f"Managed destination marker is invalid: {marker}")
    return value


def _free_bytes(path: Path) -> int:
    existing = path
    while not existing.exists():
        existing = existing.parent
    return shutil.disk_usage(existing).free


def _copy_installation(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for root, dirs, files in os.walk(source, followlinks=False):
        source_dir = Path(root)
        relative = source_dir.relative_to(source)
        output_dir = destination / relative
        output_dir.mkdir(parents=True, exist_ok=True)
        for directory in dirs:
            (output_dir / directory).mkdir(exist_ok=True)
        for filename in files:
            shutil.copy2(source_dir / filename, output_dir / filename)


def _find_bridge(source: Path) -> Path | None:
    candidate = source / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge" / "HkAgentBridge.dll"
    return candidate if candidate.is_file() else None


def _mods_status(install: Path) -> dict[str, bool]:
    mods = install / "Hollow Knight_Data" / "Managed" / "Mods"
    managed = install / "Hollow Knight_Data" / "Managed"
    bridge = any(mods.rglob("HkAgentBridge.dll")) if mods.is_dir() else False
    # Modding API v77 is installed into Managed as patched assemblies; it does not
    # ship a ModdingAPI.dll under Mods. Require the API's hook/runtime assemblies.
    api = all(
        (managed / name).is_file()
        for name in ("MMHOOK_Assembly-CSharp.dll", "MMHOOK_PlayMaker.dll", "MonoMod.RuntimeDetour.dll")
    )
    return {"hk_agent_bridge": bridge, "modding_api": api}


def _sync_bridge(bridge: Path, destination: Path) -> str:
    if not bridge.is_file():
        raise InstanceInstallError(f"Build artifact HkAgentBridge.dll not found: {bridge}")
    target_dir = destination / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge"
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bridge, target_dir / bridge.name)
    copied_hash = sha256_file(target_dir / bridge.name)
    if copied_hash != sha256_file(bridge):
        raise InstanceInstallError("Bridge DLL hash changed during synchronization")
    return copied_hash


def _install_modding_api(archive: Path, install: Path) -> str:
    if not archive.is_file():
        raise InstanceInstallError(f"Modding API package not found: {archive}")
    managed = install / "Hollow Knight_Data" / "Managed"
    try:
        with zipfile.ZipFile(archive) as package:
            entries = [entry for entry in package.infolist() if not entry.is_dir()]
            if not entries or any(Path(entry.filename).name != entry.filename for entry in entries):
                raise InstanceInstallError("Modding API archive must contain flat files only")
            names = {Path(entry.filename).name for entry in entries}
            required = {"MMHOOK_Assembly-CSharp.dll", "MMHOOK_PlayMaker.dll", "MonoMod.RuntimeDetour.dll"}
            if not required.issubset(names):
                raise InstanceInstallError("Modding API archive is missing required v77 assemblies")
            for entry in entries:
                if entry.filename == "README.md":
                    continue
                output = managed / entry.filename
                output.write_bytes(package.read(entry))
    except zipfile.BadZipFile as error:
        raise InstanceInstallError(f"Invalid Modding API archive: {archive}") from error
    return sha256_file(archive)


def refresh_instance_bridge_marker(
    install_root: Path, *, expected_index: int, expected_bridge_hash: str | None = None
) -> dict[str, Any]:
    """Refresh the managed marker only after the installed bridge hash is verified."""
    from hk_agent.instance_marker import InstanceMarker, load_instance_marker

    root = install_root.expanduser().resolve(strict=True)
    marker_path = root / MARKER_NAME
    bridge_path = root / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge" / "HkAgentBridge.dll"
    if marker_path.is_symlink() or bridge_path.is_symlink():
        raise InstanceInstallError("Instance marker and bridge DLL must be regular managed files")
    try:
        original_bytes = marker_path.read_bytes()
        marker = json.loads(original_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InstanceInstallError(f"Managed instance marker is invalid: {marker_path}") from error
    if not isinstance(marker, dict) or set(marker) != set(InstanceMarker.__dataclass_fields__):
        raise InstanceInstallError(f"Managed instance marker has an invalid schema: {marker_path}")
    if marker.get("index") != expected_index:
        actual_index = marker.get("index")
        raise InstanceMarkerMismatch(expected_index, f"Instance {expected_index} marker records index {actual_index}")
    if marker.get("schema_version") != 1:
        raise InstanceInstallError("Managed instance marker schema does not match")
    if marker.get("product_name") != f"Hollow Knight AI {expected_index}" or marker.get("company_name") != COMPANY_NAME:
        raise InstanceInstallError("Managed instance marker Unity identity does not match")
    hashes = installation_hashes(root)
    if marker.get("exe_sha256") != hashes.executable:
        raise InstanceInstallError("Instance executable hash differs from marker")
    if marker.get("globalgamemanagers_sha256") != hashes.globalgamemanagers:
        raise InstanceInstallError("Instance globalgamemanagers hash differs from marker")
    if not bridge_path.is_file():
        raise InstanceInstallError(f"Managed instance bridge DLL is missing: {bridge_path}")
    bridge_hash = sha256_file(bridge_path)
    if expected_bridge_hash is not None and bridge_hash.casefold() != expected_bridge_hash.casefold():
        raise InstanceInstallError("Installed bridge DLL does not match the requested build")
    updated = dict(marker)
    updated["bridge_dll_sha256"] = bridge_hash
    temporary = marker_path.with_name(f".{marker_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(updated, indent=2), encoding="utf-8")
        os.replace(temporary, marker_path)
        validated = load_instance_marker(root, expected_index=expected_index)
    except Exception as error:
        if temporary.exists():
            temporary.unlink()
        rollback = marker_path.with_name(f".{marker_path.name}.{uuid.uuid4().hex}.rollback")
        try:
            rollback.write_bytes(original_bytes)
            os.replace(rollback, marker_path)
        finally:
            if rollback.exists():
                rollback.unlink()
        raise InstanceInstallError(f"Updated instance marker failed validation: {error}") from error
    return {field: getattr(validated, field) for field in InstanceMarker.__dataclass_fields__}


def provision_instance(
    index: int,
    *,
    source: Path = ORIGINAL_INSTALL,
    instances_root: Path = INSTANCES_ROOT,
    workspace: Path | None = None,
    resync: bool = False,
    sync_mods: bool = False,
    bridge_dll: Path | None = None,
    api_archive: Path = MODDING_API_ARCHIVE,
    editor: ProductNameEditor | None = None,
    free_bytes: Any = _free_bytes,
) -> dict[str, Any]:
    if index < 1:
        raise InstanceInstallError("Instance index must be at least 1")
    if sync_mods and bridge_dll is None:
        raise InstanceInstallError("--sync-mods requires --bridge-dll with the current build artifact")
    source = source.expanduser().resolve(strict=False)
    destination = (instances_root / f"hk-ai-{index}").resolve(strict=False)
    validate_destination(
        destination, workspace=workspace or Path.cwd(), instances_root=instances_root
    )
    if not (source / "Hollow Knight.exe").is_file() or not (
        source / "Hollow Knight_Data" / "globalgamemanagers"
    ).is_file():
        raise InstanceInstallError(f"Original installation is incomplete: {source}")
    original_before = installation_hashes(source)
    size = validate_source_tree(source)
    destination_exists = destination.exists()
    if destination_exists:
        if not (destination / MARKER_NAME).is_file():
            raise InstanceInstallError(f"Refusing unmanaged destination: {destination}")
        marker = _read_marker(destination)
        if not resync:
            raise InstanceInstallError("Managed destination exists; pass --resync to synchronize it")
        if marker.get("index") != index:
            raise InstanceInstallError("Existing marker index does not match destination")
    if free_bytes(destination.parent) < size + 2 * 1024**3:
        raise InstanceInstallError("Insufficient free space; need installation size plus 2 GB")

    editor = editor or UnityPyProductNameEditor()
    product_name = f"Hollow Knight AI {index}"
    gm_rel = Path("Hollow Knight_Data") / "globalgamemanagers"
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(f".{destination.name}.stage-{uuid.uuid4().hex}")
    try:
        _copy_installation(source, stage)
        staged_gm = stage / gm_rel
        original_bytes = staged_gm.read_bytes()
        (staged_gm.parent / "globalgamemanagers.orig").write_bytes(original_bytes)
        editor.write(staged_gm, product_name)
        actual_product, company_name = editor.read(staged_gm)
        if actual_product != product_name or company_name != COMPANY_NAME:
            raise InstanceInstallError(
                f"Unity settings verification failed: productName={actual_product!r}, "
                f"companyName={company_name!r}"
            )
        if sync_mods:
            assert bridge_dll is not None
            bridge_hash = _sync_bridge(bridge_dll, stage)
        else:
            bridge_hash = sha256_file(
                stage / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge" / "HkAgentBridge.dll"
            )
        _install_modding_api(api_archive, stage)
        mods = _mods_status(stage)
        if not all(mods.values()):
            raise InstanceInstallError(
                "Managed\\Mods must contain HkAgentBridge and Managed must contain Modding API files; "
                f"found {mods}. Use --sync-mods for the bridge or provision the API first."
            )
        copied_hashes = installation_hashes(stage)
        original_after = installation_hashes(source)
        if original_before != original_after:
            raise InstanceInstallError("Original installation hashes changed during provisioning")
        data = {
            "schema_version": 1,
            "index": index,
            "product_name": product_name,
            "company_name": company_name,
            "persistent_data_path_expected": str(PERSONAL_ROOT.parent / product_name),
            "registry_key_expected": rf"HKCU\Software\{COMPANY_NAME}\{product_name}",
            "source_install": str(source),
            "exe_sha256": copied_hashes.executable,
            "globalgamemanagers_sha256": copied_hashes.globalgamemanagers,
            "bridge_dll_sha256": bridge_hash,
            "modding_api_version": MODDING_API_VERSION,
            "tool_version": TOOL_VERSION,
            "created_unix_ns": time.time_ns(),
        }
        (stage / MARKER_NAME).write_text(json.dumps(data, indent=2), encoding="utf-8")
        old_destination = destination.with_name(f".{destination.name}.previous-{uuid.uuid4().hex}")
        if destination_exists:
            destination.rename(old_destination)
        try:
            stage.rename(destination)
        except OSError:
            if destination_exists and old_destination.exists():
                old_destination.rename(destination)
            raise
        if destination_exists:
            shutil.rmtree(old_destination)
        return data
    finally:
        if stage.exists():
            shutil.rmtree(stage)
