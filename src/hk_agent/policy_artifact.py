from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import secrets
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

POLICY_ARTIFACT_SCHEMA_VERSION = 1
MAX_POLICY_BYTES = 512 * 1024 * 1024
MAX_TENSORS = 4_096
MAX_TOTAL_ELEMENTS = 250_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DTYPE_TO_SAFE = {
    "torch.bool": "BOOL",
    "torch.uint8": "U8",
    "torch.int8": "I8",
    "torch.int16": "I16",
    "torch.int32": "I32",
    "torch.int64": "I64",
    "torch.float16": "F16",
    "torch.bfloat16": "BF16",
    "torch.float32": "F32",
    "torch.float64": "F64",
}
_ALLOWED_SAFE_DTYPES = frozenset(_DTYPE_TO_SAFE.values())


class PolicyArtifactError(ValueError):
    """Raised when a portable policy artifact fails closed validation."""


class _Tensor(Protocol):
    @property
    def dtype(self) -> object: ...

    @property
    def shape(self) -> object: ...

    def numel(self) -> int: ...

    def detach(self) -> _Tensor: ...

    def cpu(self) -> _Tensor: ...

    def contiguous(self) -> _Tensor: ...

    def clone(self) -> _Tensor: ...


class _Policy(Protocol):
    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state_dict: Mapping[str, Any], *, strict: bool) -> object: ...


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_class(policy: object) -> str:
    cls = type(policy)
    return f"{cls.__module__}.{cls.__qualname__}"


def _shape(value: object, label: str) -> list[int]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise PolicyArtifactError(f"{label} has an invalid shape")
    items = list(value)
    if any(type(item) is not int for item in items):
        raise PolicyArtifactError(f"{label} has an invalid shape")
    result = [int(item) for item in items]
    if any(item < 0 for item in result):
        raise PolicyArtifactError(f"{label} has an invalid shape")
    return result


def _tensor_spec(name: str, tensor: object) -> dict[str, Any]:
    if not isinstance(name, str) or not name or len(name) > 512 or "\x00" in name:
        raise PolicyArtifactError("policy state contains an invalid tensor name")
    dtype = _DTYPE_TO_SAFE.get(str(getattr(tensor, "dtype", None)))
    if dtype is None:
        raise PolicyArtifactError(f"policy tensor {name!r} has an unsupported dtype")
    shape = _shape(getattr(tensor, "shape", None), f"policy tensor {name!r}")
    numel_method = getattr(tensor, "numel", None)
    if not callable(numel_method):
        raise PolicyArtifactError(f"policy state value {name!r} is not a tensor")
    numel = numel_method()
    if not isinstance(numel, int) or isinstance(numel, bool) or numel < 0:
        raise PolicyArtifactError(f"policy tensor {name!r} has an invalid size")
    expected = 1
    for dimension in shape:
        expected *= dimension
    if expected != numel:
        raise PolicyArtifactError(f"policy tensor {name!r} has inconsistent dimensions")
    return {"name": name, "dtype": dtype, "shape": shape, "numel": numel}


def _state_inventory(policy: _Policy) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = dict(policy.state_dict())
    if not state or len(state) > MAX_TENSORS:
        raise PolicyArtifactError("policy tensor count is outside the safe range")
    inventory = [_tensor_spec(name, tensor) for name, tensor in sorted(state.items())]
    if sum(int(item["numel"]) for item in inventory) > MAX_TOTAL_ELEMENTS:
        raise PolicyArtifactError("policy element count exceeds the safe limit")
    return state, inventory


def _schema_sha256(inventory: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical_bytes(inventory)).hexdigest()


def _validate_fingerprint(value: object) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise PolicyArtifactError("compatibility fingerprint must be lowercase SHA-256")
    return value


def _safetensors_api() -> tuple[Any, Any]:
    try:
        package = importlib.import_module("safetensors")
        torch_package = importlib.import_module("safetensors.torch")
    except ImportError as error:
        raise RuntimeError("safetensors is required for portable policy import/export") from error
    return package.safe_open, torch_package.save_file


def _ensure_finite(tensor: object, label: str) -> None:
    try:
        torch = importlib.import_module("torch")
        finite = torch.isfinite(tensor).all().item()
    except (ImportError, AttributeError, RuntimeError, TypeError) as error:
        raise PolicyArtifactError(f"{label} cannot be checked for finite values") from error
    if finite is not True:
        raise PolicyArtifactError(f"{label} contains NaN or infinity")


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise PolicyArtifactError(f"{label} must be a regular file")
    return path.resolve(strict=True)


def _private_snapshot(path: Path, expected_size: int, expected_sha256: str) -> Path:
    """Copy an artifact through one open handle to close path replacement races."""

    pending: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="voidgym-policy-", suffix=".safetensors", delete=False
        ) as target:
            pending = Path(target.name)
            copied = 0
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    copied += len(chunk)
                    if copied > MAX_POLICY_BYTES:
                        raise PolicyArtifactError("portable policy exceeds the safe size limit")
                    target.write(chunk)
        if copied != expected_size or _sha256_file(pending) != expected_sha256:
            raise PolicyArtifactError("portable policy snapshot differs from its manifest")
        return pending
    except BaseException:
        if pending is not None and pending.exists():
            pending.unlink()
        raise


def validate_policy_manifest(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "manifest_id",
        "schema_version",
        "format",
        "compatibility_fingerprint",
        "policy_class",
        "tensor_schema_sha256",
        "tensors",
        "artifact",
    }:
        raise PolicyArtifactError("policy manifest has unexpected or missing fields")
    manifest = value
    if manifest["schema_version"] != POLICY_ARTIFACT_SCHEMA_VERSION:
        raise PolicyArtifactError("unsupported policy artifact schema_version")
    if manifest["format"] != "safetensors":
        raise PolicyArtifactError("portable policy format must be safetensors")
    _validate_fingerprint(manifest["compatibility_fingerprint"])
    if not isinstance(manifest["policy_class"], str) or not manifest["policy_class"]:
        raise PolicyArtifactError("policy manifest class is invalid")
    if len(manifest["policy_class"]) > 512 or "\x00" in manifest["policy_class"]:
        raise PolicyArtifactError("policy manifest class is invalid")
    tensors = manifest["tensors"]
    if not isinstance(tensors, list) or not 1 <= len(tensors) <= MAX_TENSORS:
        raise PolicyArtifactError("policy manifest tensor count is outside the safe range")
    names: set[str] = set()
    total = 0
    for item in tensors:
        if not isinstance(item, dict) or set(item) != {"name", "dtype", "shape", "numel"}:
            raise PolicyArtifactError("policy tensor manifest has invalid fields")
        name, dtype, numel = item["name"], item["dtype"], item["numel"]
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 512
            or "\x00" in name
            or name in names
        ):
            raise PolicyArtifactError("policy tensor names must be unique")
        names.add(name)
        if dtype not in _ALLOWED_SAFE_DTYPES:
            raise PolicyArtifactError("policy tensor manifest has an unsupported dtype")
        shape = _shape(item["shape"], f"policy tensor {name!r}")
        if item["shape"] != shape or any(isinstance(value, bool) for value in item["shape"]):
            raise PolicyArtifactError("policy tensor manifest has a non-canonical shape")
        expected = 1
        for dimension in shape:
            expected *= dimension
        if not isinstance(numel, int) or isinstance(numel, bool) or numel != expected:
            raise PolicyArtifactError("policy tensor manifest has inconsistent dimensions")
        total += numel
    if [item["name"] for item in tensors] != sorted(names):
        raise PolicyArtifactError("policy tensor manifest must use canonical name order")
    if total > MAX_TOTAL_ELEMENTS:
        raise PolicyArtifactError("policy manifest element count exceeds the safe limit")
    schema_digest = manifest["tensor_schema_sha256"]
    if not isinstance(schema_digest, str) or schema_digest != _schema_sha256(tensors):
        raise PolicyArtifactError("policy tensor schema hash differs")
    artifact = manifest["artifact"]
    if not isinstance(artifact, dict) or set(artifact) != {"sha256", "size_bytes"}:
        raise PolicyArtifactError("policy artifact metadata has invalid fields")
    if not isinstance(artifact["sha256"], str) or not _SHA256.fullmatch(artifact["sha256"]):
        raise PolicyArtifactError("policy artifact hash is invalid")
    size = artifact["size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAX_POLICY_BYTES:
        raise PolicyArtifactError("policy artifact size is outside the safe range")
    body = dict(manifest)
    claimed_id = body.pop("manifest_id")
    if (
        not isinstance(claimed_id, str)
        or claimed_id != hashlib.sha256(_canonical_bytes(body)).hexdigest()
    ):
        raise PolicyArtifactError("policy manifest ID differs from its canonical content")
    return manifest


def export_policy_artifact(
    policy: _Policy,
    artifact_path: Path,
    *,
    compatibility_fingerprint: str,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Atomically export tensor-only policy state plus a closed JSON sidecar."""

    fingerprint = _validate_fingerprint(compatibility_fingerprint)
    if artifact_path.suffix != ".safetensors":
        raise PolicyArtifactError("portable policy filename must end in .safetensors")
    target = artifact_path.resolve(strict=False)
    sidecar = (manifest_path or artifact_path.with_suffix(".manifest.json")).resolve(strict=False)
    if target == sidecar or target.is_symlink() or sidecar.is_symlink():
        raise PolicyArtifactError("policy output paths must be distinct regular destinations")
    target.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    state, inventory = _state_inventory(policy)
    tensors: dict[str, Any] = {}
    for name, tensor in state.items():
        try:
            # Clone breaks shared storage aliases, which safetensors deliberately
            # refuses because aliases cannot be represented portably.
            tensors[name] = tensor.detach().cpu().contiguous().clone()
        except AttributeError as error:
            raise PolicyArtifactError(f"policy state value {name!r} is not a tensor") from error
        _ensure_finite(tensors[name], f"policy tensor {name!r}")
    schema_digest = _schema_sha256(inventory)
    _, save_file = _safetensors_api()
    pending = target.with_name(f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.pending")
    try:
        save_file(
            tensors,
            str(pending),
            metadata={
                "format": "voidgym-policy-v1",
                "compatibility_fingerprint": fingerprint,
                "tensor_schema_sha256": schema_digest,
            },
        )
        size = pending.stat().st_size
        if not 0 < size <= MAX_POLICY_BYTES:
            raise PolicyArtifactError("serialized policy size is outside the safe range")
        digest = _sha256_file(pending)
        body: dict[str, Any] = {
            "schema_version": POLICY_ARTIFACT_SCHEMA_VERSION,
            "format": "safetensors",
            "compatibility_fingerprint": fingerprint,
            "policy_class": _policy_class(policy),
            "tensor_schema_sha256": schema_digest,
            "tensors": inventory,
            "artifact": {"sha256": digest, "size_bytes": size},
        }
        manifest = {"manifest_id": hashlib.sha256(_canonical_bytes(body)).hexdigest(), **body}
        validate_policy_manifest(manifest)
        os.replace(pending, target)
        pending_json = sidecar.with_name(
            f".{sidecar.name}.{os.getpid()}.{secrets.token_hex(8)}.pending"
        )
        try:
            pending_json.write_bytes(_canonical_bytes(manifest) + b"\n")
            os.replace(pending_json, sidecar)
        finally:
            if pending_json.exists():
                pending_json.unlink()
        return manifest
    finally:
        if pending.exists():
            pending.unlink()


def import_policy_artifact(
    policy: _Policy,
    artifact_path: Path,
    *,
    expected_compatibility_fingerprint: str,
    manifest_path: Path | None = None,
    expected_artifact_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_manifest_id: str | None = None,
) -> dict[str, Any]:
    """Validate a tensor-only artifact fully, then strictly load an existing policy."""

    expected_fingerprint = _validate_fingerprint(expected_compatibility_fingerprint)
    expected_artifact = (
        _validate_fingerprint(expected_artifact_sha256)
        if expected_artifact_sha256 is not None
        else None
    )
    expected_sidecar = (
        _validate_fingerprint(expected_manifest_sha256)
        if expected_manifest_sha256 is not None
        else None
    )
    expected_id = (
        _validate_fingerprint(expected_manifest_id)
        if expected_manifest_id is not None
        else None
    )
    artifact = _regular_file(artifact_path, "policy artifact")
    sidecar = _regular_file(
        manifest_path or artifact_path.with_suffix(".manifest.json"), "policy manifest"
    )
    if artifact.stat().st_size > MAX_POLICY_BYTES or sidecar.stat().st_size > 2 * 1024 * 1024:
        raise PolicyArtifactError("policy artifact or manifest exceeds its size limit")
    try:
        sidecar_bytes = sidecar.read_bytes()
        manifest = validate_policy_manifest(json.loads(sidecar_bytes.decode("utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PolicyArtifactError("policy manifest is not readable canonical JSON") from error
    sidecar_digest = hashlib.sha256(sidecar_bytes).hexdigest()
    if expected_sidecar is not None and sidecar_digest != expected_sidecar:
        raise PolicyArtifactError("policy manifest differs from its frozen identity")
    if expected_id is not None and manifest["manifest_id"] != expected_id:
        raise PolicyArtifactError("policy manifest ID differs from its frozen identity")
    if manifest["compatibility_fingerprint"] != expected_fingerprint:
        raise PolicyArtifactError("portable policy compatibility fingerprint differs")
    if manifest["policy_class"] != _policy_class(policy):
        raise PolicyArtifactError("portable policy class differs")
    _, expected_inventory = _state_inventory(policy)
    if manifest["tensors"] != expected_inventory:
        raise PolicyArtifactError("portable policy tensor schema differs from the target policy")
    safe_open, _ = _safetensors_api()
    loaded: dict[str, Any] = {}
    snapshot = _private_snapshot(
        artifact,
        manifest["artifact"]["size_bytes"],
        manifest["artifact"]["sha256"],
    )
    if expected_artifact is not None and manifest["artifact"]["sha256"] != expected_artifact:
        snapshot.unlink(missing_ok=True)
        raise PolicyArtifactError("portable policy differs from its frozen identity")
    try:
        with safe_open(str(snapshot), framework="pt", device="cpu") as reader:
            metadata = reader.metadata()
            if metadata != {
                "format": "voidgym-policy-v1",
                "compatibility_fingerprint": expected_fingerprint,
                "tensor_schema_sha256": manifest["tensor_schema_sha256"],
            }:
                raise PolicyArtifactError("safetensors metadata differs from the manifest")
            if sorted(reader.keys()) != [item["name"] for item in expected_inventory]:
                raise PolicyArtifactError("safetensors keys differ from the target policy")
            for item in expected_inventory:
                view = reader.get_slice(item["name"])
                if view.get_dtype() != item["dtype"] or list(view.get_shape()) != item["shape"]:
                    raise PolicyArtifactError("safetensors tensor schema differs from the manifest")
            for item in expected_inventory:
                loaded[item["name"]] = reader.get_tensor(item["name"])
                if _tensor_spec(item["name"], loaded[item["name"]]) != item:
                    raise PolicyArtifactError(
                        "loaded tensor schema differs from the validated header"
                    )
                _ensure_finite(loaded[item["name"]], f"loaded policy tensor {item['name']!r}")
    except PolicyArtifactError:
        raise
    except Exception as error:
        raise PolicyArtifactError("safetensors payload could not be read safely") from error
    finally:
        snapshot.unlink(missing_ok=True)
    previous = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in policy.state_dict().items()
    }
    try:
        policy.load_state_dict(loaded, strict=True)
    except Exception as error:
        try:
            policy.load_state_dict(previous, strict=True)
        except Exception as rollback_error:
            raise PolicyArtifactError(
                "strict policy load failed and the previous state could not be restored"
            ) from rollback_error
        raise PolicyArtifactError(
            "strict policy load failed; previous state was restored"
        ) from error
    return manifest
