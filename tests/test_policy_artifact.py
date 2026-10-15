from __future__ import annotations

import importlib.util
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

from hk_agent import policy_artifact
from hk_agent.policy_artifact import (
    PolicyArtifactError,
    export_policy_artifact,
    import_policy_artifact,
)

_SAFE_TO_TORCH = {
    "BOOL": torch.bool,
    "I64": torch.int64,
    "F32": torch.float32,
    "F64": torch.float64,
}


class TinyPolicy(torch.nn.Module):
    def __init__(self, width: int = 3, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, width, dtype=dtype)
        self.register_buffer("counter", torch.tensor(7, dtype=torch.int64))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.linear(value))


class _Slice:
    def __init__(self, record: dict[str, Any]) -> None:
        self._record = record

    def get_dtype(self) -> str:
        return str(self._record["dtype"])

    def get_shape(self) -> list[int]:
        return [int(item) for item in self._record["shape"]]


class _Reader:
    def __init__(self, path: str, **_: object) -> None:
        self._document = json.loads(Path(path).read_text("utf-8"))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def metadata(self) -> dict[str, str]:
        return dict(self._document["metadata"])

    def keys(self) -> list[str]:
        return list(self._document["tensors"])

    def get_slice(self, name: str) -> _Slice:
        return _Slice(self._document["tensors"][name])

    def get_tensor(self, name: str) -> torch.Tensor:
        record = self._document["tensors"][name]
        return torch.tensor(record["values"], dtype=_SAFE_TO_TORCH[record["dtype"]]).reshape(
            record["shape"]
        )


def _fake_save_file(
    tensors: dict[str, torch.Tensor], path: str, *, metadata: dict[str, str]
) -> None:
    records: dict[str, object] = {}
    for name, tensor in tensors.items():
        dtype = policy_artifact._DTYPE_TO_SAFE[str(tensor.dtype)]
        records[name] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "values": tensor.tolist(),
        }
    Path(path).write_text(
        json.dumps({"metadata": metadata, "tensors": records}, sort_keys=True),
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def fake_safetensors(monkeypatch: pytest.MonkeyPatch) -> None:
    if importlib.util.find_spec("safetensors") is None:
        monkeypatch.setattr(policy_artifact, "_safetensors_api", lambda: (_Reader, _fake_save_file))


def test_round_trip_strictly_restores_policy_outputs(tmp_path: Path) -> None:
    torch.manual_seed(4)
    source = TinyPolicy()
    target = TinyPolicy()
    observation = torch.tensor([[0.25, -0.75]], dtype=torch.float32)
    expected = source(observation).detach().clone()
    artifact = tmp_path / "policy.safetensors"

    exported = export_policy_artifact(source, artifact, compatibility_fingerprint="a" * 64)
    imported = import_policy_artifact(target, artifact, expected_compatibility_fingerprint="a" * 64)

    assert imported == exported
    assert torch.equal(target(observation), expected)
    assert imported["tensor_schema_sha256"]
    assert imported["artifact"]["sha256"]


@pytest.mark.parametrize(
    ("fingerprint", "match"),
    [("b" * 64, "compatibility"), ("invalid", "lowercase SHA-256")],
)
def test_import_rejects_wrong_or_invalid_compatibility(
    tmp_path: Path, fingerprint: str, match: str
) -> None:
    artifact = tmp_path / "policy.safetensors"
    export_policy_artifact(TinyPolicy(), artifact, compatibility_fingerprint="a" * 64)

    with pytest.raises(PolicyArtifactError, match=match):
        import_policy_artifact(
            TinyPolicy(), artifact, expected_compatibility_fingerprint=fingerprint
        )


@pytest.mark.parametrize(
    "target",
    [TinyPolicy(width=4), TinyPolicy(dtype=torch.float64)],
    ids=["shape", "dtype"],
)
def test_import_rejects_target_shape_or_dtype_difference(
    tmp_path: Path, target: TinyPolicy
) -> None:
    artifact = tmp_path / "policy.safetensors"
    export_policy_artifact(TinyPolicy(), artifact, compatibility_fingerprint="a" * 64)

    with pytest.raises(PolicyArtifactError, match="tensor schema differs"):
        import_policy_artifact(target, artifact, expected_compatibility_fingerprint="a" * 64)


def test_import_rejects_tampered_artifact_without_mutating_policy(tmp_path: Path) -> None:
    artifact = tmp_path / "policy.safetensors"
    export_policy_artifact(TinyPolicy(), artifact, compatibility_fingerprint="a" * 64)
    target = TinyPolicy()
    before = {name: tensor.clone() for name, tensor in target.state_dict().items()}
    artifact.write_bytes(artifact.read_bytes() + b"tampered")

    with pytest.raises(PolicyArtifactError, match="snapshot differs"):
        import_policy_artifact(target, artifact, expected_compatibility_fingerprint="a" * 64)
    assert all(torch.equal(before[name], tensor) for name, tensor in target.state_dict().items())


def test_manifest_is_closed_and_content_addressed(tmp_path: Path) -> None:
    artifact = tmp_path / "policy.safetensors"
    export_policy_artifact(TinyPolicy(), artifact, compatibility_fingerprint="a" * 64)
    sidecar = artifact.with_suffix(".manifest.json")
    manifest = json.loads(sidecar.read_text("utf-8"))
    manifest["unexpected"] = "field"
    sidecar.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PolicyArtifactError, match="unexpected or missing"):
        import_policy_artifact(TinyPolicy(), artifact, expected_compatibility_fingerprint="a" * 64)


def test_manifest_rejects_non_integer_shape(tmp_path: Path) -> None:
    artifact = tmp_path / "policy.safetensors"
    export_policy_artifact(TinyPolicy(), artifact, compatibility_fingerprint="a" * 64)
    sidecar = artifact.with_suffix(".manifest.json")
    manifest = json.loads(sidecar.read_text("utf-8"))
    shaped = next(item for item in manifest["tensors"] if item["shape"])
    shaped["shape"][0] = float(shaped["shape"][0])
    sidecar.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PolicyArtifactError, match="invalid shape"):
        import_policy_artifact(TinyPolicy(), artifact, expected_compatibility_fingerprint="a" * 64)


@pytest.mark.parametrize(
    ("keyword", "match"),
    [
        ("expected_artifact_sha256", "policy differs"),
        ("expected_manifest_sha256", "manifest differs"),
        ("expected_manifest_id", "manifest ID differs"),
    ],
)
def test_import_rejects_identity_other_than_the_frozen_preflight(
    tmp_path: Path, keyword: str, match: str
) -> None:
    artifact = tmp_path / "policy.safetensors"
    export_policy_artifact(TinyPolicy(), artifact, compatibility_fingerprint="a" * 64)

    with pytest.raises(PolicyArtifactError, match=match):
        import_policy_artifact(
            TinyPolicy(),
            artifact,
            expected_compatibility_fingerprint="a" * 64,
            **{keyword: "f" * 64},
        )


def test_export_rejects_unsupported_dtype_and_tensor_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unsupported = TinyPolicy()
    unsupported.register_buffer("complex", torch.ones(1, dtype=torch.complex64))
    with pytest.raises(PolicyArtifactError, match="unsupported dtype"):
        export_policy_artifact(
            unsupported,
            tmp_path / "complex.safetensors",
            compatibility_fingerprint="a" * 64,
        )

    monkeypatch.setattr(policy_artifact, "MAX_TENSORS", 1)
    with pytest.raises(PolicyArtifactError, match="tensor count"):
        export_policy_artifact(
            TinyPolicy(),
            tmp_path / "large.safetensors",
            compatibility_fingerprint="a" * 64,
        )


def test_export_rejects_non_finite_tensor(tmp_path: Path) -> None:
    policy = TinyPolicy()
    with torch.no_grad():
        policy.linear.weight[0, 0] = float("nan")
    with pytest.raises(PolicyArtifactError, match="NaN or infinity"):
        export_policy_artifact(
            policy,
            tmp_path / "non-finite.safetensors",
            compatibility_fingerprint="a" * 64,
        )


def test_import_rejects_symlink(tmp_path: Path) -> None:
    artifact = tmp_path / "policy.safetensors"
    export_policy_artifact(TinyPolicy(), artifact, compatibility_fingerprint="a" * 64)
    link = tmp_path / "linked.safetensors"
    try:
        os.symlink(artifact, link)
    except OSError:
        pytest.skip("this Windows account cannot create symlinks")

    with pytest.raises(PolicyArtifactError, match="regular file"):
        import_policy_artifact(
            TinyPolicy(),
            link,
            expected_compatibility_fingerprint="a" * 64,
            manifest_path=artifact.with_suffix(".manifest.json"),
        )


def test_strict_loader_is_invoked(tmp_path: Path) -> None:
    class TrackingPolicy(TinyPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.strict_seen: bool | None = None

        def load_state_dict(
            self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
        ) -> object:
            del assign
            self.strict_seen = strict
            return super().load_state_dict(state_dict, strict=strict)

    source = TrackingPolicy()
    target = TrackingPolicy()
    artifact = tmp_path / "policy.safetensors"
    export_policy_artifact(source, artifact, compatibility_fingerprint="a" * 64)
    import_policy_artifact(target, artifact, expected_compatibility_fingerprint="a" * 64)
    assert target.strict_seen is True


def test_failed_strict_load_restores_previous_policy_state(tmp_path: Path) -> None:
    class FailingOncePolicy(TinyPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.fail_next_load = False

        def load_state_dict(
            self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
        ) -> object:
            result = super().load_state_dict(state_dict, strict=strict, assign=assign)
            if self.fail_next_load:
                self.fail_next_load = False
                raise RuntimeError("simulated post-mutation failure")
            return result

    source = FailingOncePolicy()
    target = FailingOncePolicy()
    artifact = tmp_path / "policy.safetensors"
    export_policy_artifact(source, artifact, compatibility_fingerprint="a" * 64)
    before = {name: tensor.clone() for name, tensor in target.state_dict().items()}
    target.fail_next_load = True

    with pytest.raises(PolicyArtifactError, match="previous state was restored"):
        import_policy_artifact(target, artifact, expected_compatibility_fingerprint="a" * 64)
    assert all(torch.equal(before[name], tensor) for name, tensor in target.state_dict().items())


def test_keyboard_interrupt_from_loader_is_not_converted(tmp_path: Path) -> None:
    class InterruptedPolicy(TinyPolicy):
        def load_state_dict(
            self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
        ) -> object:
            del state_dict, strict, assign
            raise KeyboardInterrupt

    artifact = tmp_path / "policy.safetensors"
    export_policy_artifact(InterruptedPolicy(), artifact, compatibility_fingerprint="a" * 64)

    with pytest.raises(KeyboardInterrupt):
        import_policy_artifact(
            InterruptedPolicy(), artifact, expected_compatibility_fingerprint="a" * 64
        )
