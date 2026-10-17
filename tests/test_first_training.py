from __future__ import annotations

import json
import os
import runpy
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from hk_agent import first_training
from hk_agent.first_training import FirstTrainingConfig, train_ppo
from hk_agent.training_task import InvalidTrainingTransition


class FakeModel:
    def __init__(self, *_: object, **kwargs: object) -> None:
        self.num_timesteps = 0
        self.policy = object()
        requested_device = str(kwargs.get("device", "cpu"))
        self.device = "cpu" if requested_device == "auto" else requested_device

    def learn(
        self,
        *,
        total_timesteps: int,
        reset_num_timesteps: bool,
        callback: Any = None,
    ) -> None:
        assert reset_num_timesteps is False
        if callback is not None and not callback({}, {}):
            return
        self.num_timesteps += total_timesteps

    def save(self, path: Path) -> None:
        output = path if path.suffix == ".zip" else Path(f"{path}.zip")
        output.write_bytes(b"checkpoint")

    @classmethod
    def load(cls, path: str, *, env: object, device: str) -> FakeModel:
        assert Path(path).is_file()
        assert env is not None
        assert device == "cpu"
        model = cls(device=device)
        model.num_timesteps = 128
        return model


class FakeEnvironment:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeTask:
    def task_contract(self) -> dict[str, object]:
        return {
            "reward_version": "test_curriculum_v1",
            "diagnostic_training": False,
            "claim": "test claim",
        }


def fake_torch_module() -> ModuleType:
    torch = ModuleType("torch")
    torch.__version__ = "test"  # type: ignore[attr-defined]
    torch.thread_calls = []  # type: ignore[attr-defined]
    torch.interop_thread_calls = []  # type: ignore[attr-defined]
    torch.set_num_threads = lambda value: torch.thread_calls.append(value)  # type: ignore[attr-defined]
    torch.set_num_interop_threads = (  # type: ignore[attr-defined]
        lambda value: torch.interop_thread_calls.append(value)
    )
    torch.get_num_threads = lambda: torch.thread_calls[-1]  # type: ignore[attr-defined]
    torch.get_num_interop_threads = (  # type: ignore[attr-defined]
        lambda: torch.interop_thread_calls[-1]
    )
    torch.manual_seed = lambda _: None  # type: ignore[attr-defined]
    return torch


@pytest.fixture(autouse=True)
def fake_portable_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_export(
        policy: object,
        artifact_path: Path,
        *,
        compatibility_fingerprint: str,
        manifest_path: Path | None = None,
    ) -> dict[str, Any]:
        del policy
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(b"portable policy")
        sidecar = manifest_path or artifact_path.with_suffix(".manifest.json")
        sidecar.write_text("{}", encoding="utf-8")
        return {
            "manifest_id": "b" * 64,
            "compatibility_fingerprint": compatibility_fingerprint,
            "artifact": {
                "sha256": first_training.sha256_file(artifact_path),
                "size_bytes": artifact_path.stat().st_size,
            },
        }

    def fake_import(
        policy: object,
        artifact_path: Path,
        *,
        expected_compatibility_fingerprint: str,
        manifest_path: Path | None = None,
        expected_artifact_sha256: str | None = None,
        expected_manifest_sha256: str | None = None,
        expected_manifest_id: str | None = None,
    ) -> dict[str, Any]:
        del policy, expected_artifact_sha256, expected_manifest_sha256, expected_manifest_id
        assert artifact_path.is_file()
        assert manifest_path is not None and manifest_path.is_file()
        return {
            "manifest_id": "c" * 64,
            "compatibility_fingerprint": expected_compatibility_fingerprint,
            "artifact": {
                "sha256": first_training.sha256_file(artifact_path),
                "size_bytes": artifact_path.stat().st_size,
            },
        }

    monkeypatch.setattr(first_training, "export_policy_artifact", fake_export)
    monkeypatch.setattr(first_training, "import_policy_artifact", fake_import)


class FailingConstructorModel:
    def __init__(self, *_: object, **__: object) -> None:
        raise RuntimeError("constructor exploded")


class FreshOnlyModel(FakeModel):
    @classmethod
    def load(cls, path: str, *, env: object, device: str) -> FakeModel:
        del path, env, device
        raise AssertionError("portable policy must never use PPO.load")


class InvalidTransitionModel(FakeModel):
    def learn(
        self,
        *,
        total_timesteps: int,
        reset_num_timesteps: bool,
        callback: Any = None,
    ) -> None:
        raise InvalidTrainingTransition("discarded before learner")


class StopDuringRolloutModel(FakeModel):
    stop_file: Path

    def learn(
        self,
        *,
        total_timesteps: int,
        reset_num_timesteps: bool,
        callback: Any = None,
    ) -> None:
        self.num_timesteps += 17
        self.stop_file.write_text("stop", encoding="utf-8")
        assert callback is not None
        # Cooperative stop is observed between complete PPO rollouts so the
        # newest optimizer update can be checkpointed atomically.
        assert callback({}, {}) is True


class ContinuousStopModel(FakeModel):
    stop_file: Path

    def learn(
        self,
        *,
        total_timesteps: int,
        reset_num_timesteps: bool,
        callback: Any = None,
    ) -> None:
        assert reset_num_timesteps is False
        self.num_timesteps += total_timesteps
        if self.num_timesteps >= total_timesteps * 4:
            self.stop_file.write_text("stop", encoding="utf-8")


class RunFailedModel(FakeModel):
    stop_file: Path

    def learn(
        self,
        *,
        total_timesteps: int,
        reset_num_timesteps: bool,
        callback: Any = None,
    ) -> None:
        assert callback is not None
        self.num_timesteps += total_timesteps
        callback(
            {"infos": [{"run_failed": True, "run_failure_reason": "lost save"}]},
            {},
        )
        self.stop_file.write_text("stop", encoding="utf-8")


def test_training_budget_requires_exact_rollouts() -> None:
    config = FirstTrainingConfig(total_timesteps=129, checkpoint_interval=128)
    try:
        config.validate()
    except ValueError as error:
        assert "multiple" in str(error)
    else:
        raise AssertionError("invalid rollout budget was accepted")


def test_training_resource_defaults_and_ranges() -> None:
    assert FirstTrainingConfig().training_device == "cpu"
    for config in (
        FirstTrainingConfig(training_device="metal"),
        FirstTrainingConfig(torch_threads=0),
        FirstTrainingConfig(torch_threads=9),
        FirstTrainingConfig(torch_interop_threads=0),
        FirstTrainingConfig(torch_interop_threads=5),
    ):
        with pytest.raises(ValueError):
            config.validate()


def test_training_applies_and_records_effective_resources_without_mutating_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sb3 = ModuleType("stable_baselines3")
    sb3.__version__ = "test"  # type: ignore[attr-defined]
    sb3.PPO = FakeModel  # type: ignore[attr-defined]
    torch = fake_torch_module()
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
    contrib = ModuleType("sb3_contrib")
    contrib.__version__ = "test"
    contrib.MaskablePPO = sb3.PPO
    monkeypatch.setitem(sys.modules, "sb3_contrib", contrib)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("OMP_NUM_THREADS", "11")
    monkeypatch.setenv("MKL_NUM_THREADS", "12")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "13")
    environment_before = dict(os.environ)

    result = train_ppo(
        FakeEnvironment(),  # type: ignore[arg-type]
        task=FakeTask(),
        config=FirstTrainingConfig(
            total_timesteps=128,
            checkpoint_interval=128,
            training_device="auto",
            torch_threads=4,
            torch_interop_threads=2,
        ),
        output_root=tmp_path,
    )

    assert dict(os.environ) == environment_before
    assert torch.thread_calls == [4]  # type: ignore[attr-defined]
    assert torch.interop_thread_calls == [2]  # type: ignore[attr-defined]
    assert result["runtime_resources"] == {
        "training_device_requested": "auto",
        "training_device_effective": "cpu",
        "torch_threads": 4,
        "torch_interop_threads": 2,
        "omp_num_threads": "11",
        "mkl_num_threads": "12",
        "openblas_num_threads": "13",
    }


def test_training_fails_clearly_without_torch_thread_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sb3 = ModuleType("stable_baselines3")
    sb3.__version__ = "test"  # type: ignore[attr-defined]
    sb3.PPO = FakeModel  # type: ignore[attr-defined]
    torch = ModuleType("torch")
    torch.__version__ = "test"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
    contrib = ModuleType("sb3_contrib")
    contrib.__version__ = "test"
    contrib.MaskablePPO = sb3.PPO
    monkeypatch.setitem(sys.modules, "sb3_contrib", contrib)
    monkeypatch.setitem(sys.modules, "torch", torch)

    with pytest.raises(RuntimeError, match="intra-op thread budget"):
        train_ppo(
            FakeEnvironment(),  # type: ignore[arg-type]
            task=FakeTask(),
            config=FirstTrainingConfig(total_timesteps=128, checkpoint_interval=128),
            output_root=tmp_path,
        )


def test_training_rejects_unavailable_requested_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sb3 = ModuleType("stable_baselines3")
    sb3.__version__ = "test"  # type: ignore[attr-defined]
    sb3.PPO = FakeModel  # type: ignore[attr-defined]
    torch = fake_torch_module()
    torch.cuda = ModuleType("torch.cuda")  # type: ignore[attr-defined]
    torch.cuda.is_available = lambda: False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
    contrib = ModuleType("sb3_contrib")
    contrib.__version__ = "test"
    contrib.MaskablePPO = sb3.PPO
    monkeypatch.setitem(sys.modules, "sb3_contrib", contrib)
    monkeypatch.setitem(sys.modules, "torch", torch)

    with pytest.raises(RuntimeError, match="CUDA training was requested"):
        train_ppo(
            FakeEnvironment(),  # type: ignore[arg-type]
            task=FakeTask(),
            config=FirstTrainingConfig(
                total_timesteps=128,
                checkpoint_interval=128,
                training_device="cuda",
            ),
            output_root=tmp_path,
        )


@pytest.mark.parametrize(
    "runner_name", ["run_first_training.py", "run_scene_change_training.py"]
)
def test_training_runners_accept_resource_arguments_before_project_imports(
    runner_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = Path(__file__).resolve().parents[1] / "tools" / runner_name
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(runner),
            "--training-device",
            "auto",
            "--torch-threads",
            "6",
            "--torch-interop-threads",
            "2",
        ],
    )
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.delenv(name, raising=False)

    namespace = runpy.run_path(str(runner))
    args = namespace["parse_args"]()

    assert args.training_device == "auto"
    assert args.torch_threads == 6
    assert args.torch_interop_threads == 2
    assert os.environ["OMP_NUM_THREADS"] == "6"
    assert os.environ["MKL_NUM_THREADS"] == "6"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "6"


@pytest.mark.parametrize(
    "config",
    [
        FirstTrainingConfig(warm_start_policy=Path("policy.safetensors")),
        FirstTrainingConfig(warm_start_policy_manifest=Path("policy.manifest.json")),
        FirstTrainingConfig(
            resume_checkpoint=Path("checkpoint.zip"),
            warm_start_policy=Path("policy.safetensors"),
            warm_start_policy_manifest=Path("policy.manifest.json"),
        ),
        FirstTrainingConfig(
            warm_start_policy=Path("policy.safetensors"),
            warm_start_policy_manifest=Path("policy.manifest.json"),
            warm_start_policy_sha256="a" * 64,
        ),
    ],
)
def test_training_rejects_incomplete_or_ambiguous_warm_start(
    config: FirstTrainingConfig,
) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_training_writes_atomic_checkpoints_and_honest_manifest(
    tmp_path: Path, monkeypatch: Any
) -> None:
    sb3 = ModuleType("stable_baselines3")
    sb3.__version__ = "test"  # type: ignore[attr-defined]
    sb3.PPO = FakeModel  # type: ignore[attr-defined]
    torch = fake_torch_module()
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
    contrib = ModuleType("sb3_contrib")
    contrib.__version__ = "test"
    contrib.MaskablePPO = sb3.PPO
    monkeypatch.setitem(sys.modules, "sb3_contrib", contrib)
    monkeypatch.setitem(sys.modules, "torch", torch)
    environment = FakeEnvironment()
    result = train_ppo(
        environment,  # type: ignore[arg-type]
        task=FakeTask(),
        config=FirstTrainingConfig(total_timesteps=256, checkpoint_interval=128),
        output_root=tmp_path,
        provenance={"game_profile": "test-profile"},
    )
    assert result["status"] == "COMPLETED"
    assert result["completed_timesteps"] == 256
    assert (tmp_path / "checkpoints" / "step-128.zip").read_bytes() == b"checkpoint"
    assert (tmp_path / "checkpoints" / "step-256.zip").is_file()
    manifest = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
    assert manifest["diagnostic_training"] is False
    assert manifest["reward_version"] == "test_curriculum_v1"
    assert manifest["claim"] == "test claim"
    assert manifest["resumable_checkpoint"] is True
    assert manifest["provenance"]["game_profile"] == "test-profile"
    assert [entry["step"] for entry in manifest["checkpoints"]] == [128, 256]
    assert all(len(entry["sha256"]) == 64 for entry in manifest["checkpoints"])
    assert [entry["step"] for entry in manifest["portable_policies"]] == [128, 256]
    assert all(
        Path(entry["path"]).is_file()
        and Path(entry["manifest_path"]).is_file()
        and entry["format"] == "safetensors"
        for entry in manifest["portable_policies"]
    )
    assert environment.closed


def test_portable_policy_warm_starts_a_fresh_model_without_loading_zip(
    tmp_path: Path, monkeypatch: Any
) -> None:
    sb3 = ModuleType("stable_baselines3")
    sb3.__version__ = "test"  # type: ignore[attr-defined]
    sb3.PPO = FreshOnlyModel  # type: ignore[attr-defined]
    torch = fake_torch_module()
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
    contrib = ModuleType("sb3_contrib")
    contrib.__version__ = "test"
    contrib.MaskablePPO = sb3.PPO
    monkeypatch.setitem(sys.modules, "sb3_contrib", contrib)
    monkeypatch.setitem(sys.modules, "torch", torch)
    artifact = tmp_path / "source" / "policy.safetensors"
    sidecar = tmp_path / "source" / "policy.policy.manifest.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"received tensors")
    sidecar.write_text("{}", encoding="utf-8")
    imported: list[tuple[Path, Path]] = []

    def record_import(
        policy: object,
        artifact_path: Path,
        *,
        expected_compatibility_fingerprint: str,
        manifest_path: Path | None = None,
        expected_artifact_sha256: str | None = None,
        expected_manifest_sha256: str | None = None,
        expected_manifest_id: str | None = None,
    ) -> dict[str, Any]:
        del policy, expected_artifact_sha256, expected_manifest_sha256, expected_manifest_id
        assert len(expected_compatibility_fingerprint) == 64
        assert manifest_path is not None
        imported.append((artifact_path, manifest_path))
        return {
            "manifest_id": "c" * 64,
            "compatibility_fingerprint": expected_compatibility_fingerprint,
            "artifact": {
                "sha256": first_training.sha256_file(artifact_path),
                "size_bytes": artifact_path.stat().st_size,
            },
        }

    monkeypatch.setattr(first_training, "import_policy_artifact", record_import)
    result = train_ppo(
        FakeEnvironment(),  # type: ignore[arg-type]
        task=FakeTask(),
        config=FirstTrainingConfig(
            total_timesteps=128,
            checkpoint_interval=128,
            warm_start_policy=artifact,
            warm_start_policy_manifest=sidecar,
        ),
        output_root=tmp_path / "run",
    )

    assert imported == [(artifact.resolve(), sidecar.resolve())]
    assert result["completed_timesteps"] == 128
    assert result["warm_started_from"]["manifest_id"] == "c" * 64
    assert "path" not in result["warm_started_from"]


def test_constructor_failure_marks_manifest_failed_and_closes_environment(
    tmp_path: Path, monkeypatch: Any
) -> None:
    sb3 = ModuleType("stable_baselines3")
    sb3.__version__ = "test"  # type: ignore[attr-defined]
    sb3.PPO = FailingConstructorModel  # type: ignore[attr-defined]
    torch = fake_torch_module()
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
    contrib = ModuleType("sb3_contrib")
    contrib.__version__ = "test"
    contrib.MaskablePPO = sb3.PPO
    monkeypatch.setitem(sys.modules, "sb3_contrib", contrib)
    monkeypatch.setitem(sys.modules, "torch", torch)
    environment = FakeEnvironment()
    try:
        train_ppo(
            environment,  # type: ignore[arg-type]
            task=FakeTask(),
            config=FirstTrainingConfig(total_timesteps=128, checkpoint_interval=128),
            output_root=tmp_path,
        )
    except RuntimeError as error:
        assert "constructor exploded" in str(error)
    else:
        raise AssertionError("constructor failure was swallowed")
    manifest = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
    assert manifest["status"] == "FAILED"
    assert manifest["completed_timesteps"] == 0
    assert "constructor exploded" in manifest["failure"]
    assert environment.closed
    assert not (tmp_path / "checkpoints").exists()


def test_training_resumes_only_from_hash_validated_compatible_checkpoint(
    tmp_path: Path, monkeypatch: Any
) -> None:
    sb3 = ModuleType("stable_baselines3")
    sb3.__version__ = "test"  # type: ignore[attr-defined]
    sb3.PPO = FakeModel  # type: ignore[attr-defined]
    torch = fake_torch_module()
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
    contrib = ModuleType("sb3_contrib")
    contrib.__version__ = "test"
    contrib.MaskablePPO = sb3.PPO
    monkeypatch.setitem(sys.modules, "sb3_contrib", contrib)
    monkeypatch.setitem(sys.modules, "torch", torch)
    source = tmp_path / "source"
    provenance = {"game_profile": "development", "build": "same"}
    train_ppo(
        FakeEnvironment(),  # type: ignore[arg-type]
        task=FakeTask(),
        config=FirstTrainingConfig(total_timesteps=128, checkpoint_interval=128),
        output_root=source,
        provenance=provenance,
    )
    checkpoint = source / "checkpoints" / "step-128.zip"
    portable_bundle = tmp_path / "downloaded-candidate"
    portable_checkpoint = portable_bundle / "checkpoints" / checkpoint.name
    portable_checkpoint.parent.mkdir(parents=True)
    shutil.copyfile(checkpoint, portable_checkpoint)
    shutil.copyfile(source / "manifest.json", portable_bundle / "manifest.json")
    resumed = train_ppo(
        FakeEnvironment(),  # type: ignore[arg-type]
        task=FakeTask(),
        config=FirstTrainingConfig(
            total_timesteps=256,
            checkpoint_interval=128,
            resume_checkpoint=portable_checkpoint,
        ),
        output_root=tmp_path / "resumed",
        provenance={**provenance, "local_install_path": r"D:\Games\Hollow Knight"},
    )
    assert resumed["completed_timesteps"] == 256
    assert resumed["resumed_from"]["sha256"] == resumed["checkpoints"][-1]["sha256"]

    incompatible_environment = FakeEnvironment()
    try:
        train_ppo(
            incompatible_environment,  # type: ignore[arg-type]
            task=FakeTask(),
            config=FirstTrainingConfig(
                total_timesteps=256,
                checkpoint_interval=128,
                resume_checkpoint=portable_checkpoint,
            ),
            output_root=tmp_path / "incompatible",
            provenance={"game_profile": "different-build", "build": "same"},
        )
    except ValueError as error:
        assert "provenance differs" in str(error)
    else:
        raise AssertionError("incompatible cross-machine checkpoint was accepted")
    assert incompatible_environment.closed


def test_invalid_transition_fails_run_without_checkpoint(
    tmp_path: Path, monkeypatch: Any
) -> None:
    sb3 = ModuleType("stable_baselines3")
    sb3.__version__ = "test"  # type: ignore[attr-defined]
    sb3.PPO = InvalidTransitionModel  # type: ignore[attr-defined]
    torch = fake_torch_module()
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
    contrib = ModuleType("sb3_contrib")
    contrib.__version__ = "test"
    contrib.MaskablePPO = sb3.PPO
    monkeypatch.setitem(sys.modules, "sb3_contrib", contrib)
    monkeypatch.setitem(sys.modules, "torch", torch)
    environment = FakeEnvironment()
    try:
        train_ppo(
            environment,  # type: ignore[arg-type]
            task=FakeTask(),
            config=FirstTrainingConfig(total_timesteps=128, checkpoint_interval=128),
            output_root=tmp_path,
        )
    except InvalidTrainingTransition:
        pass
    else:
        raise AssertionError("invalid transition did not abort training")
    manifest = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
    assert manifest["status"] == "FAILED"
    assert manifest["completed_timesteps"] == 0
    assert "InvalidTrainingTransition" in manifest["failure"]
    assert manifest["checkpoints"] == []
    assert environment.closed
    assert not (tmp_path / "checkpoints").exists()


def test_stop_request_cancels_with_final_resumable_checkpoint(
    tmp_path: Path, monkeypatch: Any
) -> None:
    stop_file = tmp_path / "stop.request"
    StopDuringRolloutModel.stop_file = stop_file
    sb3 = ModuleType("stable_baselines3")
    sb3.__version__ = "test"  # type: ignore[attr-defined]
    sb3.PPO = StopDuringRolloutModel  # type: ignore[attr-defined]
    torch = fake_torch_module()
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
    contrib = ModuleType("sb3_contrib")
    contrib.__version__ = "test"
    contrib.MaskablePPO = sb3.PPO
    monkeypatch.setitem(sys.modules, "sb3_contrib", contrib)
    monkeypatch.setitem(sys.modules, "torch", torch)
    environment = FakeEnvironment()

    result = train_ppo(
        environment,  # type: ignore[arg-type]
        task=FakeTask(),
        config=FirstTrainingConfig(total_timesteps=128, checkpoint_interval=128),
        output_root=tmp_path / "run",
        stop_file=stop_file,
    )

    assert result["status"] == "CANCELLED"
    assert result["completed_timesteps"] == 17
    assert [entry["step"] for entry in result["checkpoints"]] == [17]
    assert result["resumable_checkpoint"] is True
    assert environment.closed
    assert (tmp_path / "run" / "checkpoints" / "step-17.zip").is_file()


def test_continuous_training_runs_until_stop_and_rotates_checkpoints(
    tmp_path: Path, monkeypatch: Any
) -> None:
    stop_file = tmp_path / "stop.request"
    ContinuousStopModel.stop_file = stop_file
    sb3 = ModuleType("stable_baselines3")
    sb3.__version__ = "test"  # type: ignore[attr-defined]
    sb3.PPO = ContinuousStopModel  # type: ignore[attr-defined]
    torch = fake_torch_module()
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
    contrib = ModuleType("sb3_contrib")
    contrib.__version__ = "test"
    contrib.MaskablePPO = sb3.PPO
    monkeypatch.setitem(sys.modules, "sb3_contrib", contrib)
    monkeypatch.setitem(sys.modules, "torch", torch)

    result = train_ppo(
        FakeEnvironment(),  # type: ignore[arg-type]
        task=FakeTask(),
        config=FirstTrainingConfig(
            total_timesteps=128,
            checkpoint_interval=128,
            continuous=True,
            max_checkpoints=2,
        ),
        output_root=tmp_path / "run",
        stop_file=stop_file,
    )

    assert result["status"] == "CANCELLED"
    assert result["duration_mode"] == "continuous"
    assert result["total_timesteps"] is None
    assert result["completed_timesteps"] == 512
    assert [entry["step"] for entry in result["checkpoints"]] == [384, 512]
    assert [entry["step"] for entry in result["pruned_checkpoints"]] == [128, 256]
    assert not (tmp_path / "run" / "checkpoints" / "step-128.zip").exists()
    assert [entry["step"] for entry in result["portable_policies"]] == [384, 512]
    assert not (tmp_path / "run" / "policies" / "step-128.safetensors").exists()
    assert not (
        tmp_path / "run" / "policies" / "step-128.policy.manifest.json"
    ).exists()


def test_environment_run_failed_does_not_abort_continuous_campaign(
    tmp_path: Path, monkeypatch: Any
) -> None:
    sb3 = ModuleType("stable_baselines3")
    sb3.__version__ = "test"  # type: ignore[attr-defined]
    sb3.PPO = RunFailedModel  # type: ignore[attr-defined]
    torch = fake_torch_module()
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
    contrib = ModuleType("sb3_contrib")
    contrib.__version__ = "test"
    contrib.MaskablePPO = sb3.PPO
    monkeypatch.setitem(sys.modules, "sb3_contrib", contrib)
    monkeypatch.setitem(sys.modules, "torch", torch)

    stop_file = tmp_path / "stop.request"
    RunFailedModel.stop_file = stop_file
    result = train_ppo(
        FakeEnvironment(),  # type: ignore[arg-type]
        task=FakeTask(),
        config=FirstTrainingConfig(continuous=True),
        output_root=tmp_path / "run",
        stop_file=stop_file,
    )
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text("utf-8"))
    assert result["status"] == "CANCELLED"
    assert manifest["gameplay_run_failures"] == 1
