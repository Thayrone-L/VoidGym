from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from hk_agent.first_training import FirstTrainingConfig, train_ppo
from hk_agent.training_task import InvalidTrainingTransition


class FakeModel:
    def __init__(self, *_: object, **__: object) -> None:
        self.num_timesteps = 0

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
        assert device == "auto"
        model = cls()
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


class FailingConstructorModel:
    def __init__(self, *_: object, **__: object) -> None:
        raise RuntimeError("constructor exploded")


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


def test_training_writes_atomic_checkpoints_and_honest_manifest(
    tmp_path: Path, monkeypatch: Any
) -> None:
    sb3 = ModuleType("stable_baselines3")
    sb3.__version__ = "test"  # type: ignore[attr-defined]
    sb3.PPO = FakeModel  # type: ignore[attr-defined]
    torch = ModuleType("torch")
    torch.__version__ = "test"  # type: ignore[attr-defined]
    torch.manual_seed = lambda _: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
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
    assert environment.closed


def test_constructor_failure_marks_manifest_failed_and_closes_environment(
    tmp_path: Path, monkeypatch: Any
) -> None:
    sb3 = ModuleType("stable_baselines3")
    sb3.__version__ = "test"  # type: ignore[attr-defined]
    sb3.PPO = FailingConstructorModel  # type: ignore[attr-defined]
    torch = ModuleType("torch")
    torch.__version__ = "test"  # type: ignore[attr-defined]
    torch.manual_seed = lambda _: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
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
    torch = ModuleType("torch")
    torch.__version__ = "test"  # type: ignore[attr-defined]
    torch.manual_seed = lambda _: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
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
    torch = ModuleType("torch")
    torch.__version__ = "test"  # type: ignore[attr-defined]
    torch.manual_seed = lambda _: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
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
    torch = ModuleType("torch")
    torch.__version__ = "test"  # type: ignore[attr-defined]
    torch.manual_seed = lambda _: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
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
    torch = ModuleType("torch")
    torch.__version__ = "test"  # type: ignore[attr-defined]
    torch.manual_seed = lambda _: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
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


def test_environment_run_failed_does_not_abort_continuous_campaign(
    tmp_path: Path, monkeypatch: Any
) -> None:
    sb3 = ModuleType("stable_baselines3")
    sb3.__version__ = "test"  # type: ignore[attr-defined]
    sb3.PPO = RunFailedModel  # type: ignore[attr-defined]
    torch = ModuleType("torch")
    torch.__version__ = "test"  # type: ignore[attr-defined]
    torch.manual_seed = lambda _: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
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
