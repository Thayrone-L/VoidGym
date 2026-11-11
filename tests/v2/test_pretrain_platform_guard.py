from __future__ import annotations

import csv
import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest
from tools.v2 import pretrain_platform
from voidgym.core.hardware import HardwareProfile


def _profile(*, free_vram: float | None) -> HardwareProfile:
    return HardwareProfile(8, 16, 16.0, 5.0, "NVIDIA RTX", 12.0,
                           free_vram, "8.6", 100.0, "Windows")


def test_nvidia_smi_gpu_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pretrain_platform.subprocess, "run", lambda *args, **kwargs:  # type: ignore[attr-defined]
                        CompletedProcess(args[0], 0, "GPU 0: NVIDIA RTX\n", ""))
    assert pretrain_platform._nvidia_smi_available()


def test_curriculum_stop_level_is_configurable(tmp_path: Path) -> None:
    args = pretrain_platform._parse([
        "--runtime-root", str(tmp_path), "--stop-at-curriculum-level", "6",
    ])
    assert args.stop_at_curriculum_level == 6


def test_cpu_torch_on_nvidia_machine_requires_allow_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(pretrain_platform, "_priority_below_normal", lambda: None)
    monkeypatch.setattr(pretrain_platform, "_nvidia_smi_available", lambda: True)
    monkeypatch.setattr(pretrain_platform.torch.cuda, "is_available", lambda: False)  # type: ignore[attr-defined]
    monkeypatch.setattr(pretrain_platform, "detect_hardware",
                        lambda _root: _profile(free_vram=None))
    args = pretrain_platform._parse(["--runtime-root", str(tmp_path)])
    with pytest.raises(SystemExit, match="--allow-cpu"):
        pretrain_platform.run(args)
    assert "\033[91m" in capsys.readouterr().err
    assert not (tmp_path / "v2" / "pretrain").exists()


def test_resource_plan_cpu_fallback_on_nvidia_machine_requires_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pretrain_platform, "_priority_below_normal", lambda: None)
    monkeypatch.setattr(pretrain_platform, "_nvidia_smi_available", lambda: True)
    monkeypatch.setattr(pretrain_platform.torch.cuda, "is_available", lambda: True)  # type: ignore[attr-defined]
    monkeypatch.setattr(pretrain_platform, "detect_hardware",
                        lambda _root: _profile(free_vram=5.0))
    args = pretrain_platform._parse(["--runtime-root", str(tmp_path)])
    with pytest.raises(SystemExit, match="--allow-cpu"):
        pretrain_platform.run(args)
    assert not (tmp_path / "v2" / "pretrain").exists()


def test_short_training_writes_quick_evaluation_and_terminal_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pretrain_platform, "_priority_below_normal", lambda: None)
    monkeypatch.setattr(pretrain_platform, "_nvidia_smi_available", lambda: False)
    monkeypatch.setattr(pretrain_platform, "detect_hardware", lambda _root: _profile(free_vram=None))
    monkeypatch.setattr(pretrain_platform.torch.cuda, "is_available", lambda: False)  # type: ignore[attr-defined]
    monkeypatch.setattr(pretrain_platform, "_quick_eval", lambda *_args, **_kwargs: .5)
    monkeypatch.setattr(pretrain_platform, "_save_checkpoint", lambda *_args, **_kwargs: None)

    class TinyPolicy(pretrain_platform.torch.nn.Module):
        def initial_state(self, count, device):
            return (pretrain_platform.torch.zeros((count, 1), device=device),
                    pretrain_platform.torch.zeros((count, 1), device=device))

        def forward(self, _observations, state, _starts, masks=None, privileged=None):
            del privileged
            if masks is None:
                batch = state[0].shape[0]
                logits = pretrain_platform.torch.zeros((batch, 43), device=state[0].device)
            else:
                logits = pretrain_platform.torch.where(
                    masks.bool(),
                    pretrain_platform.torch.zeros_like(masks, dtype=pretrain_platform.torch.float32),
                    pretrain_platform.torch.full_like(masks, -1e9, dtype=pretrain_platform.torch.float32),
                )
            values = pretrain_platform.torch.zeros((state[0].shape[0],), device=state[0].device)
            return {"logits": logits, "values": values, "state": state}

    monkeypatch.setattr(pretrain_platform, "UniversalPolicy", TinyPolicy)

    class TinyTrainer:
        optimizer = None

        def __init__(self, *_args, **_kwargs):
            pass

        def update(self, *_args, **_kwargs) -> dict[str, float]:
            return {key: 0.0 for key in ("L_ppo", "L_semantic", "L_inverse_dynamics",
                                         "L_hero_motion", "entropy", "kl", "explained_variance")}

    monkeypatch.setattr(pretrain_platform, "RecurrentPPOTrainer", TinyTrainer)
    args = pretrain_platform._parse([
        "--runtime-root", str(tmp_path), "--run-name", "quick-eval-test",
        "--total-steps", "12000", "--envs", "1", "--rollout-steps", "256",
        "--seq-len", "32", "--batch-size", "32", "--n-epochs", "1",
        "--device", "cpu", "--allow-cpu", "--no-augment",
        "--quick-eval-interval-steps", "5000",
    ])
    root = pretrain_platform.run(args)
    with (root / "log.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    evaluations = [row["quick_eval_success"] for row in rows if row["quick_eval_success"]]
    assert len(evaluations) >= 2
    assert all(value == "0.5" for value in evaluations)
    assert (root / "heartbeat.json").is_file()
    assert json.loads((root / "status.json").read_text())["status"] == "COMPLETED"


def test_training_exception_records_failed_status_and_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "v2" / "pretrain" / "forced-failure"

    def fail_after_heartbeat(_args) -> Path:
        root.mkdir(parents=True)
        pretrain_platform._ACTIVE_RUN_INITIALIZED = True
        (root / "heartbeat.json").write_text('{"steps":256}', encoding="utf-8")
        raise RuntimeError("forced update failure")

    monkeypatch.setattr(pretrain_platform, "_run_impl", fail_after_heartbeat)
    args = pretrain_platform._parse(["--runtime-root", str(tmp_path), "--run-name", "forced-failure"])
    with pytest.raises(RuntimeError, match="forced update failure"):
        pretrain_platform.run(args)
    assert json.loads((root / "status.json").read_text())["status"] == "FAILED"
    assert "RuntimeError: forced update failure" in (root / "stderr.log").read_text()


def test_old_training_log_header_is_extended_without_losing_rows(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    path.write_text("steps,quick_eval_success\n1000,0.5\n", encoding="utf-8")
    pretrain_platform._ensure_csv_header(
        path, ["steps", "quick_eval_success", "success_rate_level_2"],
    )
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [{"steps": "1000", "quick_eval_success": "0.5", "success_rate_level_2": ""}]


def test_pretrain_commit_guard_waits_for_current_commit_recovery(tmp_path: Path) -> None:
    gib = 2**30
    snapshots = iter([
        {"commit_available_bytes": 2 * gib, "potential_commit_available_bytes": 4 * gib},
        {"commit_available_bytes": 4 * gib, "potential_commit_available_bytes": 4 * gib},
    ])
    now = [0.0]
    heartbeat = tmp_path / "heartbeat.json"
    heartbeat.write_text(json.dumps({"success_by_level": {"9": 0.5}}), encoding="utf-8")

    result = pretrain_platform._wait_for_pretrain_commit(
        tmp_path, 1234, next(snapshots), snapshot_fn=lambda: next(snapshots),
        timeout_seconds=10, poll_seconds=5, monotonic_fn=lambda: now[0],
        sleep_fn=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    state = json.loads(heartbeat.read_text(encoding="utf-8"))
    assert result["commit_available_bytes"] == 4 * gib
    assert now[0] == 5
    assert state["steps"] == 1234
    assert state["phase"] == "training"
    assert state["waiting_memory"] is False
    assert state["success_by_level"] == {"9": 0.5}
    assert "memory_wait_started_at" not in state


def test_pretrain_commit_guard_fails_after_wait_timeout(tmp_path: Path) -> None:
    gib = 2**30
    low = {"commit_available_bytes": 2 * gib, "potential_commit_available_bytes": 4 * gib}
    now = [0.0]

    with pytest.raises(MemoryError, match="stayed below 3 GiB"):
        pretrain_platform._wait_for_pretrain_commit(
            tmp_path, 987, low, snapshot_fn=lambda: low,
            timeout_seconds=10, poll_seconds=5, monotonic_fn=lambda: now[0],
            sleep_fn=lambda seconds: now.__setitem__(0, now[0] + seconds),
        )

    failure = json.loads((tmp_path / "memory_guard_failure.json").read_text(encoding="utf-8"))
    assert failure["steps"] == 987
    assert failure["wait_seconds"] == 10
    assert "timed out" in failure["reason"]


def test_pretrain_commit_guard_fails_when_potential_commit_is_unsafe(tmp_path: Path) -> None:
    gib = 2**30
    low = {"commit_available_bytes": gib, "potential_commit_available_bytes": 2 * gib}

    with pytest.raises(MemoryError, match="potential commit fell below 3 GiB"):
        pretrain_platform._wait_for_pretrain_commit(
            tmp_path, 654, low, snapshot_fn=lambda: low,
        )

    failure = json.loads((tmp_path / "memory_guard_failure.json").read_text(encoding="utf-8"))
    assert failure["steps"] == 654
    assert failure["wait_seconds"] == 0
