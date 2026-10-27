from __future__ import annotations

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
