"""Sobe supervisor, learner e dois atores; verifica falha, parada e retomada."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from voidgym.core.experience_store import ExperienceReader

ROOT = Path(__file__).resolve().parents[3]
SUPERVISOR = ROOT / "tools" / "v2" / "run_supervisor.py"


def _wait_for_actor(root: Path, timeout: float = 20.0) -> int:
    pid_file = root / "actors" / "actor-1" / "pid"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pid_file.exists():
            return int(pid_file.read_text(encoding="utf-8"))
        time.sleep(0.02)
    raise TimeoutError("O supervisor não iniciou o ator 1")


def _run_supervisor(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SUPERVISOR), "--root", str(root), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=150,
        check=False,
    )


@pytest.mark.e2e
def test_supervisor_actor_failure_stop_and_resume(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    process = subprocess.Popen(
        [sys.executable, str(SUPERVISOR), "--root", str(root), "--rounds", "3"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    actor_pid = _wait_for_actor(root)
    os.kill(actor_pid, signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=150)
    assert process.returncode == 0, f"{stdout}\n{stderr}"
    progress = json.loads((root / "state" / "progress.json").read_text(encoding="utf-8"))
    assert progress["completed_rounds"] == 3
    assert progress["actor_restarts"] >= 1
    _assert_campaign_artifacts(root)
    resumed = _run_supervisor(root, "--rounds", "4", "--resume")
    assert resumed.returncode == 0, f"{resumed.stdout}\n{resumed.stderr}"
    progress = json.loads((root / "state" / "progress.json").read_text(encoding="utf-8"))
    assert progress["completed_rounds"] == 4
    assert (root / "checkpoints" / "round-000004.json").exists()


def _assert_campaign_artifacts(root: Path) -> None:
    for name in ("config.json", "hardware.json", "resource_plan.json"):
        assert (root / name).is_file()
    assert (root / "checkpoints" / "round-000003.json").is_file()
    for actor in ("actor-1", "actor-2"):
        store = root / "runtime" / "v2" / "experience" / "fake" / f"e2e-{actor}"
        blocks = list(ExperienceReader(store).iter_blocks())
        assert len(blocks) >= 3
