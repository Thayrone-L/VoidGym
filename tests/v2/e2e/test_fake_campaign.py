"""Supervisor com processos learner/atores, relançamento e retomada sem o jogo."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from voidgym.core.config import config_from_dict, save_campaign_config
from voidgym.core.experience_store import ExperienceReader
from voidgym.core.hardware import HardwareProfile, plan_resources, save_resource_profile

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ACTOR = PROJECT_ROOT / "tools" / "v2" / "run_actor.py"
LEARNER = PROJECT_ROOT / "tools" / "v2" / "run_learner.py"


def _start(script: Path, *args: str) -> subprocess.Popen[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)))
    return subprocess.Popen([sys.executable, str(script), *args], cwd=PROJECT_ROOT,
                            env=environment, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)


def _wait_checkpoint(path: Path, expected_round: int, timeout: float = 60.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except PermissionError:
                time.sleep(0.02)
                continue
            if value.get("round") == expected_round:
                return value
        time.sleep(0.02)
    raise TimeoutError(f"learner não chegou à rodada {expected_round}")


def _finish(process: subprocess.Popen[str]) -> None:
    stdout, stderr = process.communicate(timeout=20)
    assert process.returncode == 0, f"{process.args}\nstdout={stdout}\nstderr={stderr}"


@pytest.mark.e2e
@pytest.mark.parametrize("game_id,adapter_name", [("fake_platformer", "fake"), ("platform_gym", "platform_gym")])
def test_campaign_process_restart_safe_stop_and_resume(tmp_path: Path, game_id: str, adapter_name: str) -> None:
    campaign = config_from_dict({"schema_version": 1, "campaign_id": "e2e",
                                 "kind": "e2e", "game_id": game_id})
    root = tmp_path / "campaign"
    save_campaign_config(campaign, root / "config.json")
    profile = HardwareProfile(8, 16, 16, 12, None, None, None, None, 100, "test")
    save_resource_profile(profile, plan_resources(profile), root)
    checkpoint = root / "checkpoint.json"
    processes: list[subprocess.Popen[str]] = []
    try:
        learner = _start(LEARNER, "--root", str(root), "--checkpoint", str(checkpoint),
                         "--stop-after-round", "4")
        processes.append(learner)
        actor_a = _start(ACTOR, "--root", str(root), "--actor-id", "actor-a",
                         "--first-round", "1", "--last-round", "4", "--pause-after", "3", "--adapter", adapter_name)
        actor_b = _start(ACTOR, "--root", str(root), "--actor-id", "actor-b",
                         "--first-round", "1", "--last-round", "3", "--pause-after", "3", "--adapter", adapter_name)
        processes.extend((actor_a, actor_b))
        _wait_checkpoint(checkpoint, 3)
        actor_b.terminate()
        actor_b.communicate(timeout=10)
        assert actor_b.returncode != 0  # encerrado pelo supervisor de teste para simular falha.
        actor_b_restarted = _start(ACTOR, "--root", str(root), "--actor-id", "actor-b",
                                   "--first-round", "4", "--last-round", "4", "--adapter", adapter_name)
        processes.append(actor_b_restarted)
        (root / "resume").mkdir(parents=True, exist_ok=True)
        (root / "resume" / "actor-a").write_text("resume", encoding="utf-8")
        final = _wait_checkpoint(checkpoint, 4)
        assert final["updates"] == 4
        _finish(learner)
        _finish(actor_a)
        _finish(actor_b_restarted)

        learner_resume = _start(LEARNER, "--root", str(root), "--checkpoint", str(checkpoint),
                                "--stop-after-round", "5")
        actor_a_resume = _start(ACTOR, "--root", str(root), "--actor-id", "actor-a",
                                "--first-round", "5", "--last-round", "5", "--adapter", adapter_name)
        actor_b_resume = _start(ACTOR, "--root", str(root), "--actor-id", "actor-b",
                                "--first-round", "5", "--last-round", "5", "--adapter", adapter_name)
        processes.extend((learner_resume, actor_a_resume, actor_b_resume))
        resumed = _wait_checkpoint(checkpoint, 5)
        assert resumed["updates"] == 5
        _finish(learner_resume)
        _finish(actor_a_resume)
        _finish(actor_b_resume)

        assert (root / "config.json").is_file()
        assert (root / "hardware.json").is_file()
        assert (root / "resource_plan.json").is_file()
        assert (root / "policy" / "round-5.json").is_file()
        assert list(ExperienceReader(root / "experience"))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
