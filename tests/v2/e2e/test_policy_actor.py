from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter
from voidgym.models.policy import UniversalPolicy
from voidgym.training.checkpoints import save_universal


def test_platform_actor_runs_universal_checkpoint_on_cpu(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[3]
    checkpoint = tmp_path / "universal"
    action_table = tuple(PlatformGymAdapter(level=0).action_table())
    save_universal(checkpoint, UniversalPolicy(), action_table, steps=0)
    root = tmp_path / "campaign"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(project_root / "src"), str(project_root)))
    result = subprocess.run([
        sys.executable, str(project_root / "tools" / "v2" / "run_actor.py"),
        "--root", str(root), "--actor-id", "policy-test", "--first-round", "1",
        "--last-round", "1", "--adapter", "platform_gym",
        "--policy-checkpoint", str(checkpoint),
    ], check=False, cwd=project_root, env=environment, text=True,
        capture_output=True, timeout=60)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert (root / "rollouts" / "round-1-policy-test.json").is_file()
    assert (root / "experience").is_dir()
