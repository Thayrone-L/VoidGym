from __future__ import annotations

import os
import subprocess
import sys

from tools.v2.probe_hk_policy import ROOT, _worker_pythonpath


def test_episode_worker_can_import_checkout_from_external_working_directory(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    worker_env = os.environ.copy()
    worker_env["PYTHONPATH"] = _worker_pythonpath("")

    result = subprocess.run(
        [sys.executable, "-c", "import hk_agent.episode_worker"],
        cwd=tmp_path,
        env=worker_env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert str(ROOT / "src") in worker_env["PYTHONPATH"].split(os.pathsep)
