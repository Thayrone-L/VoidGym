from __future__ import annotations

import json
from pathlib import Path
import sys

from tools.v2.supervise_pretrain import supervise


def _write_fake(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_unclean_exit_is_resumed_from_latest_checkpoint(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    checkpoint = tmp_path / "initial"
    checkpoint.mkdir()
    fake = _write_fake(tmp_path / "fake_resume.py", """
import json, os, pathlib, sys
attempt = int(sys.argv[1])
resume = pathlib.Path(sys.argv[2])
root = pathlib.Path(sys.argv[3])
if attempt == 1:
    ckpt = root / 'universal-004600000'
    ckpt.mkdir()
    (ckpt / 'manifest.json').write_text(json.dumps({'trained_steps': 4600000}))
    for name in ('encoder.pt', 'core.pt', 'perception.pt', 'aux_heads.pt'):
        (ckpt / name).write_bytes(b'ok')
    os._exit(3)
assert resume.name == 'universal-004600000'
(root / 'status.json').write_text(json.dumps({'status': 'COMPLETED'}))
""")
    seen: list[str] = []

    def builder(resume: Path, attempt: int) -> list[str]:
        seen.append(resume.name)
        return [sys.executable, str(fake), str(attempt), str(resume), str(run_root)]

    result = supervise(builder, run_root=run_root, initial_checkpoint=checkpoint)
    assert result == 0
    assert seen == ["initial", "universal-004600000"]
    assert json.loads((run_root / "status.json").read_text()) == {"status": "COMPLETED"}
    assert len(json.loads((run_root / "supervisor-history.json").read_text())["crashes"]) == 1


def test_three_consecutive_native_deaths_end_failed(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    checkpoint = tmp_path / "initial"
    checkpoint.mkdir()
    fake = _write_fake(tmp_path / "fake_crash.py", "import os; os._exit(3)\n")
    attempts: list[int] = []

    def builder(resume: Path, attempt: int) -> list[str]:
        del resume
        attempts.append(attempt)
        return [sys.executable, str(fake)]

    result = supervise(builder, run_root=run_root, initial_checkpoint=checkpoint,
                       max_restarts=3)
    status = json.loads((run_root / "status.json").read_text())
    assert result == 3
    assert attempts == [1, 2, 3]
    assert status["status"] == "FAILED"
    assert status["crash_count_in_window"] == 3
