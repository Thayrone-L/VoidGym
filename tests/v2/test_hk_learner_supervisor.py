from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from tools.v2.supervise_hk_learner import _latest_progress, supervise


def test_latest_progress_accepts_integral_decimal_shared_round(tmp_path: Path) -> None:
    (tmp_path / "latest.json").write_text(json.dumps({
        "trained_steps": 12, "metrics": {"shared_round": 1.0},
    }), "utf-8")
    assert _latest_progress(tmp_path)["round"] == 1
    (tmp_path / "latest.json").write_text(json.dumps({
        "trained_steps": 12, "metrics": {"shared_round": 1.5},
    }), "utf-8")
    assert _latest_progress(tmp_path)["round"] == 0


def _checkpoint(path: Path, *, steps: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(json.dumps({
        "trained_steps": steps, "metrics": {"shared_round": steps // 2},
        "experts": ["platform"], "game_ids": [],
    }), "utf-8")
    for name in ("encoder.pt", "perception.pt", "semantic_skip.pt", "aux_heads.pt",
                 "core.pt", "critic.pt", "actor/platform.pt"):
        item = path / name
        item.parent.mkdir(parents=True, exist_ok=True)
        item.write_bytes(b"valid")


def test_supervisor_resumes_from_latest_complete_policy_after_learner_crash(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    fallback = tmp_path / "base"
    _checkpoint(fallback, steps=10)
    script = tmp_path / "fake_learner.py"
    script.write_text("""
import json, pathlib, sys
root, counter = map(pathlib.Path, sys.argv[1:])
attempt = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(attempt))
if attempt == 1:
    ckpt = root / 'policies' / 'universal-000020'
    ckpt.mkdir(parents=True)
    (ckpt / 'manifest.json').write_text(json.dumps({'trained_steps': 20, 'metrics': {'shared_round': 4}, 'experts': ['platform'], 'game_ids': []}))
    for name in ('encoder.pt', 'perception.pt', 'semantic_skip.pt', 'aux_heads.pt', 'core.pt', 'critic.pt', 'actor/platform.pt'):
        path = ckpt / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'ok')
    (root / 'latest.json').write_text(json.dumps({'round': 4, 'policy': 'policies/universal-000020', 'trained_steps': 20}))
    (root / 'status.json').write_text(json.dumps({'status': 'FAILED'}))
    sys.exit(3)
latest = json.loads((root / 'latest.json').read_text())
assert latest['policy'] == 'policies/universal-000020'
(root / 'status.json').write_text(json.dumps({'status': 'COMPLETED'}))
""", encoding="utf-8")
    counter = tmp_path / "attempts"
    result = supervise([sys.executable, str(script), str(shared), str(counter)],
                       shared_root=shared, stop_file=tmp_path / "STOP",
                       fallback_checkpoint=fallback)
    assert result == 0
    assert counter.read_text() == "2"
    assert json.loads((shared / "supervisor.json").read_text())["restart_count"] == 1
    assert (shared / "learner-restarts.jsonl").is_file()


def test_supervisor_stops_after_three_restarts_in_rolling_24_hours(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    fallback = tmp_path / "base"
    _checkpoint(fallback, steps=10)
    checkpoint = shared / "policies" / "universal-000020"
    _checkpoint(checkpoint, steps=20)
    checkpoint_manifest = json.loads((checkpoint / "manifest.json").read_text("utf-8"))
    checkpoint_manifest["metrics"]["shared_round"] = 4
    (checkpoint / "manifest.json").write_text(json.dumps(checkpoint_manifest), "utf-8")
    (shared / "latest.json").write_text(json.dumps({
        "round": 4, "policy": "policies/universal-000020", "trained_steps": 20,
    }), "utf-8")
    (shared / "learner.json").write_text(json.dumps({
        "status": "FAILED", "completed_timesteps": 10, "round": 1,
    }), "utf-8")
    script = shared / "fake_crash.py"
    script.write_text(
        "import json, pathlib; pathlib.Path(__file__).parent.joinpath('status.json').write_text(json.dumps({'status':'FAILED'})); raise SystemExit(3)\n",
        encoding="utf-8",
    )
    result = supervise([sys.executable, str(script)], shared_root=shared,
                       stop_file=tmp_path / "STOP", fallback_checkpoint=fallback)
    status = json.loads((shared / "status.json").read_text())
    assert result == 3
    assert status["status"] == "FAILED"
    assert status["restart_count_24h"] == 4
    assert "3 per 24 hours" in status["reason"]
    assert status["completed_timesteps"] == 20
    assert status["round"] == 4
    assert status["policy"] == "policies/universal-000020"


def test_supervisor_recovery_window_is_fixed(tmp_path: Path) -> None:
    fallback = tmp_path / "base"
    _checkpoint(fallback, steps=10)
    with pytest.raises(ValueError, match="allows 0–3 restarts in a rolling 24 hours"):
        supervise([sys.executable, "-c", "pass"], shared_root=tmp_path / "shared",
                  stop_file=tmp_path / "STOP", fallback_checkpoint=fallback,
                  max_restarts=4)


def test_campaign_owner_mode_exits_after_one_learner_failure(tmp_path: Path) -> None:
    shared = tmp_path / "shared-owner"
    shared.mkdir()
    fallback = tmp_path / "base-owner"
    _checkpoint(fallback, steps=10)
    script = tmp_path / "owner_failure.py"
    script.write_text(
        "import json, pathlib, sys; pathlib.Path(sys.argv[1], 'learner.json').write_text(json.dumps({'status':'FAILED'})); raise SystemExit(3)\n",
        encoding="utf-8",
    )
    result = supervise([sys.executable, str(script), str(shared)], shared_root=shared,
                       stop_file=tmp_path / "STOP-owner", fallback_checkpoint=fallback,
                       max_restarts=0)
    assert result == 3
    assert json.loads((shared / "status.json").read_text(encoding="utf-8"))["status"] == "FAILED"
    assert json.loads((shared / "supervisor.json").read_text(encoding="utf-8"))["status"] == "FAILED"
    assert json.loads((shared / "learner.json").read_text(encoding="utf-8"))["status"] == "FAILED"
