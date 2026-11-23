from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools import soak_parallel_training as soak


@pytest.mark.parametrize("campaign_type", ["visual_motion_training", "scene_change_training"])
def test_campaign_is_continuous_cpu_sized_training(campaign_type: str) -> None:
    payload = soak._campaign_payload("unit-soak", campaign_type)
    assert payload["campaign_type"] == campaign_type
    assert payload["duration_mode"] == "continuous"
    assert payload["checkpoint_interval"] == 2048
    assert payload["max_checkpoints"] == 3
    assert payload["window_width"] == 960
    assert payload["window_height"] == 540
    assert payload["save_source"] == "existing_save"
    assert payload["save_slot"] == 1


def test_shared_exploration_campaign_uses_prompt_45_parameters() -> None:
    payload = soak._shared_campaign_payload("shared-test", "test")
    assert payload["campaign_type"] == "exploration_training"
    assert payload["shared_learner"] is True
    assert payload["actors"] == 2
    assert payload["rollout_steps"] == 512
    assert payload["batch_size"] == 256
    assert payload["horizon"] == 3000
    assert payload["checkpoint_interval"] == 2048
    assert payload["weights"]["novelty_reward"] == 0.05
    assert payload["weights"]["death_penalty"] == -5.0


def test_soak_control_places_output_under_launch_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    control = object.__new__(soak._SoakControlPlane)
    control.runtime_root = tmp_path / "runtime"
    monkeypatch.setattr(soak.ControlPlane, "build_command", lambda *args, **kwargs: ["runner"])
    command = soak._SoakControlPlane.build_command(
        control,
        None,
        execute=True,
        stop_file=tmp_path / "control" / "launches" / "launch-abc" / "stop.request",
    )
    assert command[-2:] == ["--output-root", str(tmp_path / "runtime" / "launches" / "launch-abc" / soak.SOAK_OUTPUT_DIR)]


def test_status_json_reader_retries_atomic_replace_sharing_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "learner.json"
    path.write_text('{"status":"WAITING"}', encoding="utf-8")
    original = Path.read_text
    attempts = 0

    def flaky_read(self: Path, *args: Any, **kwargs: Any) -> str:
        nonlocal attempts
        if self == path and attempts == 0:
            attempts += 1
            raise PermissionError("simulated Windows atomic replace lock")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read)
    assert soak._load_json_retry(path, delay_seconds=0) == {"status": "WAITING"}
    assert attempts == 1


def test_read_training_manifest_and_tree_size(tmp_path: Path) -> None:
    manifest = tmp_path / "runtime" / "launches" / "launch-a" / soak.SOAK_OUTPUT_DIR / "run-a" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"completed_timesteps": 123, "checkpoints": [{"step": 100}]}))
    assert soak._read_training({"launch_id": "launch-a"}, tmp_path / "runtime")["completed_timesteps"] == 123
    assert soak._tree_size(manifest.parent) == manifest.stat().st_size


def test_read_training_uses_transition_log_until_manifest_publishes_steps(tmp_path: Path) -> None:
    training = tmp_path / "runtime" / "launches" / "launch-b" / soak.SOAK_OUTPUT_DIR / "run-b" / "training"
    training.mkdir(parents=True)
    (training / "manifest.json").write_text(json.dumps({"status": "RUNNING", "checkpoints": []}))
    (training / "task-transitions.jsonl").write_text('{"step":1}\n{"step":2}\n{"step":3}\n')
    result = soak._read_training({"launch_id": "launch-b"}, tmp_path / "runtime")
    assert result["completed_timesteps"] == 3
    assert result["training_transitions"] == 3


def test_rss_slope_uses_post_warmup_samples() -> None:
    start = 1_000_000_000_000
    samples = []
    for minute, rss in ((0, 100), (2, 100), (3, 100), (4, 110), (5, 120)):
        samples.append({"unix_ns": start + minute * 60_000_000_000,
                        "instances": {"0": {"ram_bytes": rss * 1024 * 1024}}})
    assert soak._rss_slope(samples, "0", start) == pytest.approx(10.0)


def test_report_records_campaign_and_failure(tmp_path: Path) -> None:
    target = tmp_path / "29_soak.txt"
    soak._write_report({"status": "FAILED", "campaign_type": "visual_motion_training",
                        "campaign_reason": "gate not approved", "failure": "synthetic"}, target)
    assert "visual_motion_training" in target.read_text()
    assert "synthetic" in target.read_text()



def _rss_samples(values_mib: list[float], *, start: int, step_s: int = 10) -> list[dict[str, Any]]:
    return [{"unix_ns": start + i * step_s * 1_000_000_000,
             "instances": {"0": {"ram_bytes": int(v * 1024 * 1024)}}}
            for i, v in enumerate(values_mib)]


def test_shutdown_checkpoint_spike_does_not_fail_steady_rss() -> None:
    """Regression: the real soak plateaued at ~1.1-1.15 GiB but failed because the
    final checkpoint/export spike (1.3 GiB) was compared against a mid-run dip."""
    start = 1_000_000_000_000
    warmup = [900.0] * 18                       # first 3 minutes
    steady = [1120, 1135, 1091, 1130, 1142, 1114, 1126, 1152] * 12
    shutdown = [1317.0, 1300.0]
    samples = _rss_samples(warmup + steady + shutdown, start=start)
    stop_ns = samples[len(warmup) + len(steady)]["unix_ns"]
    result = soak.evaluate_rss_growth(samples, "0", start, stop_ns)
    assert result["ok"] is True
    assert result["peak_mib"] == pytest.approx(1317.0, abs=1)
    assert abs(result["growth"]) < 0.05


def test_real_leak_fails_on_trend() -> None:
    start = 1_000_000_000_000
    leak = [900.0] * 18 + [1000 + 4 * i for i in range(96)]  # +4 MiB every 10 s
    samples = _rss_samples(leak, start=start)
    result = soak.evaluate_rss_growth(samples, "0", start, None)
    assert result["ok"] is False
    assert result["projected_hourly_growth"] > 0.20


def test_too_few_steady_samples_fail_closed() -> None:
    start = 1_000_000_000_000
    samples = _rss_samples([900.0] * 20, start=start)
    assert soak.evaluate_rss_growth(samples, "0", start, None)["ok"] is False



def _progress(points: list[tuple[float, int]]) -> list[dict[str, Any]]:
    return [{"elapsed_seconds": t, "completed_timesteps": steps} for t, steps in points]


def test_progress_ignores_boot_and_calibration_warmup() -> None:
    """Regression: the first minute (game boot, menu bootstrap, calibration) has no
    PPO steps by design and must not fail the soak."""
    rows = _progress([(0, 0), (61, 0), (122, 53), (183, 146), (244, 219), (305, 313),
                      (366, 403), (427, 487)])
    assert soak.evaluate_progress(rows) == []


def test_progress_detects_a_real_stall_in_steady_state() -> None:
    rows = _progress([(0, 0), (183, 100), (244, 200), (305, 200), (366, 200), (427, 300)])
    problems = soak.evaluate_progress(rows)
    assert problems and "no progress between" in problems[0]


def test_progress_after_stop_is_not_judged() -> None:
    rows = _progress([(183, 100), (244, 200), (305, 300), (366, 300), (427, 300), (488, 300)])
    assert soak.evaluate_progress(rows, stop_elapsed=310) == []


def test_progress_checker_accepts_configured_restart_bound() -> None:
    rows = _progress([(183, 100), (244, 100), (305, 200), (366, 200)])
    assert soak.evaluate_progress(rows, max_gap_seconds=121) == []
    assert soak.evaluate_progress(rows, max_gap_seconds=50)


def test_shared_telemetry_reports_actor_occupancy_and_learner_wait(tmp_path: Path) -> None:
    telemetry = tmp_path / "telemetry"
    telemetry.mkdir()
    (telemetry / "actor-slot-0.jsonl").write_text(
        '{"round":0,"collect_ms":800,"wait_policy_ms":180,"publish_ms":20,"resets":1,"relaunches":0}\n',
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        '{"round_timings":[{"wait_rollouts_ms":100,"straggler_wait_ms":50,'
        '"train_ms":200,"actors_received":[0],"stale_rollouts":0}],"completed_timesteps":512}',
        encoding="utf-8",
    )
    summary = soak._shared_telemetry_summary(tmp_path)
    actor = summary["actors"]["0"]
    assert actor["collect_percent"] == 80
    assert actor["wait_policy_percent"] == 18
    assert actor["resets"] == 1
    assert summary["learner"]["train_ms"] == 200
    assert summary["recommendation"]["rollout_steps"] == 512


def test_no_progress_at_all_fails() -> None:
    assert soak.evaluate_progress(_progress([(0, 0), (200, 0), (400, 0)])) == [
        "no training progress at all"
    ]
