import json

from tools.measure_step_latency import aggregate


def test_aggregate_summarizes_components_and_leases(tmp_path):
    path = tmp_path / "transitions.jsonl"
    rows = [
        {"timing_ms": {"backend": 100, "apply_rtt": 10}, "requested_action": {"lease_ms": 50}},
        {"timing_ms": {"backend": 200, "apply_rtt": 20}, "requested_action": {"lease_ms": 80}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    result = aggregate(path)
    assert result["backend"]["mean"] == 150
    assert result["apply_rtt"]["p50"] == 10
    assert result["lease_backend_mean_ms"] == {"50": 100.0, "80": 200.0}



def test_aggregate_reads_a_run_directory_and_bridge_prefixed_metrics(tmp_path):
    episode = tmp_path / "experience" / "instances" / "i" / "episodes" / "e"
    episode.mkdir(parents=True)
    rows = [
        {"timing_ms": {"backend": 300, "bridge_capture_ms": 40, "bridge_encode_ms": 60,
                       "bridge_payload_bytes": 90,
                       "bridge_unity_fps": 30, "bridge_target_frame_rate_default": 1},
         "requested_action": {"lease_ms": 80}},
        {"timing_ms": {"backend": 100, "bridge_capture_ms": 20, "bridge_encode_ms": 20,
                       "bridge_unity_fps": 60},
         "requested_action": {"lease_ms": 50}},
    ]
    (episode / "transitions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    result = aggregate(tmp_path)
    assert result["backend"]["mean"] == 200
    assert result["bridge_capture_ms"]["mean"] == 30
    assert result["bridge_encode_ms"]["total_fraction"] == 0.2
    assert result["gauges"]["bridge_payload_bytes"]["mean"] == 90
    # FPS is a gauge, never part of the time budget.
    assert "bridge_unity_fps" not in result
    assert result["gauges"]["bridge_unity_fps"]["min"] == 30
