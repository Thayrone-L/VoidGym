from __future__ import annotations

from types import SimpleNamespace

from tools.v2.diagnose_memory import _torch_process_memory, summarize_process_memory


def test_process_summary_reports_visible_total_and_residual_without_claiming_cause() -> None:
    processes = [
        {"name": "learner.exe", "pid": 1, "private_bytes": 30, "working_set_bytes": 40},
        {"name": "claude.exe", "pid": 2, "private_bytes": 20, "working_set_bytes": 25},
        {"name": "protected.exe", "pid": 3, "private_bytes": 10, "working_set_bytes": 15},
    ]

    result = summarize_process_memory(processes, committed_bytes=100)

    assert result["process_count"] == 3
    assert result["private_total_bytes"] == 60
    assert result["committed_minus_process_private_bytes"] == 40
    assert [row["name"] for row in result["processes_top_private"]] == [
        "learner.exe", "claude.exe", "protected.exe",
    ]


def test_process_summary_keeps_missing_commit_value_unknown() -> None:
    result = summarize_process_memory([{"name": "python.exe", "private_bytes": 5}], None)

    assert result["private_total_bytes"] == 5
    assert result["committed_minus_process_private_bytes"] is None


def test_torch_memory_probe_reads_allocator_counters_from_loaded_module() -> None:
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        memory_allocated=lambda _device: 123,
        memory_reserved=lambda _device: 456,
    ))
    counters = _torch_process_memory(fake_torch)
    assert counters["cuda_available"]
    assert counters["allocated_bytes"] == 123
    assert counters["reserved_bytes"] == 456
