"""Summarize per-step latency components from transitions.jsonl."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

COMPONENTS = (
    "apply_rtt", "lease_sleep", "release_rtt", "capture_rtt", "decode", "backend",
    # Bridge-side metrics, forwarded by the backend with a "bridge_" prefix.
    "bridge_queued_to_start_ms", "bridge_capture_ms", "bridge_encode_ms", "bridge_log_ms",
)
# Not durations: summarized separately so they never count toward the time budget.
GAUGES = ("bridge_unity_fps", "bridge_png_bytes", "bridge_target_frame_rate",
          "bridge_target_frame_rate_default", "bridge_vsync_count")


def _transition_files(path: Path) -> list[Path]:
    """A single transitions.jsonl, or every one found below a run directory."""
    return [path] if path.is_file() else sorted(path.rglob("transitions.jsonl"))


def aggregate(path: Path) -> dict[str, dict[str, float | int]]:
    samples: dict[str, list[float]] = defaultdict(list)
    gauges: dict[str, list[float]] = defaultdict(list)
    total: list[float] = []
    leases: dict[int, list[float]] = defaultdict(list)
    for file in _transition_files(path):
        for line in file.open(encoding="utf-8"):
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            timing = row.get("timing_ms", {})
            backend = timing.get("backend")
            if isinstance(backend, (int, float)):
                total.append(float(backend))
            lease = row.get("requested_action", {}).get("lease_ms")
            if isinstance(lease, int) and isinstance(backend, (int, float)):
                leases[lease].append(float(backend))
            for name in COMPONENTS:
                value = timing.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    samples[name].append(float(value))
            for name in GAUGES:
                value = timing.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    gauges[name].append(float(value))
    result: dict[str, dict[str, float | int]] = {}
    for name, values in samples.items():
        ordered = sorted(values)
        mean = statistics.fmean(ordered)
        result[name] = {
            "n": len(ordered), "mean": mean,
            "p50": _percentile(ordered, 0.50), "p90": _percentile(ordered, 0.90),
            "p99": _percentile(ordered, 0.99),
            "total_fraction": mean / statistics.fmean(total) if total and name != "backend" else (1.0 if name == "backend" else 0.0),
        }
    result["lease_backend_mean_ms"] = {str(k): statistics.fmean(v) for k, v in sorted(leases.items())}  # type: ignore[assignment]
    result["gauges"] = {  # type: ignore[assignment]
        name: {"n": len(values), "mean": statistics.fmean(values),
               "min": min(values), "max": max(values)}
        for name, values in gauges.items()
    }
    return result


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    return values[min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transitions", type=Path)
    args = parser.parse_args()
    summary = aggregate(args.transitions)
    print("component | n | mean ms | p50 | p90 | p99 | fraction total")
    for name, values in summary.items():
        if "mean" not in values:
            continue
        print(f"{name} | {values['n']} | {values['mean']:.2f} | {values['p50']:.2f} | {values['p90']:.2f} | {values['p99']:.2f} | {100 * float(values['total_fraction']):.1f}%")
    print("backend mean by lease_ms:", summary.get("lease_backend_mean_ms", {}))
    print("gauges (not durations):", summary.get("gauges", {}))


if __name__ == "__main__":
    main()
