"""Measure two concurrent real PPO training launches for a bounded soak."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import statistics
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import smoke_parallel_instances as smoke

from hk_agent.config import load_config
from hk_agent.control_plane import ControlPlane

# Short on purpose: the per-launch training tree is deep (see gym_env.extended_length_path).
SOAK_OUTPUT_DIR = "out"


def _campaign_payload(campaign_id: str, campaign_type: str) -> dict[str, Any]:
    return {
        "campaign_id": campaign_id,
        "name": "Two instance PPO training soak",
        "campaign_type": campaign_type,
        "seed": 29,
        "timesteps": 1,
        "checkpoint_interval": 2048,
        "max_checkpoints": 3,
        "episodes": 1,
        "horizon": 10_000,
        "action_repeat": 1,
        "lease_ms": 80,
        "window_width": 960,
        "window_height": 540,
        "duration_mode": "continuous",
        "save_source": "existing_save",
        "save_slot": 1,
        "weights": {},
    }


class _SoakControlPlane(ControlPlane):
    """Place each runner's output below its own managed runtime launch directory."""

    def build_command(self, spec: Any, **kwargs: Any) -> list[str]:
        command = super().build_command(spec, **kwargs)
        stop_file = kwargs.get("stop_file")
        if kwargs.get("execute") and stop_file is not None:
            launch_id = Path(stop_file).parent.name
            output = self.runtime_root / "launches" / launch_id / SOAK_OUTPUT_DIR
            command += ["--output-root", str(output)]
        return command


def _read_training(launch: Mapping[str, Any], runtime_root: Path) -> dict[str, Any]:
    launch_dir = runtime_root / "launches" / str(launch["launch_id"])
    candidates = list(launch_dir.rglob("manifest.json"))
    if not candidates:
        path = None
        data: dict[str, Any] = {}
    else:
        path = max(candidates, key=lambda item: item.stat().st_mtime_ns)
        data = smoke._load_json(path)
    checkpoints = data.get("checkpoints", [])
    valid_checkpoints: list[dict[str, Any]] = []
    for entry in checkpoints if isinstance(checkpoints, list) else []:
        if not isinstance(entry, Mapping):
            continue
        filename = entry.get("path")
        digest = entry.get("sha256")
        checkpoint = Path(filename) if isinstance(filename, str) else Path()
        try:
            checkpoint.resolve(strict=False).relative_to((path.parent / "checkpoints").resolve(strict=False))
            owned = True
        except ValueError:
            owned = False
        if (owned and isinstance(filename, str) and isinstance(digest, str) and checkpoint.is_file()
                and hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest):
            valid_checkpoints.append(dict(entry))
    training_transitions = 0
    for transition_log in launch_dir.rglob("task-transitions.jsonl"):
        if transition_log.parent.name != "training":
            continue
        try:
            training_transitions += transition_log.read_bytes().count(b"\n")
        except OSError:
            continue
    manifest_steps = int(data.get("completed_timesteps", 0))
    return {
        "completed_timesteps": max(manifest_steps, training_transitions),
        "training_transitions": training_transitions,
        "checkpoints": valid_checkpoints,
        "manifest_path": str(path) if path is not None else None,
        "status": data.get("status"),
        "policy_exported": bool(data.get("portable_policies")),
    }


def _tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _rss_slope(samples: list[dict[str, Any]], slot: str, started_ns: int) -> float:
    rows = [(sample["unix_ns"] - started_ns, sample["instances"][slot]["ram_bytes"])
            for sample in samples if slot in sample["instances"]
            and sample["instances"][slot]["ram_bytes"] > 0]
    rows = [(x / 60_000_000_000, y / (1024 * 1024)) for x, y in rows if x >= 180_000_000_000]
    if len(rows) < 2:
        return 0.0
    xbar = statistics.fmean(x for x, _ in rows)
    ybar = statistics.fmean(y for _, y in rows)
    denominator = sum((x - xbar) ** 2 for x, _ in rows)
    return sum((x - xbar) * (y - ybar) for x, y in rows) / denominator if denominator else 0.0


RSS_WARMUP_NS = 180_000_000_000
RSS_MAX_GROWTH = 0.20


def evaluate_rss_growth(
    samples: list[dict[str, Any]], slot: str, started_ns: int, stop_requested_ns: int | None
) -> dict[str, Any]:
    """Judge memory growth on the steady-state window only.

    The window runs from the end of warm-up until this slot's stop was requested,
    so the legitimate spike of saving the final checkpoint and exporting the policy
    is reported (peak) but cannot fail the run. A leak shows up as a trend: the
    median of the last quarter versus the first quarter, and the fitted slope
    projected over one hour, must both stay within RSS_MAX_GROWTH.
    """
    all_rows = [(s["unix_ns"], s["instances"][slot]["ram_bytes"]) for s in samples
                if slot in s["instances"] and s["instances"][slot]["ram_bytes"] > 0]
    peak = max((value for _, value in all_rows), default=0)
    window = [(t, v) for t, v in all_rows
              if t - started_ns >= RSS_WARMUP_NS
              and (stop_requested_ns is None or t < stop_requested_ns)]
    if len(window) < 8:
        return {"ok": False, "reason": "not enough steady-state samples",
                "samples": len(window), "peak_mib": peak / 2**20}
    quarter = max(2, len(window) // 4)
    baseline = statistics.median(v for _, v in window[:quarter])
    final = statistics.median(v for _, v in window[-quarter:])
    growth = final / baseline - 1.0
    xs = [(t - started_ns) / 60_000_000_000 for t, _ in window]
    ys = [v / 2**20 for _, v in window]
    xbar, ybar = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((x - xbar) ** 2 for x in xs)
    slope = (sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys, strict=True)) / denominator
             if denominator else 0.0)
    projected_hourly = slope * 60 / (baseline / 2**20)
    ok = growth <= RSS_MAX_GROWTH and projected_hourly <= RSS_MAX_GROWTH
    return {"ok": ok, "baseline_mib": baseline / 2**20, "final_mib": final / 2**20,
            "growth": growth, "slope_mib_per_min": slope,
            "projected_hourly_growth": projected_hourly, "peak_mib": peak / 2**20,
            "samples": len(window)}


PROGRESS_WINDOW_SECONDS = 120


def evaluate_progress(
    rows: list[dict[str, Any]],
    *,
    warmup_seconds: float = RSS_WARMUP_NS / 1_000_000_000,
    stop_elapsed: float | None = None,
) -> list[str]:
    """Require training progress in every two-minute window of the steady state.

    Game boot, menu bootstrap and calibration happen before the first PPO step,
    so the warm-up is excluded, and so is everything after this slot's stop.
    """
    if not rows or max(int(row.get("completed_timesteps", 0)) for row in rows) <= 0:
        return ["no training progress at all"]
    usable = [row for row in rows
              if row["elapsed_seconds"] >= warmup_seconds
              and (stop_elapsed is None or row["elapsed_seconds"] <= stop_elapsed)]
    if len(usable) < 2:
        return ["not enough progress samples after warm-up"]
    problems: list[str] = []
    for index, current in enumerate(usable):
        earlier = [row for row in usable[:index]
                   if current["elapsed_seconds"] - row["elapsed_seconds"] >= PROGRESS_WINDOW_SECONDS]
        if earlier and int(current["completed_timesteps"]) <= int(earlier[-1]["completed_timesteps"]):
            problems.append(
                f"no progress between {earlier[-1]['elapsed_seconds']:.0f}s "
                f"and {current['elapsed_seconds']:.0f}s"
            )
    return problems


def _verify_training_isolation(launches: list[dict[str, Any]], runtime_root: Path) -> None:
    launch_dirs = [runtime_root / "launches" / str(item["launch_id"]) for item in launches]
    checkpoint_roots: list[Path] = []
    route_paths: list[Path] = []
    for item, launch_dir in zip(launches, launch_dirs, strict=True):
        train = _read_training(item, runtime_root)
        manifest_path = train.get("manifest_path")
        if not manifest_path:
            raise AssertionError(f"training manifest missing for {item['launch_id']}")
        root = Path(manifest_path).parent / "checkpoints"
        checkpoint_roots.append(root.resolve(strict=False))
        for entry in train["checkpoints"]:
            candidate = Path(str(entry.get("path", "")))
            if not candidate.is_absolute():
                candidate = Path(manifest_path).parent / candidate
            resolved = candidate.resolve(strict=False)
            if resolved.exists() and root.resolve(strict=False) not in resolved.parents:
                raise AssertionError(f"checkpoint escaped its launch: {resolved}")
        resources = item.get("resources", {})
        route_value = resources.get("route_memory") if isinstance(resources, Mapping) else None
        if item.get("campaign_type") == "scene_change_training":
            if not route_value:
                raise AssertionError("scene-change route memory path missing")
            route = Path(str(route_value)).resolve(strict=False)
            expected = (runtime_root / "route-memory" / "existing_save-slot-1" /
                        f"instance-{item['slot_id']}.json").resolve(strict=False)
            if route != expected:
                raise AssertionError(f"route memory is not isolated by slot: {route}")
            route_paths.append(route)
    smoke.verify_distinct(checkpoint_roots, "checkpoint roots")
    if route_paths:
        smoke.verify_distinct(route_paths, "route memory paths")
    for index, directory in enumerate(launch_dirs):
        other = launch_dirs[1 - index]
        if any(other.resolve(strict=False) in path.resolve(strict=False).parents
               for path in directory.rglob("*") if path.is_file()):
            raise AssertionError(f"launch {directory} contains files under the other launch")


def _run_live_soak(project_root: Path, duration_seconds: int = 1200) -> dict[str, Any]:
    preflight = smoke.preflight(project_root)
    config = load_config(project_root / "config.training.local.toml", allow_training=True)
    probe = _SoakControlPlane(project_root, runtime_root=config.runtime_root).scene_change_gate()
    campaign_type = "scene_change_training" if probe.get("passed") is True else "visual_motion_training"
    reason = "approved scene-change gate" if probe.get("passed") is True else "scene-change gate not approved"
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    soak_root = config.runtime_root / "soak" / stamp
    soak_root.mkdir(parents=True, exist_ok=False)
    personal = smoke.hash_tree(config.personal_save_root)
    baseline = smoke.hash_tree(config.experimental_save_root)
    data_before = {f"slot{i}": smoke.hash_tree(root) for i, root in enumerate(config.instance_data_roots)}
    registry = smoke.registry_snapshot(soak_root / "registry-before.reg")
    backup = soak_root / "backup" / "personal-saves"
    import shutil
    shutil.copytree(config.personal_save_root, backup)
    control = _SoakControlPlane(
        project_root, allow_execute=True, max_active_launches=2, parallel_enabled=True,
        runtime_root=config.runtime_root, training_device="cpu",
        torch_threads=config.torch_threads, torch_interop_threads=config.torch_interop_threads,
    )
    campaign = control.create_campaign(_campaign_payload("ppo-soak-" + stamp, campaign_type))
    launches: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    progress: dict[str, list[dict[str, Any]]] = {"0": [], "1": []}
    manifest: dict[str, Any] = {
        "schema_version": 1, "artifact_type": "parallel_training_soak", "status": "RUNNING",
        "started_unix_ns": time.time_ns(), "preflight": preflight, "campaign_type": campaign_type,
        "campaign_reason": reason, "duration_target_seconds": duration_seconds,
        "snapshots_before": {"personal": personal, "baseline": baseline,
                             "data_roots": data_before, "registry": registry},
        "launches": launches, "resource_samples": samples, "progress_samples": progress,
    }
    manifest_path = soak_root / "soak-manifest.json"
    started = time.monotonic()
    stopped: set[int] = set()
    observed: dict[str, dict[int, float]] = {"0": {}, "1": {}}

    def run_safety_checks() -> dict[str, Any]:
        """Personal/baseline/registry/isolation/orphan checks; always run, never skipped."""
        found: list[str] = []
        registry_after = smoke.registry_snapshot(soak_root / "registry-after.reg")
        personal_after = smoke.hash_tree(config.personal_save_root)
        baseline_after = smoke.hash_tree(config.experimental_save_root)
        data_after = {f"slot{i}": smoke.hash_tree(root)
                      for i, root in enumerate(config.instance_data_roots)}
        if personal != personal_after:
            found.append("personal saves changed (restore manually from the soak backup)")
        if baseline != baseline_after:
            found.append("experimental baseline changed")
        if registry["protected_sha256"] != registry_after["protected_sha256"]:
            found.append("protected registry changed")
        try:
            smoke.verify_root_isolation(data_before, data_after, {"slot0": "slot0", "slot1": "slot1"})
        except AssertionError as error:
            found.append(str(error))
        rows = [{"pid": pid, "alive": smoke._same_process_alive(pid, created)}
                for pids in observed.values() for pid, created in pids.items()]
        try:
            smoke.verify_no_orphans(rows, {row["pid"] for row in rows})
        except AssertionError as error:
            found.append(str(error))
        return {"problems": found, "owned_processes": rows,
                "snapshots_after": {"personal": personal_after, "baseline": baseline_after,
                                    "data_roots": data_after, "registry": registry_after}}

    try:
        if control.status().get("active_count") != 0:
            raise RuntimeError("stop existing launches before soak")
        for _ in range(2):
            record = control.launch(str(campaign["campaign_id"]), version=int(campaign["version"]), execute=True)
            resources = record.get("resources")
            runner = record.get("runner_process")
            if not isinstance(resources, Mapping) or not isinstance(runner, Mapping):
                raise TypeError("launch is missing slot resources or runner identity")
            slot_value = resources.get("slot_id")
            if not isinstance(slot_value, int) or isinstance(slot_value, bool):
                raise TypeError("launch has invalid slot id")
            launches.append({"launch_id": str(record["launch_id"]), "slot_id": slot_value,
                             "runner_pid": int(runner["pid"]), "runner_process": smoke._safe_process_record(runner),
                             "resources": dict(resources), "campaign_type": campaign_type,
                             "launch_dir": str(config.runtime_root / "launches" / str(record["launch_id"]))})
        launches.sort(key=lambda item: item["slot_id"])
        if [x["slot_id"] for x in launches] != [0, 1]:
            raise AssertionError("expected slots 0 and 1")
        started = time.monotonic()
        manifest["started_unix_ns"] = time.time_ns()
        deadline = started + duration_seconds
        next_minute = started
        slot0_stopped_at: float | None = None
        slot1_steps_at_slot0_stop = 0
        slot1_progress_after_stop = False
        while True:
            now = time.monotonic()
            sample = smoke._sample_resources(launches)
            samples.append(sample)
            for slot, row in sample["instances"].items():
                observed[slot].update({int(pid): float(created) for pid, created in row["process_birth_times"].items()})
            if now >= next_minute:
                for item in launches:
                    train = _read_training(item, config.runtime_root)
                    progress[str(item["slot_id"])].append({"elapsed_seconds": now - started, **train,
                        "launch_disk_bytes": _tree_size(Path(item["launch_dir"])),
                        "training_output_bytes": _tree_size(Path(str(train["manifest_path"])).parent)
                            if train.get("manifest_path") else 0})
                next_minute += 60
            if now >= deadline and 0 not in stopped:
                slot1_steps_at_slot0_stop = int(
                    _read_training(launches[1], config.runtime_root)["completed_timesteps"]
                )
                launches[0]["stop_requested_unix_ns"] = time.time_ns()
                launches[0]["stop_requested_elapsed"] = time.monotonic() - started
                control.stop(launches[0]["launch_id"])
                smoke._wait_terminal(control, launches[0]["launch_id"], 90)
                stopped.add(0)
                slot0_stopped_at = time.monotonic()
            if slot0_stopped_at is not None:
                current = _read_training(launches[1], config.runtime_root)
                if int(current["completed_timesteps"]) > slot1_steps_at_slot0_stop:
                    slot1_progress_after_stop = True
                if time.monotonic() - slot0_stopped_at >= 60:
                    launches[1]["stop_requested_unix_ns"] = time.time_ns()
                    launches[1]["stop_requested_elapsed"] = time.monotonic() - started
                    control.stop(launches[1]["launch_id"])
                    smoke._wait_terminal(control, launches[1]["launch_id"], 90)
                    stopped.add(1)
                    break
            time.sleep(max(0.1, min(10.0, deadline - time.monotonic())) if 0 not in stopped else 5.0)
        problems: list[str] = []
        for item in launches:
            record = control._launches[item["launch_id"]]
            item["terminal"] = {key: record.get(key) for key in
                ("status", "return_code", "final_save_status", "final_save", "failure")}
            if record.get("status") != "CANCELLED" or record.get("final_save_status") != "captured":
                problems.append(f"launch did not cancel cleanly: {item['launch_id']}")
            if not Path(str(record.get("final_save", ""))).is_dir():
                problems.append(f"final save directory missing: {item['launch_id']}")
            item["training_final"] = _read_training(item, config.runtime_root)
            if not item["training_final"].get("checkpoints"):
                problems.append(f"cancelled training has no valid checkpoint: {item['launch_id']}")
            if item["training_final"].get("status") not in (None, "CANCELLED"):
                problems.append(f"training manifest is not CANCELLED: {item['launch_id']}")
        if not slot1_progress_after_stop:
            problems.append("slot 1 did not progress after slot 0 stopped")
        try:
            _verify_training_isolation(launches, config.runtime_root)
        except AssertionError as error:
            problems.append(str(error))
        slopes = {slot: _rss_slope(samples, slot, samples[0]["unix_ns"] if samples else time.time_ns())
                  for slot in ("0", "1")}
        for item in launches:
            slot = str(item["slot_id"])
            rss = evaluate_rss_growth(
                samples, slot, samples[0]["unix_ns"], item.get("stop_requested_unix_ns")
            )
            item["rss"] = rss
            item["rss_slope_mib_per_min"] = slopes[slot]
            if not rss["ok"]:
                problems.append(f"slot {slot} steady-state RSS check failed: {rss}")
            item["progress"] = progress[slot]
            for problem in evaluate_progress(
                progress[slot], stop_elapsed=item.get("stop_requested_elapsed")
            ):
                problems.append(f"slot {slot}: {problem}")
        safety = run_safety_checks()
        manifest["safety"] = safety
        problems.extend(safety["problems"])
        manifest["problems"] = problems
        if problems:
            raise AssertionError("; ".join(problems))
        manifest.update({"status": "PASSED", "duration_real_seconds": time.monotonic() - started,
            "slot1_progress_after_slot0_stop": slot1_progress_after_stop,
            "rss_slope_mib_per_min": slopes,
            "snapshots_after": safety["snapshots_after"],
            "resource_summary": smoke._resource_summary(samples),
            "owned_processes": safety["owned_processes"],
            "checks": {"personal_unchanged": True, "baseline_unchanged": True,
                "protected_registry_unchanged": True, "data_root_isolation": True,
                "no_orphans": True, "training_file_isolation": True,
                "steady_state_rss": True, "progress_every_two_minutes": True}})
    except BaseException as error:  # noqa: BLE001
        for item in launches:
            try:
                if control._launches.get(item["launch_id"], {}).get("status") in {"PREPARING", "RUNNING", "STOP_REQUESTED"}:
                    control.stop(item["launch_id"])
                    smoke._wait_terminal(control, item["launch_id"], 90)
            except (RuntimeError, ValueError, OSError, TimeoutError):
                pass
        manifest.update({"status": "FAILED", "failure": f"{type(error).__name__}: {error}",
                         "duration_real_seconds": time.monotonic() - started,
                         "resource_summary": smoke._resource_summary(samples)})
        if "safety" not in manifest:
            try:
                manifest["safety"] = run_safety_checks()
            except Exception as safety_error:  # noqa: BLE001 - record, never mask the failure
                manifest["safety_error"] = f"{type(safety_error).__name__}: {safety_error}"
    finally:
        manifest["finished_unix_ns"] = time.time_ns()
        manifest["manifest_path"] = str(manifest_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_report(manifest, PROJECT_ROOT / "agent-prompts" / "relatorios" / "29_soak.txt")
    return manifest


def _write_report(manifest: Mapping[str, Any], path: Path) -> None:
    summary = manifest.get("resource_summary", {})
    samples = manifest.get("resource_samples", [])
    launches = manifest.get("launches", [])
    duration = float(manifest.get("duration_real_seconds", 0))
    lines = ["AGENTE 29 — SOAK DE TREINO PARALELO", "=" * 39,
             f"Status: {manifest.get('status')}", f"Campanha: {manifest.get('campaign_type')} ({manifest.get('campaign_reason')})",
             f"Duração real: {duration:.1f}s", "", "MÉTRICAS POR SLOT"]
    for slot in ("0", "1"):
        row = summary.get(slot, {}) if isinstance(summary, Mapping) else {}
        cpu = row.get("cpu_percent", {})
        slot_progress = manifest.get("progress_samples", {}).get(slot, [])
        steps = max((int(entry.get("completed_timesteps", 0)) for entry in slot_progress), default=0)
        launch = next((entry for entry in launches if str(entry.get("slot_id")) == slot), {})
        initial = next((sample.get("instances", {}).get(slot, {}).get("ram_bytes") for sample in samples
                        if sample.get("instances", {}).get(slot, {}).get("ram_bytes", 0) > 0), 0)
        final = next((sample.get("instances", {}).get(slot, {}).get("ram_bytes") for sample in reversed(samples)
                      if sample.get("instances", {}).get(slot, {}).get("ram_bytes", 0) > 0), 0)
        peak = max((sample.get("instances", {}).get(slot, {}).get("ram_bytes", 0) for sample in samples), default=0)
        checkpoint_count = len(launch.get("training_final", {}).get("checkpoints", []))
        disk_bytes = _tree_size(Path(str(launch["launch_dir"]))) if launch.get("launch_dir") else 0
        steps_per_second = steps / duration if duration > 0 else 0.0
        lines.append(f"slot {slot}: passos={steps}, passos/s={steps_per_second:.3f}; checkpoints={checkpoint_count}; "
                     f"CPU média/pico={cpu.get('average')} / {cpu.get('peak')}%; RSS inicial/final/pico="
                     f"{initial} / {final} / {peak} bytes; inclinação={launch.get('rss_slope_mib_per_min')} MiB/min; "
                     f"disco={disk_bytes} bytes")
    for slot, rows in manifest.get("progress_samples", {}).items():
        if rows:
            last = rows[-1]
            lines.append(f"slot {slot} última amostra: output treino={last.get('training_output_bytes')} bytes")
    gpu = [sample.get("gpu_utilization_percent") for sample in samples
           if isinstance(sample.get("gpu_utilization_percent"), (int, float))]
    lines.append(f"GPU total média/pico: {statistics.fmean(gpu) if gpu else 'n/a'} / {max(gpu) if gpu else 'n/a'}%")
    peaks = [max((sample.get("instances", {}).get(slot, {}).get("ram_bytes", 0) for sample in samples), default=0)
             for slot in ("0", "1")]
    ram_capacity = int(16 * 1024**3 * 0.75)
    cpu_average = sum(float(summary.get(slot, {}).get("cpu_percent", {}).get("average") or 0) for slot in ("0", "1"))
    ram_instances = min((ram_capacity // peak for peak in peaks if peak > 0), default=0)
    cpu_instances = int(8 * 100 * 0.75 / (cpu_average / 2)) if cpu_average > 0 else 0
    lines.append(f"Estimativa sob limite de 16 GB/8 núcleos com 25% de folga: {min(ram_instances, cpu_instances)} instâncias")
    checks = manifest.get("checks", {})
    if checks:
        lines.extend(["", "VERIFICAÇÕES"])
        lines.extend(f"{key}: {'PASSOU' if value else 'FALHOU'}" for key, value in checks.items())
    if manifest.get("failure"):
        lines.extend(["", f"Falha: {manifest['failure']}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-open-games", action="store_true",
                        help="authorize this script to open both managed Hollow Knight instances")
    args = parser.parse_args()
    if not args.confirm_open_games:
        print("No games opened. Run with --confirm-open-games after authorization.")
        return 2
    result = _run_live_soak(PROJECT_ROOT)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
