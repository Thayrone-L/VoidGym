"""Measure two concurrent real PPO training launches for a bounded soak."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import numpy as np
import smoke_parallel_instances as smoke

from hk_agent.config import load_config
from hk_agent.control_plane import ControlPlane

# Short on purpose: the per-launch training tree is deep (see gym_env.extended_length_path).
SOAK_OUTPUT_DIR = "out"


def _load_json_retry(path: Path, *, attempts: int = 12, delay_seconds: float = 0.025) -> dict[str, Any]:
    """Read atomically replaced status JSON despite brief Windows sharing locks."""
    last_error: OSError | json.JSONDecodeError | None = None
    for attempt in range(attempts):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise TypeError(f"expected JSON object in {path}")
            return value
        except (OSError, json.JSONDecodeError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError("JSON retry had no attempts")


def _shared_rejections(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"rejected rollout entry {line_number} must be an object")
        rows.append(row)
    return rows


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


def _shared_campaign_payload(campaign_id: str, name: str) -> dict[str, Any]:
    payload = _campaign_payload(campaign_id, "exploration_training")
    payload.update({
        "name": name, "shared_learner": True, "actors": 2,
        "rollout_steps": 512, "batch_size": 256, "horizon": 3000,
        "ent_coef": 0.01, "gamma": 0.995, "gae_lambda": 0.95,
        "n_epochs": 4, "target_kl": 0.03, "learning_rate": 2.5e-4,
        "clip_range": 0.2,
        "checkpoint_interval": 2048, "lease_ms": 80,
        "straggler_timeout_seconds": 45.0,
        "duration_mode": "continuous", "timesteps": 10_000_000,
        "weights": {
            "time_cost": -0.001, "novelty_reward": 0.05, "novelty_budget": 20.0,
            "scene_change_reward": 5.0, "enemy_kill_reward": 1.0,
            "enemy_kill_reward_budget": 5.0, "shade_recovery_reward": 1.0,
            "damage_penalty": -0.5, "geo_gain_reward": 0.01,
            "geo_spend_reward": 0.02, "death_penalty": -5.0,
            "run_failed_penalty": -5.0,
        },
    })
    return payload


def _deployment_hashes(config: Any, project_root: Path) -> dict[str, str]:
    from hk_agent.instance_marker import load_instance_marker

    built = project_root / "bridge" / "HkAgentBridge.Mod" / "bin" / "Release" / "HkAgentBridge.dll"
    if not built.is_file():
        raise FileNotFoundError(f"Release bridge DLL is missing: {built}")
    expected = hashlib.sha256(built.read_bytes()).hexdigest()
    result: dict[str, str] = {"release": expected}
    for index, install in enumerate(config.install_roots, start=1):
        marker = load_instance_marker(install, expected_index=index)
        expected_data = Path(marker.persistent_data_path_expected).resolve(strict=False)
        if expected_data != config.instance_data_roots[index - 1].resolve(strict=False):
            raise AssertionError(f"instance {index} ModLog/persistentDataPath does not match its configured data root")
        dll = install / "Hollow Knight_Data" / "Managed" / "Mods" / "HkAgentBridge" / "HkAgentBridge.dll"
        digest = hashlib.sha256(dll.read_bytes()).hexdigest() if dll.is_file() else None
        if digest != expected:
            raise AssertionError(f"instance {index} deployed bridge DLL hash differs from Release: {digest}")
        result[f"instance_{index}"] = str(digest)
    return result


def _process_preflight_no_steam() -> list[dict[str, Any]]:
    import psutil

    found: list[dict[str, Any]] = []
    for process in psutil.process_iter(["name", "exe"]):
        name = str(process.info.get("name") or "").casefold()
        if name in {"steam.exe", "steamwebhelper.exe"}:
            found.append({"pid": process.pid, "name": process.info.get("name"), "exe": process.info.get("exe")})
    if found:
        raise RuntimeError(f"soak requires Steam to be closed: {found}")
    return found


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
    max_gap_seconds: float = PROGRESS_WINDOW_SECONDS,
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
                   if current["elapsed_seconds"] - row["elapsed_seconds"] >= max_gap_seconds]
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


def _run_shared_learner_soak(
    project_root: Path, duration_seconds: int = 1200, restart_at_seconds: int = 600,
    *, start_long_campaign: bool = True,
) -> tuple[dict[str, Any], ControlPlane | None]:
    """Run the 20-minute shared soak, then leave the long run under panel ownership."""
    from hk_agent.scene_change_task import SCENE_CHANGE_ACTION_NAMES

    preflight = smoke.preflight(project_root)
    steam_processes = _process_preflight_no_steam()
    config = load_config(project_root / "config.training.local.toml", allow_training=True)
    if config.instance_count != 2 or not config.parallel_enabled:
        raise RuntimeError("shared soak requires exactly two enabled actor slots")
    hashes = _deployment_hashes(config, project_root)
    try:
        with socket.create_connection(("127.0.0.1", 8765), timeout=0.25):
            raise RuntimeError("ControlPlane port 8765 is occupied; refusing to start a second owner")
    except OSError:
        pass

    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    soak_root = config.runtime_root / "soak" / f"shared-{stamp}"
    soak_root.mkdir(parents=True, exist_ok=False)
    personal = smoke.hash_tree(config.personal_save_root)
    baseline = smoke.hash_tree(config.experimental_save_root)
    data_before = {f"slot{i}": smoke.hash_tree(root) for i, root in enumerate(config.instance_data_roots)}
    registry = smoke.registry_snapshot(soak_root / "registry-before.reg")
    import shutil
    shutil.copytree(config.personal_save_root, soak_root / "backup" / "personal-saves")
    control = _SoakControlPlane(
        project_root, allow_execute=True, max_active_launches=2, parallel_enabled=True,
        runtime_root=config.runtime_root, training_device="cpu",
        torch_threads=config.torch_threads, torch_interop_threads=config.torch_interop_threads,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1, "artifact_type": "shared_learner_soak", "status": "RUNNING",
        "soak_root": str(soak_root),
        "started_unix_ns": time.time_ns(), "duration_target_seconds": duration_seconds,
        "restart_at_seconds": restart_at_seconds,
        "parameters": {"campaign_type": "exploration_training", "task": "explore_v1",
                       "shared_learner": True, "actors": 2, "rollout_steps": 512, "batch_size": 256,
                       "horizon": 3000, "ent_coef": 0.01, "gamma": 0.995, "gae_lambda": 0.95,
                       "n_epochs": 4, "target_kl": 0.03, "learning_rate": 2.5e-4, "clip_range": 0.2,
                       "weights": _shared_campaign_payload("manifest", "manifest")["weights"],
                       "lease_ms": 80, "duration_mode": "continuous", "checkpoint_interval": 2048,
                       "straggler_timeout_seconds": 45.0,
                       "training_device": "cpu"},
        "preflight": preflight, "steam_processes": steam_processes, "bridge_sha256": hashes,
        "snapshots_before": {"personal": personal, "baseline": baseline,
                             "data_roots": data_before, "registry": registry},
        "launches": [], "resource_samples": [], "progress_samples": [],
    }
    manifest_path = soak_root / "soak-manifest.json"
    shared_root: Path | None = None
    group_id: str | None = None
    long_group_id: str | None = None
    started = time.monotonic()
    actor_records: list[dict[str, Any]] = []
    learner_record: dict[str, Any] = {}
    learner_id = ""
    restarted = False
    training_timings: list[dict[str, Any]] = []
    waiting_seconds = {"0": 0.0, "1": 0.0}
    samples: list[dict[str, Any]] = manifest["resource_samples"]
    last_sample = started
    last_steps = 0
    last_round = 0
    modlog_start_wall = time.time()

    def persist() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def safety_check() -> dict[str, Any]:
        registry_after = smoke.registry_snapshot(soak_root / "registry-after.reg")
        personal_after = smoke.hash_tree(config.personal_save_root)
        baseline_after = smoke.hash_tree(config.experimental_save_root)
        data_after = {f"slot{i}": smoke.hash_tree(root) for i, root in enumerate(config.instance_data_roots)}
        problems: list[str] = []
        if personal != personal_after:
            problems.append("personal saves changed")
        if baseline != baseline_after:
            problems.append("experimental baseline changed")
        if registry["protected_sha256"] != registry_after["protected_sha256"]:
            problems.append("protected registry changed")
        try:
            smoke.verify_root_isolation(data_before, data_after, {"slot0": "slot0", "slot1": "slot1"})
        except AssertionError as error:
            problems.append(str(error))
        process_map: dict[int, float] = {}
        for sample in samples:
            for instance in sample.get("instances", {}).values():
                process_map.update({int(pid): float(created)
                                    for pid, created in instance.get("process_birth_times", {}).items()})
        process_status = [{"pid": pid, "alive": smoke._same_process_alive(pid, birth)}
                          for pid, birth in process_map.items()]
        alive = [row for row in process_status if row["alive"]]
        if alive:
            problems.append(f"owned processes remain after teardown: {alive}")
        transition_count = 0
        novelty_steps = 0
        reward_component_totals: dict[str, float] = {}
        for actor in manifest["launches"]:
            root = Path(str(actor.get("launch_dir", "")))
            for log in root.rglob("task-transitions.jsonl"):
                if log.parent.name != "training":
                    continue
                for index, text in enumerate(log.read_text("utf-8", errors="replace").splitlines(), 1):
                    try:
                        row = json.loads(text)
                    except json.JSONDecodeError:
                        problems.append(f"invalid transition JSON at {log}:{index}")
                        continue
                    mask = row.get("action_mask", [])
                    requested = row.get("requested_task_action")
                    executed = row.get("executed_task_action")
                    if isinstance(mask, list) and requested in mask and executed != SCENE_CHANGE_ACTION_NAMES.index("NOOP"):
                        problems.append(f"unavailable masked action was executed at {log}:{index}")
                    transition_count += 1
                    reward = row.get("reward", {})
                    components = reward.get("components", {}) if isinstance(reward, Mapping) else {}
                    if isinstance(components, Mapping):
                        if float(components.get("novelty", 0.0)) > 0:
                            novelty_steps += 1
                        for name, value in components.items():
                            if isinstance(value, (int, float)):
                                reward_component_totals[str(name)] = reward_component_totals.get(str(name), 0.0) + float(value)
        return {"problems": problems, "owned_processes": process_status,
                "reward_variability": {"transition_count": transition_count,
                    "novelty_steps": novelty_steps,
                    "novelty_step_percent": (100.0 * novelty_steps / transition_count) if transition_count else None,
                    "reward_component_totals": reward_component_totals},
                "snapshots_after": {"personal": personal_after, "baseline": baseline_after,
                                    "data_roots": data_after, "registry": registry_after}}

    try:
        if control.status().get("active_count") != 0:
            raise RuntimeError("stop active launches before shared learner soak")
        campaign = control.create_campaign(
            _shared_campaign_payload("shared-soak-" + stamp, "Shared learner 20 minute soak")
        )
        group = control.launch(str(campaign["campaign_id"]), version=int(campaign["version"]), execute=True)
        group_id = str(group["shared_group"])
        shared_root = Path(str(group["shared_root"]))
        learner = group["learner"]
        learner_id = str(learner["launch_id"])
        identity = learner.get("runner_process")
        if not isinstance(identity, Mapping):
            raise TypeError("learner runner identity is missing")
        learner_record = {"launch_id": learner_id, "slot_id": "learner", "runner_pid": int(identity["pid"]),
                          "runner_process": dict(identity),
                          "launch_dir": str(config.runtime_root / "shared" / str(campaign["campaign_id"]))}
        for item in group["actors"]:
            resources, runner = item.get("resources"), item.get("runner_process")
            if not isinstance(resources, Mapping) or not isinstance(runner, Mapping):
                raise TypeError("actor process identity or slot is missing")
            actor = {"launch_id": str(item["launch_id"]), "slot_id": int(resources["slot_id"]),
                     "runner_pid": int(runner["pid"]), "runner_process": dict(runner),
                     "launch_dir": str(config.runtime_root / "launches" / str(item["launch_id"]))}
            actor_records.append(actor)
            manifest["launches"].append(dict(actor))
        actor_records.sort(key=lambda row: int(row["slot_id"]))
        manifest["learner"] = learner_record
        manifest["campaign"] = {"campaign_id": campaign["campaign_id"], "version": campaign["version"],
                                 "shared_group": group_id, "shared_root": str(shared_root)}
        started = time.monotonic()
        deadline = started + duration_seconds
        while time.monotonic() < deadline:
            now = time.monotonic()
            control.status()
            status_path = shared_root / "learner.json"
            learner_state = _load_json_retry(status_path) if status_path.is_file() else {}
            active_ids = {learner_id, *(str(actor["launch_id"]) for actor in actor_records)}
            for active_id in active_ids:
                launch_status = control._launches.get(active_id, {}).get("status")
                if launch_status in {"FAILED", "COMPLETED", "CANCELLED"}:
                    raise RuntimeError(f"shared group member ended during soak: {active_id} {launch_status}")
            if learner_state.get("status") in {"FAILED", "CANCELLED", "COMPLETED"}:
                raise RuntimeError(f"learner ended unexpectedly: {learner_state}")
            rejection_path = shared_root / "rejected.jsonl"
            rejected_rows = _shared_rejections(rejection_path)
            if rejected_rows:
                allowed_stale = {
                    "round": manifest.get("actor_restart", {}).get("learner_round_before", -1),
                    "slot": manifest.get("actor_restart", {}).get("slot_id", -1),
                    "after_unix_ns": manifest.get("actor_restart", {}).get("restarted_unix_ns", 0),
                }
                bug_rejections = [row for row in rejected_rows if not (
                    restarted
                    and row.get("reason") == "rollout belongs to a different policy round"
                    and row.get("slot") == allowed_stale["slot"]
                    and isinstance(row.get("time_unix_ns"), int)
                    and row["time_unix_ns"] >= allowed_stale["after_unix_ns"]
                )]
                if bug_rejections:
                    raise AssertionError(f"learner rejected rollout(s) for a bug: {bug_rejections[:10]}")
            if now - last_sample >= 10:
                latest_actors = [actor for actor in actor_records if actor.get("active", True)]
                sample = smoke._sample_resources([*latest_actors, learner_record])
                samples.append(sample)
                actor_progress: dict[str, Any] = {}
                for actor in latest_actors:
                    manifest_candidates = list(Path(actor["launch_dir"]).rglob("manifest.json"))
                    actor_manifest = max(manifest_candidates, key=lambda path: path.stat().st_mtime_ns) if manifest_candidates else None
                    data = _load_json_retry(actor_manifest) if actor_manifest is not None else {}
                    slot = str(actor["slot_id"])
                    actor_round = int(data.get("last_policy_round", -1))
                    actor_progress[slot] = {"last_policy_round": actor_round,
                                            "rounds_published": data.get("rounds_published", 0),
                                            "status": data.get("status")}
                    if int(learner_state.get("round", 0)) > actor_round:
                        waiting_seconds[slot] += now - last_sample
                manifest["progress_samples"].append({
                    "elapsed_seconds": now - started, "round": learner_state.get("round", 0),
                    "completed_timesteps": learner_state.get("completed_timesteps", 0),
                    "accepted_rollouts": learner_state.get("accepted_rollouts", 0),
                    "rejected_rollouts": learner_state.get("rejected_rollouts", 0),
                    "last_training_seconds": learner_state.get("last_training_seconds"),
                    "actors": actor_progress,
                })
                steps, round_number = int(learner_state.get("completed_timesteps", 0)), int(learner_state.get("round", 0))
                if steps < last_steps or round_number < last_round:
                    raise AssertionError("learner progress regressed")
                last_steps, last_round = steps, round_number
                learner_manifest_path = shared_root / "manifest.json"
                learner_manifest = _load_json_retry(learner_manifest_path) if learner_manifest_path.is_file() else {}
                timings = learner_manifest.get("round_timings", [])
                if isinstance(timings, list):
                    training_timings = [dict(row) for row in timings if isinstance(row, Mapping)]
                last_sample = now
                persist()
            if not restarted and now - started >= restart_at_seconds:
                actor = actor_records[0]
                old_id = str(actor["launch_id"])
                old_launch_info = dict(actor)
                before_steps = int(learner_state.get("completed_timesteps", 0))
                replacement = control.restart_shared_actor(old_id, reason="planned mid-soak restart", timeout=90)
                resources, runner = replacement.get("resources"), replacement.get("runner_process")
                if not isinstance(resources, Mapping) or not isinstance(runner, Mapping):
                    raise TypeError("restarted actor metadata missing")
                old_launch_info["active"] = False
                manifest["launches"].append(old_launch_info)
                actor.update({"launch_id": str(replacement["launch_id"]), "runner_pid": int(runner["pid"]),
                              "runner_process": dict(runner),
                              "launch_dir": str(config.runtime_root / "launches" / str(replacement["launch_id"]))})
                manifest["launches"].append(dict(actor))
                manifest["actor_restart"] = {
                    "old_launch_id": old_id, "new_launch_id": actor["launch_id"],
                    "learner_steps_before": before_steps,
                    "learner_round_before": int(learner_state.get("round", 0)),
                    "slot_id": int(actor["slot_id"]), "restarted_unix_ns": time.time_ns(),
                    "learner_pid_unchanged": learner_record["runner_pid"] == int(identity["pid"]),
                }
                restarted = True
                persist()
            time.sleep(1)

        soak_stop_requested_ns = time.time_ns()
        soak_stop_elapsed = time.monotonic() - started
        control.stop(str(actor_records[0]["launch_id"]))
        stop_deadline = time.monotonic() + 120
        while time.monotonic() < stop_deadline:
            control.status()
            if control.shared_group_stopped(group_id):
                break
            time.sleep(0.5)
        else:
            raise TimeoutError("shared group did not stop cleanly")
        final_state = _load_json_retry(shared_root / "learner.json")
        final_manifest = _load_json_retry(shared_root / "manifest.json")
        if final_state.get("status") != "CANCELLED":
            raise AssertionError(f"learner did not cancel cooperatively: {final_state}")
        if int(final_state.get("round", 0)) < 1 or int(final_state.get("completed_timesteps", 0)) < 512:
            raise AssertionError("learner did not complete a two-actor training round")
        actors_per_round = final_manifest.get("actors_per_round", {})
        if not isinstance(actors_per_round, Mapping) or not any(
            set(slots) == {0, 1} for slots in actors_per_round.values() if isinstance(slots, list)
        ):
            raise AssertionError("no completed learner round contains both actor slots")
        if not restarted or not manifest.get("actor_restart", {}).get("learner_pid_unchanged"):
            raise AssertionError("actor was not restarted while the same learner remained alive")
        rejected = _shared_rejections(shared_root / "rejected.jsonl")
        expected_stale = [row for row in rejected if (
            restarted
            and row.get("reason") == "rollout belongs to a different policy round"
            and row.get("slot") == manifest.get("actor_restart", {}).get("slot_id")
            and isinstance(row.get("time_unix_ns"), int)
            and row["time_unix_ns"] >= manifest.get("actor_restart", {}).get("restarted_unix_ns", 0)
        )]
        if len(expected_stale) != len(rejected):
            raise AssertionError(f"learner rejected rollouts unexpectedly: {rejected[:10]}")
        manifest["expected_stale_rejections"] = expected_stale
        checkpoint = Path(str(final_manifest.get("latest_checkpoint", "")))
        if not checkpoint.is_file():
            raise AssertionError("learner checkpoint ZIP missing after safe stop")
        checkpoint_check = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "tools" / "verify_shared_checkpoint.py"),
             "--shared-root", str(shared_root), "--checkpoint", str(checkpoint)],
            cwd=PROJECT_ROOT, check=True, capture_output=True, text=True, timeout=180,
        )
        checkpoint_validation = json.loads(checkpoint_check.stdout.strip().splitlines()[-1])
        muting: dict[str, bool] = {}
        modlog_paths: dict[str, str] = {}
        for slot, root in enumerate(config.instance_data_roots):
            modlog = root / "ModLog.txt"
            text = modlog.read_text("utf-8", errors="replace")
            if modlog.stat().st_mtime < modlog_start_wall:
                raise AssertionError(f"ModLog for instance {slot} predates this soak")
            muting[str(slot)] = "Audio muted for training instance" in text and "Audio mute unavailable" not in text
            if not muting[str(slot)]:
                raise AssertionError(f"no audio-mute confirmation in slot {slot} persistentDataPath ModLog")
            modlog_paths[str(slot)] = str(modlog)
        safety = safety_check()
        if safety["problems"]:
            raise AssertionError("safety checks failed: " + "; ".join(safety["problems"]))
        variability = safety["reward_variability"]
        if not variability["transition_count"] or not any(
            abs(value) > 0 for value in variability["reward_component_totals"].values()
        ) or len([value for value in variability["reward_component_totals"].values() if value != 0]) < 2:
            raise AssertionError(f"reward appears constant or transition logs are missing: {variability}")
        elapsed = time.monotonic() - started
        rss_by_slot: dict[str, Any] = {}
        for slot in ("0", "1"):
            rss_by_slot[slot] = evaluate_rss_growth(
                samples, slot, samples[0]["unix_ns"], soak_stop_requested_ns
            )
            if not rss_by_slot[slot]["ok"]:
                raise AssertionError(f"slot {slot} RSS did not reach a stable regime: {rss_by_slot[slot]}")
        train_seconds = [float(row["training_seconds"]) for row in training_timings
                         if isinstance(row.get("training_seconds"), (int, float))]
        p95_train_seconds = float(np.percentile(train_seconds, 95)) if train_seconds else 0.0
        maximum_progress_gap = min(
            120.0, 45.0 + p95_train_seconds + 60.0
        )
        progress_problems = evaluate_progress(
            manifest["progress_samples"], stop_elapsed=soak_stop_elapsed,
            max_gap_seconds=maximum_progress_gap,
        )
        if progress_problems:
            raise AssertionError("combined learner progress check failed: " + "; ".join(progress_problems))
        manifest.update({
            "status": "PASSED", "duration_real_seconds": elapsed, "finished_unix_ns": time.time_ns(),
            "learner_final": final_state, "learner_manifest": final_manifest,
            "rejected_rollouts": rejected,
            "rounds_per_hour": int(final_state["round"]) * 3600 / elapsed,
            "combined_steps_per_second": int(final_state["completed_timesteps"]) / elapsed,
            "learner_training_seconds_per_round": {
                "samples": train_seconds, "average": statistics.fmean(train_seconds) if train_seconds else None,
                "p95": float(np.percentile(train_seconds, 95)) if train_seconds else None,
            },
            "actor_wait_seconds_estimate": {
                slot: float(value.get("wait_policy_ms", 0.0)) / 1000
                for slot, value in _shared_telemetry_summary(shared_root)["actors"].items()
            }, "audio_muted_from_modlogs": muting,
            "straggler_timeout_seconds": 45.0,
            "maximum_progress_gap_seconds": maximum_progress_gap,
            "checkpoint_reload_subprocess": checkpoint_validation,
            "modlog_paths": modlog_paths, "modlog_session_start_unix": modlog_start_wall,
            "bridge_sha256_after": _deployment_hashes(config, project_root),
            "resource_summary": smoke._resource_summary(samples), "safety": safety,
            "reward_variability": variability,
            "rss_by_slot": rss_by_slot, "progress_problems": progress_problems,
            "actor_occupancy": _shared_telemetry_summary(shared_root)["actors"],
            "learner_telemetry": _shared_telemetry_summary(shared_root)["learner"],
            "recommendation": _shared_telemetry_summary(shared_root)["recommendation"],
            "checks": {"learner_rounds_advanced": True, "timesteps_increased": True,
                       "two_actor_round": True, "zero_bug_rollout_rejections": True,
                       "checkpoint_zip_reloaded": checkpoint_validation.get("status") == "PASSED",
                       "zero_unavailable_masked_actions_executed": True, "audio_muted": True,
                       "actor_restarted_and_learner_continued": True,
                       "personal_save_baseline_registry_isolation_orphans": True},
        })
        if start_long_campaign:
            long_payload = _shared_campaign_payload("longa-learner-1-" + stamp, "longa-learner-1")
            long_payload["max_checkpoints"] = 5
            long_campaign = control.create_campaign(long_payload)
            long_launch = control.launch(str(long_campaign["campaign_id"]),
                                         version=int(long_campaign["version"]), execute=True)
            long_group_id = str(long_launch["shared_group"])
            long_root = Path(str(long_launch["shared_root"]))
            long_status: dict[str, Any] = {}
            for _ in range(600):
                try:
                    long_status = _load_json_retry(long_root / "learner.json")
                except (OSError, json.JSONDecodeError):
                    long_status = {}
                if long_status.get("status") in {"FAILED", "CANCELLED", "COMPLETED"}:
                    raise RuntimeError(f"long learner ended during startup: {long_status}")
                if long_status.get("status") in {"WAITING", "TRAINING", "RUNNING"}:
                    break
                time.sleep(1)
            else:
                raise TimeoutError("long learner did not publish a live heartbeat")
            manifest["long_campaign"] = {
                "campaign_id": long_campaign["campaign_id"], "version": long_campaign["version"],
                "shared_group": long_group_id, "shared_root": str(long_root),
                "launch_ids": [str(long_launch["learner"]["launch_id"]),
                               *[str(actor["launch_id"]) for actor in long_launch["actors"]]],
                "status": long_status["status"], "panel_url": "http://127.0.0.1:8765",
                "stop_instructions": "No painel, selecione qualquer membro ativo do grupo e clique em Parar.",
            }
        else:
            manifest["long_campaign"] = None
        path = project_root / "agent-prompts" / "relatorios" / "46_resoak_learner.txt"
        _write_shared_report(manifest, path)
        manifest["report_path"] = str(path)
        persist()
        return manifest, control
    except BaseException as error:  # noqa: BLE001 - soak failure must trigger safe group teardown
        for failed_group in (long_group_id, group_id):
            if not failed_group:
                continue
            try:
                active = [row for row in control._launches.values()
                          if row.get("shared_group") == failed_group
                          and row.get("status") not in {"COMPLETED", "CANCELLED", "FAILED"}]
                if active:
                    control.stop(str(active[0]["launch_id"]))
                    deadline = time.monotonic() + 120
                    while time.monotonic() < deadline:
                        control.status()
                        if control.shared_group_stopped(failed_group):
                            break
                        time.sleep(0.5)
            except Exception as stop_error:  # noqa: BLE001 - preserve primary failure and report cleanup
                manifest.setdefault("teardown_errors", []).append(
                    f"{failed_group}: {type(stop_error).__name__}: {stop_error}"
                )
        manifest.update({"status": "FAILED", "failure": f"{type(error).__name__}: {error}",
                         "duration_real_seconds": time.monotonic() - started,
                         "finished_unix_ns": time.time_ns()})
        if shared_root is not None:
            telemetry = _shared_telemetry_summary(shared_root)
            manifest["actor_occupancy"] = telemetry["actors"]
            manifest["learner_telemetry"] = telemetry["learner"]
            manifest["recommendation"] = telemetry["recommendation"]
            manifest["straggler_timeout_seconds"] = 45.0
            manifest["actor_wait_seconds_estimate"] = {
                slot: float(value.get("wait_policy_ms", 0.0)) / 1000
                for slot, value in telemetry["actors"].items()
            }
            if isinstance(manifest.get("learner_manifest"), Mapping):
                timings = manifest["learner_manifest"].get("round_timings", [])
                if isinstance(timings, list):
                    train_seconds = [
                        float(row["training_seconds"]) for row in timings
                        if isinstance(row, Mapping) and isinstance(row.get("training_seconds"), (int, float))
                    ]
                    manifest["learner_training_seconds_per_round"] = {
                        "samples": train_seconds,
                        "average": statistics.fmean(train_seconds) if train_seconds else None,
                        "p95": float(np.percentile(train_seconds, 95)) if train_seconds else None,
                    }
        try:
            manifest["safety"] = safety_check()
        except Exception as safety_error:  # noqa: BLE001 - keep the primary soak failure
            manifest["safety_error"] = f"{type(safety_error).__name__}: {safety_error}"
        manifest["resource_summary"] = smoke._resource_summary(samples) if samples else {}
        if shared_root is not None:
            try:
                manifest["learner_final"] = _load_json_retry(shared_root / "learner.json")
                manifest["learner_manifest"] = _load_json_retry(shared_root / "manifest.json")
                manifest["rejected_rollouts"] = _shared_rejections(shared_root / "rejected.jsonl")
                final_round = int(manifest["learner_final"].get("round", 0))
                final_steps = int(manifest["learner_final"].get("completed_timesteps", 0))
                elapsed = max(float(manifest["duration_real_seconds"]), 1.0)
                manifest["rounds_per_hour"] = final_round * 3600 / elapsed
                manifest["combined_steps_per_second"] = final_steps / elapsed
                manifest["reward_variability"] = manifest.get("safety", {}).get(
                    "reward_variability", {}
                )
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as evidence_error:
                manifest["evidence_error"] = f"{type(evidence_error).__name__}: {evidence_error}"
            audio: dict[str, bool] = {}
            modlog_paths: dict[str, str] = {}
            for slot, root in enumerate(config.instance_data_roots):
                path = root / "ModLog.txt"
                modlog_paths[str(slot)] = str(path)
                try:
                    text = path.read_text("utf-8", errors="replace")
                    audio[str(slot)] = (
                        path.stat().st_mtime >= modlog_start_wall
                        and "Audio muted for training instance" in text
                        and "Audio mute unavailable" not in text
                    )
                except OSError:
                    audio[str(slot)] = False
            manifest["audio_muted_from_modlogs"] = audio
            manifest["modlog_paths"] = modlog_paths
            manifest["checks"] = {
                "learner_rounds_advanced": int(manifest.get("learner_final", {}).get("round", 0)) > 0,
                "zero_bug_rollout_rejections": not manifest.get("rejected_rollouts"),
                "actor_restarted_and_learner_continued": bool(manifest.get("actor_restart", {}).get("learner_pid_unchanged")),
                "checkpoint_zip_reloaded": bool(manifest.get("checkpoint_reload_subprocess", {}).get("status") == "PASSED"),
                "audio_muted": bool(audio) and all(audio.values()),
                "safety": not bool(manifest.get("safety", {}).get("problems")),
                "shared_group_stopped": bool(group_id and control.shared_group_stopped(group_id)),
            }
        persist()
        _write_shared_report(manifest, project_root / "agent-prompts" / "relatorios" / "46_resoak_learner.txt")
        control.close()
        return manifest, None


def _write_shared_report(manifest: Mapping[str, Any], path: Path) -> None:
    timing = manifest.get("learner_training_seconds_per_round", {})
    lines = [
        "SOAK LEARNER COMPARTILHADO — PROMPT 46", "=" * 43,
        f"Status: {manifest.get('status')}",
        f"Duração real: {manifest.get('duration_real_seconds', 0):.1f}s",
        f"Rodadas/hora: {manifest.get('rounds_per_hour', 'n/a')}",
        f"Passos combinados/s: {manifest.get('combined_steps_per_second', 'n/a')}",
        f"Treino PPO/rodada: média {timing.get('average')}s, p95 {timing.get('p95')}s",
        f"Espera estimada dos atores: {manifest.get('actor_wait_seconds_estimate')}",
        f"Ocupação real: {json.dumps(manifest.get('actor_occupancy', {}), ensure_ascii=False)}",
        f"Telemetria learner: {json.dumps(manifest.get('learner_telemetry', {}), ensure_ascii=False)}",
        f"Straggler timeout: {manifest.get('straggler_timeout_seconds', 45)}s",
        f"Novidade: {json.dumps(manifest.get('reward_variability', {}), ensure_ascii=False)}",
        f"Recomendação rollout/straggler: {json.dumps(manifest.get('recommendation', {}), ensure_ascii=False)}",
        f"CPU/RSS: {json.dumps(manifest.get('resource_summary', {}), ensure_ascii=False)}",
        f"Rollouts rejeitados: {json.dumps(manifest.get('rejected_rollouts', []), ensure_ascii=False)}",
        f"DLL SHA-256: {json.dumps(manifest.get('bridge_sha256'), ensure_ascii=False)}",
        f"Áudio e ModLog por persistentDataPath: {json.dumps(manifest.get('audio_muted_from_modlogs'), ensure_ascii=False)}",
        f"Campanha longa: {json.dumps(manifest.get('long_campaign', {}), ensure_ascii=False)}",
        f"Checks: {json.dumps(manifest.get('checks', {}), ensure_ascii=False)}",
        f"Falha: {manifest.get('failure', 'nenhuma')}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _shared_telemetry_summary(shared_root: Path) -> dict[str, Any]:
    actors: dict[str, Any] = {}
    for slot in (0, 1, 2):
        path = shared_root / "telemetry" / f"actor-slot-{slot}.jsonl"
        rows: list[dict[str, Any]] = []
        if path.is_file():
            for line in path.read_text("utf-8", errors="replace").splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    rows.append(item)
        if not rows:
            continue
        collect = sum(float(row.get("collect_ms", 0.0)) for row in rows)
        wait = sum(float(row.get("wait_policy_ms", 0.0)) for row in rows)
        publish = sum(float(row.get("publish_ms", 0.0)) for row in rows)
        total = collect + wait + publish
        actors[str(slot)] = {
            "rounds_measured": sum("collect_ms" in row for row in rows),
            "collect_ms": collect, "wait_policy_ms": wait, "publish_ms": publish,
            "collect_percent": 100.0 * collect / total if total else None,
            "wait_policy_percent": 100.0 * wait / total if total else None,
            "publish_percent": 100.0 * publish / total if total else None,
            "stale_rollouts": sum(int(row.get("stale_rollouts", 0)) for row in rows),
            "resets": sum(int(row.get("resets", 0)) for row in rows),
            "relaunches": sum(int(row.get("relaunches", 0)) for row in rows),
        }
    try:
        learner = json.loads((shared_root / "manifest.json").read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        learner = {}
    timings = learner.get("round_timings", [])
    if not isinstance(timings, list):
        timings = []
    stale_path = shared_root / "telemetry" / "stale-rollouts.jsonl"
    try:
        with stale_path.open("rb") as stream:
            stale_total = sum(1 for _ in stream)
    except OSError:
        stale_total = 0
    waiting_percentages = [
        float(value["wait_policy_percent"]) for value in actors.values()
        if isinstance(value.get("wait_policy_percent"), (int, float))
    ]
    return {
        "actors": actors,
        "recommendation": {
            "rollout_steps": 512 if waiting_percentages and max(waiting_percentages) <= 20 else 1024,
            "straggler_timeout_seconds": 45.0,
            "basis": "keep 512 when measured actor policy-wait is at most 20%; otherwise test 1024",
            "validated": bool(waiting_percentages),
        },
        "learner": {
            "rounds": len(timings),
            "wait_rollouts_ms": sum(float(row.get("wait_rollouts_ms", 0.0))
                                     for row in timings if isinstance(row, Mapping)),
            "straggler_wait_ms": sum(float(row.get("straggler_wait_ms", 0.0))
                                      for row in timings if isinstance(row, Mapping)),
            "train_ms": sum(float(row.get("train_ms", 0.0))
                             for row in timings if isinstance(row, Mapping)),
            "actors_received": [row.get("actors_received", []) for row in timings
                                if isinstance(row, Mapping)],
            "stale_rollouts": stale_total,
            "completed_timesteps": learner.get("completed_timesteps", 0),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-open-games", action="store_true",
                        help="authorize this script to open both managed Hollow Knight instances")
    parser.add_argument("--shared-learner", action="store_true",
                        help="run prompt 44's shared learner soak and long campaign")
    parser.add_argument("--no-long-campaign", action="store_true",
                        help="stop after the soak and leave the shared group safely stopped")
    args = parser.parse_args()
    if not args.confirm_open_games:
        print("No games opened. Run with --confirm-open-games after authorization.")
        return 2
    if args.shared_learner:
        result, control = _run_shared_learner_soak(
            PROJECT_ROOT, start_long_campaign=not args.no_long_campaign
        )
        if result.get("status") != "PASSED" or control is None:
            print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
            return 1
        if args.no_long_campaign:
            print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
            return 0
        from hk_agent.control_plane_server import ControlPlaneHTTPServer

        static_root = PROJECT_ROOT / "src" / "hk_agent" / "control_plane_web"
        server: Any = None
        server_thread: threading.Thread | None = None
        try:
            server = ControlPlaneHTTPServer(("127.0.0.1", 8765), control, static_root)
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            long_campaign = result["long_campaign"]
            long_root = Path(str(long_campaign["shared_root"]))
            learner_id = str(long_campaign["launch_ids"][0])
            started_at = time.monotonic()
            last_round_progress: float | None = None
            initial_state = _load_json_retry(long_root / "learner.json")
            initial_round = int(initial_state.get("round", 0))
            last_round = initial_round
            samples: list[dict[str, Any]] = []
            next_panel_poll = started_at
            while time.monotonic() - started_at < 600:
                elapsed = time.monotonic() - started_at
                try:
                    state = _load_json_retry(long_root / "learner.json")
                except (OSError, json.JSONDecodeError):
                    state = {}
                if state.get("status") in {"FAILED", "CANCELLED", "COMPLETED"}:
                    raise RuntimeError(f"long learner terminated during 10-minute panel check: {state}")
                current_round = int(state.get("round", 0))
                if current_round > last_round:
                    last_round = current_round
                    last_round_progress = time.monotonic()
                elif last_round_progress is not None and time.monotonic() - last_round_progress > 120:
                    raise TimeoutError("long learner made no round progress through panel for over 120 seconds")
                if time.monotonic() >= next_panel_poll:
                    with urllib.request.urlopen("http://127.0.0.1:8765/api/status", timeout=5) as response:
                        panel_status = json.loads(response.read().decode("utf-8"))
                    active = panel_status.get("active_launches", [])
                    panel_learner = next(
                        (row for row in active if row.get("launch_id") == learner_id), None
                    )
                    if panel_learner is None:
                        raise RuntimeError("ControlPlane panel no longer lists the long learner")
                    shared_status = panel_learner.get("shared_status", {})
                    samples.append({
                        "elapsed_seconds": elapsed,
                        "panel_round": int(shared_status.get("round", 0)),
                        "learner_round": current_round,
                        "completed_timesteps": int(state.get("completed_timesteps", 0)),
                    })
                    next_panel_poll = time.monotonic() + 10
                time.sleep(1)
            if last_round <= initial_round:
                raise AssertionError("long learner did not advance a round during panel verification")
            long_campaign.update({
                "panel_verification_seconds": time.monotonic() - started_at,
                "round_at_verification_start": initial_round,
                "round_at_verification_end": last_round,
                "panel_samples": samples,
            })
            result["finished_unix_ns"] = time.time_ns()
            manifest_path = Path(str(result["soak_root"])) / "soak-manifest.json"
            manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            _write_shared_report(result, PROJECT_ROOT / "agent-prompts" / "relatorios" / "46_resoak_learner.txt")
            print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
            print("Campanha longa em execução e painel disponível em http://127.0.0.1:8765", flush=True)
            server_thread.join()
        except KeyboardInterrupt:
            try:
                control.stop(str(result["long_campaign"]["launch_ids"][0]))
                stop_deadline = time.monotonic() + 120
                while time.monotonic() < stop_deadline and not control.shared_group_stopped(
                    str(result["long_campaign"]["shared_group"])
                ):
                    time.sleep(0.25)
            finally:
                if server is not None:
                    server.shutdown()
                    server.server_close()
                control.close()
            print("Parada segura da campanha longa solicitada; grupo encerrado.", flush=True)
            return 130
        except (OSError, RuntimeError, ValueError, AssertionError, KeyError) as error:
            try:
                control.stop(str(result["long_campaign"]["launch_ids"][0]))
                stop_deadline = time.monotonic() + 120
                while time.monotonic() < stop_deadline and not control.shared_group_stopped(
                    str(result["long_campaign"]["shared_group"])
                ):
                    time.sleep(0.25)
            except (OSError, RuntimeError, ValueError) as cleanup_error:
                result["teardown_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
            if server is not None:
                server.shutdown()
                server.server_close()
            control.close()
            result.update({"status": "FAILED", "failure": f"{type(error).__name__}: {error}"})
            result["long_campaign"]["status"] = "STOP_REQUESTED_AFTER_VALIDATION_FAILURE"
            manifest_path = Path(str(result["soak_root"])) / "soak-manifest.json"
            manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            _write_shared_report(result, PROJECT_ROOT / "agent-prompts" / "relatorios" / "46_resoak_learner.txt")
            print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
            return 1
        finally:
            if server is not None and server.fileno() >= 0 and server_thread is not None and not server_thread.is_alive():
                server.server_close()
        return 0
    result = _run_live_soak(PROJECT_ROOT)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
