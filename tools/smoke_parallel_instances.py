"""Run and verify a supervised two-instance bridge smoke test."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path, PureWindowsPath
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hk_agent.config import load_config
from hk_agent.control_plane import ControlPlane
from hk_agent.instance_marker import load_instance_marker


def hash_tree(root: Path) -> dict[str, str]:
    """Return stable relative-path to SHA-256 mapping; missing roots are errors."""
    if not root.is_dir():
        raise FileNotFoundError(root)
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def tree_changes(before: Mapping[str, str], after: Mapping[str, str]) -> dict[str, list[str]]:
    return {
        "added": sorted(after.keys() - before.keys()),
        "removed": sorted(before.keys() - after.keys()),
        "changed": sorted(key for key in before.keys() & after.keys() if before[key] != after[key]),
    }


def verify_distinct(values: Iterable[object], label: str, expected: int = 2) -> None:
    items = list(values)
    if len(items) != expected or len(set(items)) != expected:
        raise AssertionError(f"{label} must contain {expected} distinct values: {items!r}")


def verify_root_isolation(
    before: Mapping[str, Mapping[str, str]],
    after: Mapping[str, Mapping[str, str]],
    owners: Mapping[str, str],
) -> None:
    """Reject changes in roots not explicitly owned by their corresponding launch."""
    if set(before) != set(after):
        raise AssertionError("data-root snapshot identities changed")
    if set(owners) != set(before):
        raise AssertionError("each data root must have exactly one declared owner")
    for root, old in before.items():
        changes = tree_changes(old, after[root])
        changed_paths = [path for paths in changes.values() for path in paths]
        if changed_paths and owners[root] != root:
            raise AssertionError(f"data root {root!r} changed outside its owning launch")
        unexpected = [path for path in changed_paths if not _is_expected_instance_file(path)]
        if unexpected:
            raise AssertionError(f"unexpected non-save changes in data root {root!r}: {unexpected}")


_INSTANCE_RUNTIME_FILES = frozenset(
    {"modlog.txt", "appconfig.ini", "player.log", "player-prev.log", "output_log.txt"}
)
_INSTANCE_RUNTIME_DIRS = frozenset({"old modlogs", "crashes"})


def _is_expected_instance_file(path: str) -> bool:
    """Files the game and Modding API legitimately write in their own data root."""
    parts = Path(path).parts
    name = Path(path).name.casefold()
    if _is_save_file(path) or name in _INSTANCE_RUNTIME_FILES:
        return True
    if parts and parts[0].casefold() in _INSTANCE_RUNTIME_DIRS:
        return True
    return name.endswith((".globalsettings.json", ".globalsettings.json.bak"))


def verify_runner_identity(
    launched: Mapping[str, Any], reported: Mapping[str, Any]
) -> str:
    """Match the bridge evidence's runner to the process the panel launched.

    Returns "direct" when they are the same process, or "venv-redirector" when the
    panel launched the Windows venv redirector (Scripts\\python.exe) and the
    evidence comes from its direct child, the base interpreter. Anything else fails.
    """
    if all(reported.get(key) == launched.get(key) for key in ("pid", "creation_time", "executable")):
        return "direct"
    launched_pid = launched.get("pid")
    launched_exe = str(launched.get("executable") or "")
    reported_exe = str(reported.get("executable") or "")
    launched_path = PureWindowsPath(launched_exe)
    reported_path = PureWindowsPath(reported_exe)
    is_redirector = launched_path.parent.name.casefold() == "scripts" and (
        launched_path.name.casefold() in {"python.exe", "pythonw.exe"}
    )
    launched_created = launched.get("creation_time")
    reported_created = reported.get("creation_time")
    if (
        is_redirector
        and isinstance(launched_pid, int)
        and reported.get("parent_pid") == launched_pid
        and reported.get("pid") != launched_pid
        and reported_path.name.casefold() in {"python.exe", "pythonw.exe"}
        and reported_path.parent != launched_path.parent
        and _not_older(reported_created, launched_created)
    ):
        return "venv-redirector"
    raise AssertionError("runner identity differs between launch and bridge evidence")


def verify_frame_captures(evidence: Mapping[str, Any], *, require_after_action: bool) -> None:
    """Check the bridge captures declared by the probe evidence."""
    frame_paths = evidence.get("frame_paths")
    if not isinstance(frame_paths, list) or len(frame_paths) != 2:
        raise AssertionError("bridge evidence does not declare its two frame captures")
    if not Path(str(frame_paths[0])).is_file():
        raise AssertionError("initial bridge capture is missing")
    if require_after_action and not Path(str(frame_paths[1])).is_file():
        raise AssertionError("post-action bridge capture is missing")


def _not_older(child: object, parent: object) -> bool:
    if isinstance(child, (int, float)) and isinstance(parent, (int, float)):
        # Both values come from the same clock source (FILETIME on Windows,
        # psutil elsewhere); a child can never be created before its parent.
        return child >= parent
    return False


def _is_save_file(path: str) -> bool:
    name = Path(path).name.casefold()
    return name.startswith("user") and name.endswith(
        (".dat", ".modded.json", ".dat.bak", ".modded.json.bak")
    )


def verify_no_orphans(processes: Iterable[Mapping[str, Any]], known_pids: set[int]) -> None:
    remaining = [p for p in processes if p.get("pid") in known_pids and p.get("alive")]
    if remaining:
        raise AssertionError(f"orphan launch processes remain: {remaining!r}")


def registry_snapshot(destination: Path) -> dict[str, object]:
    """Export the protected Team Cherry key to a temporary evidence file and hash it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["reg.exe", "export", r"HKCU\Software\Team Cherry", str(destination), "/y"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"reg export failed ({result.returncode}): {result.stderr.strip()}")
    raw = destination.read_bytes()
    return {
        "path": str(destination),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "protected_sha256": protected_registry_digest(raw),
    }


_INSTANCE_KEY = re.compile(r"\\Team Cherry\\Hollow Knight AI \d+\]$", re.IGNORECASE)


def protected_registry_digest(raw: bytes) -> str:
    """Hash every Team Cherry section except the managed instance keys.

    The instance keys (Hollow Knight AI <n>) are expected to change while the
    copies run; the personal key and anything else must stay byte-identical.
    """
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8")
    kept: list[str] = []
    skipping = False
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.startswith("["):
            skipping = bool(_INSTANCE_KEY.search(line))
        if not skipping:
            kept.append(line)
    return hashlib.sha256("\n".join(kept).encode("utf-8")).hexdigest()


def preflight(project_root: Path) -> dict[str, object]:
    if importlib.util.find_spec("psutil") is None:
        raise RuntimeError("psutil is required before starting the two games")
    import psutil

    config = load_config(project_root / "config.training.local.toml", allow_training=True)
    if config.instance_count != 2 or not config.parallel_enabled:
        raise RuntimeError("smoke requires instances.count=2 and parallel_enabled=true")
    for index, install in enumerate(config.install_roots, start=1):
        marker = load_instance_marker(install, expected_index=index)
        if not marker.product_name or not marker.persistent_data_path_expected:
            raise RuntimeError(f"instance marker {index} is incomplete")
    try:
        game_processes = [
            {"pid": process.pid, "name": process.info.get("name"),
             "executable": process.info.get("exe")}
            for process in psutil.process_iter(["name", "exe"])
            if str(process.info.get("name", "")).casefold() == "hollow knight.exe"
        ]
    except psutil.Error as error:
        raise RuntimeError(f"could not verify running Hollow Knight processes: {error}") from error
    if game_processes:
        raise RuntimeError(f"preflight requires no Hollow Knight.exe process: {game_processes}")
    usage = shutil.disk_usage(config.runtime_root.parent if config.runtime_root.parent.exists() else project_root)
    if usage.free < 2 * 1024**3:
        raise RuntimeError("at least 2 GiB of free space is required for evidence and save copies")
    return {"instance_count": config.instance_count, "parallel_enabled": config.parallel_enabled,
            "free_bytes": usage.free, "data_roots": [str(p) for p in config.instance_data_roots],
            "hollow_knight_processes": game_processes}


def _campaign_payload(campaign_id: str) -> dict[str, object]:
    return {
        "campaign_id": campaign_id,
        "name": "Two instance bridge smoke",
        "campaign_type": "parallel_smoke",
        "seed": 17,
        "timesteps": 1,
        "checkpoint_interval": 1,
        "episodes": 1,
        "horizon": 10_000,
        "action_repeat": 1,
        "lease_ms": 80,
        "window_width": 960,
        "window_height": 540,
        "duration_mode": "limited",
        "max_checkpoints": 2,
        "save_source": "existing_save",
        "save_slot": 1,
        "weights": {},
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:
        output = subprocess.run(["tasklist.exe", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True, check=False).stdout
        return f'"{pid}"' in output


def _same_process_alive(pid: int, creation_time: float) -> bool:
    """True only if this exact process (PID + creation time) is still running."""
    try:
        import psutil
    except ImportError:
        return False
    try:
        return abs(psutil.Process(pid).create_time() - creation_time) < 0.01
    except (psutil.Error, OSError, ValueError):
        # psutil.NoSuchProcess is not an OSError: an exited process means
        # "not alive", which is exactly the orphan check's success case.
        return False


def _gpu_usage_by_pid() -> tuple[dict[int, int], int | None]:
    command = shutil.which("nvidia-smi")
    if command is None:
        return {}, None
    result = subprocess.run(
        [command, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False, timeout=5,
    )
    memory: dict[int, int] = {}
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", 1)]
            try:
                memory[int(parts[0])] = int(parts[1])
            except (IndexError, ValueError):
                continue
    gpu_result = subprocess.run(
        [command, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False, timeout=5,
    )
    try:
        utilization = int(gpu_result.stdout.splitlines()[0].strip()) if gpu_result.returncode == 0 else None
    except (IndexError, ValueError):
        utilization = None
    return memory, utilization


# psutil reports CPU between two calls on the SAME Process object; a fresh
# object always returns 0.0, so processes are cached across samples.
_PROCESS_CACHE: dict[tuple[int, float], Any] = {}


def _cached_process(process: Any) -> Any:
    try:
        key = (process.pid, process.create_time())
    except Exception:  # noqa: BLE001 - process vanished between enumeration and lookup
        return process
    cached = _PROCESS_CACHE.get(key)
    if cached is None:
        _PROCESS_CACHE[key] = process
        process.cpu_percent(interval=None)  # prime the baseline
        return process
    return cached


def _sample_resources(launches: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("psutil is required for CPU/RAM measurements") from error
    by_instance: dict[str, Any] = {}
    gpu_memory, gpu_utilization = _gpu_usage_by_pid()
    total_cpu = 0.0
    total_ram = 0
    total_vram = 0
    known: dict[str, list[int]] = {}
    for launch in launches:
        key = str(launch["slot_id"])
        root_pid = int(launch["runner_pid"])
        try:
            root = psutil.Process(root_pid)
            descendants = [_cached_process(p) for p in (root, *root.children(recursive=True))]
        except psutil.Error:
            descendants = []
        cpu = 0.0
        ram = 0
        pids: list[int] = []
        process_birth_times: dict[int, float] = {}
        vram = 0
        for process in descendants:
            try:
                cpu += process.cpu_percent(interval=None)
                ram += process.memory_info().rss
                pids.append(process.pid)
                process_birth_times[process.pid] = process.create_time()
                vram += gpu_memory.get(process.pid, 0)
            except psutil.Error:
                continue
        known[key] = sorted(set(pids))
        by_instance[key] = {"cpu_percent": cpu, "ram_bytes": ram,
                            "vram_mib": vram, "pids": pids,
                            "process_birth_times": process_birth_times}
        total_cpu += cpu
        total_ram += ram
        total_vram += vram
    return {"unix_ns": time.time_ns(), "instances": by_instance,
            "total": {"cpu_percent": total_cpu, "ram_bytes": total_ram,
                      "vram_mib": total_vram},
            "gpu_utilization_percent": gpu_utilization, "known_pids": known}


def _resource_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    keys = [*sorted(samples[0]["instances"]), "total"] if samples else []
    for key in keys:
        rows = [sample["total"] if key == "total" else sample["instances"][key]
                for sample in samples]
        result[key] = {}
        for metric in ("cpu_percent", "ram_bytes", "vram_mib"):
            numbers = [float(row[metric]) for row in rows if row.get(metric) is not None]
            result[key][metric] = {
                "average": statistics.fmean(numbers) if numbers else None,
                "peak": max(numbers) if numbers else None,
            }
    return result


def _log_tail(control: ControlPlane | None, launch_id: str | None, limit: int = 4000) -> str:
    if control is None or launch_id is None:
        return ""
    record = control._launches.get(launch_id) or {}
    log_value = record.get("log_path")
    if not isinstance(log_value, str):
        return ""
    try:
        text = (control.project_root / log_value).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-limit:]


def _wait_for_evidence(
    path: Path,
    timeout: float,
    *,
    control: ControlPlane | None = None,
    launch_id: str | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Fail fast: a runner that already exited will never produce a HELLO.
        if control is not None and launch_id is not None:
            record = control._launches.get(launch_id) or {}
            if record.get("status") in {"COMPLETED", "CANCELLED", "FAILED"}:
                raise RuntimeError(
                    f"runner for {launch_id} ended ({record.get('status')}, "
                    f"return_code={record.get('return_code')}) before the bridge HELLO; "
                    f"process.log tail:\n{_log_tail(control, launch_id)}"
                )
        if path.is_file():
            try:
                value = _load_json(path)
                if value.get("run_id") and value.get("session_id"):
                    return value
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        time.sleep(0.25)
    raise TimeoutError(f"bridge HELLO evidence timeout: {path}")


def _wait_terminal(control: ControlPlane, launch_id: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = control._launches.get(launch_id)
        if record is not None and record.get("status") in {"COMPLETED", "CANCELLED", "FAILED"}:
            return record
        time.sleep(0.25)
    raise TimeoutError(f"launch did not stop cooperatively: {launch_id}")


def _safe_process_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in ("pid", "creation_time", "executable")}


def _run_live_smoke(project_root: Path) -> dict[str, Any]:
    preflight_record = preflight(project_root)
    config = load_config(project_root / "config.training.local.toml", allow_training=True)
    smoke_root = config.runtime_root / "smoke" / dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    smoke_root.mkdir(parents=True, exist_ok=False)
    personal_backup = smoke_root / "backup" / "personal-saves"
    shutil.copytree(config.personal_save_root, personal_backup)
    personal_before = hash_tree(config.personal_save_root)
    baseline_before = hash_tree(config.experimental_save_root)
    data_before = {f"slot{index}": hash_tree(root)
                   for index, root in enumerate(config.instance_data_roots)}
    registry_before = registry_snapshot(smoke_root / "registry-before.reg")
    control = ControlPlane(
        project_root, allow_execute=True, max_active_launches=2, parallel_enabled=True,
        runtime_root=config.runtime_root, training_device="cpu",
        torch_threads=config.torch_threads, torch_interop_threads=config.torch_interop_threads,
    )
    campaign_id = "parallel-smoke-" + dt.datetime.now().strftime("%H%M%S")
    campaign = control.create_campaign(_campaign_payload(campaign_id))
    launched: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "schema_version": 1, "artifact_type": "parallel_instances_smoke",
        "status": "RUNNING", "started_unix_ns": time.time_ns(),
        "preflight": preflight_record,
        "protected_paths": {"personal": str(config.personal_save_root),
                            "baseline": str(config.experimental_save_root)},
        "snapshots_before": {"personal": personal_before, "baseline": baseline_before,
                             "data_roots": data_before, "registry": registry_before},
        "personal_backup": str(personal_backup), "launches": [], "resource_samples": samples,
    }
    manifest_path = smoke_root / "smoke-manifest.json"
    try:
        status = control.status()
        if status.get("active_count") != 0:
            raise RuntimeError("stop existing control-plane launches before running this smoke")
        for _ in range(2):
            record = control.launch(str(campaign["campaign_id"]), version=int(campaign["version"]), execute=True)
            resources = record.get("resources")
            if not isinstance(resources, Mapping) or not isinstance(resources.get("slot_id"), int):
                raise TypeError("control plane did not assign a real instance slot")
            slot = int(resources["slot_id"])
            launch_id = str(record["launch_id"])
            runtime_launch = config.runtime_root / "launches" / launch_id
            game_root = control.control_root / "launches" / launch_id
            runner = record.get("runner_process")
            if not isinstance(runner, Mapping) or not isinstance(runner.get("pid"), int):
                raise TypeError(f"runner PID missing for {launch_id}")
            launch_info = {
                "launch_id": launch_id, "slot_id": slot,
                "runner_pid": int(runner["pid"]),
                "runner_process": _safe_process_record(runner),
                "game_record_path": str(game_root / "game-process.json"),
                "data_root": str(resources["instance_data_root"]),
                "product_name_expected": resources.get("product_name"),
                "player_log": resources.get("log_file"),
                "evidence_path": str(runtime_launch / "smoke-bridge-evidence.json"),
            }
            launched.append(launch_info)
        launched.sort(key=lambda item: item["slot_id"])
        verify_distinct([item["runner_pid"] for item in launched], "runner PIDs")
        if [item["slot_id"] for item in launched] != [0, 1]:
            raise AssertionError("control plane did not use slots 0 and 1")
        bridge_evidence = [
            _wait_for_evidence(
                Path(item["evidence_path"]),
                120,
                control=control,
                launch_id=str(item["launch_id"]),
            )
            for item in launched
        ]
        for item, evidence in zip(launched, bridge_evidence, strict=True):
            game = _load_json(Path(item["game_record_path"]))
            item["game_process"] = _safe_process_record(game)
            item["bridge"] = evidence
            reported_runner = evidence.get("runner_process")
            if not isinstance(reported_runner, Mapping):
                raise TypeError("runner process identity missing from bridge evidence")
            verify_runner_identity(item["runner_process"], reported_runner)
            expected_executable = (config.install_roots[int(item["slot_id"])] / "Hollow Knight.exe").resolve(strict=False)
            if Path(str(item["game_process"].get("executable", ""))).resolve(strict=False) != expected_executable:
                raise AssertionError(f"game process executable does not match slot {item['slot_id']}")
            if item["game_process"].get("creation_time") is None:
                raise AssertionError(f"game creation time missing for slot {item['slot_id']}")
            expected_root = config.instance_data_roots[int(item["slot_id"])].resolve(strict=False)
            actual_root = Path(str(evidence.get("persistent_data_path", ""))).resolve(strict=False)
            if actual_root != expected_root:
                raise AssertionError(f"HELLO persistentDataPath crossed slots: {actual_root} != {expected_root}")
            if evidence.get("product_name") != item["product_name_expected"]:
                raise AssertionError("HELLO productName does not match the reserved instance")
            # At HELLO time only the initial capture can exist; the post-action
            # capture is verified after the run, from the final evidence.
            verify_frame_captures(evidence, require_after_action=False)
        for field in ("run_id", "token_sha256", "port", "instance_id", "persistent_data_path", "product_name"):
            verify_distinct([entry[field] for entry in bridge_evidence], field)
        verify_distinct([item["game_process"].get("pid") for item in launched], "game PIDs")
        verify_distinct([item["runner_pid"] for item in launched]
                        + [item["game_process"]["pid"] for item in launched], "all launch PIDs", 4)
        started = time.monotonic()
        stop_zero_at = started + 60
        last_slot1_steps = 0
        slot1_progress_at_stop = False
        stopped_zero = False
        stopped_one = False
        post_slot0_deadline: float | None = None
        observed_pids: dict[str, dict[int, float]] = {"0": {}, "1": {}}
        while not stopped_one:
            now = time.monotonic()
            sample = _sample_resources(launched)
            samples.append(sample)
            for slot, row in sample["instances"].items():
                observed_pids[slot].update({int(pid): float(created)
                                            for pid, created in row["process_birth_times"].items()})
            if not stopped_zero and now >= stop_zero_at:
                control.stop(str(launched[0]["launch_id"]))
                launched[0]["stop_requested_unix_ns"] = time.time_ns()
                _wait_terminal(control, str(launched[0]["launch_id"]), 30)
                stopped_zero = True
                last_slot1_steps = int(_load_json(Path(launched[1]["evidence_path"])).get("steps", 0))
                post_slot0_deadline = time.monotonic() + 10
            if stopped_zero:
                current = _load_json(Path(launched[1]["evidence_path"]))
                if int(current.get("steps", 0)) > last_slot1_steps:
                    slot1_progress_at_stop = True
                if not _alive(int(launched[1]["runner_pid"])):
                    raise AssertionError("slot 1 exited when slot 0 was stopped")
            if post_slot0_deadline is not None and time.monotonic() >= post_slot0_deadline:
                control.stop(str(launched[1]["launch_id"]))
                launched[1]["stop_requested_unix_ns"] = time.time_ns()
                _wait_terminal(control, str(launched[1]["launch_id"]), 30)
                stopped_one = True
                break
            next_sample = stop_zero_at if not stopped_zero else post_slot0_deadline
            if next_sample is not None:
                time.sleep(max(0.0, min(5.0, next_sample - time.monotonic())))
            else:
                time.sleep(min(5.0, max(0.0, stop_zero_at - time.monotonic())))
        for item in launched:
            terminal = control._launches[str(item["launch_id"])]
            item["terminal"] = {key: terminal.get(key) for key in
                                 ("status", "return_code", "final_save_status", "final_save", "failure")}
            if terminal.get("status") != "CANCELLED" or terminal.get("final_save_status") != "captured":
                raise AssertionError(f"launch did not stop cleanly and capture final save: {item['launch_id']}")
            if not Path(str(terminal.get("final_save", ""))).is_dir():
                raise AssertionError(f"final-save directory missing for {item['launch_id']}")
            item["bridge_final"] = _load_json(Path(item["evidence_path"]))
            verify_frame_captures(item["bridge_final"], require_after_action=True)
        if not slot1_progress_at_stop:
            raise AssertionError("slot 1 did not continue responding after slot 0 stopped")
        registry_after = registry_snapshot(smoke_root / "registry-after.reg")
        personal_after = hash_tree(config.personal_save_root)
        baseline_after = hash_tree(config.experimental_save_root)
        data_after = {f"slot{index}": hash_tree(root)
                      for index, root in enumerate(config.instance_data_roots)}
        if personal_before != personal_after:
            manifest["personal_changes"] = tree_changes(personal_before, personal_after)
            raise AssertionError("personal save tree changed; preserve backup and restore manually")
        if registry_before["protected_sha256"] != registry_after["protected_sha256"]:
            raise AssertionError("HKCU\\Software\\Team Cherry changed")
        if baseline_before != baseline_after:
            raise AssertionError("experimental baseline changed")
        verify_root_isolation(data_before, data_after, {"slot0": "slot0", "slot1": "slot1"})
        for item in launched:
            log_path = Path(str(item["player_log"]))
            if not log_path.is_file():
                raise AssertionError(f"Player.log missing for launch {item['launch_id']}")
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            other_launch = str(launched[1 - int(item["slot_id"])]["launch_id"])
            if other_launch in log_text:
                raise AssertionError(f"Player.log for {item['launch_id']} contains other launch ID")
            item["player_log_exists"] = log_path.is_file()
        verify_distinct([item["player_log"] for item in launched], "Player.log paths")
        for item in launched:
            slot = str(item["slot_id"])
            if int(item["runner_pid"]) not in observed_pids[slot]:
                raise AssertionError(f"runner PID {item['runner_pid']} was not observed in its process tree")
            if int(item["game_process"]["pid"]) not in observed_pids[slot]:
                raise AssertionError(f"game PID {item['game_process']['pid']} was not observed in its process tree")
        process_rows = [
            {"pid": pid, "alive": _same_process_alive(pid, created)}
            for pids in observed_pids.values() for pid, created in pids.items()
        ]
        verify_no_orphans(process_rows, {int(row["pid"]) for row in process_rows})
        manifest["owned_process_pids"] = {slot: sorted(pids)
                                          for slot, pids in observed_pids.items()}
        manifest.update({
            "status": "PASSED", "launches": launched,
            "snapshots_after": {"personal": personal_after, "baseline": baseline_after,
                                "data_roots": data_after, "registry": registry_after},
            "resource_summary": _resource_summary(samples),
            "slot1_continued_after_slot0_stop": slot1_progress_at_stop,
        })
    except BaseException as error:  # noqa: BLE001 - stop owned launches on every exit path
        for item in launched:
            launch_id = str(item["launch_id"])
            try:
                if control._launches.get(launch_id, {}).get("status") in {"PREPARING", "RUNNING", "STOP_REQUESTED"}:
                    control.stop(launch_id)
                    _wait_terminal(control, launch_id, 30)
            except (RuntimeError, ValueError, OSError):
                pass
        manifest.update({"status": "FAILED", "failure": f"{type(error).__name__}: {error}",
                         "launches": launched, "resource_summary": _resource_summary(samples)})
        try:
            personal_after = hash_tree(config.personal_save_root)
            if personal_before != personal_after:
                changes = tree_changes(personal_before, personal_after)
                manifest["personal_changes"] = changes
                manifest["restore_guidance"] = f"Personal saves changed: {changes}. Backup: {personal_backup}. Restore manually."
        except OSError:
            pass
    finally:
        if "snapshots_after" not in manifest:
            try:
                registry_after = registry_snapshot(smoke_root / "registry-after.reg")
                personal_after = hash_tree(config.personal_save_root)
                baseline_after = hash_tree(config.experimental_save_root)
                data_after = {f"slot{index}": hash_tree(root)
                              for index, root in enumerate(config.instance_data_roots)}
                manifest["snapshots_after"] = {
                    "personal": personal_after, "baseline": baseline_after,
                    "data_roots": data_after, "registry": registry_after,
                }
                if personal_before != personal_after:
                    changes = tree_changes(personal_before, personal_after)
                    manifest["personal_changes"] = changes
                    manifest["restore_guidance"] = (
                        f"Personal saves changed: {changes}. Backup: {personal_backup}. "
                        "Restore manually; the smoke did not restore them."
                    )
                    manifest["status"] = "FAILED"
                if baseline_before != baseline_after:
                    manifest["baseline_changes"] = tree_changes(baseline_before, baseline_after)
                    manifest["status"] = "FAILED"
                if registry_before["protected_sha256"] != registry_after["protected_sha256"]:
                    manifest["registry_changed"] = True
                    manifest["status"] = "FAILED"
                verify_root_isolation(data_before, data_after,
                                      {"slot0": "slot0", "slot1": "slot1"})
            except (OSError, RuntimeError, ValueError, AssertionError) as error:
                manifest["post_snapshot_error"] = f"{type(error).__name__}: {error}"
                manifest["status"] = "FAILED"
        manifest["finished_unix_ns"] = time.time_ns()
        manifest["manifest_path"] = str(manifest_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_report(manifest, smoke_root / "smoke-manifest.json")
    return manifest


def _write_report(manifest: Mapping[str, Any], manifest_path: Path) -> None:
    report_path = PROJECT_ROOT / "agent-prompts" / "relatorios" / "17_smoke.txt"
    summary = manifest.get("resource_summary", {})
    lines = ["AGENTE 17 — SMOKE REAL COM DUAS INSTÂNCIAS", "=" * 45,
             f"Data: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             f"Status: {manifest.get('status')}", f"Manifesto: {manifest_path}", "",
             "RECURSOS (média / pico)", "Instância | CPU % | RAM MiB | VRAM MiB"]
    for key, label in (("0", "slot 0"), ("1", "slot 1"), ("total", "total")):
        metrics = summary.get(key, {}) if isinstance(summary, Mapping) else {}
        cells = []
        for metric, scale in (("cpu_percent", 1), ("ram_bytes", 1024 * 1024), ("vram_mib", 1)):
            values = metrics.get(metric, {})
            average, peak = values.get("average"), values.get("peak")
            cells.append("n/a" if average is None else f"{average / scale:.1f} / {peak / scale:.1f}")
        lines.append(f"{label} | " + " | ".join(cells))
    failure = manifest.get("failure")
    if failure:
        lines.extend(["", f"Falha: {failure}"])
    if manifest.get("restore_guidance"):
        lines.extend(["", str(manifest["restore_guidance"])])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-open-games", action="store_true",
                        help="explicitly authorize this script to open both game instances")
    args = parser.parse_args()
    if not args.confirm_open_games:
        print("No games opened. Re-run with --confirm-open-games only after reviewing the preflight.")
        return 2
    try:
        manifest = _run_live_smoke(PROJECT_ROOT)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0 if manifest.get("status") == "PASSED" else 1
    except (OSError, RuntimeError, ValueError) as error:
        print(f"SMOKE NOT RUN: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
