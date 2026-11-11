from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from run_scene_change_probe import build_task

from hk_agent.campaign_source import CampaignOrigin, resolve_campaign_bootstrap
from hk_agent.config import load_config
from hk_agent.scene_change_task import SceneChangeTaskConfig
from hk_agent.temporal_actions import TEMPORAL_COMBAT_ACTIONS


def _telemetry(info: dict[str, Any]) -> dict[str, Any]:
    value = info.get("operator_telemetry")
    return dict(value) if isinstance(value, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure in-game jump height using an isolated game slot")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.training.local.toml")
    parser.add_argument("--slot", type=int, choices=(0, 1), default=0)
    parser.add_argument("--save-slot", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "artifacts" / "scripted-validation" / "70" / "jump-height")
    parser.add_argument("--bootstrap-plan", type=Path, default=PROJECT_ROOT / "configs" / "menu-bootstrap" / "continue-kings-pass-v1.json")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"will_launch_game": False, "slot": args.slot, "cases_ms": [50, 160, 300, 450, 600]}, indent=2))
        return 0

    config = load_config(args.config.resolve(strict=True), allow_training=True)
    game_install = config.install_roots[args.slot]
    data_root = config.instance_data_roots[args.slot]
    plan = resolve_campaign_bootstrap(
        continue_plan_path=args.bootstrap_plan,
        new_game_plan_path=PROJECT_ROOT / "configs" / "menu-bootstrap" / "new-game-kings-pass-v1.json",
        origin=CampaignOrigin("existing_save", args.save_slot),
    ).plan
    run_root = args.output_root.resolve(strict=False) / f"run-{time.strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(3)}"
    run_root.mkdir(parents=True, exist_ok=False)
    task_args = Namespace(
        game_install=game_install, instance_data_root=data_root,
        window_width=960, window_height=540, log_file=run_root / "Player.log",
        experimental_save_root=None, output_root=run_root,
        save_source="existing_save", save_slot=args.save_slot, lease_ms=80,
        height=338, width=600, frame_stack=4,
        route_memory=run_root / "route-memory.json",
    )
    os.environ["HK_AGENT_SLOT_ID"] = str(args.slot)
    task = None
    owner = None
    measurements: list[dict[str, Any]] = []
    started = time.time()
    try:
        task, owner = build_task(
            project_config=config, bootstrap_plan=plan,
            bootstrap_plan_provider=None, output_root=run_root,
            task_config=SceneChangeTaskConfig(horizon_steps=20_000, time_cost=0.0),
            args=task_args,
        )
        _observation, reset_info = task.environment.reset(seed=70_000 + args.slot)
        initial = _telemetry(reset_info)
        scene = initial.get("scene_name")
        capture = getattr(task.environment.backend, "capture_png", None)
        if callable(capture):
            (run_root / "initial.png").write_bytes(capture())

        def step(action_name: str, lease_ms: int) -> dict[str, Any]:
            action = next(item for item in TEMPORAL_COMBAT_ACTIONS if item.name == action_name)
            _obs, _reward, terminated, truncated, info = task.environment.step(
                action.raw_action_index, lease_ms=lease_ms,
            )
            telemetry = _telemetry(info)
            if terminated or truncated:
                raise RuntimeError(f"game ended during probe: {info.get('termination_reason')}")
            if telemetry.get("scene_name") != scene:
                raise RuntimeError(f"scene changed during probe: {telemetry.get('scene_name')!r}")
            return telemetry

        def settle() -> float:
            recent: list[float] = []
            for _ in range(60):
                telemetry = step("NOOP", 80)
                value = telemetry.get("y")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    recent.append(float(value))
                    recent = recent[-4:]
                    if len(recent) == 4 and max(recent) - min(recent) <= 0.35:
                        return sum(recent) / len(recent)
            raise RuntimeError("hero did not settle on a stable surface within 4.8 seconds")

        def measure(label: str, hold_ms: int, repeated: bool = False) -> dict[str, Any]:
            baseline = settle()
            button_steps = 2 if repeated else 1
            readings: list[dict[str, Any]] = []
            for _ in range(button_steps):
                telemetry = step("JUMP_HOLD" if hold_ms > 50 else "JUMP_TAP", hold_ms)
                readings.append({"phase": "held", "y": telemetry.get("y"), "timestamp_unix": time.time()})
            for _ in range(40):
                telemetry = step("NOOP", 50)
                readings.append({"phase": "released", "y": telemetry.get("y"), "timestamp_unix": time.time()})
            vertical_values = [float(row["y"]) for row in readings if isinstance(row.get("y"), (int, float))]
            if not vertical_values:
                raise RuntimeError("bridge did not return hero_y telemetry")
            first_delta = vertical_values[0] - baseline
            sign = 1.0 if first_delta >= 0 else -1.0
            peak = max((sign * (value - baseline) for value in vertical_values), default=0.0)
            return {
                "case": label, "hold_ms": hold_ms, "repeated_steps": button_steps,
                "baseline_y": baseline, "peak_y": baseline + sign * peak,
                "height_world_units": max(0.0, peak), "scene_name": scene,
                "samples": readings,
            }

        for name, duration, repeated in (
            ("jump_tap", 50, False), ("jump_hold_current", 160, False),
            ("jump_hold_repeated", 160, True), ("jump_hold_300", 300, False),
            ("jump_hold_450", 450, False), ("jump_hold_600", 600, False),
        ):
            measurements.append(measure(name, duration, repeated))
        result = {
            "schema_version": 1, "status": "COMPLETED", "slot": args.slot,
            "scene_name": scene, "started_unix": started,
            "duration_seconds": time.time() - started, "measurements": measurements,
            "limitations": ["height measured from post-release telemetry samples; target ledge geometry must be reviewed from the captured frame"],
        }
        if callable(capture):
            (run_root / "final.png").write_bytes(capture())
    except BaseException as error:
        result = {"schema_version": 1, "status": "FAILED", "slot": args.slot,
                  "started_unix": started, "failure": f"{type(error).__name__}: {error}",
                  "measurements": measurements}
        raise
    finally:
        if task is not None:
            task.close()
        if owner is not None:
            owner.close()
        (run_root / "jump-height.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"run_root": str(run_root), "result": result}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
