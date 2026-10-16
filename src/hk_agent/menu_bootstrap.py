from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from hk_agent.protocol import ProtocolError
from hk_agent.supervisor import AcceptedBridge, CapturedFrame, SupervisorServer

MENU_BUTTONS = frozenset({"UP", "DOWN", "LEFT", "RIGHT", "SUBMIT", "CANCEL"})


class MenuBootstrapError(RuntimeError):
    """The operational menu bootstrap could not prove its readiness contract."""


@dataclass(frozen=True, slots=True)
class ReadinessVerdict:
    ready: bool
    verifier_id: str
    detail: str
    metrics: dict[str, float | int | str]


ReadinessVerifier = Callable[[CapturedFrame], ReadinessVerdict]


@dataclass(frozen=True, slots=True)
class ReferenceReadinessSpec:
    verifier_id: str
    reference_image: Path
    maximum_mean_absolute_error: float


@dataclass(frozen=True, slots=True)
class MenuBootstrapStep:
    button: str
    lease_ms: int = 50
    settle_seconds: float = 2.0
    label: str = "menu-step"

    @classmethod
    def from_mapping(cls, value: object, index: int) -> MenuBootstrapStep:
        if not isinstance(value, dict):
            raise TypeError(f"steps[{index}] must be an object")
        button = value.get("button")
        lease_ms = value.get("lease_ms", 50)
        settle = value.get("settle_seconds", 2.0)
        label = value.get("label", f"step-{index}-{str(button).lower()}")
        if not isinstance(button, str) or button not in MENU_BUTTONS:
            raise ValueError(f"steps[{index}].button is not an allowed menu button")
        if isinstance(lease_ms, bool) or not isinstance(lease_ms, int) or not 1 <= lease_ms <= 100:
            raise ValueError(f"steps[{index}].lease_ms must be between 1 and 100")
        if isinstance(settle, bool) or not isinstance(settle, (int, float)) or not 0 <= settle <= 30:
            raise ValueError(f"steps[{index}].settle_seconds must be between 0 and 30")
        if not isinstance(label, str) or not label or len(label) > 64:
            raise ValueError(f"steps[{index}].label must be a short non-empty string")
        return cls(button=button, lease_ms=lease_ms, settle_seconds=float(settle), label=label)


@dataclass(frozen=True, slots=True)
class MenuBootstrapPlan:
    plan_id: str
    steps: tuple[MenuBootstrapStep, ...]
    initial_wait_seconds: float = 5.0
    final_wait_seconds: float = 5.0
    timeout_seconds: float = 180.0
    minimum_distinct_captures: int = 2
    readiness: ReferenceReadinessSpec | None = None

    @classmethod
    def load(cls, path: Path) -> MenuBootstrapPlan:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid menu bootstrap plan: {path}") from error
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("menu bootstrap plan schema_version must be 1")
        plan_id = value.get("plan_id")
        raw_steps = value.get("steps")
        if not isinstance(plan_id, str) or not plan_id or len(plan_id) > 128:
            raise ValueError("plan_id must be a short non-empty string")
        if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= 16:
            raise ValueError("steps must contain between 1 and 16 menu actions")
        steps = tuple(MenuBootstrapStep.from_mapping(item, i) for i, item in enumerate(raw_steps))
        initial = _bounded_number(value.get("initial_wait_seconds", 5.0), "initial_wait_seconds", 30)
        final = _bounded_number(value.get("final_wait_seconds", 5.0), "final_wait_seconds", 180)
        timeout = _bounded_number(value.get("timeout_seconds", 180.0), "timeout_seconds", 300, positive=True)
        distinct = value.get("minimum_distinct_captures", 2)
        if isinstance(distinct, bool) or not isinstance(distinct, int) or not 2 <= distinct <= len(steps) + 2:
            raise ValueError("minimum_distinct_captures is outside the capture budget")
        planned_seconds = initial + final + sum(step.settle_seconds for step in steps)
        if planned_seconds >= timeout:
            raise ValueError("planned waits must leave time for commands and captures")
        raw_readiness = value.get("readiness")
        if not isinstance(raw_readiness, dict):
            raise TypeError("an explicit versioned readiness verifier is required")
        verifier_id = raw_readiness.get("verifier_id")
        reference_value = raw_readiness.get("reference_image")
        maximum_error = raw_readiness.get("maximum_mean_absolute_error")
        if verifier_id != "reference-rgb-mae-v1":
            raise ValueError("unsupported readiness verifier")
        if not isinstance(reference_value, str) or not reference_value:
            raise ValueError("readiness.reference_image must be a path")
        reference = (path.parent / reference_value).resolve(strict=True)
        if not reference.is_file():
            raise ValueError("readiness reference image is not a file")
        if (
            isinstance(maximum_error, bool)
            or not isinstance(maximum_error, (int, float))
            or not 0 <= float(maximum_error) <= 255
        ):
            raise ValueError("maximum_mean_absolute_error must be between 0 and 255")
        readiness = ReferenceReadinessSpec(verifier_id, reference, float(maximum_error))
        return cls(plan_id, steps, initial, final, timeout, distinct, readiness)


@dataclass(slots=True)
class MenuBootstrapReport:
    schema_version: int = 1
    plan_id: str = ""
    episode_id: str = ""
    status: str = "STARTING"
    started_monotonic_ns: int = 0
    finished_monotonic_ns: int | None = None
    steps_completed: int = 0
    captures: list[dict[str, object]] = field(default_factory=list)
    events: list[dict[str, object]] = field(default_factory=list)
    failure: str | None = None
    readiness: dict[str, object] | None = None


def _bounded_number(value: object, name: str, maximum: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if result < 0 or result > maximum or (positive and result == 0):
        raise ValueError(f"{name} is outside its safe range")
    return result


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def build_readiness_verifier(plan: MenuBootstrapPlan) -> ReadinessVerifier:
    spec = plan.readiness
    if spec is None:
        raise ValueError("menu bootstrap plan has no explicit readiness verifier")
    if spec.verifier_id != "reference-rgb-mae-v1":
        raise ValueError(f"unsupported readiness verifier: {spec.verifier_id}")
    reference_bytes = spec.reference_image.read_bytes()

    def verify(frame: CapturedFrame) -> ReadinessVerdict:
        # Imported lazily to keep the protocol types independent and avoid a
        # module cycle while reusing the environment's strict PNG decoder.
        from hk_agent.supervisor_backend import decode_rgb8_png

        reference = decode_rgb8_png(reference_bytes)
        if (frame.width, frame.height) != (reference.shape[1], reference.shape[0]):
            raise ProtocolError(
                f"menu frame is {frame.width}x{frame.height} but the readiness reference is "
                f"{reference.shape[1]}x{reference.shape[0]}; the game window client area "
                "changed size (check window placement and -screen-width/-height)"
            )
        candidate = decode_rgb8_png(
            frame.data,
            expected_width=reference.shape[1],
            expected_height=reference.shape[0],
        )
        mean_error = float(
            np.mean(np.abs(candidate.astype(np.int16) - reference.astype(np.int16)))
        )
        ready = mean_error <= spec.maximum_mean_absolute_error
        return ReadinessVerdict(
            ready=ready,
            verifier_id=spec.verifier_id,
            detail=(
                "candidate matches the calibrated gameplay reference"
                if ready
                else "candidate does not match the calibrated gameplay reference"
            ),
            metrics={
                "mean_absolute_error": mean_error,
                "maximum_mean_absolute_error": spec.maximum_mean_absolute_error,
                "reference_image": str(spec.reference_image),
            },
        )

    return verify


async def run_menu_bootstrap(
    plan: MenuBootstrapPlan,
    *,
    supervisor: SupervisorServer,
    bridge: AcceptedBridge,
    episode_id: str,
    artifact_root: Path,
    command_timeout_seconds: float,
    readiness_verifier: ReadinessVerifier | None,
) -> MenuBootstrapReport:
    """Run bounded operational menu actions before BEGIN_EPISODE.

    No action from this function can enter an experience log: the episode has
    not begun, and only APPLY_MENU_ACTION is emitted.
    """

    if readiness_verifier is None:
        raise MenuBootstrapError("an explicit versioned readiness verifier is required")
    output_root = artifact_root / episode_id / "menu-bootstrap"
    report_path = output_root / "report.json"
    report = MenuBootstrapReport(
        plan_id=plan.plan_id,
        episode_id=episode_id,
        started_monotonic_ns=time.monotonic_ns(),
    )

    def persist() -> None:
        _write_json_atomic(report_path, asdict(report))

    async def capture(index: int, label: str) -> CapturedFrame:
        frame = await supervisor.capture_once(timeout=command_timeout_seconds, bridge=bridge)
        safe_label = "".join(char if char.isalnum() or char in "-_" else "-" for char in label)
        path = output_root / f"{index:03d}-{safe_label}.png"
        _write_bytes_atomic(path, frame.data)
        report.captures.append(
            {
                "index": index,
                "label": label,
                "path": str(path),
                "frame_id": frame.frame_id,
                "capture_monotonic_ns": frame.capture_monotonic_ns,
                "width": frame.width,
                "height": frame.height,
                "sha256": hashlib.sha256(frame.data).hexdigest(),
                "scene_transition": frame.scene_transition,
                "duplicate_of": frame.duplicate_of,
            }
        )
        persist()
        return frame

    async def release_all() -> None:
        response = await supervisor.send_command(
            "RELEASE_ALL", {}, timeout=command_timeout_seconds, bridge=bridge
        )
        if response.message_type != "ACK":
            raise MenuBootstrapError("RELEASE_ALL was not acknowledged")

    persist()
    try:
        async with asyncio.timeout(plan.timeout_seconds):
            await asyncio.sleep(plan.initial_wait_seconds)
            await capture(0, "before")
            for index, step in enumerate(plan.steps, start=1):
                report.status = "ACTING"
                report.events.append(
                    {
                        "index": index,
                        "label": step.label,
                        "button": step.button,
                        "lease_ms": step.lease_ms,
                        "started_monotonic_ns": time.monotonic_ns(),
                    }
                )
                persist()
                response = None
                try:
                    response = await supervisor.send_command(
                        "APPLY_MENU_ACTION",
                        {"buttons": [step.button], "lease_ms": step.lease_ms},
                        timeout=command_timeout_seconds,
                        bridge=bridge,
                    )
                    if response.message_type == "ERROR":
                        raise MenuBootstrapError(
                            "menu action rejected: "
                            f"{response.payload.get('error_code', 'unknown')}"
                        )
                    if response.message_type != "ACK":
                        raise MenuBootstrapError("menu action was not acknowledged")
                    if response.payload.get("buttons") != [step.button] or response.payload.get(
                        "lease_ms"
                    ) != step.lease_ms:
                        raise MenuBootstrapError("menu action acknowledgement mismatch")
                    await asyncio.sleep(step.settle_seconds)
                finally:
                    await release_all()
                await capture(index, step.label)
                report.steps_completed = index
                persist()
            if plan.final_wait_seconds:
                await asyncio.sleep(plan.final_wait_seconds)
            final_frame = await capture(len(plan.steps) + 1, "ready-candidate")
            await release_all()
            distinct = len({item["sha256"] for item in report.captures})
            if distinct < plan.minimum_distinct_captures:
                raise MenuBootstrapError(
                    f"readiness evidence has only {distinct} distinct captures; "
                    f"expected {plan.minimum_distinct_captures}"
                )
            verdict = readiness_verifier(final_frame)
            report.readiness = {
                "ready": verdict.ready,
                "verifier_id": verdict.verifier_id,
                "detail": verdict.detail,
                "metrics": verdict.metrics,
            }
            persist()
            if not verdict.ready:
                raise MenuBootstrapError(
                    f"readiness verifier {verdict.verifier_id} rejected the final frame: "
                    f"{verdict.detail}"
                )
            report.status = "READY"
    except BaseException as error:
        try:
            await release_all()
        except Exception as release_error:  # noqa: BLE001
            error.add_note(f"final RELEASE_ALL failed: {release_error}")
        report.status = "FAILED"
        report.failure = f"{type(error).__name__}: {error}"
        report.finished_monotonic_ns = time.monotonic_ns()
        persist()
        if isinstance(error, asyncio.TimeoutError):
            raise MenuBootstrapError("menu bootstrap exceeded its deadline") from error
        raise
    report.finished_monotonic_ns = time.monotonic_ns()
    persist()
    return report
