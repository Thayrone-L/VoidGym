"""Shared heartbeat contract for the v2 learner and HK control plane."""

from __future__ import annotations

import time
from typing import Any


def learner_heartbeat_timestamp(payload: dict[str, Any]) -> int | None:
    """Read the canonical learner beat, with compatibility for older runtimes."""
    for key in ("heartbeat_unix_ns", "updated_unix_ns"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def learner_monitor_state(
    *, status: str, round_id: int, completed_steps: int,
    resources: dict[str, Any], metrics: dict[str, float] | None = None,
    failure: str | None = None, waiting_memory: bool = False,
    last_wait_rollouts_ms: float = 0.0, last_train_ms: float = 0.0,
    last_straggler_wait_ms: float = 0.0,
) -> dict[str, Any]:
    """Build the common learner/heartbeat payload with one timestamp value."""
    now = time.time_ns()
    payload: dict[str, Any] = {
        "status": status,
        "round": int(round_id),
        "completed_timesteps": int(completed_steps),
        "heartbeat_unix_ns": now,
        "updated_unix_ns": now,
        "last_wait_rollouts_ms": float(last_wait_rollouts_ms),
        "last_train_ms": float(last_train_ms),
        "last_straggler_wait_ms": float(last_straggler_wait_ms),
        "waiting_memory": bool(waiting_memory),
        **resources,
    }
    if metrics:
        payload["metrics"] = metrics
    if failure:
        payload["failure"] = failure[:2000]
    return payload
