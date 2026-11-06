from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hk_agent.menu_bootstrap import (
    MenuBootstrapPlan,
    MenuBootstrapStep,
    ReferenceReadinessSpec,
)

_CHARM_ID = re.compile(r"^gotCharm_([1-9]|[1-3][0-9]|40)$")


@dataclass(frozen=True, slots=True)
class CharmMenuState:
    at_bench: bool
    menu_open: bool
    charm_menu_open: bool
    cursor: int
    equipped: frozenset[str]
    notch_capacity: int


def build_charm_menu_plan(
    *,
    plan_id: str,
    state: CharmMenuState,
    target: Sequence[str],
    acquired_notch_costs: Mapping[str, int],
    step_references: Sequence[Path],
    gameplay_reference: Path,
    maximum_mean_absolute_error: float = 0.0,
    grid_columns: int = 5,
    step_settle_seconds: float = 0.15,
) -> MenuBootstrapPlan:
    """Compile an owned, capacity-safe loadout into a frame-guarded menu macro.

    The macro only runs while the live state says the player is seated at a bench
    with the charm page open. Every key pulse must match its calibrated PNG before
    the next pulse is allowed; an explicit gameplay reference guards the exit.
    """
    if not state.at_bench or not state.menu_open or not state.charm_menu_open:
        raise ValueError("charm macro requires a seated player on the open charm page")
    if not plan_id or len(plan_id) > 128:
        raise ValueError("plan_id must be non-empty and at most 128 characters")
    if not 0 <= state.cursor < 40:
        raise ValueError("charm cursor must be within the 40-slot inventory")
    if not 1 <= grid_columns <= 10:
        raise ValueError("grid_columns must be between 1 and 10")
    if not 0 <= state.notch_capacity <= 12:
        raise ValueError("notch capacity must be between 0 and 12")
    owned: dict[str, int] = {}
    for charm_id, cost in acquired_notch_costs.items():
        if _CHARM_ID.fullmatch(charm_id) is None:
            raise ValueError("acquired charm ids must be known gotCharm_N flags")
        if isinstance(cost, bool) or not isinstance(cost, int) or not 1 <= cost <= 6:
            raise ValueError("charm notch costs must be between 1 and 6")
        owned[charm_id] = cost
    if len(owned) > 40 or not state.equipped.issubset(owned):
        raise ValueError("equipped charms must belong to the acquired inventory")
    desired = tuple(sorted(set(target)))
    if len(desired) != len(target):
        raise ValueError("target loadout cannot contain duplicate charms")
    if any(charm_id not in owned for charm_id in desired):
        raise ValueError("target loadout contains a charm that has not been acquired")
    if sum(owned[charm_id] for charm_id in desired) > state.notch_capacity:
        raise ValueError("target loadout exceeds observed notch capacity")
    if not 0 <= maximum_mean_absolute_error <= 255:
        raise ValueError("maximum image error must be between 0 and 255")
    if not 0 <= step_settle_seconds <= 30:
        raise ValueError("step settle time must be between 0 and 30 seconds")

    actions: list[tuple[str, str]] = []
    cursor = state.cursor
    for charm_id in sorted(state.equipped - set(desired)):
        cursor = _append_charm_toggle(actions, charm_id, cursor, grid_columns)
    for charm_id in sorted(set(desired) - state.equipped):
        cursor = _append_charm_toggle(actions, charm_id, cursor, grid_columns)
    actions.append(("CANCEL", "exit-charm-menu"))
    if len(actions) > 128:
        raise ValueError("charm macro exceeds the 128-step menu safety bound")
    if len(step_references) != len(actions):
        raise ValueError("provide exactly one calibrated frame reference for every macro step")
    if not gameplay_reference.is_file() or any(not path.is_file() for path in step_references):
        raise FileNotFoundError("all macro frame references must exist before execution")
    steps = tuple(
        MenuBootstrapStep(
            button=button,
            lease_ms=50,
            settle_seconds=step_settle_seconds,
            label=label,
            readiness=ReferenceReadinessSpec(
                "reference-rgb-mae-v1", reference.resolve(), maximum_mean_absolute_error
            ),
        )
        for (button, label), reference in zip(actions, step_references, strict=True)
    )
    final = ReferenceReadinessSpec(
        "reference-rgb-mae-v1", gameplay_reference.resolve(), maximum_mean_absolute_error
    )
    planned = 0.5 + sum(step.settle_seconds for step in steps)
    timeout = max(30.0, planned + 15.0)
    return MenuBootstrapPlan(
        plan_id=plan_id,
        steps=steps,
        initial_wait_seconds=0.5,
        final_wait_seconds=0.0,
        timeout_seconds=timeout,
        minimum_distinct_captures=2,
        readiness=final,
    )


def _append_charm_toggle(
    actions: list[tuple[str, str]], charm_id: str, cursor: int, columns: int
) -> int:
    match = _CHARM_ID.fullmatch(charm_id)
    if match is None:
        raise ValueError("invalid acquired charm id")
    target = int(match.group(1)) - 1
    current_row, current_column = divmod(cursor, columns)
    target_row, target_column = divmod(target, columns)
    while current_row < target_row:
        actions.append(("DOWN", f"move-to-{charm_id}-down"))
        current_row += 1
    while current_row > target_row:
        actions.append(("UP", f"move-to-{charm_id}-up"))
        current_row -= 1
    while current_column < target_column:
        actions.append(("RIGHT", f"move-to-{charm_id}-right"))
        current_column += 1
    while current_column > target_column:
        actions.append(("LEFT", f"move-to-{charm_id}-left"))
        current_column -= 1
    actions.append(("SUBMIT", f"toggle-{charm_id}"))
    return target
