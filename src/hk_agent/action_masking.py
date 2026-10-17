from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hk_agent.protocol import ProtocolError


@dataclass(frozen=True, slots=True)
class ActionAvailability:
    mp_charge: int
    spell_cost: int
    focus_cost: int
    fireball_level: int
    quake_level: int
    scream_level: int
    has_nail_art: bool
    has_cyclone: bool
    has_dash_slash: bool
    has_upward_slash: bool


_INT_FIELDS = ("mp_charge", "spell_cost", "focus_cost", "fireball_level", "quake_level", "scream_level")
_BOOL_FIELDS = ("has_nail_art", "has_cyclone", "has_dash_slash", "has_upward_slash")


def parse_action_availability(value: object) -> ActionAvailability | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ProtocolError("FRAME action_availability must be an object")
    expected = set(_INT_FIELDS + _BOOL_FIELDS)
    if set(value) != expected:
        raise ProtocolError("FRAME action_availability has invalid fields")
    for name in _INT_FIELDS:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ProtocolError(f"FRAME action_availability.{name} must be a non-negative integer")
    for name in _BOOL_FIELDS:
        if not isinstance(value[name], bool):
            raise ProtocolError(f"FRAME action_availability.{name} must be boolean")
    return ActionAvailability(**value)


def build_action_mask(names: tuple[str, ...], availability: ActionAvailability | None) -> NDArray[np.bool_]:
    mask = np.ones(len(names), dtype=np.bool_)
    if availability is None:
        return mask
    for index, name in enumerate(names):
        if name == "FOCUS_HOLD":
            mask[index] = availability.mp_charge >= availability.focus_cost
        elif name == "CAST":
            mask[index] = availability.fireball_level > 0 and availability.mp_charge >= availability.spell_cost
        elif name == "UP+CAST":
            mask[index] = availability.scream_level > 0 and availability.mp_charge >= availability.spell_cost
        elif name == "DOWN+CAST":
            mask[index] = availability.quake_level > 0 and availability.mp_charge >= availability.spell_cost
        elif "NAIL_ART_CHARGE" in name:
            mask[index] = availability.has_nail_art
    return mask


def availability_dict(value: ActionAvailability | None) -> dict[str, Any] | None:
    return asdict(value) if value is not None else None


@dataclass(frozen=True, slots=True)
class ActionMaskDecision:
    requested_index: int
    executed_index: int
    mask: NDArray[np.bool_]
    substituted: bool
    availability: ActionAvailability | None


def decide_masked_action(
    requested_index: int,
    action_names: tuple[str, ...],
    availability: ActionAvailability | None,
) -> ActionMaskDecision:
    mask = build_action_mask(action_names, availability)
    if not 0 <= requested_index < len(action_names):
        raise ValueError("requested action index is outside the task action space")
    executed = requested_index if mask[requested_index] else action_names.index("NOOP")
    return ActionMaskDecision(
        requested_index=requested_index,
        executed_index=executed,
        mask=mask,
        substituted=executed != requested_index,
        availability=availability,
    )


def action_mask_audit(decision: ActionMaskDecision, unavailable_count: int) -> dict[str, Any]:
    return {
        "action_mask": np.flatnonzero(~decision.mask).tolist(),
        "action_availability": availability_dict(decision.availability),
        "masked_action_substituted": decision.substituted,
        "requested_task_action": decision.requested_index,
        "executed_task_action": decision.executed_index,
        "mask_unavailable": decision.availability is None,
        "mask_unavailable_count": unavailable_count,
    }
