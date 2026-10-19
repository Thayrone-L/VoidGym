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
    has_dash: bool = False
    has_walljump: bool = False
    has_double_jump: bool = False
    has_super_dash: bool = False
    has_dream_nail: bool = False
    has_dashmaster: bool = False
    dialog_open: bool = False
    menu_open: bool = False
    at_bench: bool = False
    charm_menu_open: bool = False
    charm_cursor: int = 0
    equipped_charm_mask: int = 0
    charm_notches_used: int = 0
    charm_notches_total: int = 0


_INT_FIELDS = ("mp_charge", "spell_cost", "focus_cost", "fireball_level", "quake_level", "scream_level", "charm_cursor", "equipped_charm_mask", "charm_notches_used", "charm_notches_total")
_BOOL_FIELDS = ("has_nail_art", "has_cyclone", "has_dash_slash", "has_upward_slash", "has_dash", "has_walljump", "has_double_jump", "has_super_dash", "has_dream_nail", "has_dashmaster", "dialog_open", "menu_open", "at_bench", "charm_menu_open")


def parse_action_availability(value: object) -> ActionAvailability | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ProtocolError("FRAME action_availability must be an object")
    expected = set(_INT_FIELDS + _BOOL_FIELDS)
    # Accept v1 bridge frames without the newer progression fields.
    new_fields = {"at_bench", "charm_menu_open", "charm_cursor", "equipped_charm_mask", "charm_notches_used", "charm_notches_total"}
    previous = expected - new_fields
    previous_without_ui = previous - {"dialog_open", "menu_open"}
    legacy = previous_without_ui - {"has_dash", "has_walljump", "has_double_jump", "has_super_dash", "has_dream_nail", "has_dashmaster"}
    if frozenset(value) not in {frozenset(expected), frozenset(previous), frozenset(previous_without_ui), frozenset(legacy)}:
        raise ProtocolError("FRAME action_availability has invalid fields")
    normalized = dict(value)
    for name in expected - set(normalized):
        normalized[name] = 0 if name in _INT_FIELDS else False
    for name in _INT_FIELDS:
        item = normalized[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ProtocolError(f"FRAME action_availability.{name} must be a non-negative integer")
    for name in _BOOL_FIELDS:
        if not isinstance(normalized[name], bool):
            raise ProtocolError(f"FRAME action_availability.{name} must be boolean")
    if normalized["charm_notches_used"] > 12 or normalized["charm_notches_total"] > 12:
        raise ProtocolError("FRAME action_availability charm notch values exceed the game bound")
    if normalized["equipped_charm_mask"] >= (1 << 40):
        raise ProtocolError("FRAME action_availability equipped charm mask exceeds 40 charms")
    return ActionAvailability(**normalized)


def build_action_mask(names: tuple[str, ...], availability: ActionAvailability | None) -> NDArray[np.bool_]:
    mask = np.ones(len(names), dtype=np.bool_)
    if availability is None:
        for index, name in enumerate(names):
            if "DASH" in name or name in {"SUPER_DASH_HOLD", "DREAM_NAIL"} or name.startswith("MENU_"):
                mask[index] = False
            if name == "OPEN_CHARMS":
                mask[index] = False
        return mask
    for index, name in enumerate(names):
        if name == "INTERACT":
            mask[index] = True
        elif name == "OPEN_CHARMS":
            mask[index] = availability.at_bench and not availability.menu_open and not availability.dialog_open
        elif name.startswith("MENU_"):
            mask[index] = availability.charm_menu_open or availability.menu_open or availability.dialog_open
            continue
        elif (availability.menu_open or availability.dialog_open or availability.charm_menu_open) and name not in {"NOOP", "INTERACT"}:
            mask[index] = False
        if name == "FOCUS_HOLD":
            mask[index] = mask[index] and availability.mp_charge >= availability.focus_cost
        elif name == "CAST":
            mask[index] = mask[index] and availability.fireball_level > 0 and availability.mp_charge >= availability.spell_cost
        elif name == "UP+CAST":
            mask[index] = mask[index] and availability.scream_level > 0 and availability.mp_charge >= availability.spell_cost
        elif name == "DOWN+CAST":
            mask[index] = mask[index] and availability.quake_level > 0 and availability.mp_charge >= availability.spell_cost
        elif "NAIL_ART_CHARGE" in name:
            mask[index] = mask[index] and availability.has_nail_art
        elif "DASH" in name and "SUPER_DASH" not in name:
            mask[index] = mask[index] and availability.has_dash and (
                name != "DOWN+DASH" or availability.has_dashmaster
            )
        elif name == "SUPER_DASH_HOLD":
            mask[index] = mask[index] and availability.has_super_dash
        elif name == "DREAM_NAIL":
            mask[index] = mask[index] and availability.has_dream_nail
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
