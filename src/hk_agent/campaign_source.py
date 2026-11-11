from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, replace
from pathlib import Path

from hk_agent.menu_bootstrap import MenuBootstrapPlan, MenuBootstrapStep

SAVE_SOURCES = frozenset({"existing_save", "new_game", "continuous_playthrough"})


@dataclass(frozen=True, slots=True)
class CampaignOrigin:
    source: str = "existing_save"
    slot: int = 1
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.source not in SAVE_SOURCES:
            raise ValueError("save_source must be existing_save, new_game or continuous_playthrough")
        if isinstance(self.slot, bool) or not isinstance(self.slot, int) or not 1 <= self.slot <= 4:
            raise ValueError("save_slot must be an integer in [1, 4]")

    @property
    def save_file_name(self) -> str:
        return f"user{self.slot}.dat"


@dataclass(frozen=True, slots=True)
class ResolvedCampaignBootstrap:
    origin: CampaignOrigin
    base_path: Path
    plan: MenuBootstrapPlan
    canonical_sha256: str

    def provenance(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "save_source": self.origin.source,
            "save_slot": self.origin.slot,
            "save_file_name": self.origin.save_file_name,
            "base_plan_path": str(self.base_path),
            "resolved_plan_id": self.plan.plan_id,
            "resolved_plan_sha256": self.canonical_sha256,
            "resolved_steps": [
                {
                    "button": step.button,
                    "lease_ms": step.lease_ms,
                    "settle_seconds": step.settle_seconds,
                    "label": step.label,
                }
                for step in self.plan.steps
            ],
        }


class CampaignBootstrapSequence:
    """Use the new-game bootstrap once, then resume the selected save slot."""

    def __init__(
        self, initial: MenuBootstrapPlan, subsequent: MenuBootstrapPlan | None = None
    ) -> None:
        self._initial = initial
        self._subsequent = subsequent or initial
        self._calls = 0
        self._lock = threading.Lock()

    def __call__(self) -> MenuBootstrapPlan:
        with self._lock:
            plan = self._initial if self._calls == 0 else self._subsequent
            self._calls += 1
            return plan


def campaign_origin_provenance(
    initial: ResolvedCampaignBootstrap,
    subsequent: ResolvedCampaignBootstrap | None = None,
) -> dict[str, object]:
    record = initial.provenance()
    if subsequent is not None:
        record["subsequent_episode_bootstrap"] = subsequent.provenance()
    return record


def resolve_campaign_bootstrap(
    *,
    continue_plan_path: Path,
    new_game_plan_path: Path,
    origin: CampaignOrigin,
) -> ResolvedCampaignBootstrap:
    base_path = (
        new_game_plan_path if origin.source == "new_game" else continue_plan_path
    ).resolve(strict=True)
    base = MenuBootstrapPlan.load(base_path)
    if origin.slot == 1:
        plan = base
    elif origin.source != "new_game":
        plan = _existing_slot_plan(base, origin.slot)
    else:
        plan = _new_game_slot_plan(base, origin.slot)
    canonical = _canonical_plan(plan)
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ResolvedCampaignBootstrap(origin, base_path, plan, digest)


def validate_save_precondition(experimental_root: Path, origin: CampaignOrigin) -> Path:
    save_path = experimental_root / origin.save_file_name
    exists = save_path.is_file()
    if origin.source in {"existing_save", "continuous_playthrough"} and not exists:
        raise RuntimeError(f"selected existing save slot is empty: {origin.save_file_name}")
    if origin.source == "new_game" and exists:
        raise RuntimeError(
            f"new-game slot is occupied and will not be overwritten: {origin.save_file_name}"
        )
    return save_path


def _existing_slot_plan(base: MenuBootstrapPlan, slot: int) -> MenuBootstrapPlan:
    steps = list(base.steps)
    index = _step_index(steps, "select-slot-one")
    steps[index:index] = tuple(
        MenuBootstrapStep("DOWN", 50, 0.35, f"move-to-slot-{number}")
        for number in range(2, slot + 1)
    )
    steps[index + slot - 1] = replace(steps[index + slot - 1], label=f"select-slot-{slot}")
    return replace(base, plan_id=f"{base.plan_id}-slot-{slot}", steps=tuple(steps))


def _new_game_slot_plan(base: MenuBootstrapPlan, slot: int) -> MenuBootstrapPlan:
    steps = list(base.steps)
    index = _step_index(steps, "select-new-game-slot-1")
    steps[index:index] = tuple(
        MenuBootstrapStep("DOWN", 50, 0.35, f"move-to-empty-slot-{number}")
        for number in range(2, slot + 1)
    )
    steps[index + slot - 1] = replace(
        steps[index + slot - 1], label=f"select-new-game-slot-{slot}"
    )
    return replace(base, plan_id=f"{base.plan_id}-slot-{slot}", steps=tuple(steps))


def _step_index(steps: list[MenuBootstrapStep], label: str) -> int:
    for index, step in enumerate(steps):
        if step.label == label:
            return index
    raise ValueError(f"bootstrap base plan is missing required step: {label}")


def _canonical_plan(plan: MenuBootstrapPlan) -> dict[str, object]:
    readiness = plan.readiness
    return {
        "plan_id": plan.plan_id,
        "initial_wait_seconds": plan.initial_wait_seconds,
        "final_wait_seconds": plan.final_wait_seconds,
        "timeout_seconds": plan.timeout_seconds,
        "minimum_distinct_captures": plan.minimum_distinct_captures,
        "steps": [
            {
                "button": step.button,
                "lease_ms": step.lease_ms,
                "settle_seconds": step.settle_seconds,
                "label": step.label,
            }
            for step in plan.steps
        ],
        "readiness": (
            {
                "verifier_id": readiness.verifier_id,
                "reference_image": str(readiness.reference_image),
                "maximum_mean_absolute_error": readiness.maximum_mean_absolute_error,
            }
            if readiness is not None
            else None
        ),
    }
