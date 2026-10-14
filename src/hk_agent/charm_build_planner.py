from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CHARM_BUILD_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PlannerScope:
    """Identity boundary that prevents evidence leaking between incompatible runs."""

    game_build_hash: str
    save_slot_id: str
    campaign_id: str

    def __post_init__(self) -> None:
        if not all((self.game_build_hash, self.save_slot_id, self.campaign_id)):
            raise ValueError("planner scope fields must be non-empty")


@dataclass(frozen=True)
class PlannedCharmBuild:
    candidate_ids: tuple[str, ...]
    notch_cost: int
    predicted_outcome: float
    exploration_bonus: float
    evidence_runs: int


class CharmBuildPlanner:
    """Bounded empirical planner for visually distinguished charm candidates.

    Candidate identifiers are deliberately opaque.  Callers derive the descriptor,
    notch cost and available capacity from pixels; this class never reads game state,
    a wiki, charm names, or a hand-authored effects table.  Outcome estimates are
    correlations learned from completed runs, not claims about a charm's true effect.
    """

    def __init__(
        self,
        path: Path,
        scope: PlannerScope,
        *,
        match_distance: float = 0.08,
        beam_width: int = 64,
        max_candidates: int = 40,
        max_runs: int = 10_000,
    ) -> None:
        if not 0 < match_distance < 1:
            raise ValueError("match_distance must be in (0, 1)")
        if beam_width < 2 or max_candidates < 1 or max_runs < 1:
            raise ValueError("search and history limits must be positive")
        self.path = path
        self.scope = scope
        self.match_distance = float(match_distance)
        self.beam_width = int(beam_width)
        self.max_candidates = int(max_candidates)
        self.max_runs = int(max_runs)
        self._descriptor_shape: tuple[int, ...] | None = None
        self._candidates: dict[str, dict[str, Any]] = {}
        self._runs: list[dict[str, Any]] = []
        self._next_candidate = 1
        if path.is_file():
            self._load()

    @classmethod
    def contract(cls) -> dict[str, Any]:
        return {
            "schema_version": CHARM_BUILD_SCHEMA_VERSION,
            "inputs": [
                "pixel_derived_charm_descriptor",
                "pixel_observed_notch_cost",
                "pixel_observed_notch_capacity",
                "completed_run_outcome",
            ],
            "candidate_identity": "nearest_visual_descriptor_within_match_distance",
            "effect_learning": "empirical_present_vs_absent_run_outcomes",
            "synergy_learning": "empirical_pairwise_factorial_contrast",
            "uncertainty": "count_based_upper_confidence_exploration",
            "search": "capacity_constrained_bounded_beam",
            "exhaustive_power_set_enumeration": False,
            "menu_integration": "planner_only_not_implemented",
            "information_boundary": "pixel_only",
            "forbidden_inputs": [
                "PlayerData",
                "memory_reading",
                "wiki_or_external_charm_database",
                "hand_authored_charm_effects",
            ],
            "persistence_scope": ["game_build_hash", "save_slot_id", "campaign_id"],
        }

    def observe_candidate(
        self, descriptor: Sequence[float], *, notch_cost: int
    ) -> str:
        if not isinstance(notch_cost, int) or isinstance(notch_cost, bool) or notch_cost < 0:
            raise ValueError("notch_cost must be a non-negative integer")
        vector = self._validate_descriptor(descriptor)
        nearest: str | None = None
        nearest_distance = math.inf
        for candidate_id, record in self._candidates.items():
            prototype = record["descriptor"]
            distance = sum(abs(a - b) for a, b in zip(prototype, vector, strict=True)) / len(vector)
            if distance < nearest_distance:
                nearest, nearest_distance = candidate_id, distance
        if nearest is not None and nearest_distance <= self.match_distance:
            record = self._candidates[nearest]
            if record["notch_cost"] != notch_cost:
                raise ValueError("visual candidate was observed with a conflicting notch cost")
            record["observations"] += 1
            self._persist()
            return nearest
        if len(self._candidates) >= self.max_candidates:
            raise ValueError("candidate limit reached; refusing unbounded planner growth")
        candidate_id = f"visual-charm-{self._next_candidate:04d}"
        self._next_candidate += 1
        self._candidates[candidate_id] = {
            "descriptor": vector,
            "notch_cost": notch_cost,
            "observations": 1,
        }
        self._persist()
        return candidate_id

    def record_run(
        self,
        candidate_ids: Iterable[str],
        *,
        outcome: float,
        context: Mapping[str, float] | None = None,
    ) -> None:
        if not math.isfinite(outcome):
            raise ValueError("run outcome must be finite")
        build = tuple(sorted(set(candidate_ids)))
        unknown = set(build).difference(self._candidates)
        if unknown:
            raise KeyError(f"unknown visual charm candidates: {sorted(unknown)}")
        safe_context: dict[str, float] = {}
        for key, value in (context or {}).items():
            numeric = float(value)
            if not isinstance(key, str) or not key or not math.isfinite(numeric):
                raise ValueError("run context must contain named finite metrics")
            safe_context[key] = numeric
        self._runs.append(
            {"candidate_ids": list(build), "outcome": float(outcome), "context": safe_context}
        )
        if len(self._runs) > self.max_runs:
            del self._runs[: len(self._runs) - self.max_runs]
        self._persist()

    def estimate_effect(self, candidate_id: str) -> dict[str, float | int]:
        self._require_candidate(candidate_id)
        present = [r["outcome"] for r in self._runs if candidate_id in r["candidate_ids"]]
        absent = [r["outcome"] for r in self._runs if candidate_id not in r["candidate_ids"]]
        effect = _mean(present) - _mean(absent) if present and absent else 0.0
        uncertainty = _contrast_uncertainty(present, absent)
        return {
            "effect": effect,
            "uncertainty": uncertainty,
            "present_runs": len(present),
            "absent_runs": len(absent),
        }

    def estimate_synergy(self, first: str, second: str) -> dict[str, float | int]:
        self._require_candidate(first)
        self._require_candidate(second)
        if first == second:
            raise ValueError("synergy requires two distinct candidates")
        cells: dict[tuple[bool, bool], list[float]] = {
            (False, False): [], (True, False): [], (False, True): [], (True, True): []
        }
        for run in self._runs:
            members = run["candidate_ids"]
            cells[(first in members, second in members)].append(run["outcome"])
        complete = all(cells.values())
        synergy = 0.0
        if complete:
            synergy = (
                _mean(cells[(True, True)])
                - _mean(cells[(True, False)])
                - _mean(cells[(False, True)])
                + _mean(cells[(False, False)])
            )
        return {
            "synergy": synergy,
            "uncertainty": sum(1 / math.sqrt(max(1, len(values))) for values in cells.values()),
            "joint_runs": len(cells[(True, True)]),
            "complete_contrast": int(complete),
        }

    def suggest_build(
        self,
        *,
        notch_capacity: int,
        available_candidate_ids: Iterable[str] | None = None,
        exploration_weight: float = 1.0,
    ) -> PlannedCharmBuild:
        if not isinstance(notch_capacity, int) or notch_capacity < 0:
            raise ValueError("notch_capacity must be a non-negative integer")
        if not math.isfinite(exploration_weight) or exploration_weight < 0:
            raise ValueError("exploration_weight must be finite and non-negative")
        available = (
            sorted(self._candidates)
            if available_candidate_ids is None
            else sorted(set(available_candidate_ids))
        )
        unknown = set(available).difference(self._candidates)
        if unknown:
            raise KeyError(f"unknown visual charm candidates: {sorted(unknown)}")
        if len(available) > self.max_candidates:
            raise ValueError("available candidate count exceeds configured search bound")

        # Keep only the best bounded partial builds at every inclusion decision.
        beam: list[tuple[str, ...]] = [()]
        for candidate_id in available:
            cost = self._candidates[candidate_id]["notch_cost"]
            expanded = list(beam)
            expanded.extend(
                (*build, candidate_id)
                for build in beam
                if self._build_cost((*build, candidate_id)) <= notch_capacity and cost <= notch_capacity
            )
            beam = sorted(
                set(expanded),
                key=lambda build: (-self._ucb_score(build, exploration_weight), build),
            )[: self.beam_width]

        best = min(
            beam,
            key=lambda build: (-self._ucb_score(build, exploration_weight), build),
        )
        predicted, bonus = self._score_parts(best, exploration_weight)
        evidence_runs = sum(
            1 for run in self._runs if tuple(run["candidate_ids"]) == best
        )
        return PlannedCharmBuild(
            candidate_ids=best,
            notch_cost=self._build_cost(best),
            predicted_outcome=predicted,
            exploration_bonus=bonus,
            evidence_runs=evidence_runs,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": CHARM_BUILD_SCHEMA_VERSION,
            "contract": self.contract(),
            "scope": asdict(self.scope),
            "match_distance": self.match_distance,
            "beam_width": self.beam_width,
            "max_candidates": self.max_candidates,
            "max_runs": self.max_runs,
            "descriptor_shape": list(self._descriptor_shape) if self._descriptor_shape else None,
            "next_candidate": self._next_candidate,
            "candidates": self._candidates,
            "runs": self._runs,
        }

    def _score_parts(self, build: tuple[str, ...], exploration_weight: float) -> tuple[float, float]:
        baseline = _mean([run["outcome"] for run in self._runs])
        predicted = baseline
        uncertainty_sq = 0.0
        for candidate_id in build:
            estimate = self.estimate_effect(candidate_id)
            predicted += float(estimate["effect"])
            uncertainty_sq += float(estimate["uncertainty"]) ** 2
        for index, first in enumerate(build):
            for second in build[index + 1 :]:
                estimate = self.estimate_synergy(first, second)
                predicted += float(estimate["synergy"])
                uncertainty_sq += float(estimate["uncertainty"]) ** 2
        bonus = exploration_weight * math.sqrt(uncertainty_sq)
        return predicted, bonus

    def _ucb_score(self, build: tuple[str, ...], exploration_weight: float) -> float:
        predicted, bonus = self._score_parts(build, exploration_weight)
        return predicted + bonus

    def _build_cost(self, build: tuple[str, ...]) -> int:
        return sum(self._candidates[item]["notch_cost"] for item in build)

    def _require_candidate(self, candidate_id: str) -> None:
        if candidate_id not in self._candidates:
            raise KeyError(f"unknown visual charm candidate: {candidate_id}")

    def _validate_descriptor(self, descriptor: Sequence[float]) -> list[float]:
        vector = [float(value) for value in descriptor]
        if not vector or not all(math.isfinite(value) and 0 <= value <= 1 for value in vector):
            raise ValueError("visual descriptor must contain finite values in [0, 1]")
        shape = (len(vector),)
        if self._descriptor_shape is None:
            self._descriptor_shape = shape
        elif shape != self._descriptor_shape:
            raise ValueError("visual descriptor shape differs from persisted planner")
        return vector

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(self.snapshot(), indent=2, allow_nan=False), "utf-8")
        os.replace(temporary, self.path)

    def _load(self) -> None:
        payload = json.loads(self.path.read_text("utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != CHARM_BUILD_SCHEMA_VERSION:
            raise ValueError("unsupported charm build planner schema version")
        if payload.get("scope") != asdict(self.scope):
            raise ValueError("persisted charm evidence belongs to a different scope")
        for field in ("match_distance", "beam_width", "max_candidates", "max_runs"):
            if payload.get(field) != getattr(self, field):
                raise ValueError(f"persisted planner {field} differs from configured value")
        raw_shape = payload.get("descriptor_shape")
        if raw_shape is not None and (
            not isinstance(raw_shape, list) or len(raw_shape) != 1 or not isinstance(raw_shape[0], int)
        ):
            raise ValueError("invalid persisted descriptor shape")
        self._descriptor_shape = tuple(raw_shape) if raw_shape is not None else None
        candidates, runs = payload.get("candidates"), payload.get("runs")
        if not isinstance(candidates, dict) or not isinstance(runs, list):
            raise TypeError("invalid persisted charm planner payload")
        self._candidates = candidates
        self._runs = runs
        self._next_candidate = int(payload.get("next_candidate", 0))
        # Validate loaded data through the same public invariants without rewriting it.
        if self._next_candidate < 1 or len(candidates) > self.max_candidates or len(runs) > self.max_runs:
            raise ValueError("persisted planner exceeds configured bounds")
        for candidate_id, record in candidates.items():
            if not isinstance(candidate_id, str) or not isinstance(record, dict):
                raise TypeError("invalid persisted charm candidate")
            descriptor = record.get("descriptor")
            cost, observations = record.get("notch_cost"), record.get("observations")
            if (
                not isinstance(descriptor, list)
                or self._descriptor_shape != (len(descriptor),)
                or not all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1 for value in descriptor)
                or not isinstance(cost, int) or cost < 0
                or not isinstance(observations, int) or observations < 1
            ):
                raise ValueError("invalid persisted charm candidate fields")
        for run in runs:
            if not isinstance(run, dict):
                raise TypeError("invalid persisted run evidence")
            members, outcome, context = run.get("candidate_ids"), run.get("outcome"), run.get("context")
            if (
                not isinstance(members, list)
                or members != sorted(set(members))
                or any(member not in candidates for member in members)
                or not isinstance(outcome, (int, float)) or not math.isfinite(outcome)
                or not isinstance(context, dict)
                or any(not isinstance(k, str) or not isinstance(v, (int, float)) or not math.isfinite(v) for k, v in context.items())
            ):
                raise ValueError("invalid persisted run evidence fields")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _contrast_uncertainty(present: Sequence[float], absent: Sequence[float]) -> float:
    # A finite prior keeps unseen/one-sided candidates explorable.  As both cells
    # accumulate evidence the bonus shrinks without pretending the estimate is exact.
    return math.sqrt(1 / max(1, len(present)) + 1 / max(1, len(absent)))
