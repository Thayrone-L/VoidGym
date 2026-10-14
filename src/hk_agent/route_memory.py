from __future__ import annotations

import json
import os
from collections import deque
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROUTE_MEMORY_SCHEMA_VERSION = 1


class TopologicalRouteMemory:
    """Persistent graph of visually recognised landmarks and observed traversals.

    The memory consumes only the same pixel descriptors used by the conservative
    scene-change detector.  It is deliberately an external planner artifact: it
    neither changes rewards nor policy observations, so PPO remains Markov with
    respect to its declared observation contract.  Repeated visits add evidence;
    they are never rejected or penalised.
    """

    def __init__(
        self,
        path: Path,
        *,
        scope: Mapping[str, str | int],
        match_distance: float = 0.06,
    ) -> None:
        if not 0 < match_distance < 1:
            raise ValueError("match_distance must be in (0, 1)")
        self.path = path
        self.scope = self._validated_scope(scope)
        self.match_distance = float(match_distance)
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: dict[tuple[str, str], dict[str, Any]] = {}
        self._descriptor_shape: tuple[int, ...] | None = None
        self._next_node = 1
        self._current_node: str | None = None
        self._revision = 0
        if path.is_file():
            self._load()

    @classmethod
    def contract(cls) -> dict[str, Any]:
        return {
            "schema_version": ROUTE_MEMORY_SCHEMA_VERSION,
            "inputs": [
                "pixel_scene_descriptor",
                "confirmed_stable_visual_regime_change",
            ],
            "node_identity": "nearest_visual_landmark_within_match_distance",
            "edges": "directed_observed_traversals",
            "revisits": "counted_and_allowed",
            "policy_integration": "external_planner_only",
            "reward_integration": "none",
            "edge_payload": "shortest_observed_temporal_action_trace",
            "confidence": "observed_traversal_count_capped_at_three",
            "single_writer": "optimistic_revision_check",
            "privileged_state": [],
        }

    @property
    def current_node(self) -> str | None:
        return self._current_node

    def configuration(self) -> dict[str, Any]:
        return {
            **self.contract(),
            "match_distance": self.match_distance,
            "scope": dict(self.scope),
        }

    def reset_episode(self) -> None:
        self._current_node = None

    def observe_start(self, descriptor: NDArray[np.floating[Any]]) -> dict[str, Any]:
        node, created, match_distance, ambiguities = self._recognise(descriptor)
        self._nodes[node]["visits"] += 1
        self._current_node = node
        self._persist()
        return self._event(
            node=node,
            created=created,
            source=None,
            new_edge=False,
            match_distance=match_distance,
            ambiguities=ambiguities,
        )

    def observe_transition(
        self,
        descriptor: NDArray[np.floating[Any]],
        *,
        action_trace: tuple[int, ...],
    ) -> dict[str, Any]:
        if self._current_node is None:
            raise RuntimeError("route memory has no episode start landmark")
        source = self._current_node
        if not action_trace or any(action < 0 for action in action_trace):
            raise ValueError("route transition requires a non-empty non-negative action trace")
        destination, created, match_distance, ambiguities = self._recognise(descriptor)
        self._nodes[destination]["visits"] += 1
        edge = (source, destination)
        new_edge = edge not in self._edges
        if new_edge:
            self._edges[edge] = {
                "traversals": 0,
                "best_action_trace": list(action_trace),
            }
        edge_record = self._edges[edge]
        edge_record["traversals"] += 1
        if len(action_trace) < len(edge_record["best_action_trace"]):
            edge_record["best_action_trace"] = list(action_trace)
        self._current_node = destination
        self._persist()
        return self._event(
            node=destination,
            created=created,
            source=source,
            new_edge=new_edge,
            match_distance=match_distance,
            ambiguities=ambiguities,
        )

    def route_plan(self, source: str, destination: str) -> list[dict[str, Any]] | None:
        path = self.shortest_path(source, destination)
        if path is None:
            return None
        segments: list[dict[str, Any]] = []
        for edge_source, edge_destination in pairwise(path):
            record = self._edges[(edge_source, edge_destination)]
            traversals = int(record["traversals"])
            segments.append(
                {
                    "source": edge_source,
                    "destination": edge_destination,
                    "temporal_action_trace": list(record["best_action_trace"]),
                    "observed_traversals": traversals,
                    "confidence": min(1.0, traversals / 3),
                }
            )
        return segments

    def shortest_path(self, source: str, destination: str) -> tuple[str, ...] | None:
        if source not in self._nodes or destination not in self._nodes:
            raise KeyError("route endpoint is not present in memory")
        frontier: deque[tuple[str, tuple[str, ...]]] = deque([(source, (source,))])
        visited = {source}
        adjacency: dict[str, list[str]] = {}
        for edge_source, edge_destination in self._edges:
            adjacency.setdefault(edge_source, []).append(edge_destination)
        while frontier:
            node, path = frontier.popleft()
            if node == destination:
                return path
            for neighbor in sorted(adjacency.get(node, [])):
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append((neighbor, (*path, neighbor)))
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": ROUTE_MEMORY_SCHEMA_VERSION,
            "current_node": self._current_node,
            "known_nodes": len(self._nodes),
            "known_edges": len(self._edges),
            "observed_traversals": sum(
                int(record["traversals"]) for record in self._edges.values()
            ),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": ROUTE_MEMORY_SCHEMA_VERSION,
            "revision": self._revision,
            "contract": self.contract(),
            "scope": self.scope,
            "match_distance": self.match_distance,
            "descriptor_shape": (
                list(self._descriptor_shape) if self._descriptor_shape is not None else None
            ),
            "next_node": self._next_node,
            "nodes": [
                {
                    "id": node,
                    "descriptor_u8": record["descriptor_u8"],
                    "visits": record["visits"],
                }
                for node, record in sorted(self._nodes.items())
            ],
            "edges": [
                {
                    "source": source,
                    "destination": destination,
                    "traversals": record["traversals"],
                    "best_action_trace": record["best_action_trace"],
                }
                for (source, destination), record in sorted(self._edges.items())
            ],
        }

    def _recognise(
        self, descriptor: NDArray[np.floating[Any]]
    ) -> tuple[str, bool, float | None, list[str]]:
        candidate = np.asarray(descriptor, dtype=np.float32)
        if candidate.size == 0 or not np.all(np.isfinite(candidate)):
            raise ValueError("route descriptor must be finite and non-empty")
        if np.any(candidate < 0) or np.any(candidate > 1):
            raise ValueError("route descriptor values must be in [0, 1]")
        shape = tuple(int(item) for item in candidate.shape)
        if self._descriptor_shape is None:
            self._descriptor_shape = shape
        elif shape != self._descriptor_shape:
            raise ValueError("route descriptor shape differs from persisted memory")
        distances: list[tuple[float, str]] = []
        for node, record in self._nodes.items():
            prototype = np.asarray(record["descriptor_u8"], dtype=np.float32).reshape(shape) / 255
            distance = float(np.mean(np.abs(prototype - candidate)))
            distances.append((distance, node))
        distances.sort()
        nearest_distance, nearest = distances[0] if distances else (float("inf"), None)
        if nearest is not None and nearest_distance <= self.match_distance:
            ambiguities = [
                node
                for distance, node in distances[1:]
                if distance <= self.match_distance
                and distance - nearest_distance <= self.match_distance * 0.2
            ]
            return nearest, False, nearest_distance, ambiguities
        node = f"landmark-{self._next_node:06d}"
        self._next_node += 1
        quantised = np.rint(candidate * 255).astype(np.uint8).reshape(-1).tolist()
        self._nodes[node] = {"descriptor_u8": quantised, "visits": 0}
        return node, True, None, []

    def _event(
        self,
        *,
        node: str,
        created: bool,
        source: str | None,
        new_edge: bool,
        match_distance: float | None = None,
        ambiguities: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "source": source,
            "destination": node,
            "new_landmark": created,
            "revisit": not created,
            "new_edge": new_edge,
            "edge_traversals": (
                int(self._edges[(source, node)]["traversals"])
                if source is not None
                else 0
            ),
            "match_distance": match_distance,
            "ambiguous_candidates": ambiguities or [],
            **self.summary(),
        }

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            current = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(current, dict) or current.get("revision") != self._revision:
                raise RuntimeError("route memory was modified by another writer")
        self._revision += 1
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(self.snapshot(), indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, self.path)

    def _load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("route memory must be a JSON object")
        if payload.get("schema_version") != ROUTE_MEMORY_SCHEMA_VERSION:
            raise ValueError("unsupported route memory schema version")
        if payload.get("scope") != self.scope:
            raise ValueError("route memory scope differs from the configured campaign/save")
        stored_distance = payload.get("match_distance")
        if stored_distance != self.match_distance:
            raise ValueError("route memory match_distance differs from configured value")
        raw_shape = payload.get("descriptor_shape")
        if raw_shape is not None:
            if not isinstance(raw_shape, list) or not all(
                isinstance(item, int) and item > 0 for item in raw_shape
            ):
                raise ValueError("invalid route descriptor_shape")
            self._descriptor_shape = tuple(raw_shape)
        raw_nodes = payload.get("nodes")
        raw_edges = payload.get("edges")
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            raise TypeError("route memory nodes and edges must be arrays")
        expected_size = (
            int(np.prod(self._descriptor_shape)) if self._descriptor_shape is not None else 0
        )
        for record in raw_nodes:
            if not isinstance(record, dict):
                raise TypeError("invalid route node")
            node = record.get("id")
            descriptor = record.get("descriptor_u8")
            visits = record.get("visits")
            if (
                not isinstance(node, str)
                or not isinstance(descriptor, list)
                or len(descriptor) != expected_size
                or not all(isinstance(item, int) and 0 <= item <= 255 for item in descriptor)
                or not isinstance(visits, int)
                or visits < 0
                or node in self._nodes
            ):
                raise ValueError("invalid route node fields")
            self._nodes[node] = {"descriptor_u8": descriptor, "visits": visits}
        for record in raw_edges:
            if not isinstance(record, dict):
                raise TypeError("invalid route edge")
            source = record.get("source")
            destination = record.get("destination")
            count = record.get("traversals")
            trace = record.get("best_action_trace")
            if (
                source not in self._nodes
                or destination not in self._nodes
                or not isinstance(count, int)
                or count < 1
                or not isinstance(trace, list)
                or not trace
                or not all(isinstance(action, int) and action >= 0 for action in trace)
                or (source, destination) in self._edges
            ):
                raise ValueError("invalid route edge fields")
            self._edges[(source, destination)] = {
                "traversals": count,
                "best_action_trace": trace,
            }
        next_node = payload.get("next_node")
        if not isinstance(next_node, int) or next_node < 1:
            raise ValueError("invalid route next_node")
        used_numbers = [
            int(node.removeprefix("landmark-"))
            for node in self._nodes
            if node.startswith("landmark-") and node.removeprefix("landmark-").isdigit()
        ]
        if used_numbers and next_node <= max(used_numbers):
            raise ValueError("route next_node would collide with a persisted landmark")
        self._next_node = next_node
        revision = payload.get("revision")
        if not isinstance(revision, int) or revision < 0:
            raise ValueError("invalid route revision")
        self._revision = revision

    @staticmethod
    def _validated_scope(scope: Mapping[str, str | int]) -> dict[str, str | int]:
        required = {
            "game_profile",
            "save_source",
            "save_slot",
            "campaign_id",
            "detector_signature",
        }
        if set(scope) != required:
            raise ValueError(f"route scope must contain exactly {sorted(required)}")
        result = dict(scope)
        if (
            not isinstance(result["game_profile"], str)
            or not result["game_profile"]
            or result["save_source"] not in {"existing_save", "new_game"}
            or not isinstance(result["save_slot"], int)
            or not 1 <= result["save_slot"] <= 4
            or not isinstance(result["campaign_id"], str)
            or not result["campaign_id"]
            or not isinstance(result["detector_signature"], str)
            or not result["detector_signature"]
        ):
            raise ValueError("invalid route scope fields")
        return result
