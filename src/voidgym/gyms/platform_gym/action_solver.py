"""Search platform rooms using only actions exposed by ``PlatformGymAdapter``.

Unlike :mod:`solver`, this search never chooses controls at physics-tick
granularity. Each edge is one legal, fixed-duration adapter action, including
the adapter's held-jump continuation and button-release behavior.
"""

from __future__ import annotations

import heapq
import itertools
from copy import deepcopy
from dataclasses import dataclass
from math import ceil
from typing import cast

from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter
from voidgym.gyms.platform_gym.solver import _buttons


@dataclass(frozen=True, slots=True)
class ActionSolveResult:
    """Outcome and search effort for one adapter episode."""

    success: bool
    actions: tuple[str, ...]
    expanded: int
    duration_ms: int
    reason: str


@dataclass(slots=True)
class _Node:
    adapter: PlatformGymAdapter
    elapsed_ms: int
    guided_cost_ms: int
    key: tuple[object, ...]


def _trace_path(parent: dict[tuple[object, ...], tuple[tuple[object, ...], int]],
                key: tuple[object, ...], final_action: int | None = None) -> tuple[int, ...]:
    path = [] if final_action is None else [final_action]
    while key in parent:
        key, action = parent[key]
        path.append(action)
    path.reverse()
    return tuple(path)


def _q(value: float, resolution: float) -> int:
    return round(value / resolution)


def _signature(adapter: PlatformGymAdapter) -> tuple[object, ...]:
    """Discretize the full mutable physics state, including jump continuation."""
    player = adapter.player
    p = adapter.room.physics
    player_state: tuple[object, ...] = (
        _q(player.x, .1), _q(player.y, .1), _q(player.vx, .5), _q(player.vy, .5),
        player.on_ground, player.facing, player.hp, _q(player.invuln_timer, .04),
        _q(player.jump_hold_time, .04), player.jumping, tuple(sorted(player.abilities)),
        (_q(player.last_safe_pos[0], .1), _q(player.last_safe_pos[1], .1)),
        player.jump_was_pressed, _q(player.coyote_timer, .04),
        _q(player.jump_buffer_timer, .04), player.air_jump_available,
        _q(player.dash_timer, .04), _q(player.dash_cooldown, .04),
        _q(player.attack_cooldown, .04), _q(player.attack_timer, .04),
        player.attack_was_pressed, player.interact_was_pressed, player.dash_was_pressed,
        tuple(sorted(player.attack_targets_hit)), player.coins,
    )
    room_state = (
        tuple((item.hp, item.broken) for item in adapter.room.breakables),
        tuple((_q(item.rect.x, .1), _q(item.rect.y, .1), item.hp, item.alive,
               _q(item.speed, .25)) for item in adapter.room.enemies),
        tuple((item.kind, _q(item.rect.x, .1), _q(item.rect.y, .1), item.falling,
               _q(item.velocity_y, .5)) for item in adapter.room.pickups),
        tuple((item.opened,) for item in adapter.room.doors),
        tuple((item.hp, item.broken) for item in adapter.room.treasure_rocks),
    )
    # Physics parameters are fixed per generated episode, but including them
    # makes signatures safe if callers ever combine nodes from distinct rooms.
    physics = (ceil(p.run_speed * 100), ceil(p.gravity * 100),
               ceil(p.jump_v0 * 100), ceil(p.jump_hold_max_seconds * 100))
    return (adapter._room_index, adapter._room_counter, adapter._episode_steps,
            adapter._steps_since_new_cell, tuple(sorted(adapter._visited)),
            player_state, room_state, physics)


def _heuristic(adapter: PlatformGymAdapter) -> float:
    room = adapter.room
    exit_ = min(room.exits, key=lambda item: abs(item.rect.x + item.rect.w / 2 - adapter.player.x))
    distance = abs(exit_.rect.x + exit_.rect.w / 2 - adapter.player.x)
    rooms_left = max(0, len(adapter._rooms) - adapter._room_index - 1)
    estimate = distance / room.physics.run_speed + rooms_left * 1.5
    return cast(float, estimate)


def solve_with_actions(adapter: PlatformGymAdapter, *, max_expansions: int = 30_000,
                       max_game_seconds: float = 45.0,
                       heuristic_weight: float = 2.5,
                       stop_after_room_change: bool = False) -> ActionSolveResult:
    """Find a route with discrete actions applied through the real adapter.

    The caller supplies a fresh adapter. Legal edges are obtained from its
    action mask, so the search respects unavailable abilities and current
    interaction prompts exactly as a policy does. The existing continuous
    solver's next control is used only as an ordering preference; it never
    removes an action from the search or advances the simulated adapter.
    """
    if max_expansions < 1 or max_game_seconds <= 0 or heuristic_weight < 1:
        raise ValueError("search limits must be positive")
    names = tuple(adapter.action_table())
    root = deepcopy(adapter)
    root_key = _signature(root)
    serial = itertools.count()
    frontier: list[tuple[float, int, _Node]] = [
        (heuristic_weight * _heuristic(root), next(serial), _Node(root, 0, 0, root_key))
    ]
    best_costs = {root_key: [(0, 0)]}
    parent: dict[tuple[object, ...], tuple[tuple[object, ...], int]] = {}
    expanded = 0
    max_ms = int(max_game_seconds * 1000)
    while frontier and expanded < max_expansions:
        _, _, node = heapq.heappop(frontier)
        if (node.elapsed_ms, node.guided_cost_ms) not in best_costs.get(node.key, []):
            continue
        expanded += 1
        legal = node.adapter.action_mask()
        room = node.adapter.room
        guidance = _buttons(room, node.adapter.player)
        for action_index in (int(i) for i in legal.nonzero()[0]):
            child = deepcopy(node.adapter)
            child.apply(action_index)
            elapsed = node.elapsed_ms + child._last_step_game_ms
            action = node.adapter.to_universal(action_index)
            mismatch = (int(guidance["left"] != (action.move < 0)) +
                        int(guidance["right"] != (action.move > 0)) +
                        int(guidance["jump"] != (action.jump is not None)) +
                        int(guidance["attack"] != action.attack))
            guided_cost = node.guided_cost_ms + mismatch * 250
            signals = child.signals()
            room_changed = bool(signals.get("room_changed"))
            if signals.get("terminated") or (room_changed and
                                               (child.level != 8 or stop_after_room_change)):
                path = (*_trace_path(parent, node.key), action_index)
                return ActionSolveResult(True, tuple(names[i] for i in path), expanded,
                                         elapsed, "exit" if signals.get("terminated")
                                         else "room_exit")
            if (signals.get("died") or signals.get("truncated") or child.player.hp <= 0
                    or child.player.y < -2):
                continue
            if elapsed >= max_ms:
                continue
            child_key = _signature(child)
            prior_costs = best_costs.get(child_key, [])
            if any(prior_elapsed <= elapsed and prior_guided <= guided_cost
                   for prior_elapsed, prior_guided in prior_costs):
                continue
            best_costs[child_key] = [
                (prior_elapsed, prior_guided)
                for prior_elapsed, prior_guided in prior_costs
                if not (elapsed <= prior_elapsed and guided_cost <= prior_guided)
            ] + [(elapsed, guided_cost)]
            parent[child_key] = (node.key, action_index)
            priority = guided_cost / 1000 + heuristic_weight * _heuristic(child)
            heapq.heappush(frontier, (priority, next(serial),
                                      _Node(child, elapsed, guided_cost, child_key)))

    reason = "node_limit" if expanded >= max_expansions else "frontier_exhausted"
    return ActionSolveResult(False, (), expanded, 0, reason)
