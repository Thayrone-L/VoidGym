from __future__ import annotations

from time import perf_counter

import pytest
import numpy as np

from voidgym.gyms.platform_gym.generator import (
    _make,
    _mandatory_gap_width,
    _mandatory_jump_wall,
    generate_room,
    generate_room_set,
)
from voidgym.gyms.platform_gym.solver import solve_room, solve_room_set


def test_generated_rooms_certified_deterministic_and_fast() -> None:
    exit_directions: set[str] = set()
    tall_rooms = 0
    tall_by_level: dict[int, int] = {}
    for level in range(9):
        started = perf_counter()
        rooms = [generate_room(level, seed) for seed in range(200)]
        # Leave headroom for shared-host scheduling while still guarding regressions.
        assert (perf_counter() - started) / len(rooms) < 0.08
        for seed, room in enumerate(rooms):
            assert solve_room(room)
            tall_rooms += room.height > room.width
            tall_by_level[level] = tall_by_level.get(level, 0) + int(room.height > room.width)
            exit_ = room.exits[0].rect
            if exit_.y < 0:
                exit_directions.add("down")
            elif exit_.y > .7:
                exit_directions.add("up")
            elif exit_.x + exit_.w / 2 < room.spawn[0]:
                exit_directions.add("left")
            else:
                exit_directions.add("right")
            if seed == 0:
                repeated = generate_room(level, seed)
                assert room.width == repeated.width
                assert room.physics == repeated.physics
                assert room.solids == repeated.solids
            if level == 7:
                assert solve_room(room, require_breakables=True)
    assert exit_directions == {"left", "right", "up", "down"}
    assert tall_rooms > 0
    assert all(tall_by_level[level] > 0 for level in (0, 1, 2, 3, 4, 5, 7))


def test_metroidvania_rooms_include_linked_ability_and_bench() -> None:
    for seed in range(50):
        rooms = generate_room_set(seed)
        assert 3 <= len(rooms) <= 6
        assert [room.id for room in rooms] == list(range(len(rooms)))
        assert rooms[0].exits[0].to_room == 1
        assert rooms[-1].exits[0].to_room == 0
        assert any(p.kind == "ability_double_jump" for p in rooms[0].pickups)
        assert any(d.requires == "ability_double_jump" for d in rooms[-1].doors)
        assert any(room.interactives for room in rooms)
        assert solve_room_set(rooms)


def test_metroidvania_retries_a_seed_whose_first_room_set_is_not_solvable() -> None:
    rooms = generate_room_set(155230)
    assert 3 <= len(rooms) <= 6
    assert solve_room_set(rooms)


def test_pit_and_gap_levels_have_recoverable_lower_routes_and_level_seven_requires_breaking() -> None:
    gap = generate_room(2, 31)
    lower = next(s.rect for s in gap.solids if s.rect.y < 0.1 and s.rect.h < .6)
    gap.spawn = (lower.x + lower.w / 2, lower.top)
    assert solve_room(gap)

    pit = generate_room(4, 17)
    assert pit.spawn[1] < next(s.rect.top for s in pit.solids if s.rect.x > pit.spawn[0])
    assert solve_room(pit)

    secret = generate_room(7, 22)
    block = secret.breakables[0]
    direction = 1 if secret.exits[0].rect.x > secret.spawn[0] else -1
    assert direction * (block.rect.x + block.rect.w / 2 - secret.spawn[0]) > 0
    assert direction * (secret.exits[0].rect.x + .5 - block.rect.x - block.rect.w / 2) > 0
    assert block.rect.h >= secret.height - 1
    assert solve_room(secret, require_breakables=True)


def test_level_six_is_attack_gated_and_treasure_rocks_have_two_to_four_hits() -> None:
    for seed in range(200):
        room = generate_room(6, seed)
        blockers = [enemy for enemy in room.enemies if enemy.blocks_path]
        assert 1 <= len(blockers) <= 2
        assert all(enemy.hp == 2 for enemy in blockers)
        assert 1 <= len(room.treasure_rocks) <= 2
        assert all(2 <= rock.hp <= 4 for rock in room.treasure_rocks)
        assert solve_room(room)
        assert not solve_room(room, allow_attack=False)


def test_mandatory_jump_obstacles_respect_physics_limit_and_level_two_runup() -> None:
    for level in (2, 3, 4, 5):
        for seed in range(100):
            room = _make(level, seed, allow_vertical=False)
            challenge_x = room.width * .68
            challenge = ([solid for solid in room.solids
                          if abs(solid.rect.x - challenge_x) < 1e-8 or
                          (abs(solid.rect.x - room.width * .82) < 1e-8 and
                           abs(solid.rect.y - max(item.rect.top for item in room.solids
                                                  if item.rect.x > 1 and item.rect.h <= .7)) < 1e-8) or
                          (abs(solid.rect.x - 4.5) < 1e-8 and abs(solid.rect.y - .6) < 1e-8)]
                         if level == 4 else
                         [solid for solid in room.solids if solid.rect.y == .6
                          and abs(solid.rect.w - .65) < 1e-9 and solid.rect.h > .3]
                         if level == 5 else
                         [solid for solid in room.solids if solid.rect.y == .6
                          and abs(solid.rect.w - .65) < 1e-9]
                         if level == 3 else
                         [solid for solid in room.solids if solid.rect.y == room.spawn[1]
                          and abs(solid.rect.w - .65) < 1e-9])
            assert challenge
            assert max(solid.rect.h for solid in challenge) <= .85 * room.physics.jump_max_height
            if level == 2:
                reach = room.physics.run_speed * (
                    room.physics.jump_hold_max_seconds + 2 * room.physics.jump_v0 / room.physics.gravity
                )
                pit = next(solid.rect for solid in room.solids
                           if solid.rect.y == 0)
                gap = pit.w / .62
                wall = max(challenge, key=lambda solid: solid.rect.x)
                assert .35 * reach <= gap <= .70 * reach
                runup = (wall.rect.x - (pit.x + gap) if wall.rect.x > pit.x else
                         pit.x - (gap - pit.w) - (wall.rect.x + wall.rect.w))
                assert runup >= 3.0 - 1e-8


def test_required_wall_helper_rejects_near_limit_challenges() -> None:
    room = _make(2, 51, allow_vertical=False)
    with pytest.raises(ValueError, match="85%"):
        _mandatory_jump_wall(np.random.default_rng(4), room.physics,
                             .86, .92, 10.0, 3.0)
    with pytest.raises(ValueError, match="85%"):
        _mandatory_gap_width(np.random.default_rng(4), room.physics,
                             .86, .92)
