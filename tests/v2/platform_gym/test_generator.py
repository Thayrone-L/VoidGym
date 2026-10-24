from __future__ import annotations

from time import perf_counter

from voidgym.gyms.platform_gym.generator import generate_room, generate_room_set
from voidgym.gyms.platform_gym.solver import solve_room, solve_room_set


def test_generated_rooms_certified_deterministic_and_fast() -> None:
    exit_directions: set[str] = set()
    tall_rooms = 0
    tall_by_level: dict[int, int] = {}
    for level in range(7):
        started = perf_counter()
        rooms = [generate_room(level, seed) for seed in range(200)]
        assert (perf_counter() - started) / len(rooms) < 0.05
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
            if level == 6:
                assert solve_room(room, require_breakables=True)
    assert exit_directions == {"left", "right", "up", "down"}
    assert tall_rooms > 0
    assert all(tall_by_level[level] > 0 for level in range(7))


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


def test_pit_and_gap_levels_have_recoverable_lower_routes_and_level_six_requires_breaking() -> None:
    gap = generate_room(2, 31)
    lower = next(s.rect for s in gap.solids if s.rect.y < 0.1 and s.rect.h < .6)
    gap.spawn = (lower.x + lower.w / 2, lower.top)
    assert solve_room(gap)

    pit = generate_room(4, 17)
    assert pit.spawn[1] < next(s.rect.top for s in pit.solids if s.rect.x > pit.spawn[0])
    assert solve_room(pit)

    secret = generate_room(6, 22)
    block = secret.breakables[0]
    direction = 1 if secret.exits[0].rect.x > secret.spawn[0] else -1
    assert direction * (block.rect.x + block.rect.w / 2 - secret.spawn[0]) > 0
    assert direction * (secret.exits[0].rect.x + .5 - block.rect.x - block.rect.w / 2) > 0
    assert block.rect.h >= secret.height - 1
    assert solve_room(secret, require_breakables=True)
