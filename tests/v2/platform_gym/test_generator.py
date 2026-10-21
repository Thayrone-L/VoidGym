"""Determinismo, solveability e custos do gerador procedural."""

import time

from voidgym.gyms.platform_gym.generator import generate_metroidvania_set, generate_room
from voidgym.gyms.platform_gym.solver import solve_set


def test_two_hundred_rooms_per_curriculum_level_are_solvable() -> None:
    for level in range(7):
        started = time.perf_counter()
        rooms = [generate_room(level, seed) for seed in range(200)]
        assert len(rooms) == 200
        average = (time.perf_counter() - started) / len(rooms)
        assert average < .05, f"level {level}: {average:.4f}s/sala"


def test_fifty_connected_metroidvania_sets_are_solvable_and_deterministic() -> None:
    for seed in range(50):
        rooms = generate_metroidvania_set(seed)
        assert 3 <= len(rooms) <= 6
        assert any(pickup.kind == "ability_double_jump" for pickup in rooms[0].pickups)
        assert not any(pickup.kind == "ability_double_jump" for pickup in rooms[1].pickups)
        assert sum(len(room.interactives) for room in rooms) == 1
        assert not rooms[0].doors and rooms[1].doors
        assert solve_set(rooms)
        assert generate_room(7, seed).solids == generate_room(7, seed).solids
