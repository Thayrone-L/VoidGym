"""Geração determinística de salas de treino sempre com rota aberta."""

import random

from voidgym.gyms.platform_gym.params import PhysicsParams
from voidgym.gyms.platform_gym.world import (
    Breakable,
    Door,
    Enemy,
    Exit,
    Hazard,
    Interactive,
    Pickup,
    Rect,
    Room,
    Solid,
)


def generate_room(level: int, seed: int) -> Room:
    from voidgym.gyms.platform_gym.solver import solve_room

    for retry in range(20):
        room = _build_room(level, seed + retry * 1000)
        if solve_room(room):
            return room
    raise ValueError("Não foi possível gerar uma sala resolvível após 20 tentativas")


def generate_metroidvania_set(seed: int) -> list[Room]:
    from voidgym.gyms.platform_gym.solver import solve_set

    rng = random.Random(seed)
    count = rng.randint(3, 6)
    rooms = [generate_room(7, seed * 10 + index) for index in range(count)]
    for index, room in enumerate(rooms):
        if index > 0:
            room.pickups = [pickup for pickup in room.pickups if pickup.kind != "ability_double_jump"]
        if index != 1:
            room.doors.clear()
        if index > 0:
            room.interactives.clear()
    for index, room in enumerate(rooms):
        room.id = index
        room.exits[0].to_room = index + 1 if index + 1 < count else -1
    if not solve_set(rooms):
        raise ValueError("O conjunto metroidvania gerado não é resolvível")
    return rooms


def _build_room(level: int, seed: int) -> Room:
    if not 0 <= level <= 7:
        raise ValueError("nível deve estar entre 0 e 7")
    rng = random.Random(seed)
    physics = PhysicsParams.sample(rng)
    width, height = rng.uniform(20, 40), rng.uniform(9, 14)
    room = Room(seed, width, height, physics=physics, style_seed=seed)
    room.solids.extend((Solid(Rect(0, 0, width, .5)), Solid(Rect(0, 0, .5, height)), Solid(Rect(width - .5, 0, .5, height))))
    room.spawn = (1, .5)
    room.exits.append(Exit(Rect(width - 1, .5, .5, 2), -1, "right"))
    if level == 1:
        for x in range(5, int(width - 4), 5):
            room.solids.append(Solid(Rect(x, min(2.0, physics.jump_short_height * .55), 4, .5), "platform"))
    if level in {2, 4}:
        # Gaps stay below 70% of the computed maximum air range.
        gap_start = 4.0
        gap: float = min(physics.max_jump_range * .35, 2.5)
        room.solids = [solid for solid in room.solids if solid.rect.y != 0 or solid.rect.w < 1]
        room.solids.extend((Solid(Rect(0, 0, gap_start, .5)), Solid(Rect(gap_start + gap, 0, width - gap_start - gap, .5))))
        if level == 4:
            room.spawn = (gap_start + .2, -1.5)
            room.solids.append(Solid(Rect(gap_start - .2, -2.0, gap + .4, .5)))
    if level == 3:
        room.solids.append(Solid(Rect(width / 2, .5, .5, physics.jump_max_height * .75)))
    if 5 <= level <= 6:
        room.hazards.append(Hazard(Rect(width * .55, .5, 1.0, .25), rng.choice(("spikes", "thorns", "acid"))))
        for _ in range(rng.randint(1, 3)):
            enemy_x: float = rng.uniform(7, width - 6)
            room.enemies.append(Enemy(Rect(enemy_x, .5, .7, .8), "walker", 2, max(1, enemy_x - 2), min(width - 2, enemy_x + 2), rng.choice((-1.5, 1.5))))
    if level == 6:
        block = Breakable(Rect(width - 2, 2.5, .5, 2), 3, "passage", rng.random() < .7)
        room.breakables.append(block)
        # Guaranteed route stays open; breakable is an optional shortcut.
    if level == 7:
        room.interactives.append(Interactive(Rect(width * .45, .5, 1, 1)))
        room.pickups.append(Pickup(Rect(width * .3, .5, .8, .8), "ability_double_jump"))
        room.doors.append(Door(Rect(width * .65, .5, .5, 2), "ability_double_jump"))
        room.pickups.append(Pickup(Rect(width * .1, .5, .5, .5), "coin"))
    # Stable extra platforms make routes visually varied without narrowing the safe floor route.
    for _ in range(rng.randint(1, 3)):
        platform_x: float = rng.uniform(3, max(4, width - 6))
        room.solids.append(Solid(Rect(platform_x, rng.uniform(2, 5), rng.uniform(2, 5), .25), "platform"))
    return room
