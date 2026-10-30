"""Procedural platform rooms with actual traversal challenges at each level."""

from __future__ import annotations

import numpy as np

from voidgym.gyms.platform_gym.params import PhysicsParams
from voidgym.gyms.platform_gym.solver import solve_room, solve_room_set
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
    TreasureRock,
)


def _mandatory_jump_wall(rng: np.random.Generator, physics: PhysicsParams,
                         minimum_ratio: float, maximum_ratio: float, x: float,
                         floor: float) -> Solid:
    """Build a required jump obstacle with a strict 85% physical-limit ceiling."""
    if not 0 <= minimum_ratio <= maximum_ratio <= .85:
        raise ValueError("mandatory jump obstacles must stay at or below 85% of jump height")
    height = float(rng.uniform(minimum_ratio, maximum_ratio) * physics.jump_max_height)
    return Solid(Rect(x, floor, .65, height))


def _mandatory_gap_width(rng: np.random.Generator, physics: PhysicsParams,
                        minimum_ratio: float, maximum_ratio: float) -> float:
    """Build a required gap with the same strict 85% reach ceiling as jump walls."""
    if not 0 <= minimum_ratio <= maximum_ratio <= .85:
        raise ValueError("mandatory gaps must stay at or below 85% of jump reach")
    reach = physics.run_speed * (
        physics.jump_hold_max_seconds + 2 * physics.jump_v0 / physics.gravity
    )
    return float(rng.uniform(minimum_ratio, maximum_ratio) * reach)


def _make(level: int, seed: int, *, allow_vertical: bool = True) -> Room:
    rng = np.random.default_rng(seed)
    physics = PhysicsParams.sample(rng)
    vertical_rate = .005 if level in {1, 5, 7} else \
        .08 if level in {0, 2, 3, 4} else 0.0
    vertical_variant = allow_vertical and vertical_rate > 0 and rng.random() < vertical_rate
    width = float(rng.uniform(14, 17) if vertical_variant else
                  rng.uniform(24, 36) if level == 0 else rng.uniform(22, 30))
    height = float(rng.uniform(22, 28) if vertical_variant else rng.uniform(10, 17))
    floor = 0.6
    if level == 2:
        floor = 3.0
    floor_y = floor
    spawn_y = floor
    solids: list[Solid]
    if level == 2:
        gap_width = _mandatory_gap_width(rng, physics, .35, .70)
        gap_left = width * .42
        pit_width = gap_width * .62
        solids = [Solid(Rect(0, floor - .6, gap_left, .6)),
                  Solid(Rect(gap_left, 0, pit_width, .5)),
                  Solid(Rect(gap_left + gap_width, floor - .6, width - gap_left - gap_width, .6))]
        # At least 3 units of run-up and a moderate wall teach a held jump
        # without putting the required height near the physical limit.
        wall_x = gap_left + gap_width + 3.0
        solids.append(_mandatory_jump_wall(rng, physics, .55, .75, wall_x, floor))
    elif level == 4:
        pit_floor = .6
        ledge_x = 4.5
        ledge_rise = float(rng.uniform(.55, .75) * physics.jump_max_height)
        upper_floor = pit_floor + ledge_rise
        spawn_y = pit_floor
        solids = [Solid(Rect(0, pit_floor - .6, ledge_x, .6)),
                  Solid(Rect(0, pit_floor, .55, 3.6)),
                  _mandatory_jump_wall(rng, physics, .65, .75, ledge_x, pit_floor),
                  Solid(Rect(ledge_x + .65, upper_floor - .6,
                             width - ledge_x - .65, .6))]
        floor_y = upper_floor
    else:
        solids = [Solid(Rect(0, 0, width, floor))]
    hazards, breakables, enemies, pickups, interactives, doors, treasure_rocks = _make_entities(
        level, rng, physics, width, height, floor, solids,
    )
    exit_rect = _make_exit(level, rng, vertical_variant and level != 4, width, floor_y, solids)
    spawn_x = 2.2 if level == 4 else 1.5
    room = Room(level, width, height, solids, hazards, breakables, enemies, pickups,
                interactives, [Exit(exit_rect, level + 1)], doors, (spawn_x, spawn_y), physics,
                seed, treasure_rocks)
    if level != 4 and rng.random() < .25:
        room = _mirror_room(room)
    return room


def _make_entities(level: int, rng: np.random.Generator, physics: PhysicsParams,
                   width: float, height: float, floor: float, solids: list[Solid]
                   ) -> tuple[list[Hazard], list[Breakable], list[Enemy], list[Pickup],
                              list[Interactive], list[Door], list[TreasureRock]]:
    hazards: list[Hazard] = []
    breakables: list[Breakable] = []
    enemies: list[Enemy] = []
    pickups: list[Pickup] = []
    interactives: list[Interactive] = []
    doors: list[Door] = []
    treasure_rocks: list[TreasureRock] = []
    if level == 1:
        for fraction in (.27, .48, .70):
            height_u = float(rng.uniform(.5, .6) * physics.jump_short_height)
            solids.append(Solid(Rect(width * fraction, floor, 1.2, height_u)))
    if level == 3:
        solids.append(_mandatory_jump_wall(rng, physics, .6, .82, width * .48, floor))
    if level == 4:
        wall_x = width * .68
        solids.append(_mandatory_jump_wall(rng, physics, .65, .82, wall_x, floor))
        # A second, moderate wall near the raised exit makes level 4 require
        # two deliberate jumps instead of a single lucky traversal.
        upper_wall_x = width * .82
        upper_floor = max(solid.rect.top for solid in solids
                          if solid.rect.x > 1.0 and solid.rect.h <= .7)
        solids.append(_mandatory_jump_wall(rng, physics, .65, .75,
                                           upper_wall_x, upper_floor))
    if level == 5:
        # Level 5 needs a deliberate full jump; short-tap policies used to
        # walk to the exit too often.
        wall_x = width * .68
        solids.append(_mandatory_jump_wall(rng, physics, .55, .82, wall_x, floor))
    if level >= 5:
        _add_hazards_and_enemies(level, rng, physics, width, floor, solids,
                                 hazards, enemies, treasure_rocks)
    if level == 7:
        block_x = width * .68
        breakables.append(Breakable(Rect(block_x, floor, .75, height - floor), hp=3, reveals="passage",
                                    cue=bool(rng.random() < .7)))
        pickups.append(Pickup(Rect(width * .2, floor, .4, .5), "coin"))
    if level == 8:
        pickups.append(Pickup(Rect(width * .2, floor, .5, .5), "ability_double_jump"))
        doors.append(Door(Rect(width * .62, floor, .8, 2.2), "ability_double_jump"))
        interactives.append(Interactive(Rect(width * .4, floor, 1.0, .8)))
        # Multi-room topology is recorded as forward links; the adapter can advance the set.
    if rng.random() < .7:
        # Decorative one-way shelf: it changes routes and silhouettes without sealing the exit.
        shelf_x = float(rng.uniform(width * .12, width * .28))
        solids.append(Solid(Rect(shelf_x, floor + float(rng.uniform(2.0, 3.5)), 2.0, .22), "platform"))
    if rng.random() < .4:
        ceiling_x = float(rng.uniform(width * .12, width * .3))
        solids.append(Solid(Rect(ceiling_x, height - 1.0, float(rng.uniform(3.0, 5.0)), .5)))
    return hazards, breakables, enemies, pickups, interactives, doors, treasure_rocks


def _add_combat_entities(rng: np.random.Generator, physics: PhysicsParams, width: float,
                         floor: float, solids: list[Solid], enemies: list[Enemy],
                         rocks: list[TreasureRock]) -> None:
    """Place a low-roof mandatory blocker and optional breakable treasure rocks."""
    solids.append(Solid(Rect(0, floor + physics.player_height + .04, width, .28)))
    for index in range(int(rng.integers(1, 3))):
        x = width * (.43 + .16 * index)
        enemies.append(Enemy(
            Rect(x, floor, .8, .8), kind=str(rng.choice(("walker", "spiky", "eye"))),
            hp=2, patrol_left=x, patrol_right=x, speed=0.0, blocks_path=True,
        ))
    for index in range(int(rng.integers(1, 3))):
        x = width * (.25 + .10 * index)
        rocks.append(TreasureRock(
            Rect(x, floor, 1.05, 1.05), hp=int(rng.integers(2, 5)),
            visual_variant=int(rng.integers(0, 4)),
        ))


def _add_hazards_and_enemies(level: int, rng: np.random.Generator, physics: PhysicsParams,
                             width: float, floor: float, solids: list[Solid],
                             hazards: list[Hazard], enemies: list[Enemy],
                             rocks: list[TreasureRock]) -> None:
    hazards.append(Hazard(Rect(width * .4, floor, 1.4, .35), "spikes"))
    if level == 6:
        _add_combat_entities(rng, physics, width, floor, solids, enemies, rocks)
        return
    for _ in range(int(rng.integers(1, 4))):
        x = float(rng.uniform(width * .22, width * .78))
        enemies.append(Enemy(Rect(x, floor, .8, .8), kind=str(rng.choice(("walker", "orb", "spiky", "eye"))),
                             hp=2, patrol_left=max(0, x - 1),
                             patrol_right=min(width, x + 2.8), speed=float(rng.uniform(.7, 1.6))))


def _make_exit(level: int, rng: np.random.Generator, vertical_variant: bool, width: float,
               floor: float, solids: list[Solid]) -> Rect:
    exit_rect = Rect(width - 1, floor, 1, 2)
    if vertical_variant:
        solids.extend((Solid(Rect(width * .28, floor + 1.9, 3.2, .22), "platform"),
                       Solid(Rect(width * .56, floor + 3.6, 4.2, .22), "platform")))
        exit_rect = Rect(width - 2.5, floor + 3.0, 2.5, 1.4)
    elif level == 0:
        exit_variant = rng.choice(("right", "right", "up", "down"))
        if exit_variant == "up":
            solids.append(Solid(Rect(width - 3.2, floor + 1.8, 2.7, .22), "platform"))
            exit_rect = Rect(width - 2.0, floor + 1.2, 2.0, 1.2)
        elif exit_variant == "down":
            solids[0] = Solid(Rect(0, floor - .6, width - 3.2, .6))
            exit_rect = Rect(width - 2.0, -1.5, 2.0, 2.2)
    return exit_rect


def _mirror_rect(rect: Rect, width: float) -> Rect:
    return Rect(width - rect.right, rect.y, rect.w, rect.h)


def _mirror_room(room: Room) -> Room:
    """Reflect all geometry and patrol direction around the room's vertical axis."""
    width = room.width
    for solid in room.solids:
        solid.rect = _mirror_rect(solid.rect, width)
    for hazard in room.hazards:
        hazard.rect = _mirror_rect(hazard.rect, width)
    for block in room.breakables:
        block.rect = _mirror_rect(block.rect, width)
    for rock in room.treasure_rocks:
        rock.rect = _mirror_rect(rock.rect, width)
    for enemy in room.enemies:
        old_left, old_right = enemy.patrol_left, enemy.patrol_right
        enemy.rect = _mirror_rect(enemy.rect, width)
        enemy.patrol_left, enemy.patrol_right = width - old_right, width - old_left
        enemy.speed *= -1
    for pickup in room.pickups:
        pickup.rect = _mirror_rect(pickup.rect, width)
    for item in room.interactives:
        item.rect = _mirror_rect(item.rect, width)
    for exit_ in room.exits:
        exit_.rect = _mirror_rect(exit_.rect, width)
        exit_.to_entry = "right"
    for door in room.doors:
        door.rect = _mirror_rect(door.rect, width)
    room.spawn = (width - room.spawn[0], room.spawn[1])
    return room


def generate_room(level: int, seed: int, *, allow_vertical: bool = True) -> Room:
    """Generate a seeded level and retain it only when the physics solver completes it."""
    if not 0 <= level <= 8:
        raise ValueError("level must be in [0, 8]")
    for attempt in range(20):
        room = _make(level, seed + 1000 * attempt, allow_vertical=allow_vertical)
        if solve_room(room, require_breakables=level == 7):
            return room
    raise RuntimeError(f"could not solve level {level} room within 20 seed attempts")


def generate_room_set(seed: int) -> list[Room]:
    """Create and solve a 3–6 room metroidvania chain with a remote key ability."""
    for attempt in range(20):
        attempt_seed = seed + attempt * 100_003
        rng = np.random.default_rng(attempt_seed)
        count = int(rng.integers(3, 7))
        levels = [3, *[int(rng.integers(1, 7)) for _ in range(count - 2)], 5]
        rooms = [generate_room(level, attempt_seed + i * 7919, allow_vertical=False)
                 for i, level in enumerate(levels)]
        for room_id, room in enumerate(rooms):
            room.id = room_id
            room.exits[0].to_room = (room_id + 1) % count
            room.exits[0].to_entry = "left"
        first = rooms[0]
        rooms[0].pickups.append(Pickup(Rect(first.spawn[0] - .25, first.spawn[1], .5, .5),
                                       "ability_double_jump"))
        last = rooms[-1]
        direction = 1 if last.exits[0].rect.x > last.spawn[0] else -1
        door_x = last.width * (.84 if direction > 0 else .16)
        rooms[-1].doors.append(Door(Rect(door_x, last.exits[0].rect.y, .8, 2.2),
                                    "ability_double_jump"))
        middle = rooms[count // 2]
        middle.interactives.append(Interactive(Rect(middle.width * .5, .6, 1.0, .8)))
        if solve_room_set(rooms):
            return rooms
    raise RuntimeError("generated room-set route failed its physics proof within 20 attempts")
