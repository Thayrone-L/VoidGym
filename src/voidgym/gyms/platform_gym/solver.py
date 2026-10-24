"""Scripted level solver that advances the same physics as the training adapter."""

from __future__ import annotations

from copy import deepcopy
from math import sqrt

from voidgym.gyms.platform_gym.physics import Event, step_physics
from voidgym.gyms.platform_gym.world import Player, Rect, Room

DT = 0.02


def _direction(room: Room, player: Player) -> int:
    exit_ = min(room.exits, key=lambda item: abs(item.rect.x + item.rect.w / 2 - player.x))
    return 1 if exit_.rect.x + exit_.rect.w / 2 >= player.x else -1


def _floor_ahead(room: Room, player: Player, direction: int) -> tuple[float, float, float] | None:
    floor_sides = sorted(((s.rect.x, s.rect.right, s.rect.top) for s in room.solids
                         if s.rect.h <= 0.7), key=lambda span: span[0])
    current = next((span for span in floor_sides
                    if span[0] - room.physics.player_width <= player.x <= span[1] + room.physics.player_width
                    and abs(span[2] - player.y) <= .25), None)
    if current is None:
        return None
    candidates = [span for span in floor_sides
                  if (span[0] > current[1] + 1e-5 if direction > 0 else span[1] < current[0] - 1e-5)
                  and abs(span[2] - current[2]) <= room.physics.jump_max_height]
    return min(candidates, key=lambda span: (span[0] if direction > 0 else -span[1]), default=None)


def _jump_trigger_distance(room: Room, player: Player, obstacle_height: float) -> float:
    physics = room.physics
    needed = max(0.0, obstacle_height)
    hold_t = min(physics.jump_hold_max_seconds, needed / physics.jump_v0)
    height_during_hold = physics.jump_v0 * hold_t
    remaining = max(0.0, needed - height_during_hold)
    discriminant = max(0.0, physics.jump_v0**2 - 2 * physics.gravity * remaining)
    after_hold = (physics.jump_v0 - sqrt(discriminant)) / physics.gravity
    # Include a generous lead distance so a buffered landing jump starts before
    # the feet reach a step's vertical face.
    return physics.run_speed * (hold_t + after_hold) + physics.player_width / 2 + 2.0


def _is_ahead(rect: Rect, player: Player, direction: int) -> bool:
    return direction * (rect.x + rect.w / 2 - player.x) >= -rect.w / 2


def _should_jump(room: Room, player: Player) -> bool:
    direction = _direction(room, player)
    reachable_top = player.y + room.physics.jump_max_height + room.physics.player_height
    obstacles = [s.rect for s in room.solids if s.kind == "solid" and s.rect.h > .3
                 and s.rect.y <= reachable_top and _is_ahead(s.rect, player, direction)]
    obstacles += [h.rect for h in room.hazards if _is_ahead(h.rect, player, direction)]
    obstacles += [e.rect for e in room.enemies if e.alive and _is_ahead(e.rect, player, direction)]
    target = min(obstacles, key=lambda rect: direction * (rect.x + rect.w / 2 - player.x), default=None)
    target_height = target.top - player.last_safe_pos[1] if target is not None else 0.0
    target_distance = direction * (target.x + target.w / 2 - player.x) if target is not None else 0.0
    if target is not None and target_distance <= _jump_trigger_distance(room, player, target_height):
        if player.on_ground:
            return True
        if player.vy < 0 and player.y <= target.y + .8 and player.jump_buffer_timer <= 0:
            return True
    next_floor = _floor_ahead(room, player, direction)
    gap_distance = room.physics.run_speed * (room.physics.jump_hold_max_seconds +
                                               2 * room.physics.jump_v0 / room.physics.gravity)
    if next_floor is not None:
        distance = direction * (next_floor[0] - player.x if direction > 0 else player.x - next_floor[1])
    else:
        distance = float("inf")
    if next_floor is not None and distance <= max(1.2, gap_distance * .85):
        return True
    raised = [s.rect for s in room.solids if player.y + .7 < s.rect.y <= reachable_top
              and _is_ahead(s.rect, player, direction)]
    return bool(raised and min(direction * (r.x + r.w / 2 - player.x) for r in raised) <=
                _jump_trigger_distance(room, player, min(r.top - player.y for r in raised)))


def _attack_target(room: Room, player: Player) -> bool:
    direction = _direction(room, player)
    targets = [b.rect for b in room.breakables if not b.broken]
    return any(-.1 <= direction * (rect.x + rect.w / 2 - player.x) <= 1.25
               and abs(rect.y - player.y) < 1.5 for rect in targets)


def _buttons(room: Room, player: Player) -> dict[str, bool]:
    direction = _direction(room, player)
    jump = (player.jumping and player.jump_hold_time < room.physics.jump_hold_max_seconds)
    if _should_jump(room, player):
        jump = True
    attack = _attack_target(room, player) and player.attack_cooldown <= DT
    block = next((item for item in room.breakables if not item.broken
                  and 0 <= direction * (item.rect.x + item.rect.w / 2 - player.x) <= 2.0), None)
    if block is not None:
        jump = False
        attack = player.attack_cooldown <= DT
        if not player.on_ground:
            return {"left": False, "right": False, "jump": False, "attack": attack}
    locked_door = next((door for door in room.doors
                        if not door.opened and door.requires in player.abilities
                        and direction * (door.rect.x + door.rect.w / 2 - player.x) >= 0
                        and direction * (door.rect.x + door.rect.w / 2 - player.x) <= 2.5), None)
    if locked_door is not None:
        # Touch the key gate on the ground; don't leap over it while clearing
        # a nearby enemy or obstacle.
        jump = False
        if not player.on_ground:
            return {"left": False, "right": False, "jump": False, "attack": attack}
    waiting_for_land = not player.on_ground and player.jump_buffer_timer > 0 and not player.jumping
    left = direction < 0 and not waiting_for_land
    right = direction > 0 and not waiting_for_land
    exit_ = min(room.exits, key=lambda item: abs(item.rect.x + item.rect.w / 2 - player.x))
    exit_distance = direction * (exit_.rect.x + exit_.rect.w / 2 - player.x)
    box_top = player.y + room.physics.player_height
    vertically_aligned = box_top > exit_.rect.y and player.y < exit_.rect.top
    if -0.5 <= exit_distance <= 1.25 and not vertically_aligned:
        # Stop at the edge and wait until the player's vertical path intersects the exit.
        left = right = False
        jump = exit_.rect.y > box_top and player.on_ground
    return {"left": left, "right": right, "jump": jump, "attack": attack}


def _run_room(room: Room, *, max_steps: int, abilities: set[str],
              must_break: bool = False) -> Player | None:
    if not room.exits or not room.solids:
        return None
    room = deepcopy(room)
    player = Player(*room.spawn, on_ground=True)
    player.last_safe_pos = room.spawn
    player.abilities.update(abilities)
    for _ in range(max_steps):
        events: list[Event] = step_physics(room, player, _buttons(room, player), DT)
        if any(event.kind == "exit" for event in events):
            return None if must_break and any(not block.broken for block in room.breakables) else player
        if player.y < -2 or player.x > room.width + 2 or player.hp <= 0:
            return None
    return None


def solve_room(room: Room, *, max_steps: int = 20_000, abilities: set[str] | None = None,
               require_breakables: bool = False) -> bool:
    """Run a deterministic greedy controller through a room using ``step_physics``."""
    return _run_room(room, max_steps=max_steps, abilities=abilities or set(),
                     must_break=require_breakables) is not None


def solve_room_set(rooms: list[Room], *, max_steps: int = 20_000) -> bool:
    """Prove a linked room chain, carrying only abilities actually collected."""
    if not 3 <= len(rooms) <= 6:
        return False
    abilities: set[str] = set()
    for index, room in enumerate(rooms):
        expected_next = (index + 1) % len(rooms)
        if not room.exits or room.exits[0].to_room != expected_next:
            return False
        player = _run_room(room, max_steps=max_steps, abilities=abilities)
        if player is None:
            return False
        abilities.update(player.abilities)
    return all(door.requires in abilities for door in rooms[-1].doors)
