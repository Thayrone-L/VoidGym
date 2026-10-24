"""Small deterministic AABB platform physics used by the procedural gym."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from voidgym.gyms.platform_gym.world import Door, Enemy, Pickup, Player, Rect, Room, Solid


@dataclass(frozen=True, slots=True)
class Event:
    kind: str
    target: str | None = None
    value: Any = None


def player_rect(room: Room, player: Player) -> Rect:
    p = room.physics
    return Rect(player.x - p.player_width / 2, player.y, p.player_width, p.player_height)


def interaction_prompt(room: Room, player: Player) -> bool:
    box = player_rect(room, player)
    return any(_near_interactive(box, item.rect) for item in room.interactives)


def _near_interactive(box: Rect, target: Rect) -> bool:
    return (box.intersects(target) or
            abs((box.x + box.w / 2) - (target.x + target.w / 2)) <= 1.5
            and abs(box.y - target.y) <= 1.2)


def _blockers(room: Room) -> list[tuple[Rect, str, object]]:
    result: list[tuple[Rect, str, object]] = [(s.rect, s.kind, s) for s in room.solids]
    result.extend((b.rect, "solid", b) for b in room.breakables if not b.broken)
    result.extend((d.rect, "solid", d) for d in room.doors if not d.opened)
    return result


def _prepare_motion(room: Room, player: Player, buttons: dict[str, bool], dt: float) -> tuple[float, float, bool]:
    physics = room.physics
    grounded = player.on_ground
    if grounded:
        player.coyote_timer = 0.08
        player.air_jump_available = True
        player.dash_cooldown = 0.0
    else:
        player.coyote_timer = max(0.0, player.coyote_timer - dt)
    jump_held = player.jump_was_pressed
    jump_started = buttons["jump"] and not jump_held
    player.jump_buffer_timer = 0.08 if jump_started else max(0.0, player.jump_buffer_timer - dt)
    player.jump_was_pressed = buttons["jump"]
    direction = int(buttons["right"]) - int(buttons["left"])
    if direction:
        player.facing = direction
    player.vx = direction * physics.run_speed
    if player.jump_buffer_timer > 0 and (grounded or player.coyote_timer > 0):
        player.vy, player.jumping, player.jump_hold_time = physics.jump_v0, True, 0.0
        player.jump_buffer_timer = player.coyote_timer = 0.0
    elif player.jump_buffer_timer > 0 and "ability_double_jump" in player.abilities and player.air_jump_available:
        player.vy, player.jumping, player.jump_hold_time = physics.jump_v0, True, 0.0
        player.jump_buffer_timer, player.air_jump_available = 0.0, False
    dash_started = buttons["dash"] and not player.dash_was_pressed
    if dash_started and "ability_dash" in player.abilities and player.dash_cooldown <= 0:
        player.dash_timer = player.dash_cooldown = 0.15
    player.dash_was_pressed = buttons["dash"]
    if player.dash_timer > 0:
        player.vx = player.facing * physics.dash_distance / 0.15
        player.vy = 0.0
        player.dash_timer = max(0.0, player.dash_timer - dt)
        return 0.0, player.x, jump_started
    dy = 0.0
    if not jump_started:
        hold_dt = min(dt, max(0.0, physics.jump_hold_max_seconds - player.jump_hold_time))
        if buttons["jump"] and player.jumping and jump_held and hold_dt > 0:
            dy = physics.jump_v0 * hold_dt
            player.vy = physics.jump_v0
            player.jump_hold_time += hold_dt
        else:
            hold_dt = 0.0
        gravity_dt = dt - hold_dt
        if gravity_dt > 0:
            dy += player.vy * gravity_dt - 0.5 * physics.gravity * gravity_dt**2
            player.vy = max(-physics.max_fall_speed, player.vy - physics.gravity * gravity_dt)
        if not buttons["jump"] or player.jump_hold_time >= physics.jump_hold_max_seconds:
            player.jumping = False
    return dy, player.x, jump_started


def _open_touched_doors(room: Room, player: Player, candidate: Rect, events: list[Event]) -> None:
    for door in room.doors:
        if not door.opened and door.requires in player.abilities and candidate.intersects(door.rect):
            door.opened = True
            events.append(Event("door_open", door.requires))


def _resolve_x(room: Room, player: Player, old: Rect, start_x: float, dt: float, events: list[Event]) -> list[tuple[Rect, str, object]]:
    blockers = _blockers(room)
    candidate = Rect(old.x + player.vx * dt, old.y, old.w, old.h)
    _open_touched_doors(room, player, candidate, events)
    blockers = [(r, k, o) for r, k, o in blockers if not (isinstance(o, Door) and o.opened)]
    for rect, kind, _ in blockers:
        if kind == "platform" or not candidate.intersects(rect):
            continue
        if player.vx > 0:
            player.x = rect.x - room.physics.player_width / 2
        elif player.vx < 0:
            player.x = rect.right + room.physics.player_width / 2
        player.vx = 0.0
        break
    else:
        player.x = start_x + player.vx * dt
    return blockers


def _resolve_y(room: Room, player: Player, blockers: list[tuple[Rect, str, object]],
               start_y: float, dy: float, was_grounded: bool, events: list[Event]) -> None:
    target_y = start_y + dy
    player.y, player.on_ground = target_y, False
    candidate = player_rect(room, player)
    for rect, kind, _ in blockers:
        if kind == "platform":
            crosses_from_above = start_y >= rect.top and target_y <= rect.top
            horizontal_overlap = candidate.x < rect.right and candidate.right > rect.x
            if player.vy > 0 or not crosses_from_above or not horizontal_overlap:
                continue
        else:
            lands_on_top = player.vy <= 0 and start_y >= rect.top and target_y <= rect.top
            horizontal_overlap = candidate.x < rect.right and candidate.right > rect.x
            if lands_on_top and horizontal_overlap:
                pass
            elif not candidate.intersects(rect):
                continue
        if player.vy <= 0:
            player.y, player.on_ground = rect.top, True
            if not was_grounded:
                events.append(Event("landed"))
        else:
            player.y = rect.y - room.physics.player_height
        player.vy = 0.0
        candidate = player_rect(room, player)
    if player.on_ground and player.jump_buffer_timer > 0:
        player.vy, player.jumping, player.jump_hold_time = room.physics.jump_v0, True, 0.0
        player.jump_buffer_timer, player.on_ground = 0.0, False


def _attack(room: Room, player: Player, buttons: dict[str, bool], dt: float, events: list[Event]) -> None:
    player.attack_cooldown = max(0.0, player.attack_cooldown - dt)
    player.attack_timer = max(0.0, player.attack_timer - dt)
    if buttons["attack"] and not player.attack_was_pressed and player.attack_cooldown <= 0:
        player.attack_timer, player.attack_cooldown = 0.1, 0.35
        player.attack_targets_hit.clear()
    player.attack_was_pressed = buttons["attack"]
    if player.attack_timer <= 0:
        return
    box = player_rect(room, player)
    if buttons["up"]:
        hitbox = Rect(player.x - 0.4, box.top, 0.8, 1.2)
    elif buttons["down"]:
        hitbox = Rect(player.x - 0.4, box.y - 1.2, 0.8, 1.2)
    elif player.facing < 0:
        hitbox = Rect(box.x - 1.2, player.y + 0.2, 1.2, 0.8)
    else:
        hitbox = Rect(box.right, player.y + 0.2, 1.2, 0.8)
    _hit_enemies(room, player, hitbox, events)
    _hit_breakables(room, player, hitbox, events)
    pogo_target = any(hitbox.intersects(h.rect) for h in room.hazards) or any(
        enemy.alive and hitbox.intersects(enemy.rect) for enemy in room.enemies
    )
    if buttons["down"] and not player.on_ground and pogo_target:
        player.vy = room.physics.jump_v0 * 0.9


def _hit_enemies(room: Room, player: Player, hitbox: Rect, events: list[Event]) -> None:
    for i, enemy in enumerate(room.enemies):
        key = ("enemy", i)
        if enemy.alive and key not in player.attack_targets_hit and hitbox.intersects(enemy.rect):
            enemy.hp -= 1
            player.attack_targets_hit.add(key)
            enemy.rect = Rect(enemy.rect.x + player.facing * 0.3, enemy.rect.y, enemy.rect.w, enemy.rect.h)
            if enemy.hp <= 0:
                enemy.alive = False
                events.append(Event("kill", str(i)))


def _hit_breakables(room: Room, player: Player, hitbox: Rect, events: list[Event]) -> None:
    for i, block in enumerate(room.breakables):
        key = ("breakable", i)
        if block.broken or key in player.attack_targets_hit or not hitbox.intersects(block.rect):
            continue
        block.hp -= 1
        player.attack_targets_hit.add(key)
        if block.hp <= 0:
            block.broken = True
            events.append(Event("break", str(i), block.reveals))
            if block.reveals == "pickup":
                room.pickups.append(Pickup(Rect(block.rect.x, block.rect.y, 0.5, 0.5), "coin"))


def _patrol_enemies(room: Room, player: Player, dt: float, events: list[Event]) -> None:
    for enemy in room.enemies:
        if not enemy.alive:
            continue
        direction = 1 if enemy.speed >= 0 else -1
        nx = enemy.rect.x + direction * abs(enemy.speed) * dt
        edge = not _enemy_has_floor(room, enemy, nx + (.1 if direction > 0 else enemy.rect.w - .1))
        if nx < enemy.patrol_left or nx + enemy.rect.w > enemy.patrol_right or edge:
            direction *= -1
            nx = enemy.rect.x + direction * abs(enemy.speed) * dt
            if edge:
                support = _enemy_support(room, enemy)
                if support is not None:
                    nx = min(nx, support.rect.right - enemy.rect.w) if direction < 0 else max(nx, support.rect.x)
        enemy.speed = direction * abs(enemy.speed)
        enemy.rect = Rect(nx, enemy.rect.y, enemy.rect.w, enemy.rect.h)
        if player_rect(room, player).intersects(enemy.rect) and player.invuln_timer <= 0:
            player.hp -= 1
            player.invuln_timer = room.physics.invulnerable_after_hit
            player.vx = -player.facing * 3.0
            events.append(Event("damage", "enemy"))


def _enemy_has_floor(room: Room, enemy: Enemy, x: float) -> bool:
    return any(s.rect.x <= x <= s.rect.right and abs(s.rect.top - enemy.rect.y) <= .25
               for s in room.solids)


def _enemy_support(room: Room, enemy: Enemy) -> Solid | None:
    return next((s for s in room.solids
                 if s.rect.x < enemy.rect.right and s.rect.right > enemy.rect.x
                 and abs(s.rect.top - enemy.rect.y) <= .25), None)


def _hazard_damage_and_death(room: Room, player: Player, dt: float, events: list[Event]) -> None:
    player.invuln_timer = max(0.0, player.invuln_timer - dt)
    for hazard in room.hazards:
        if player_rect(room, player).intersects(hazard.rect) and player.invuln_timer <= 0:
            player.hp -= 1
            player.invuln_timer = room.physics.invulnerable_after_hit
            player.x, player.y = player.last_safe_pos
            player.vx = player.vy = 0.0
            events.append(Event("damage", hazard.kind))
            break
    if player.hp <= 0:
        player.hp = player.max_hp
        player.x, player.y = player.bench_pos or room.spawn
        player.vx = player.vy = 0.0
        player.invuln_timer = room.physics.invulnerable_after_hit
        events.append(Event("death"))


def _collect_and_interact(room: Room, player: Player, buttons: dict[str, bool], events: list[Event]) -> None:
    box = player_rect(room, player)
    for pickup in tuple(room.pickups):
        if not box.intersects(pickup.rect):
            continue
        room.pickups.remove(pickup)
        if pickup.kind.startswith("ability_"):
            player.abilities.add(pickup.kind)
            events.append(Event("ability", pickup.kind))
        elif pickup.kind == "coin":
            player.coins += 1
            events.append(Event("pickup", "coin"))
        else:
            player.hp = min(player.max_hp, player.hp + 1)
            events.append(Event("pickup", "heal"))
    if buttons["interact"] and not player.interact_was_pressed:
        for item in room.interactives:
            if _near_interactive(box, item.rect):
                player.hp, player.bench_pos = player.max_hp, (player.x, player.y)
                events.append(Event("bench"))
                break
    player.interact_was_pressed = buttons["interact"]
    _open_touched_doors(room, player, box, events)


def _record_exit_and_safe_position(room: Room, player: Player, events: list[Event]) -> None:
    box = player_rect(room, player)
    if any(box.intersects(exit_.rect) for exit_ in room.exits):
        events.append(Event("exit"))
    safe = all(not Rect(h.rect.x - 1, h.rect.y - 1, h.rect.w + 2, h.rect.h + 2).intersects(box)
               for h in room.hazards)
    if player.on_ground and safe:
        player.last_safe_pos = (player.x, player.y)


def step_physics(room: Room, player: Player, buttons: dict[str, bool], dt: float) -> list[Event]:
    """Advance one fixed physics step; coordinates use bottom-center player origin."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    keys = ("left", "right", "up", "down", "jump", "attack", "dash", "interact")
    pressed = {name: bool(buttons.get(name, False)) for name in keys}
    events: list[Event] = []
    was_grounded = player.on_ground
    dy, start_x, _ = _prepare_motion(room, player, pressed, dt)
    start_y = player.y
    # A jump edge sets velocity at the current frame boundary; displacement starts next tick.
    if pressed["jump"] and player.jump_hold_time == 0 and player.jumping:
        # Keep any preexisting airborne momentum only when this was not a new launch.
        pass
    old = Rect(start_x - room.physics.player_width / 2, start_y,
               room.physics.player_width, room.physics.player_height)
    blockers = _resolve_x(room, player, old, start_x, dt, events)
    _resolve_y(room, player, blockers, start_y, dy, was_grounded, events)
    _attack(room, player, pressed, dt, events)
    _patrol_enemies(room, player, dt, events)
    _hazard_damage_and_death(room, player, dt, events)
    _collect_and_interact(room, player, pressed, events)
    _record_exit_and_safe_position(room, player, events)
    return events
