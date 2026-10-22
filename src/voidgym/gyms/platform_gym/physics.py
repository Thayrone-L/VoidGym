"""Física AABB determinista executada em passos de 20 ms."""

from dataclasses import dataclass

from voidgym.gyms.platform_gym.params import PHYSICS_DT, PhysicsParams
from voidgym.gyms.platform_gym.world import Pickup, Player, Rect, Room


@dataclass(frozen=True)
class Event:
    kind: str
    detail: str = ""


def step_physics(room: Room, player: Player, buttons: dict[str, bool], dt: float = PHYSICS_DT) -> list[Event]:
    """Avança uma simulação interna, resolvendo movimento e eventos."""
    physics = room.physics if isinstance(room.physics, PhysicsParams) else PhysicsParams()
    events: list[Event] = []
    was_grounded = player.on_ground
    player.invuln_timer = max(0.0, player.invuln_timer - dt)
    player.attack_cooldown = max(0.0, player.attack_cooldown - dt)
    player.jump_buffer = .08 if buttons.get("jump", False) and not player.jumping else max(0, player.jump_buffer - dt)
    player.coyote_time = .08 if was_grounded else max(0, player.coyote_time - dt)
    if was_grounded:
        player.double_jump_used = False
        player.dash_ready = True
    player.vx = physics.run_speed * (int(buttons.get("right", False)) - int(buttons.get("left", False)))
    if player.vx:
        player.facing = 1 if player.vx > 0 else -1
    launched = False
    if buttons.get("jump", False) and player.jumping and player.jump_hold_time < physics.hold_max:
        player.vy = physics.jump_v0
        player.jump_hold_time += dt
    elif player.jump_buffer and (was_grounded or player.coyote_time > 0 or "ability_double_jump" in player.abilities and not player.double_jump_used):
        if not was_grounded and player.coyote_time <= 0:
            player.double_jump_used = True
        player.vy = physics.jump_v0
        player.jumping = True
        player.jump_hold_time = 0.0
        launched = True
        player.jump_buffer = 0.0
    else:
        player.jumping = False
        player.vy = max(-physics.max_fall_speed, player.vy - physics.gravity * dt)
    if buttons.get("dash", False) and "ability_dash" in player.abilities and player.dash_ready:
        player.vx = player.facing * physics.dash_distance / .15
        player.vy = 0.0
        player.dash_time = .15
        player.dash_ready = False
    elif player.dash_time > 0:
        player.vx = player.facing * physics.dash_distance / .15
        player.vy = 0.0
        player.dash_time = max(0, player.dash_time - dt)
    _move_x(room, player, dt)
    gravity_active = player.dash_time <= 0 and (not buttons.get("jump", False) or player.jump_hold_time >= physics.hold_max)
    landed = _move_y(room, player, dt, was_grounded, physics.gravity if gravity_active and player.vy < physics.jump_v0 else 0.0, launched)
    if landed:
        events.append(Event("landed"))
    if player.on_ground and not any(h.rect.intersects(player.rect(room).expanded(1)) for h in room.hazards):
        player.last_safe_pos = (player.x, player.y)
    _attack(room, player, buttons, physics, events)
    _damage(room, player, events, physics)
    _interactions(room, player, buttons, events)
    for exit_item in room.exits:
        if player.rect(room).intersects(exit_item.rect):
            events.append(Event("exit", str(exit_item.to_room)))
    _move_enemies(room, dt)
    return events


def _solids(room: Room, player: Player) -> list[tuple[Rect, bool]]:
    shapes = [(item.rect, item.kind == "platform") for item in room.solids]
    shapes.extend((item.rect, False) for item in room.breakables if not item.broken)
    shapes.extend((item.rect, False) for item in room.doors if not item.opened and item.requires not in player.abilities)
    return shapes


def _move_x(room: Room, player: Player, dt: float) -> None:
    player.x += player.vx * dt
    body = player.rect(room)
    for rect, one_way in _solids(room, player):
        if one_way:
            continue
        if body.intersects(rect):
            player.x = rect.x - body.w if player.vx > 0 else rect.x + rect.w
            player.vx = 0
            body = player.rect(room)


def _move_y(room: Room, player: Player, dt: float, was_grounded: bool, gravity: float, launched: bool) -> bool:
    old_y = player.y
    if not launched:
        player.y += (player.vy + gravity * dt / 2) * dt
    body = player.rect(room)
    player.on_ground = False
    for rect, one_way in _solids(room, player):
        if one_way and (player.vy > 0 or old_y < rect.y + rect.h - .02):
            continue
        if body.intersects(rect):
            if player.vy <= 0:
                player.y = rect.y + rect.h
                player.on_ground = True
            else:
                player.y = rect.y - body.h
            player.vy = 0
            body = player.rect(room)
    return player.on_ground and not was_grounded


def _attack(room: Room, player: Player, buttons: dict[str, bool], physics: PhysicsParams, events: list[Event]) -> None:
    if not buttons.get("attack") or player.attack_cooldown > 0:
        return
    aim = "up" if buttons.get("up") else "down" if buttons.get("down") else "side"
    body = player.rect(room)
    hit = Rect(body.x, body.y + body.h, 1.2, .8) if aim == "up" else Rect(body.x, body.y - .8, 1.2, .8) if aim == "down" else Rect(body.x + (body.w if player.facing > 0 else -1.2), body.y + .2, 1.2, .8)
    for enemy in room.enemies:
        if enemy.alive and hit.intersects(enemy.rect):
            enemy.hp -= 1
            if enemy.hp <= 0:
                enemy.alive = False
                events.append(Event("kill"))
    for block in room.breakables:
        if not block.broken and hit.intersects(block.rect):
            block.hp -= 1
            if block.hp <= 0:
                block.broken = True
                events.append(Event("break", f"{block.reveals}:{block.rect.x:.2f}"))
                if block.reveals == "pickup":
                    room.pickups.append(Pickup(Rect(block.rect.x, block.rect.y, .6, .6), "coin"))
    for enemy in room.enemies:
        if enemy.alive and hit.intersects(enemy.rect):
            enemy.rect = Rect(enemy.rect.x + player.facing * .2, enemy.rect.y, enemy.rect.w, enemy.rect.h)
    target_hit = any(enemy.alive and hit.intersects(enemy.rect) for enemy in room.enemies)
    target_hit = target_hit or any(hit.intersects(hazard.rect) for hazard in room.hazards)
    if aim == "down" and not player.on_ground and target_hit:
        player.vy = physics.jump_v0 * .9
    player.attack_cooldown = .35


def _damage(room: Room, player: Player, events: list[Event], physics: PhysicsParams) -> None:
    body = player.rect(room)
    hazard_hit = any(body.intersects(item.rect) for item in room.hazards)
    enemy_hit = any(item.alive and body.intersects(item.rect) for item in room.enemies)
    if (hazard_hit or enemy_hit) and player.invuln_timer <= 0:
        player.hp -= 1
        player.invuln_timer = physics.invulnerable_after_hit
        events.append(Event("damage"))
        if hazard_hit:
            player.x, player.y = player.last_safe_pos
        elif enemy_hit:
            player.x -= player.facing * .2
        if player.hp <= 0:
            events.append(Event("death"))
            player.x, player.y = player.respawn or room.spawn
            player.hp = player.max_hp


def _interactions(room: Room, player: Player, buttons: dict[str, bool], events: list[Event]) -> None:
    body = player.rect(room)
    for pickup in room.pickups[:]:
        if body.intersects(pickup.rect):
            room.pickups.remove(pickup)
            if pickup.kind.startswith("ability_"):
                player.abilities.add(pickup.kind)
                events.append(Event("ability", f"{pickup.kind}:{pickup.rect.x:.2f}"))
            else:
                if pickup.kind == "heal":
                    player.hp = player.max_hp
                events.append(Event("pickup", f"{pickup.kind}:{pickup.rect.x:.2f}"))
    for bench in room.interactives:
        if buttons.get("interact") and body.expanded(.5).intersects(bench.rect):
            player.hp = player.max_hp
            player.respawn = (player.x, player.y)
            events.append(Event("bench"))
    for door in room.doors:
        if door.requires in player.abilities and body.intersects(door.rect):
            door.opened = True
            events.append(Event("door_open", door.requires))


def _move_enemies(room: Room, dt: float) -> None:
    for enemy in room.enemies:
        if not enemy.alive:
            continue
        enemy.rect = Rect(enemy.rect.x + enemy.speed * dt, enemy.rect.y, enemy.rect.w, enemy.rect.h)
        if enemy.rect.x < enemy.patrol_left or enemy.rect.x + enemy.rect.w > enemy.patrol_right:
            enemy.speed *= -1
            enemy.rect = Rect(min(max(enemy.rect.x, enemy.patrol_left), enemy.patrol_right - enemy.rect.w), enemy.rect.y, enemy.rect.w, enemy.rect.h)


