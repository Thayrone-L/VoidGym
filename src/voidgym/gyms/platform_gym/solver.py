"""Roteiro de prova para as salas geradas, executado com a física compartilhada."""

import copy

from voidgym.gyms.platform_gym.params import SUBSTEPS, PhysicsParams
from voidgym.gyms.platform_gym.physics import step_physics
from voidgym.gyms.platform_gym.world import Player, Room


def solve_room(room: Room, max_steps: int = 20_000, trace: list[tuple[float, float]] | None = None, collected_abilities: set[str] | None = None) -> bool:
    """Percorre o piso seguro e aplica saltos de margem ao detectar obstáculo."""
    room = copy.deepcopy(room)
    physics = room.physics if isinstance(room.physics, PhysicsParams) else PhysicsParams()
    player = Player(*room.spawn)
    player.abilities.update(collected_abilities or set())
    player.last_safe_pos = room.spawn
    if trace is not None:
        trace.append((player.x, player.y))
    steps = 0
    while steps < max_steps:
        if any(player.rect(room).intersects(exit_item.rect) for exit_item in room.exits):
            if collected_abilities is not None:
                collected_abilities.update(player.abilities)
            return True
        buttons = {"right": True}
        ahead = player.x + .9
        obstacle = any(s.kind == "solid" and s.rect.x <= ahead <= s.rect.x + s.rect.w and s.rect.y + s.rect.h > player.y + .5 for s in room.solids)
        obstacle = obstacle or any(h.rect.x <= ahead <= h.rect.x + h.rect.w for h in room.hazards)
        obstacle = obstacle or any(e.alive and e.rect.x - 1 <= ahead <= e.rect.x + e.rect.w + 1 for e in room.enemies)
        gap = not any(s.rect.x <= ahead <= s.rect.x + s.rect.w and s.rect.y <= .5 for s in room.solids)
        if (obstacle or gap or player.y < 0) and player.on_ground or player.jumping and player.jump_hold_time < physics.hold_max:
            buttons["jump"] = True
        for _ in range(SUBSTEPS):
            step_physics(room, player, buttons)
            steps += 1
            if trace is not None and steps % SUBSTEPS == 0:
                trace.append((player.x, player.y))
            if player.y < -4 or player.hp < player.max_hp - 2:
                return False
            if any(player.rect(room).intersects(exit_item.rect) for exit_item in room.exits):
                if collected_abilities is not None:
                    collected_abilities.update(player.abilities)
                return True
    return False


def solve_set(rooms: list[Room]) -> bool:
    """Verifica que cada sala conectada tenha uma solução local."""
    abilities: set[str] = set()
    return all(solve_room(room, collected_abilities=abilities) for room in rooms)
