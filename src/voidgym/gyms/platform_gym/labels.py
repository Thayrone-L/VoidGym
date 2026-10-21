"""Gabarito semântico alinhado às caixas físicas e independente do estilo."""

import numpy as np

from voidgym.core.semantics import SEM_GRID, Sem
from voidgym.gyms.platform_gym.params import PhysicsParams, StyleParams
from voidgym.gyms.platform_gym.render import HEIGHT, WIDTH
from voidgym.gyms.platform_gym.styles import RenderStyle
from voidgym.gyms.platform_gym.world import Player, Rect, Room

_SEMANTIC_PRIORITY = np.asarray([0, 2, 1, 7, 8, 9, 6, 5, 4, 3], dtype=np.uint8)


def _paint(grid: np.ndarray, rect: Rect, semantic: Sem, camera: tuple[float, float, float], priority: dict[Sem, int]) -> None:
    camera_x, camera_y, camera_width = camera
    pixels_per_unit = WIDTH / camera_width
    x0 = max(0, int((rect.x - camera_x) * pixels_per_unit / 4))
    x1 = min(SEM_GRID[1], int(np.ceil((rect.x + rect.w - camera_x) * pixels_per_unit / 4)))
    y0 = max(0, int((HEIGHT - (rect.y + rect.h - camera_y) * pixels_per_unit) / 4))
    y1 = min(SEM_GRID[0], int(np.ceil((HEIGHT - (rect.y - camera_y) * pixels_per_unit) / 4)))
    view = grid[y0:y1, x0:x1]
    replace = _SEMANTIC_PRIORITY[view] <= priority[semantic]
    view[replace] = semantic


def labels(room: Room, player: Player, camera_x: float = 0.0, camera_y: float = 0.0, style: RenderStyle | None = None) -> dict[str, object]:
    del style
    camera_width = StyleParams.sample(room.style_seed).camera_width
    camera = (camera_x, camera_y, camera_width)
    grid = np.full(SEM_GRID, Sem.EMPTY, dtype=np.uint8)
    priority = {
        Sem.PLAYER: 9, Sem.ENEMY: 8, Sem.HAZARD: 7, Sem.BREAKABLE: 6,
        Sem.PICKUP: 5, Sem.INTERACTIVE: 4, Sem.EXIT: 3, Sem.SOLID: 2,
        Sem.PLATFORM: 1, Sem.EMPTY: 0,
    }
    for solid in room.solids:
        _paint(grid, solid.rect, Sem.PLATFORM if solid.kind == "platform" else Sem.SOLID, camera, priority)
    for door in room.doors:
        if not door.opened and door.requires not in player.abilities:
            _paint(grid, door.rect, Sem.SOLID, camera, priority)
    for block in room.breakables:
        if not block.broken:
            _paint(grid, block.rect, Sem.BREAKABLE, camera, priority)
    for hazard in room.hazards:
        _paint(grid, hazard.rect, Sem.HAZARD, camera, priority)
    for enemy in room.enemies:
        if enemy.alive:
            _paint(grid, enemy.rect, Sem.ENEMY, camera, priority)
    for pickup in room.pickups:
        _paint(grid, pickup.rect, Sem.PICKUP, camera, priority)
    for interactive in room.interactives:
        _paint(grid, interactive.rect, Sem.INTERACTIVE, camera, priority)
    for exit_item in room.exits:
        _paint(grid, exit_item.rect, Sem.EXIT, camera, priority)
    _paint(grid, player.rect(room), Sem.PLAYER, camera, priority)
    physics = room.physics if isinstance(room.physics, PhysicsParams) else PhysicsParams()
    return {
        "sem": grid,
        "hero_screen_xy": np.asarray([np.clip((player.x - camera_x) / camera_width, 0, 1), np.clip((player.y - camera_y) / camera_width, 0, 1)], dtype=np.float32),
        "hero_velocity": np.asarray([player.vx / physics.run_speed, player.vy / physics.run_speed], dtype=np.float32),
        "on_ground": int(player.on_ground),
        "interact_prompt": int(any(player.rect(room).expanded(.5).intersects(item.rect) for item in room.interactives)),
    }
