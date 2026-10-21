"""Renderizador numpy direto a 192×108 RGB."""

import numpy as np

from voidgym.gyms.platform_gym.styles import RenderStyle, make_style
from voidgym.gyms.platform_gym.world import Player, Rect, Room

WIDTH, HEIGHT = 192, 108


def _pixel_rect(rect: Rect, camera_x: float, camera_y: float, camera_width: float) -> tuple[int, int, int, int]:
    scale = WIDTH / camera_width
    return (max(0, int((rect.x - camera_x) * scale)), max(0, int(HEIGHT - (rect.y + rect.h - camera_y) * scale)),
            min(WIDTH, int((rect.x + rect.w - camera_x) * scale) + 1), min(HEIGHT, int(HEIGHT - (rect.y - camera_y) * scale) + 1))


def _box(image: np.ndarray, rect: Rect, camera_x: float, camera_y: float, camera_width: float, color: tuple[int, int, int], texture_delta: np.ndarray | None = None) -> None:
    x0, y0, x1, y1 = _pixel_rect(rect, camera_x, camera_y, camera_width)
    if x1 > x0 and y1 > y0:
        if texture_delta is None:
            image[y0:y1, x0:x1] = color
        else:
            delta = texture_delta[np.arange(y0, y1)[:, None] % 16, np.arange(x0, x1)[None, :] % 16]
            image[y0:y1, x0:x1] = np.clip(np.asarray(color, dtype=np.int16) + delta, 0, 255).astype(np.uint8)


def _sprite(image: np.ndarray, rect: Rect, camera: tuple[float, float, float], color: tuple[int, int, int], family: str) -> None:
    x0, y0, x1, y1 = _pixel_rect(rect, *camera)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[:y1-y0, :x1-x0]
    nx = (xx + .5) / max(x1 - x0, 1)
    ny = (yy + .5) / max(y1 - y0, 1)
    if family in {"capsule", "horned_ball", "cape"}:
        mask = ((nx - .5) / .48) ** 2 + ((ny - .5) / .5) ** 2 <= 1
    elif family == "triangle":
        mask = (ny >= .08) & (np.abs(nx - .5) <= ny * .5)
    else:
        mask = np.ones((y1-y0, x1-x0), dtype=bool)
    if family == "cape":
        mask |= (nx > .42) & (nx < .9) & (ny > .55) & (ny < .95)
    if family == "horned_ball":
        mask |= ((nx < .32) | (nx > .68)) & (ny < .25)
    patch = image[y0:y1, x0:x1]
    patch[mask] = color
    if family == "square_eyes" and x1 - x0 > 3 and y1 - y0 > 3:
        patch[max(0, (y1-y0)//3-1):(y1-y0)//3+1, (x1-x0)//3:(x1-x0)//3+2] = (255,255,255)


def render(room: Room, player: Player, style: RenderStyle | None = None) -> np.ndarray:
    style = style or make_style(room.style_seed)
    camera_width = style.style_params.camera_width
    camera_x = min(max(player.x - camera_width * .42, 0), max(0, room.width - camera_width))
    camera_y = max(0, player.y - 3)
    camera = (camera_x, camera_y, camera_width)
    image = style.background_layer.copy()
    _draw_geometry(image, room, player, style, camera)
    _draw_objects(image, room, player, style, camera)
    _sprite(image, player.rect(room), camera, style.player, style.style_params.character)
    _fake_hud(image, style.hud)
    if player.invuln_timer > 0 and int(player.invuln_timer * 20) % 2:
        image[::2, ::2] = 255 - image[::2, ::2]
    return image


def _draw_geometry(image: np.ndarray, room: Room, player: Player, style: RenderStyle, camera: tuple[float, float, float]) -> None:
    for solid in room.solids:
        _box(image, solid.rect, *camera, style.platform if solid.kind == "platform" else style.solid, style.texture_delta)
    for block in room.breakables:
        if not block.broken:
            _box(image, block.rect, *camera, style.platform if block.cue else style.solid, style.texture_delta)
    for door in room.doors:
        if not door.opened and door.requires not in player.abilities:
            _box(image, door.rect, *camera, style.solid, style.texture_delta)
    for hazard in room.hazards:
        _hazard(image, hazard.rect, camera, style.hazard, hazard.kind)
    for enemy in room.enemies:
        if enemy.alive:
            _sprite(image, enemy.rect, camera, style.enemy, enemy.kind)


def _draw_objects(image: np.ndarray, room: Room, player: Player, style: RenderStyle, camera: tuple[float, float, float]) -> None:
    for pickup in room.pickups:
        _box(image, pickup.rect, *camera, style.pickup, style.texture_delta)
    for exit_item in room.exits:
        _box(image, exit_item.rect, *camera, style.pickup, style.texture_delta)
    for interactive in room.interactives:
        _box(image, interactive.rect, *camera, style.pickup, style.texture_delta)
        if player.rect(room).expanded(.5).intersects(interactive.rect):
            prompt = Rect(interactive.rect.x + .3, interactive.rect.y + interactive.rect.h + .45, .35, .35)
            _box(image, prompt, *camera, (255, 255, 235))


def _fake_hud(image: np.ndarray, items: tuple[tuple[int, int, int, int, tuple[int, int, int]], ...]) -> None:
    for x, y, width, height, color in items:
        image[y:y+height, x:x+width] = color


def _hazard(image: np.ndarray, rect: Rect, camera: tuple[float, float, float], color: tuple[int, int, int], kind: str) -> None:
    x0, y0, x1, y1 = _pixel_rect(rect, *camera)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[:y1-y0, :x1-x0]
    nx = (xx + .5) / (x1 - x0)
    ny = (yy + .5) / (y1 - y0)
    if kind == "acid":
        mask = ny > .35
    else:
        mask = ny >= np.abs(nx - .5) * (4 if kind == "thorns" else 2)
    patch = image[y0:y1, x0:x1]
    patch[mask] = color
