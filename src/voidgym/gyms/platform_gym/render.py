"""Fast direct 192x108 NumPy renderer for invented platform rooms."""

from __future__ import annotations

from functools import lru_cache
from typing import cast

import numpy as np

from voidgym.gyms.platform_gym.params import Split
from voidgym.gyms.platform_gym.physics import interaction_prompt, player_rect
from voidgym.gyms.platform_gym.styles import Palette, sample_style
from voidgym.gyms.platform_gym.world import Player, Rect, Room

WIDTH, HEIGHT = 192, 108
_YGRID, _XGRID = np.mgrid[0:HEIGHT, 0:WIDTH]
_Y_ROWS = np.arange(HEIGHT)[:, None]


# Episode seeds are intentionally unique. Keep only a small recent window so
# persistent spawned workers do not retain every episode's image arrays.
@lru_cache(maxsize=64)
def _static_effects(seed: int, split: Split) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, bool]:
    """Precompute seeded particles, lighting, fog, noise and scanline for each room."""
    style, _ = sample_style(seed, split=split)
    rng = np.random.default_rng(seed)
    particle_count = int(rng.integers(0, 41))
    px = rng.integers(0, WIDTH, particle_count)
    py = rng.integers(0, HEIGHT, particle_count)
    light_x, light_y = rng.uniform(0, WIDTH), rng.uniform(0, HEIGHT)
    radius = np.sqrt(((_XGRID - light_x) / WIDTH) ** 2 + ((_YGRID - light_y) / HEIGHT) ** 2)
    light = np.clip(1 + (style.light_strength - .5) * .36 - np.maximum(0, radius - .45) * .12,
                    .8, 1.2).astype(np.float32)
    if style.fog_strength:
        fog = np.linspace(1 - style.fog_strength * .15, 1, HEIGHT, dtype=np.float32)
        light *= fog[:, None]
    noise_level = int(rng.integers(0, 4))
    noise = (rng.integers(-noise_level, noise_level + 1, (HEIGHT, WIDTH, 3), dtype=np.int16)
             if noise_level else None)
    return px, py, light, noise, bool(rng.random() < .25)


@lru_cache(maxsize=2048)
def _background_ridge(layer: int, offset: int) -> np.ndarray:
    xs = np.arange(WIDTH) + offset
    ridge = (np.sin(xs * (.018 + layer * .007) + layer) * 8 +
             np.sin(xs * (.043 + layer * .01)) * 4).astype(np.int32)
    return cast(np.ndarray, ridge)


@lru_cache(maxsize=128)
def _background_frame(seed: int, split: Split, background: tuple[int, int, int],
                      offsets: tuple[int, ...]) -> np.ndarray:
    image = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:] = background
    for layer, offset in enumerate(offsets):
        y = int(HEIGHT * (.40 + .13 * layer))
        col: tuple[int, int, int] = tuple(int(c * (.55 + .1 * layer)) for c in background)  # type: ignore[assignment]
        skyline = np.clip(y - _background_ridge(layer, offset), 0, HEIGHT)
        image[_Y_ROWS >= skyline[None, :]] = col
    _paint_backdrop_motif(image, seed, background)
    px, py, _, _, _ = _static_effects(seed, split)
    if len(px):
        image[py, px] = np.minimum(255, image[py, px].astype(np.uint16) + 40).astype(np.uint8)
    return cast(np.ndarray, image)


def _paint_backdrop_motif(image: np.ndarray, seed: int,
                          background: tuple[int, int, int]) -> None:
    """Add a seeded architectural or organic silhouette behind the playable room."""
    rng = np.random.default_rng(seed ^ 0x5A17)
    motif = int(rng.integers(0, 4))
    count = int(rng.integers(3, 8))
    for _ in range(count):
        x = int(rng.integers(-8, WIDTH - 2))
        width = int(rng.integers(5, 19))
        top = int(rng.integers(28, 78))
        color = tuple(int(c * rng.uniform(.48, .72)) for c in background)
        x0, x1 = max(0, x), min(WIDTH, x + width)
        if x1 <= x0:
            continue
        if motif == 0:  # city blocks and narrow towers
            image[top:, x0:x1] = color
            if top > 3:
                image[top - 3:top, x0:x1] = color
        elif motif == 1:  # crystal spires
            for row in range(top, HEIGHT):
                half = max(1, int((row - top) * width / max(1, HEIGHT - top)))
                center = (x0 + x1) // 2
                image[row, max(x0, center - half):min(x1, center + half + 1)] = color
        elif motif == 2:  # archways with open centers
            image[top:, x0:min(x1, x0 + 3)] = color
            image[top:, max(x0, x1 - 3):x1] = color
            image[top:min(HEIGHT, top + 4), x0:x1] = color
        else:  # tapered trunks and roots
            center = (x0 + x1) // 2
            for row in range(top, HEIGHT):
                half = max(2, width // 5 + (row - top) // 12)
                image[row, max(x0, center - half):min(x1, center + half)] = color


def _paint_rect(image: np.ndarray, rect: Rect, camera: tuple[float, float, float],
                color: tuple[int, int, int], *, texture: str = "smooth") -> None:
    cx, cy, scale = camera
    x0 = max(0, int((rect.x - cx) * scale + WIDTH / 2))
    x1 = min(WIDTH, int(np.ceil((rect.right - cx) * scale + WIDTH / 2)))
    y0 = max(0, int(HEIGHT / 2 - (rect.top - cy) * scale))
    y1 = min(HEIGHT, int(np.ceil(HEIGHT / 2 - (rect.y - cy) * scale)))
    if x1 <= x0 or y1 <= y0:
        return
    if texture == "smooth":
        image[y0:y1, x0:x1] = color
        return
    tile = _texture_rgb_tile(texture, color)
    rows, cols = _tile_indices(y1 - y0, x1 - x0)
    image[y0:y1, x0:x1] = tile[rows, cols]


@lru_cache(maxsize=8)
def _texture_tile(texture: str) -> np.ndarray:
    """Create and cache a seamless 32x32 grayscale texture tile."""
    yy, xx = np.indices((32, 32))
    if texture == "noise":
        shade = .85 + ((xx * 73 + yy * 151) % 29) / 100
    elif texture == "bricks":
        seam = (yy % 8 == 0) | ((xx + (yy // 8 % 2) * 8) % 16 == 0)
        shade = np.where(seam, .52, 1.0)
    elif texture == "rounded_stones":
        dx = (xx % 16) - 8 + ((yy // 16) % 2) * 4
        dy = (yy % 16) - 8
        shade = np.where(dx * dx / 36 + dy * dy / 9 > .76, .58, 1.06)
    elif texture == "wood_planks":
        grain = ((xx * 17 + (yy // 8) * 23) % 31) / 90
        shade = np.where(yy % 8 == 0, .52, .83 + grain)
    elif texture == "metal_grid":
        shade = np.where((xx % 8 == 0) | (yy % 8 == 0), 1.32, .72)
    elif texture == "moss":
        spots = ((xx * 11 + yy * 17) % 37) < 11
        shade = np.where(spots, 1.3, .9)
    elif texture == "diagonal_stripes":
        shade = np.where((xx + yy) % 9 < 3, 1.24, .72)
    elif texture == "crystals":
        diamond = np.abs((xx % 16) - 8) + np.abs((yy % 16) - 8) < 5
        shade = np.where(diamond, 1.45, .65)
    else:  # hexagons
        col = xx % 16
        row = yy % 16
        seam = (row == 0) | (row == 9) | ((col == 0) & (row < 5)) | ((col == 6) & (row >= 5))
        shade = np.where(seam, .55, 1.08)
    return cast(np.ndarray, np.asarray(shade, dtype=np.float32))


@lru_cache(maxsize=512)
def _tile_indices(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    return (np.arange(height, dtype=np.intp)[:, None] % 32,
            np.arange(width, dtype=np.intp)[None, :] % 32)


@lru_cache(maxsize=256)
def _texture_rgb_tile(texture: str, color: tuple[int, int, int]) -> np.ndarray:
    shade = _texture_tile(texture)
    return cast(np.ndarray, np.clip(np.asarray(color)[None, None, :] * shade[:, :, None],
                                    0, 255).astype(np.uint8))


def _paint_character(image: np.ndarray, rect: Rect, camera: tuple[float, float, float],
                     color: tuple[int, int, int], family: str, facing: int) -> None:
    cx, cy, scale = camera
    x0 = max(0, int((rect.x - cx) * scale + WIDTH / 2))
    x1 = min(WIDTH, int(np.ceil((rect.right - cx) * scale + WIDTH / 2)))
    y0 = max(0, int(HEIGHT / 2 - (rect.top - cy) * scale))
    y1 = min(HEIGHT, int(np.ceil(HEIGHT / 2 - (rect.y - cy) * scale)))
    if x1 <= x0 or y1 <= y0:
        return
    mask = _character_shape(y1 - y0, x1 - x0, family)
    patch = image[y0:y1, x0:x1]
    patch[mask] = color
    if family in {"horned_ball", "eyed_square"}:
        eye_x = .66 if facing > 0 else .34
        ex, ey = int(eye_x * (x1 - x0)), int(.38 * (y1 - y0))
        patch[max(0, ey - 1):ey + 2, max(0, ex - 1):ex + 2] = (245, 245, 245)


def _paint_hazard(image: np.ndarray, rect: Rect, camera: tuple[float, float, float],
                  color: tuple[int, int, int], kind: str) -> None:
    _paint_rect(image, rect, camera, color, texture="crystals" if kind != "acid" else "moss")
    x0 = max(0, int((rect.x - camera[0]) * camera[2] + WIDTH / 2))
    x1 = min(WIDTH, int(np.ceil((rect.right - camera[0]) * camera[2] + WIDTH / 2)))
    y0 = max(0, int(HEIGHT / 2 - (rect.top - camera[1]) * camera[2]))
    y1 = min(HEIGHT, int(np.ceil(HEIGHT / 2 - (rect.y - camera[1]) * camera[2])))
    if x1 <= x0 or y1 <= y0:
        return
    if kind == "acid":
        image[y0:y1, x0:x1][_hazard_shape(y1 - y0, x1 - x0, kind)] = (245, 245, 185)
    else:
        image[y0:y1, x0:x1][~_hazard_shape(y1 - y0, x1 - x0, kind)] = np.asarray(color, dtype=np.uint8) // 3


@lru_cache(maxsize=128)
def _hazard_shape(height: int, width: int, kind: str) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    if kind == "acid":
        return cast(np.ndarray, ((xx * 7 + yy * 11) % 31) < 3)
    nx, ny = (xx % 10 + .5) / 10, (yy + .5) / height
    points = (np.abs(nx - .5) < ny * .5) & ((xx // 10) % 2 == 0)
    return cast(np.ndarray, points)


def _paint_enemy(image: np.ndarray, rect: Rect, camera: tuple[float, float, float],
                 color: tuple[int, int, int], kind: str) -> None:
    cx, cy, scale = camera
    x0 = max(0, int((rect.x - cx) * scale + WIDTH / 2))
    x1 = min(WIDTH, int(np.ceil((rect.right - cx) * scale + WIDTH / 2)))
    y0 = max(0, int(HEIGHT / 2 - (rect.top - cy) * scale))
    y1 = min(HEIGHT, int(np.ceil(HEIGHT / 2 - (rect.y - cy) * scale)))
    if x1 <= x0 or y1 <= y0:
        return
    patch = image[y0:y1, x0:x1]
    patch[_enemy_shape(y1 - y0, x1 - x0, kind)] = color
    if kind == "eye":
        patch[max(0, int((y1-y0)*.4)-1):int((y1-y0)*.4)+2,
              max(0, int((x1-x0)*.55)-1):int((x1-x0)*.55)+2] = (245, 245, 245)


@lru_cache(maxsize=256)
def _enemy_shape(height: int, width: int, kind: str) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx, ny = (xx + .5) / width, (yy + .5) / height
    if kind in {"orb", "bubble"}:
        mask = ((nx - .5) / .5) ** 2 + ((ny - .5) / .5) ** 2 <= 1
    elif kind == "spiky":
        mask = ((xx + yy) % 7 < 3) | ((xx - yy) % 7 < 3)
    elif kind == "eye":
        mask = (nx > .05) & (nx < .95) & (ny > .18) & (ny < .82)
    else:
        mask = (ny > .18) & (ny < .95) & ((nx - .5) ** 2 < .24)
    return cast(np.ndarray, mask)


def _paint_platform_edge(image: np.ndarray, rect: Rect, camera: tuple[float, float, float],
                         color: tuple[int, int, int], style: str) -> None:
    x0 = max(0, int((rect.x - camera[0]) * camera[2] + WIDTH / 2))
    x1 = min(WIDTH, int(np.ceil((rect.right - camera[0]) * camera[2] + WIDTH / 2)))
    y = int(HEIGHT / 2 - (rect.top - camera[1]) * camera[2])
    if x1 <= x0 or not 0 <= y < HEIGHT:
        return
    edge = np.asarray(color, dtype=np.uint8)
    if style == "moss":
        edge = np.asarray((80, 180, 100), dtype=np.uint8)
    elif style == "shadow":
        y = min(HEIGHT - 1, y + 2)
        edge = (edge.astype(np.uint16) * .45).astype(np.uint8)
    xs = np.arange(x0, x1)
    if style == "irregular":
        ys = np.clip(y + ((xs * 13 + rect.x.__int__()) % 3 == 0), 0, HEIGHT - 1)
        image[ys, xs] = edge
    elif style == "rounded":
        image[y:min(HEIGHT, y + 2), x0:x1] = edge
    elif style == "moss":
        image[y, xs[(xs * 7) % 9 < 5]] = edge
    else:
        image[y, x0:x1] = edge


@lru_cache(maxsize=256)
def _character_shape(height: int, width: int, family: str) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx + .5) / width
    ny = (yy + .5) / height
    if family in {"capsule", "horned_ball"}:
        mask = ((nx - .5) / .48) ** 2 + ((ny - .5) / .48) ** 2 <= 1
    elif family == "triangle":
        mask = (ny >= .05) & (nx >= .5 - ny * .52) & (nx <= .5 + ny * .52)
    else:
        mask = (ny > .08) & (ny < .98) & (nx > .06) & (nx < .94)
        if family == "cape_rectangle":
            mask |= (ny > .28) & (ny < .95) & (np.abs(nx - .5) < (.25 + ny * .22))
    return cast(np.ndarray, mask)


def player_pixel_mask(room: Room, player: Player,
                      camera: tuple[float, float, float] | None = None,
                      *, split: Split = "train") -> np.ndarray:
    """Return the player sprite's visible pixels at the same camera/style as render_room."""
    if player.invuln_timer > 0 and int(player.invuln_timer * 16) % 2 == 0:
        return np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
    style, _ = sample_style(room.style_seed, split=split)
    scale = WIDTH / style.camera_width
    cam = camera or (player.x, player.y + 1.0, scale)
    rect = player_rect(room, player)
    x0 = max(0, int((rect.x - cam[0]) * cam[2] + WIDTH / 2))
    x1 = min(WIDTH, int(np.ceil((rect.right - cam[0]) * cam[2] + WIDTH / 2)))
    y0 = max(0, int(HEIGHT / 2 - (rect.top - cam[1]) * cam[2]))
    y1 = min(HEIGHT, int(np.ceil(HEIGHT / 2 - (rect.y - cam[1]) * cam[2])))
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = _character_shape(y1 - y0, x1 - x0, style.character_family)
    return mask


def render_room(room: Room, player: Player, camera: tuple[float, float, float] | None = None,  # noqa: C901
                *, palette: Palette | None = None, style_seed: int | None = None,
                split: Split = "train") -> np.ndarray:
    seed = room.style_seed if style_seed is None else style_seed
    style, sampled_palette = sample_style(seed, split=split)
    texture_id = style.texture_family
    colors = palette or sampled_palette
    scale = WIDTH / style.camera_width
    cam = camera or (player.x, player.y + 1.0, scale)
    offsets = tuple(int(cam[0] * scale * (.05 + layer * .04))
                    for layer in range(style.background_layers))
    img = _background_frame(seed, split, colors.background, offsets).copy()
    _, _, light, noise, scanline = _static_effects(seed, split)
    for solid in room.solids:
        _paint_rect(img, solid.rect, cam, colors.platform if solid.kind == "platform" else colors.ground, texture=texture_id)
        _paint_platform_edge(img, solid.rect, cam, colors.foreground, style.edge_style)
    for door in room.doors:
        if not door.opened:
            _paint_rect(img, door.rect, cam, colors.foreground, texture="metal_grid")
    for block in room.breakables:
        if not block.broken:
            color = cast(tuple[int, int, int], tuple(max(0, c - (18 if block.cue else 0)) for c in colors.ground))
            _paint_rect(img, block.rect, cam, color, texture="crystals" if block.cue else "smooth")
    for hazard in room.hazards:
        _paint_hazard(img, hazard.rect, cam, colors.hazard, hazard.kind)
    for enemy in room.enemies:
        if enemy.alive:
            _paint_enemy(img, enemy.rect, cam, colors.enemy, enemy.kind)
    for pickup in room.pickups:
        _paint_rect(img, pickup.rect, cam, colors.pickup, texture="smooth")
    for item in room.interactives:
        _paint_rect(img, item.rect, cam, colors.foreground, texture="smooth")
    for exit_ in room.exits:
        _paint_rect(img, exit_.rect, cam, colors.foreground, texture="smooth")
    if not (player.invuln_timer > 0 and int(player.invuln_timer * 16) % 2 == 0):
        _paint_character(img, player_rect(room, player), cam, colors.player, style.character_family, player.facing)
    if interaction_prompt(room, player):
        sx = int(WIDTH / 2)
        img[7:12, max(0, sx - 2):sx + 2] = colors.pickup
        img[7:9, max(0, sx - 2):sx + 2] = colors.foreground
    for icon in range(style.hud_icons):
        hx = 5 + (icon % 3) * 11
        hy = 5 + (icon // 3) * 8
        img[hy:hy + 4, hx:hx + 4] = (colors.player, colors.pickup, colors.hazard,
                                      colors.enemy, colors.platform)[icon]
        img[hy + 5:hy + 6, hx:hx + 8] = colors.foreground
    img = np.clip(img.astype(np.float32) * light[:, :, None] +
                  (noise if noise is not None else 0), 0, 255).astype(np.uint8)
    if scanline:
        img[::3] = (img[::3].astype(np.float32) * .92).astype(np.uint8)
    return cast(np.ndarray, img)
