"""Per-frame semantic oracle, separate from the policy observation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from voidgym.core.semantics import SEM_GRID, Sem
from voidgym.gyms.platform_gym.params import Split
from voidgym.gyms.platform_gym.physics import interaction_prompt
from voidgym.gyms.platform_gym.render import player_pixel_mask
from voidgym.gyms.platform_gym.styles import sample_style
from voidgym.gyms.platform_gym.world import Breakable, Player, Rect, Room, TreasureRock


def _stamp_entities(room: Room, stamp: Callable[[Rect, Sem], None]) -> None:
    for solid in room.solids:
        stamp(solid.rect, Sem.PLATFORM if solid.kind == "platform" else Sem.SOLID)
    for door in room.doors:
        if not door.opened:
            stamp(door.rect, Sem.SOLID)
    for exit_ in room.exits:
        stamp(exit_.rect, Sem.EXIT)
    for item in room.interactives:
        stamp(item.rect, Sem.INTERACTIVE)
    for pickup in room.pickups:
        stamp(pickup.rect, Sem.PICKUP)
    _stamp_breakables(room, stamp)
    for hazard in room.hazards:
        stamp(hazard.rect, Sem.HAZARD)
    for enemy in room.enemies:
        if enemy.alive:
            stamp(enemy.rect, Sem.ENEMY)


def _stamp_breakables(room: Room, stamp: Callable[[Rect, Sem], None]) -> None:
    items: tuple[Breakable | TreasureRock, ...] = (
        *room.breakables,
        *room.treasure_rocks,
    )
    for item in items:
        if not item.broken:
            stamp(item.rect, Sem.BREAKABLE)


def labels(room: Room, player: Player,
           camera: tuple[float, float, float] | None = None,
           *, split: Split = "train") -> dict[str, np.ndarray | float | int]:
    style, _ = sample_style(room.style_seed, split=split)
    scale = 192 / style.camera_width if camera is None else camera[2]
    cx, cy = (player.x, player.y + 1.0) if camera is None else camera[:2]
    sem = np.full(SEM_GRID, int(Sem.EMPTY), dtype=np.uint8)

    def stamp(rect: Rect, klass: Sem) -> None:
        x0 = int(np.floor(((rect.x - cx) * scale + 96) / 4))
        x1 = int(np.ceil(((rect.right - cx) * scale + 96) / 4))
        y0 = int(np.floor((54 - (rect.top - cy) * scale) / 4))
        y1 = int(np.ceil((54 - (rect.y - cy) * scale) / 4))
        sem[max(0, y0):min(27, y1), max(0, x0):min(48, x1)] = int(klass)

    _stamp_entities(room, stamp)
    hero_mask = player_pixel_mask(room, player, (cx, cy, scale), split=split)
    player_cells = hero_mask.reshape(27, 4, 48, 4).any(axis=(1, 3))
    sem[player_cells] = int(Sem.PLAYER)
    x_norm = float(np.clip((player.x - cx) * scale / 192 + .5, 0, 1))
    y_norm = float(np.clip(.5 - ((player.y + room.physics.player_height / 2) - cy) * scale / 108, 0, 1))
    return {
        "sem": sem,
        "hero_screen_xy": np.asarray((x_norm, y_norm), dtype=np.float32),
        "hero_velocity": np.asarray((player.vx / room.physics.run_speed, player.vy / room.physics.run_speed), dtype=np.float32),
        "on_ground": int(player.on_ground),
        "interact_prompt": int(interaction_prompt(room, player)),
    }
