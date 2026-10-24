"""Generate operator previews for procedural platform gym."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import cast, Literal

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from voidgym.core.semantics import Sem
from voidgym.gyms.platform_gym.generator import generate_room, generate_room_set
from voidgym.gyms.platform_gym.labels import labels
from voidgym.gyms.platform_gym.physics import step_physics
from voidgym.gyms.platform_gym.render import render_room
from voidgym.gyms.platform_gym.solver import _buttons
from voidgym.gyms.platform_gym.world import Player

OUT = Path("relatorios/84_amostras")
COLORS = {
    Sem.EMPTY: (0, 0, 0, 0), Sem.SOLID: (70, 90, 255, 110), Sem.PLATFORM: (40, 210, 230, 110),
    Sem.HAZARD: (255, 35, 35, 145), Sem.ENEMY: (220, 40, 220, 155), Sem.PLAYER: (255, 255, 255, 220),
    Sem.BREAKABLE: (255, 140, 20, 140), Sem.PICKUP: (255, 230, 20, 160),
    Sem.INTERACTIVE: (20, 255, 120, 150), Sem.EXIT: (255, 255, 255, 100),
}


def overlay(frame: np.ndarray, sem: np.ndarray) -> Image.Image:
    base = Image.fromarray(frame).convert("RGBA")
    lay = Image.new("RGBA", (192, 108))
    for y in range(27):
        for x in range(48):
            color = COLORS[Sem(int(sem[y, x]))]
            if color[3]:
                ImageDraw.Draw(lay).rectangle((x * 4, y * 4, x * 4 + 3, y * 4 + 3), fill=color)
    return Image.alpha_composite(base, lay).convert("RGB")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    labels_frames: list[Image.Image] = []
    levels = [0, 1, 2, 3, 4, 5, 6, 7] * 3
    for i, level in enumerate(levels):
        room = generate_room_set(8400 + i)[0] if level == 7 else generate_room(level, 8400 + i)
        player = Player(*room.spawn, on_ground=True)
        frame = render_room(room, player)
        frames.append(Image.fromarray(frame).resize((384, 216), Image.Resampling.NEAREST))
        sem = cast(np.ndarray, labels(room, player)["sem"])
        labels_frames.append(overlay(frame, sem).resize((384, 216), Image.Resampling.NEAREST))
    for name, source in (("mosaico.png", frames), ("gabarito.png", labels_frames)):
        sheet = Image.new("RGB", (384 * 4, 216 * 6), (10, 10, 14))
        for i, image in enumerate(source):
            sheet.paste(image, ((i % 4) * 384, (i // 4) * 216))
        sheet.save(OUT / name)
    for level in (3, 4, 6, 7):
        rooms = generate_room_set(8400 + level) if level == 7 else [generate_room(level, 8400 + level)]
        images = []
        abilities: set[str] = set()
        for room in rooms:
            player = Player(*room.spawn, on_ground=True, abilities=abilities)
            player.last_safe_pos = room.spawn
            for tick in range(2000):
                events = step_physics(room, player, _buttons(room, player), .02)
                abilities = set(player.abilities)
                if tick % 5 == 0 or any(event.kind == "exit" for event in events):
                    image = Image.fromarray(render_room(room, player)).resize((384, 216), Image.Resampling.NEAREST)
                    images.append(image)
                if any(event.kind == "exit" for event in events):
                    break
            else:
                raise RuntimeError(f"solver preview did not exit level {level} room {room.id}")
        images[0].save(OUT / f"solver_nivel_{level}.gif", save_all=True, append_images=images[1:], duration=40, loop=0)
    comparison = []
    for split, start in (("train", 8500), ("test", 8501)):
        for i in range(8):
            room = generate_room(i % 8, start + i)
            player = Player(*room.spawn, on_ground=True)
            image = Image.fromarray(render_room(room, player, split=cast(Literal["train", "test"], split))).resize((384, 216), Image.Resampling.NEAREST)
            comparison.append((split, image))
    sheet = Image.new("RGB", (384 * 4, 216 * 4 + 28), (10, 10, 14))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), "TREINO", fill="white")
    draw.text((8, 216 * 2 + 6), "TESTE", fill="white")
    for i, (_, image) in enumerate(comparison):
        row = i // 4
        sheet.paste(image, ((i % 4) * 384, 28 + row * 216))
    sheet.save(OUT / "teste_vs_treino.png")


if __name__ == "__main__":
    main()
