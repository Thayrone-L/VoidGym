"""Gera mosaicos, gabaritos e trajetórias do solver para inspeção visual."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from voidgym.core.semantics import Sem
from voidgym.gyms.platform_gym.generator import generate_room
from voidgym.gyms.platform_gym.labels import labels
from voidgym.gyms.platform_gym.render import render
from voidgym.gyms.platform_gym.solver import solve_room
from voidgym.gyms.platform_gym.styles import make_style
from voidgym.gyms.platform_gym.world import Player

COLORS = {Sem.SOLID: (70, 80, 95), Sem.PLATFORM: (110, 180, 170), Sem.HAZARD: (240, 45, 75),
          Sem.ENEMY: (240, 130, 35), Sem.PLAYER: (255, 255, 255), Sem.BREAKABLE: (195, 130, 220),
          Sem.PICKUP: (255, 220, 60), Sem.INTERACTIVE: (70, 210, 180), Sem.EXIT: (40, 220, 100)}


def _player_at(position: tuple[float, float]) -> Player:
    return Player(position[0], position[1])


def _camera(room: Any, player: Player, style: Any) -> tuple[float, float]:
    width = style.style_params.camera_width
    return min(max(player.x - width * .42, 0), max(0, room.width - width)), max(0, player.y - 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("relatorios/84_amostras"))
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    scenes = []
    overlay_scenes = []
    for seed in range(24):
        level = seed % 8
        room = generate_room(level, seed + 101)
        player = Player(*room.spawn)
        style = make_style(seed + 101, "train")
        frame = Image.fromarray(render(room, player, style)).resize((384, 216), Image.Resampling.NEAREST)
        scenes.append(frame)
        camera_x, camera_y = _camera(room, player, style)
        data = labels(room, player, camera_x, camera_y, style)
        semantic = np.asarray(data["sem"])
        colors = np.zeros((27, 48, 3), dtype=np.uint8)
        for kind, color in COLORS.items():
            colors[semantic == kind] = color
        overlay = Image.fromarray(colors).resize((384, 216), Image.Resampling.NEAREST)
        overlay_scenes.append(Image.blend(frame, overlay, .5))
    _mosaic(scenes, output / "mosaico.png")
    _mosaic(overlay_scenes, output / "gabarito.png", legend=True)
    _holdout(output / "teste_vs_treino.png")
    for level in (3, 4, 6, 7):
        room = generate_room(level, 420 + level)
        trace: list[tuple[float, float]] = []
        if not solve_room(room, trace=trace):
            raise RuntimeError(f"Solver falhou no nível {level}")
        style = make_style(room.style_seed)
        frames = [Image.fromarray(render(room, _player_at(point), style)).resize((384, 216), Image.Resampling.NEAREST)
                  for point in trace[::max(1, len(trace)//24)]]
        frames[-1].save(output / f"solver_nivel_{level}.gif", save_all=True, append_images=frames[1:], duration=80, loop=0)
    return 0


def _mosaic(frames: list[Image.Image], path: Path, legend: bool = False) -> None:
    canvas = Image.new("RGB", (4 * 384, 6 * 216 + (30 if legend else 0)), (20, 22, 30))
    for index, frame in enumerate(frames):
        canvas.paste(frame, ((index % 4) * 384, (index // 4) * 216))
    if legend:
        draw = ImageDraw.Draw(canvas)
        y = 6 * 216
        for index, (kind, color) in enumerate(COLORS.items()):
            x = index * 180
            draw.rectangle((x, y + 8, x + 18, y + 24), fill=color)
            draw.text((x + 23, y + 7), kind.name.lower(), fill="white")
    canvas.save(path)


def _holdout(path: Path) -> None:
    frames = []
    for index in range(16):
        split = "train" if index < 8 else "test"
        seed = 100 + index - 8 if split == "test" else 11 + index
        room = generate_room(index % 8, seed)
        style = make_style(seed, split)
        frames.append(Image.fromarray(render(room, Player(*room.spawn), style)).resize((384, 216), Image.Resampling.NEAREST))
    canvas = Image.new("RGB", (4 * 384, 4 * 216 + 28), (20, 22, 30))
    for index, frame in enumerate(frames):
        canvas.paste(frame, ((index % 4) * 384, (index // 4) * 216))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4 * 216 + 6), "TRAIN: linhas 1–2 | TEST: linhas 3–4", fill="white")
    canvas.save(path)


if __name__ == "__main__":
    raise SystemExit(main())
