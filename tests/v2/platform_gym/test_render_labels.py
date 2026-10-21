"""Saída renderizada e gabaritos da cena procedural."""

import numpy as np

from voidgym.core.semantics import SEM_GRID, Sem
from voidgym.gyms.platform_gym.generator import generate_room
from voidgym.gyms.platform_gym.labels import labels
from voidgym.gyms.platform_gym.render import render
from voidgym.gyms.platform_gym.styles import make_style
from voidgym.gyms.platform_gym.world import Player, Rect, Room, Solid


def test_frame_shape_style_randomization_and_split_holdout() -> None:
    room = generate_room(0, 123)
    player = Player(*room.spawn)
    style = make_style(123)
    first = render(room, player, style)
    assert first.shape == (108, 192, 3) and first.dtype == np.uint8
    assert np.array_equal(first, render(room, player, style))
    assert np.abs(first.astype(float) - render(room, player, make_style(124)).astype(float)).mean() > 20
    assert make_style(10).style_params.split == "test"
    assert make_style(10).style_params.texture in {"crystals", "hexagons"}
    assert make_style(10).style_params.character == "horned_ball"
    assert make_style(11, "train").style_params.texture not in {"crystals", "hexagons"}


def test_semantics_cover_solid_and_player_independently_of_style() -> None:
    room = Room(1, 24, 12, solids=[Solid(Rect(4, 0, 2, 1))], physics=generate_room(0, 1).physics)
    player = Player(1, 1)
    left = labels(room, player, style=make_style(1))
    right = labels(room, player, style=make_style(2))
    assert left["sem"].shape == SEM_GRID
    assert np.array_equal(left["sem"], right["sem"])
    assert np.any(left["sem"] == Sem.SOLID)
    assert np.any(left["sem"] == Sem.PLAYER)
