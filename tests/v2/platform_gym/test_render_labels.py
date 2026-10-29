import numpy as np

from voidgym.core.semantics import SEM_GRID, Sem
from voidgym.gyms.platform_gym.generator import generate_room
from voidgym.gyms.platform_gym.labels import labels
from voidgym.gyms.platform_gym.render import (
    _background_frame, _static_effects, player_pixel_mask, render_room,
)
from voidgym.gyms.platform_gym.styles import sample_style
from voidgym.gyms.platform_gym.world import Player


def test_render_shape_dtype_and_seed_reproducibility() -> None:
    room = generate_room(2, 42)
    player = Player(*room.spawn, on_ground=True)
    a = render_room(room, player)
    b = render_room(room, player)
    assert a.shape == (108, 192, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert np.mean(np.abs(a.astype(np.int16) - render_room(room, player, style_seed=43).astype(np.int16))) > 20


def test_seeded_render_caches_remain_bounded() -> None:
    for seed in range(10_000, 10_100):
        _static_effects(seed, "train")
        _background_frame(seed, "train", (20, 30, 40), (1, 2, 3))
    assert _static_effects.cache_info().currsize <= 64
    assert _background_frame.cache_info().currsize <= 128


def test_train_split_never_uses_held_out_families() -> None:
    for seed in range(50):
        style, _ = sample_style(seed, split="train")
        assert style.texture_family not in {"crystals", "hexagons"}
        assert style.character_family != "triangle"


def test_labels_prioritize_player_and_are_style_independent() -> None:
    room = generate_room(0, 8)
    player = Player(*room.spawn, on_ground=True)
    style, _ = sample_style(room.style_seed, split="train")
    camera = (player.x, player.y + 1, 192 / style.camera_width)
    a = labels(room, player, camera)
    render_room(room, player, camera, style_seed=42)
    b = labels(room, player, camera)
    render_room(room, player, camera, style_seed=43)
    c = labels(room, player, camera)
    assert a["sem"].shape == SEM_GRID
    assert int(Sem.PLAYER) in a["sem"]
    assert np.array_equal(a["sem"], b["sem"]) and np.array_equal(a["sem"], c["sem"])
    held_out = labels(room, player, camera, split="test")["sem"]
    assert np.array_equal(a["sem"], held_out)
    sprite_cells = player_pixel_mask(room, player, camera).reshape(27, 4, 48, 4).any(axis=(1, 3))
    labeled_cells = a["sem"] == int(Sem.PLAYER)
    assert labeled_cells.sum() > 0
    assert np.count_nonzero(labeled_cells & sprite_cells) > labeled_cells.sum() / 2
