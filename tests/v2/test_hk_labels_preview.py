import numpy as np

from tools.v2.hk_labels_preview import _is_transition_frame, _largest_connected_region
from voidgym.core.semantics import Sem


def test_largest_connected_enemy_region_uses_each_frame_and_four_neighbors():
    first = np.zeros((27, 48), dtype=np.uint8)
    second = np.zeros((27, 48), dtype=np.uint8)
    first[2:6, 2:6] = Sem.ENEMY
    first[8:12, 8:12] = Sem.ENEMY
    second[1:11, 1:9] = Sem.ENEMY

    size, frame_index = _largest_connected_region(
        [(np.zeros((108, 192, 3), dtype=np.uint8), first),
         (np.zeros((108, 192, 3), dtype=np.uint8), second)], Sem.ENEMY
    )

    assert size == 80
    assert frame_index == 1


def test_largest_connected_region_accepts_any_semantic_class():
    labels = np.zeros((27, 48), dtype=np.uint8)
    labels[2:7, 4:10] = Sem.HAZARD

    size, frame_index = _largest_connected_region(
        [(np.zeros((108, 192, 3), dtype=np.uint8), labels)], Sem.HAZARD
    )

    assert size == 30
    assert frame_index == 0


def test_largest_connected_region_does_not_join_diagonal_cells():
    labels = np.zeros((27, 48), dtype=np.uint8)
    labels[0, 0] = Sem.ENEMY
    labels[1, 1] = Sem.ENEMY

    size, frame_index = _largest_connected_region(
        [(np.zeros((108, 192, 3), dtype=np.uint8), labels)], Sem.ENEMY
    )

    assert size == 1
    assert frame_index == 0


def test_transition_filter_discards_black_fades_but_keeps_dark_gameplay():
    black = np.zeros((108, 192, 3), dtype=np.uint8)
    dark_gameplay = np.full((108, 192, 3), 8, dtype=np.uint8)
    dark_gameplay[20:90, 20:170] = 28

    assert _is_transition_frame(black)
    assert not _is_transition_frame(dark_gameplay)


def test_transition_filter_rejects_malformed_frames():
    assert _is_transition_frame(np.zeros((108, 192), dtype=np.uint8))
