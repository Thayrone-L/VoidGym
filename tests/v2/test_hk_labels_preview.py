import numpy as np

from tools.v2.hk_labels_preview import _largest_connected_enemy_region
from voidgym.core.semantics import Sem


def test_largest_connected_enemy_region_uses_each_frame_and_four_neighbors():
    first = np.zeros((27, 48), dtype=np.uint8)
    second = np.zeros((27, 48), dtype=np.uint8)
    first[2:6, 2:6] = Sem.ENEMY
    first[8:12, 8:12] = Sem.ENEMY
    second[1:11, 1:9] = Sem.ENEMY

    size, frame_index = _largest_connected_enemy_region(
        [(np.zeros((108, 192, 3), dtype=np.uint8), first),
         (np.zeros((108, 192, 3), dtype=np.uint8), second)]
    )

    assert size == 80
    assert frame_index == 1


def test_largest_connected_enemy_region_does_not_join_diagonal_cells():
    labels = np.zeros((27, 48), dtype=np.uint8)
    labels[0, 0] = Sem.ENEMY
    labels[1, 1] = Sem.ENEMY

    size, frame_index = _largest_connected_enemy_region(
        [(np.zeros((108, 192, 3), dtype=np.uint8), labels)]
    )

    assert size == 1
    assert frame_index == 0
