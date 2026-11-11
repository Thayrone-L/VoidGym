from __future__ import annotations

import numpy as np
import pytest

from hk_agent.semantic_labels import pack_u4, unpack_u4


def test_semantic_u4_round_trip_preserves_class_ids_and_ignore() -> None:
    labels = np.asarray([[0, 1, 9, 255], [3, 4, 5, 6]], dtype=np.uint8)
    packed = pack_u4(labels)

    assert len(packed) == labels.size // 2
    np.testing.assert_array_equal(unpack_u4(packed, width=4, height=2), labels)


def test_semantic_u4_rejects_wrong_size_and_unknown_classes() -> None:
    with pytest.raises(ValueError, match="wrong length"):
        unpack_u4(b"\x00", width=2, height=2)
    with pytest.raises(ValueError, match="unknown class"):
        pack_u4(np.asarray([[10]], dtype=np.uint8))


def test_semantic_u4_handles_odd_cell_count() -> None:
    labels = np.asarray([[2, 255, 7]], dtype=np.uint8)
    np.testing.assert_array_equal(unpack_u4(pack_u4(labels), width=3, height=1), labels)
