from __future__ import annotations

import numpy as np

from tools.v2.eval_perception_platform import (
    _choose_action,
    _episode_quotas,
    _frame_quotas,
)


def test_default_perception_sample_is_balanced_over_nine_levels() -> None:
    quotas = _frame_quotas(5625)
    assert quotas == dict.fromkeys(range(9), 625)
    assert sum(quotas.values()) == 5625


def test_remainder_is_distributed_by_at_most_one_frame() -> None:
    quotas = _frame_quotas(5003)
    assert list(quotas.values()) == [556, 556, 556, 556, 556, 556, 556, 556, 555]


def test_per_episode_quota_spreads_samples_over_episodes() -> None:
    quotas = _episode_quotas(625, 100)
    assert len(quotas) == 100
    assert sum(quotas) == 625
    assert quotas.count(7) == 25
    assert quotas.count(6) == 75


def test_fixed_route_respects_action_mask_and_falls_back_to_right() -> None:
    table = ("NOOP", "RIGHT", "RIGHT+JUMP_HOLD")
    mask = np.asarray([True, False, True])
    assert _choose_action(table, mask, step=3, episode=0) == 2
    assert _choose_action(table, np.asarray([True, True, False]), step=3, episode=0) == 1
