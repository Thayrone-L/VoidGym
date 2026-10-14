from __future__ import annotations

import pytest

from hk_agent.shade_run_lifecycle import ShadeRunTracker, ShadeRunUpdate


def observe(
    tracker: ShadeRunTracker,
    *,
    dead: bool = False,
    pending: bool = False,
    recovered: bool = False,
    failed: bool = False,
) -> ShadeRunUpdate:
    return tracker.observe(
        player_dead=dead,
        death_persisted=dead,
        shade_pending=pending,
        shade_recovered=recovered,
        run_failed=failed,
    )


def test_first_death_recovery_and_second_death_define_logical_runs() -> None:
    tracker = ShadeRunTracker()
    assert observe(tracker).logical_run_index == 1

    first_death = observe(tracker, dead=True, pending=True)
    assert first_death.recovery_required is True
    assert first_death.run_failed is False

    # This snapshot may come from a new bridge/game process.
    after_restart = observe(tracker, pending=True)
    assert after_restart.logical_run_index == 1
    assert after_restart.recovery_required is True

    recovery = observe(tracker, recovered=True)
    assert recovery.recovery_required is False

    observe(tracker, dead=True, pending=True)
    observe(tracker, pending=True)
    second_death = observe(tracker, dead=True, pending=True, failed=True)
    assert second_death.run_failed is True
    assert second_death.logical_run_index == 2
    assert second_death.failed_runs == 1
    assert second_death.recovery_required is True


def test_tracker_initializes_from_a_save_with_pending_shade() -> None:
    tracker = ShadeRunTracker()
    initial = observe(tracker, pending=True)
    assert initial.recovery_required is True
    assert initial.logical_run_index == 1


def test_death_without_persistence_barrier_is_rejected() -> None:
    tracker = ShadeRunTracker()
    with pytest.raises(ValueError, match="persistence barrier"):
        tracker.observe(
            player_dead=True,
            death_persisted=False,
            shade_pending=True,
            shade_recovered=False,
            run_failed=False,
        )
