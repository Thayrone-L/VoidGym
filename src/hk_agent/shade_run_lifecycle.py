from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ShadeRunUpdate:
    logical_run_index: int
    recovery_required: bool
    run_failed: bool
    shade_recovered: bool
    failed_runs: int


class ShadeRunTracker:
    """Campaign-side tracker that survives episode and game-process resets.

    The values are supervisor metadata only. Callers must not add them to the
    policy observation or turn them directly into reward features.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._recovery_required = False
        self._logical_run_index = 1
        self._failed_runs = 0
        self._death_latched = False

    def observe(
        self,
        *,
        player_dead: bool,
        death_persisted: bool,
        shade_pending: bool,
        shade_recovered: bool,
        run_failed: bool,
    ) -> ShadeRunUpdate:
        values = (
            player_dead,
            death_persisted,
            shade_pending,
            shade_recovered,
            run_failed,
        )
        if any(type(value) is not bool for value in values):
            raise TypeError("Shade lifecycle signals must be bool values")
        if player_dead and not death_persisted:
            raise ValueError("a death is terminal only after its persistence barrier")
        if run_failed and (not player_dead or not shade_pending):
            raise ValueError("run_failed requires a persisted death and a new pending Shade")
        if shade_recovered and shade_pending:
            raise ValueError("a recovered Shade cannot remain pending in the same snapshot")

        if not self._initialized:
            # A campaign may attach to a save that already has a Shade.
            self._recovery_required = shade_pending
            self._initialized = True

        if shade_recovered:
            self._recovery_required = False

        failed_now = False
        if player_dead and not self._death_latched:
            if run_failed:
                failed_now = True
                self._failed_runs += 1
                self._logical_run_index += 1
            # Both first and second deaths leave a new Shade to recover. On a
            # second death this obligation belongs to the newly started run.
            self._recovery_required = True
            self._death_latched = True
        elif player_dead:
            self._recovery_required = True
        elif shade_pending:
            # Reconcile state after the game process was restarted.
            self._death_latched = False
            self._recovery_required = True
        elif not shade_pending:
            self._death_latched = False
            self._recovery_required = False

        return ShadeRunUpdate(
            logical_run_index=self._logical_run_index,
            recovery_required=self._recovery_required,
            run_failed=failed_now,
            shade_recovered=shade_recovered,
            failed_runs=self._failed_runs,
        )
