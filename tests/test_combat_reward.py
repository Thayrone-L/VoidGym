from __future__ import annotations

import pytest

from hk_agent.combat_reward import (
    CombatRewardConfig,
    CombatRewardLedger,
    CombatStep,
    ConfirmedKillEvent,
)


def _event(
    sequence: int,
    *,
    stable: str | None = "enemy/path@10,4",
    runtime: str | None = "42",
    scene: str = "Crossroads_01",
    boss: bool = False,
    session: str = "process-a",
) -> ConfirmedKillEvent:
    return ConfirmedKillEvent(session, sequence, scene, stable, runtime, boss)


def _step(
    health_lost: int = 0,
    *,
    dead: bool = False,
    failed: bool = False,
    death_sequence: int = 0,
    session: str = "process-a",
) -> CombatStep:
    return CombatStep(session, health_lost, death_sequence, dead, failed)


def _ledger(config: CombatRewardConfig | None = None) -> CombatRewardLedger:
    ledger = CombatRewardLedger(config)
    ledger.begin_gameplay_run("run-1")
    ledger.begin_episode("episode-1")
    return ledger


def test_confirmed_kill_is_paid_once_and_transport_replay_is_deduplicated() -> None:
    ledger = _ledger()
    first = ledger.apply(_step(), [_event(1)])
    replay = ledger.apply(_step(), [_event(1)])
    assert first.components["confirmed_enemy_kill"] == 0.5
    assert first.accepted_kills == 1
    assert replay.total == 0
    assert replay.duplicate_kills == 1


def test_repeat_is_diminished_and_capped_but_new_identity_is_always_rewarded() -> None:
    config = CombatRewardConfig(
        enemy_kill_reward=1,
        repeated_kill_decay=0.5,
        repeated_identity_cap=0.6,
        repeated_scene_budget=0.6,
        repeated_episode_budget=0.6,
    )
    ledger = _ledger(config)
    rewards = [ledger.apply(_step(), [_event(index)]).total for index in range(1, 5)]
    assert rewards == pytest.approx([1.0, 0.5, 0.1, 0.0])

    novel = ledger.apply(_step(), [_event(5, stable="different/path@12,4")])
    assert novel.components["confirmed_enemy_kill"] == 1.0


def test_stable_identity_survives_process_relaunch_and_state_restore() -> None:
    ledger = _ledger()
    ledger.apply(_step(), [_event(1)])
    restored = CombatRewardLedger.from_state(ledger.state())
    repeated = restored.apply(
        _step(session="process-b"),
        [_event(1, session="process-b", runtime="999")],
    )
    assert repeated.components["confirmed_enemy_kill"] == 0
    assert repeated.components["repeated_enemy_kill"] == pytest.approx(0.125)
    assert repeated.identity_fallbacks == 0


def test_runtime_identity_fallback_is_explicit_and_process_local() -> None:
    ledger = _ledger()
    first = ledger.apply(_step(), [_event(1, stable=None)])
    relaunched = ledger.apply(
        _step(session="process-b"),
        [_event(1, stable=None, session="process-b")],
    )
    assert first.identity_fallbacks == 1
    assert relaunched.identity_fallbacks == 1
    assert relaunched.components["confirmed_enemy_kill"] == 0.5


def test_damage_uses_cumulative_real_health_delta_and_rejects_regression() -> None:
    ledger = _ledger()
    assert ledger.apply(_step(1)).components["health_lost"] == -0.25
    assert ledger.apply(_step(1)).components["health_lost"] == 0
    assert ledger.apply(_step(3)).components["health_lost"] == -0.5
    with pytest.raises(ValueError, match="moved backwards"):
        ledger.apply(_step(2))


def test_terminal_death_replaces_health_loss_and_latch_is_not_repaid() -> None:
    ledger = _ledger()
    terminal = ledger.apply(_step(1, dead=True, death_sequence=1))
    latched = ledger.apply(_step(1, dead=True, death_sequence=1))
    assert terminal.components["health_lost"] == 0
    assert terminal.components["player_death"] == -2
    assert terminal.total == -2
    assert latched.total == 0


def test_run_failure_adds_larger_penalty_without_stopping_ledger() -> None:
    ledger = _ledger()
    failure = ledger.apply(_step(1, dead=True, failed=True, death_sequence=1))
    assert failure.components["player_death"] == -2
    assert failure.components["gameplay_run_failed"] == -4
    assert failure.total == -6
    ledger.begin_gameplay_run("run-2")
    ledger.begin_episode("episode-2")
    assert ledger.apply(_step(1), [_event(2, stable="new-enemy")]).total == 0.5


def test_only_confirmed_boss_classification_gets_multiplier() -> None:
    ledger = _ledger()
    ordinary = ledger.apply(_step(), [_event(1, stable="a")])
    boss = ledger.apply(_step(), [_event(2, stable="b", boss=True)])
    assert ordinary.total == 0.5
    assert boss.components["confirmed_enemy_kill"] == 0.5
    assert boss.components["confirmed_boss_kill_bonus"] == 1.5
    assert boss.total == 2.0


def test_progress_must_dominate_repeatable_combat_and_boss_reward() -> None:
    with pytest.raises(ValueError, match="repeatable combat"):
        CombatRewardConfig(
            progress_reward=1,
            repeated_scene_budget=1,
            repeated_episode_budget=1,
        ).validate()
    with pytest.raises(ValueError, match="boss-kill"):
        CombatRewardConfig(progress_reward=1, enemy_kill_reward=1).validate()
