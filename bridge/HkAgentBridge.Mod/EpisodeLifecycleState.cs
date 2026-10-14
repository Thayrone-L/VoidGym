using System.Threading;

namespace HkAgentBridge;

internal readonly struct EpisodeLifecycleSnapshot
{
    public EpisodeLifecycleSnapshot(
        bool playerDead,
        bool shadePending,
        bool shadeRecovered,
        bool runFailed,
        int enemyKillsSinceLastFrame,
        int healthLostSinceLastFrame,
        int geoGainedSinceLastFrame,
        int geoSpentSinceLastFrame
    )
    {
        PlayerDead = playerDead;
        ShadePending = shadePending;
        ShadeRecovered = shadeRecovered;
        RunFailed = runFailed;
        EnemyKillsSinceLastFrame = enemyKillsSinceLastFrame;
        HealthLostSinceLastFrame = healthLostSinceLastFrame;
        GeoGainedSinceLastFrame = geoGainedSinceLastFrame;
        GeoSpentSinceLastFrame = geoSpentSinceLastFrame;
    }

    public bool PlayerDead { get; }
    public bool ShadePending { get; }
    public bool ShadeRecovered { get; }
    public bool RunFailed { get; }
    public int EnemyKillsSinceLastFrame { get; }
    public int HealthLostSinceLastFrame { get; }
    public int GeoGainedSinceLastFrame { get; }
    public int GeoSpentSinceLastFrame { get; }
}

/// <summary>
/// Supervisor-only lifecycle state. None of these values are policy observations.
/// </summary>
internal sealed class EpisodeLifecycleState
{
    private int _playerDead;
    private int _shadePending;
    private int _shadeRecovered;
    private int _runFailed;
    private int _deathStarted;
    private int _pendingBeforeDeath;
    private int _enemyKillsSinceLastFrame;
    private int _healthLostSinceLastFrame;
    private int _geoGainedSinceLastFrame;
    private int _geoSpentSinceLastFrame;

    public EpisodeLifecycleSnapshot Snapshot() =>
        new(
            Volatile.Read(ref _playerDead) != 0,
            Volatile.Read(ref _shadePending) != 0,
            Interlocked.Exchange(ref _shadeRecovered, 0) != 0,
            Volatile.Read(ref _runFailed) != 0,
            Interlocked.Exchange(ref _enemyKillsSinceLastFrame, 0),
            Interlocked.Exchange(ref _healthLostSinceLastFrame, 0),
            Interlocked.Exchange(ref _geoGainedSinceLastFrame, 0),
            Interlocked.Exchange(ref _geoSpentSinceLastFrame, 0)
        );

    public void BeginEpisode(bool? shadePendingFromSave)
    {
        Interlocked.Exchange(ref _playerDead, 0);
        Interlocked.Exchange(ref _shadeRecovered, 0);
        Interlocked.Exchange(ref _runFailed, 0);
        Interlocked.Exchange(ref _deathStarted, 0);
        Interlocked.Exchange(ref _enemyKillsSinceLastFrame, 0);
        Interlocked.Exchange(ref _healthLostSinceLastFrame, 0);
        Interlocked.Exchange(ref _geoGainedSinceLastFrame, 0);
        Interlocked.Exchange(ref _geoSpentSinceLastFrame, 0);
        if (shadePendingFromSave.HasValue)
        {
            Interlocked.Exchange(ref _shadePending, shadePendingFromSave.Value ? 1 : 0);
        }
    }

    public void RecordGeoGained(int amount)
    {
        if (amount > 0)
        {
            Interlocked.Add(ref _geoGainedSinceLastFrame, amount);
        }
    }

    public void RecordGeoSpent(int amount)
    {
        if (amount > 0)
        {
            Interlocked.Add(ref _geoSpentSinceLastFrame, amount);
        }
    }

    public void RecordConfirmedEnemyKill()
    {
        Interlocked.Increment(ref _enemyKillsSinceLastFrame);
    }

    public void RecordHealthLost(int amount)
    {
        if (amount > 0)
        {
            Interlocked.Add(ref _healthLostSinceLastFrame, amount);
        }
    }

    public void MarkPlayerDying()
    {
        // A new death supersedes any recovery event that has not yet been
        // consumed by Snapshot. Publishing both states is contradictory.
        Interlocked.Exchange(ref _shadeRecovered, 0);
        Interlocked.Exchange(ref _pendingBeforeDeath, Volatile.Read(ref _shadePending));
        Interlocked.Exchange(ref _deathStarted, 1);
    }

    public void MarkPlayerDead()
    {
        int wasPending = Volatile.Read(ref _deathStarted) != 0
            ? Volatile.Read(ref _pendingBeforeDeath)
            : Volatile.Read(ref _shadePending);
        Interlocked.Exchange(ref _shadePending, 1);
        if (wasPending != 0)
        {
            Interlocked.Exchange(ref _runFailed, 1);
        }
        Interlocked.Exchange(ref _playerDead, 1);
    }

    public void ObserveShadePresence(bool shadePending)
    {
        // A false->true soulLimited write may occur during the death sequence
        // before BeforePlayerDeadHook. It must not turn a first death into a
        // false second-death failure. Only BEGIN_EPISODE or MarkPlayerDead may
        // create a pending recovery obligation; runtime observation may clear it.
        if (Volatile.Read(ref _deathStarted) != 0 || Volatile.Read(ref _playerDead) != 0)
        {
            return;
        }

        if (shadePending)
        {
            return;
        }

        int previous = Interlocked.Exchange(ref _shadePending, 0);
        if (previous != 0)
        {
            Interlocked.Exchange(ref _shadeRecovered, 1);
        }
    }
}
