using System;
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
        int geoSpentSinceLastFrame,
        int healthRestoredSinceLastFrame,
        int shadeHitsSinceLastFrame,
        string? sceneName,
        float? shadeDistance
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
        HealthRestoredSinceLastFrame = healthRestoredSinceLastFrame;
        ShadeHitsSinceLastFrame = shadeHitsSinceLastFrame;
        SceneName = sceneName;
        ShadeDistance = shadeDistance;
    }

    public bool PlayerDead { get; }
    public bool ShadePending { get; }
    public bool ShadeRecovered { get; }
    public bool RunFailed { get; }
    public int EnemyKillsSinceLastFrame { get; }
    public int HealthLostSinceLastFrame { get; }
    public int GeoGainedSinceLastFrame { get; }
    public int GeoSpentSinceLastFrame { get; }
    public int HealthRestoredSinceLastFrame { get; }
    public int ShadeHitsSinceLastFrame { get; }
    public string? SceneName { get; }
    public float? ShadeDistance { get; }
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
    private int _healthRestoredSinceLastFrame;
    private int _shadeHitsSinceLastFrame;
    private int _shadeHitCycleTotal;

    public bool IsDeathStarted => Volatile.Read(ref _deathStarted) != 0 || Volatile.Read(ref _playerDead) != 0;
    public bool ShadePending => Volatile.Read(ref _shadePending) != 0;

    public EpisodeLifecycleSnapshot Snapshot(string? sceneName = null, float? shadeDistance = null) =>
        new(
            Volatile.Read(ref _playerDead) != 0,
            Volatile.Read(ref _shadePending) != 0,
            Interlocked.Exchange(ref _shadeRecovered, 0) != 0,
            Volatile.Read(ref _runFailed) != 0,
            Interlocked.Exchange(ref _enemyKillsSinceLastFrame, 0),
            Interlocked.Exchange(ref _healthLostSinceLastFrame, 0),
            Interlocked.Exchange(ref _geoGainedSinceLastFrame, 0),
            Interlocked.Exchange(ref _geoSpentSinceLastFrame, 0),
            Interlocked.Exchange(ref _healthRestoredSinceLastFrame, 0),
            Interlocked.Exchange(ref _shadeHitsSinceLastFrame, 0),
            sceneName,
            shadeDistance
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
        Interlocked.Exchange(ref _healthRestoredSinceLastFrame, 0);
        Interlocked.Exchange(ref _shadeHitsSinceLastFrame, 0);
        Interlocked.Exchange(ref _shadeHitCycleTotal, 0);
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

    public void RecordHealthRestored(int amount)
    {
        if (amount > 0 && Volatile.Read(ref _deathStarted) == 0 && Volatile.Read(ref _playerDead) == 0)
        {
            Interlocked.Add(ref _healthRestoredSinceLastFrame, amount);
        }
    }

    public void RecordShadeHit()
    {
        // A Shade cycle needs at most four registered hits. Cap across the
        // complete pending cycle so over-reporting cannot create reward.
        if (Volatile.Read(ref _shadePending) == 0) return;
        while (true)
        {
            int current = Volatile.Read(ref _shadeHitCycleTotal);
            if (current >= 4) return;
            if (Interlocked.CompareExchange(ref _shadeHitCycleTotal, current + 1, current) == current)
            {
                Interlocked.Increment(ref _shadeHitsSinceLastFrame);
                return;
            }
        }
    }

    internal static int CountHealthRestored(int previousHealth, int currentHealth, bool episodeActive, bool deathOrReset, bool atBench) =>
        episodeActive && !deathOrReset && !atBench && previousHealth >= 0
            ? Math.Max(0, currentHealth - previousHealth)
            : 0;

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
            Interlocked.Exchange(ref _shadeHitCycleTotal, 0);
        }
    }
}
