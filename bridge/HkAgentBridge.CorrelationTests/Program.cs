using System;
using System.IO;
using HkAgentBridge;

var state = new EpisodeCorrelationState();
string firstEpisode = Guid.NewGuid().ToString();
string secondEpisode = Guid.NewGuid().ToString();

state.BeginEpisode(firstEpisode);
state.RecordAction(17, firstEpisode);
CorrelationSnapshot frame = state.Snapshot(firstEpisode);
Assert(frame.EpisodeId == firstEpisode, "FRAME must retain episode_id");
Assert(frame.LastAppliedActionId == 17, "FRAME must retain action_id");

state.BeginEpisode(secondEpisode);
frame = state.Snapshot(secondEpisode);
Assert(frame.EpisodeId == secondEpisode, "BEGIN_EPISODE must replace episode_id");
Assert(frame.LastAppliedActionId == null, "BEGIN_EPISODE must clear previous action_id");

AssertThrows(() => state.RecordAction(18, firstEpisode));
AssertThrows(() => state.Snapshot(firstEpisode));

var lifecycle = new EpisodeLifecycleState();
Assert(EpisodeLifecycleState.CountHealthRestored(5, 6, true, false, false) == 1, "in-episode healing must be counted");
Assert(EpisodeLifecycleState.CountHealthRestored(5, 7, true, true, false) == 0, "death/reset health changes must be ignored");
Assert(EpisodeLifecycleState.CountHealthRestored(5, 7, true, false, true) == 0, "bench healing must be ignored");
Assert(EpisodeLifecycleState.CountHealthRestored(5, 7, false, false, false) == 0, "out-of-episode healing must be ignored");
lifecycle.BeginEpisode(false);
Assert(!lifecycle.Snapshot().PlayerDead, "lifecycle must start alive");
lifecycle.ObserveShadePresence(true);
lifecycle.MarkPlayerDying();
lifecycle.ObserveShadePresence(true);
lifecycle.MarkPlayerDead();
EpisodeLifecycleSnapshot firstDeath = lifecycle.Snapshot();
Assert(firstDeath.PlayerDead, "death hook must latch the terminal state");
Assert(firstDeath.ShadePending, "first death must require Shade recovery");
Assert(!firstDeath.RunFailed, "first death must not fail the logical run");
lifecycle.RecordShadeHit();
lifecycle.RecordShadeHit();
lifecycle.RecordShadeHit();
lifecycle.RecordShadeHit();
lifecycle.RecordShadeHit();
Assert(lifecycle.Snapshot().ShadeHitsSinceLastFrame == 4, "Shade hit reward count must cap at four per cycle");
lifecycle.RecordConfirmedEnemyKill();
lifecycle.RecordHealthLost(2);
EpisodeLifecycleSnapshot combat = lifecycle.Snapshot();
Assert(combat.EnemyKillsSinceLastFrame == 1, "confirmed kills must be reported once");
Assert(combat.HealthLostSinceLastFrame == 2, "real health loss must be reported once");
combat = lifecycle.Snapshot();
Assert(combat.EnemyKillsSinceLastFrame == 0, "kill delta must drain after a frame");
Assert(combat.HealthLostSinceLastFrame == 0, "health delta must drain after a frame");

lifecycle.BeginEpisode(true);
Assert(!lifecycle.Snapshot().PlayerDead, "BEGIN_EPISODE must clear the death latch");
Assert(lifecycle.Snapshot().ShadePending, "BEGIN_EPISODE must preserve a pending Shade");
lifecycle.ObserveShadePresence(false);
EpisodeLifecycleSnapshot recovered = lifecycle.Snapshot();
Assert(recovered.ShadeRecovered, "clearing shadeScene must emit a recovery event");
Assert(!recovered.ShadePending, "recovery must clear the Shade obligation");
Assert(!lifecycle.Snapshot().ShadeRecovered, "Shade recovery must be a one-frame event");

lifecycle.BeginEpisode(true);
lifecycle.ObserveShadePresence(false);
lifecycle.MarkPlayerDying();
lifecycle.MarkPlayerDead();
EpisodeLifecycleSnapshot deathAfterRecovery = lifecycle.Snapshot();
Assert(deathAfterRecovery.PlayerDead, "a new death must remain observable");
Assert(deathAfterRecovery.ShadePending, "a new death must create a Shade obligation");
Assert(!deathAfterRecovery.ShadeRecovered, "a new death must supersede an unconsumed recovery event");
Assert(!deathAfterRecovery.RunFailed, "death after recovery must begin a fresh recovery cycle");

lifecycle.BeginEpisode(false);
lifecycle.MarkPlayerDying();
lifecycle.MarkPlayerDead();
lifecycle.BeginEpisode(true);
lifecycle.MarkPlayerDying();
lifecycle.MarkPlayerDead();
EpisodeLifecycleSnapshot secondDeath = lifecycle.Snapshot();
Assert(secondDeath.PlayerDead, "second death must still terminate its episode");
Assert(secondDeath.RunFailed, "death with an outstanding Shade must fail the logical run");
Assert(secondDeath.ShadePending, "second death starts a new Shade obligation");

lifecycle.BeginEpisode(null);
Assert(lifecycle.Snapshot().ShadePending, "an unavailable PlayerData sample must not clear state");

lifecycle.BeginEpisode(false);
lifecycle.RecordGeoGained(5);
lifecycle.RecordGeoSpent(3);
EpisodeLifecycleSnapshot geo = lifecycle.Snapshot();
Assert(geo.GeoGainedSinceLastFrame == 5, "positive Geo delta must be reported separately");
Assert(geo.GeoSpentSinceLastFrame == 3, "negative Geo delta must be reported separately");
EpisodeLifecycleSnapshot consumedGeo = lifecycle.Snapshot();
Assert(consumedGeo.GeoGainedSinceLastFrame == 0, "Geo gain delta must be consumed once");
Assert(consumedGeo.GeoSpentSinceLastFrame == 0, "Geo spend delta must be consumed once");

string saveRoot = Path.Combine(Path.GetTempPath(), $"hk-agent-save-{Guid.NewGuid():N}");
Directory.CreateDirectory(saveRoot);
try
{
    Environment.SetEnvironmentVariable("HK_AGENT_HOST", "127.0.0.1");
    Environment.SetEnvironmentVariable("HK_AGENT_PORT", "43123");
    Environment.SetEnvironmentVariable("HK_AGENT_RUN_ID", "run-1");
    Environment.SetEnvironmentVariable("HK_AGENT_INSTANCE_ID", "instance-1");
    Environment.SetEnvironmentVariable("HK_AGENT_TOKEN", "secret");
    Environment.SetEnvironmentVariable("HK_AGENT_GAME_PROFILE", "development");
    Environment.SetEnvironmentVariable("HK_AGENT_SAVE_ROOT", saveRoot);
    BridgeConfiguration bridgeConfiguration = BridgeConfiguration.FromEnvironment();
    Assert(
        string.Equals(bridgeConfiguration.SaveRoot, Path.GetFullPath(saveRoot), StringComparison.OrdinalIgnoreCase),
        "bridge configuration must canonicalize the isolated save root"
    );
    Environment.SetEnvironmentVariable("HK_AGENT_SAVE_ROOT", null);
    AssertBridgeConfigurationThrows();
}
finally
{
    foreach (string name in new[]
    {
        "HK_AGENT_HOST", "HK_AGENT_PORT", "HK_AGENT_RUN_ID", "HK_AGENT_INSTANCE_ID",
        "HK_AGENT_TOKEN", "HK_AGENT_GAME_PROFILE", "HK_AGENT_SAVE_ROOT",
    })
    {
        Environment.SetEnvironmentVariable(name, null);
    }
    Directory.Delete(saveRoot);
}

Console.WriteLine("Episode correlation tests passed");
return 0;

static void Assert(bool condition, string message)
{
    if (!condition) throw new InvalidOperationException(message);
}

static void AssertThrows(Action action)
{
    try
    {
        action();
    }
    catch (InvalidOperationException)
    {
        return;
    }
    throw new InvalidOperationException("Expected an episode mismatch to be rejected");
}

static void AssertBridgeConfigurationThrows()
{
    try
    {
        BridgeConfiguration.FromEnvironment();
    }
    catch (BridgeConfigurationException)
    {
        return;
    }
    throw new InvalidOperationException("Missing isolated save root must be rejected");
}
