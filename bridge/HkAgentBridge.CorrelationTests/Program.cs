using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using HkAgentBridge;

MenuActionSelection pauseAction = MenuActionPolicy.Validate(new[] { "PAUSE" }, 50);
Assert(pauseAction.Button == "PAUSE" && pauseAction.Mask == (1 << 6),
    "PAUSE must have a dedicated virtual-controller control distinct from menu cancel");
MenuActionSelection cancelAction = MenuActionPolicy.Validate(new[] { "CANCEL" }, 50);
Assert(cancelAction.Mask == (1 << 5), "CANCEL must remain separate from PAUSE");
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
Assert(EpisodeLifecycleState.NormalizeEnemyType("Crawler (Clone)") == "Crawler", "Clone suffix must be stripped from enemy type");
Assert(EpisodeLifecycleState.NormalizeEnemyType("Crawler (12)") == "Crawler", "instance suffix must be stripped from enemy type");
Assert(EpisodeLifecycleState.NormalizeEnemyType("Crawler") == "Crawler", "stable enemy type must be retained");
Assert(EnemyObservationState.TypeId("Crawler (Clone)") == EnemyObservationState.TypeId("Crawler"), "enemy type IDs must ignore runtime clone suffixes");
var enemyObservations = new EnemyObservationState();
enemyObservations.RecordHit(123);
enemyObservations.RecordHit(123);
Assert(enemyObservations.Hits(123) == 2, "visible enemy hit count must be tracked per runtime instance");
Assert(enemyObservations.Hits(124) == 0, "enemy hit counts must remain instance-specific");
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
lifecycle.RecordConfirmedEnemyKill("Crawler (Clone)", "Room_Test", 2);
lifecycle.RecordHealthLost(2);
EpisodeLifecycleSnapshot combat = lifecycle.Snapshot();
Assert(combat.EnemyKillsSinceLastFrame == 1, "confirmed kills must be reported once");
Assert(combat.HealthLostSinceLastFrame == 2, "real health loss must be reported once");
Assert(combat.EnemyEventsSinceLastFrame.Count == 1, "enemy type and scene event must be reported");
Assert(combat.EnemyEventsSinceLastFrame[0].EnemyType == "Crawler (Clone)", "event storage keeps bridge input before normalization test");
Assert(combat.EnemyEventsSinceLastFrame[0].HitsDealt == 2, "kill events must carry the player's observed hit count");
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

var telemetry = new OperatorTelemetryState();
var flags = new[] { new KeyValuePair<string, bool>("hasDash", false) };
telemetry.Observe("Crossroads_01", "Crossroads", 2, 3, 1, 2, new[] { 0f, 0f, 10f, 10f }, 0, 20, flags);
telemetry.Observe("Crossroads_01", "Crossroads", 3, 4, 1, 2, new[] { 0f, 0f, 10f, 10f }, 1, 22,
    new[] { new KeyValuePair<string, bool>("hasDash", true) });
Dictionary<string, object?> telemetryDelta = telemetry.Snapshot();
var milestoneEvents = (Dictionary<string, object?>[])telemetryDelta["milestone_events_since_last_frame"]!;
Assert(milestoneEvents.Length == 2, "grub count and false-to-true item flag must become confirmed events");
Assert((string)milestoneEvents[0]["category"]! == "grub", "grub delta category must be typed");
Assert((string)milestoneEvents[1]["category"]! == "item", "item flag transition must be typed");
Assert((string)telemetry.Snapshot()["scene_name"]! == "Crossroads_01", "scene telemetry remains available after deltas are consumed");
var framePayload = new Dictionary<string, object?> { ["operator_telemetry"] = telemetryDelta };
using (JsonDocument serializedFrame = JsonDocument.Parse(JsonSerializer.Serialize(framePayload)))
{
    JsonElement payloadTelemetry = serializedFrame.RootElement.GetProperty("operator_telemetry");
    Assert(payloadTelemetry.GetProperty("x").GetSingle() == 3, "FRAME payload must preserve the hero x position");
    Assert(payloadTelemetry.GetProperty("geo_current").GetInt32() == 22, "FRAME payload must preserve current Geo");
    Assert(payloadTelemetry.GetProperty("milestone_events_since_last_frame").GetArrayLength() == 2,
        "FRAME payload must preserve confirmed milestone events");
}

var passageTelemetry = new OperatorTelemetryState();
var noFlags = Array.Empty<KeyValuePair<string, bool>>();
passageTelemetry.Observe("Room_A", "Zone_A", 9, 4, 9, 4, new[] { 0f, 0f, 20f, 10f }, 0, 0, noFlags);
passageTelemetry.Observe("Room_A", "Zone_A", 10, 4, 9, 4, new[] { 0f, 0f, 20f, 10f }, 0, 0, noFlags);
passageTelemetry.Observe("Room_B", "Zone_B", 1, 4, 1, 4, new[] { -5f, 0f, 15f, 10f }, 0, 0, noFlags);
passageTelemetry.Observe("Room_B", "Zone_B", 1.1f, 4.1f, 1, 4, new[] { -5f, 0f, 15f, 10f }, 0, 0, noFlags);
Dictionary<string, object?> passageFrame = passageTelemetry.Snapshot();
var passageEvents = (Dictionary<string, object?>[])passageFrame["milestone_events_since_last_frame"]!;
var passage = Array.Find(passageEvents, item => (string)item["category"]! == "scene_transition")!;
Assert((string)passage["from"]! == "Room_A" && (string)passage["to"]! == "Room_B",
    "scene transition must preserve both room names");
Assert((float)passage["exit_x"]! == 10 && (float)passage["entry_x"]! == 1,
    "scene transition must preserve last exit and first stable entry positions");
Assert((bool)passage["spatial"]!, "valid stable passage positions must be marked spatial");
Assert((string)passage["from_map_zone"]! == "Zone_A" && (string)passage["to_map_zone"]! == "Zone_B",
    "scene transition must preserve map zones");

var respawnTelemetry = new OperatorTelemetryState();
respawnTelemetry.Observe("Room_A", "Zone_A", 10, 4, 9, 4, new[] { 0f, 0f, 20f, 10f }, 0, 0, noFlags);
respawnTelemetry.RecordEvent("death", "death_1", 10, 4);
respawnTelemetry.Observe("Room_B", "Zone_B", 1, 4, 1, 4, new[] { -5f, 0f, 15f, 10f }, 0, 0, noFlags);
respawnTelemetry.Observe("Room_B", "Zone_B", 1.1f, 4.1f, 1, 4, new[] { -5f, 0f, 15f, 10f }, 0, 0, noFlags);
var respawnEvents = (Dictionary<string, object?>[])respawnTelemetry.Snapshot()["milestone_events_since_last_frame"]!;
var respawnTransition = Array.Find(respawnEvents, item => (string)item["category"]! == "scene_transition")!;
Assert(!(bool)respawnTransition["spatial"]! && (string)respawnTransition["transition_kind"]! == "death_respawn",
    "death/respawn scene changes must not become spatial passage edges");

var checkpointTelemetry = new OperatorTelemetryState();
checkpointTelemetry.Observe("Room_Bench", "Zone_B", 1, 4, 1, 4,
    new[] { -5f, 0f, 15f, 10f }, 0, 0, noFlags, atBench: true,
    equippedCharms: new Dictionary<string, bool> { ["equippedCharm_40"] = true },
    charmNotchesUsed: 3, charmNotchesTotal: 4, charmMenuOpen: true, charmCursor: 39);
var checkpointFrame = checkpointTelemetry.Snapshot();
Assert((bool)checkpointFrame["at_bench"]!, "operator telemetry must expose the current bench state");
Assert((long)checkpointFrame["equipped_charm_mask"]! == (1L << 39),
    "equipped charm bitmask must include charm 40 without truncation");
Assert((int)checkpointFrame["charm_notches_used"]! == 3 &&
       (int)checkpointFrame["charm_notches_total"]! == 4,
    "operator telemetry must expose used and total charm notches");
Assert((bool)checkpointFrame["charm_menu_open"]! && (int)checkpointFrame["charm_cursor"]! == 39,
    "operator telemetry must expose charm menu state and cursor");
var checkpointEvents = (Dictionary<string, object?>[])checkpointFrame["milestone_events_since_last_frame"]!;
Assert(checkpointEvents.Length == 1 && (string)checkpointEvents[0]["category"]! == "checkpoint",
    "first observation at a bench must emit a checkpoint milestone");

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
