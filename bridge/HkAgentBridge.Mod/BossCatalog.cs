using System;
using System.Collections.Generic;

namespace HkAgentBridge;

internal readonly struct ConfirmedBoss
{
    public ConfirmedBoss(string key, string version)
    {
        Key = key;
        Version = version;
    }

    public string Key { get; }
    public string Version { get; }
}

/// <summary>Curated boss aliases; unknown enemies are never promoted to bosses.</summary>
internal static class BossCatalog
{
    private static readonly Dictionary<string, ConfirmedBoss> Entries = new(StringComparer.OrdinalIgnoreCase)
    {
        ["False Knight"] = new("False_Knight", "normal"),
        ["Failed Champion"] = new("False_Knight", "dream"),
        ["Gruz Mother"] = new("Gruz_Mother", "normal"),
        ["Hornet"] = new("Hornet", "Greenpath"),
        ["Hornet Sentinel"] = new("Hornet", "Kingdoms_Edge"),
        ["Mantis Lords"] = new("Mantis_Lords", "normal"),
        ["Sisters of Battle"] = new("Mantis_Lords", "Godhome"),
        ["Soul Master"] = new("Soul_Master", "normal"),
        ["Soul Tyrant"] = new("Soul_Master", "dream"),
        ["Dung Defender"] = new("Dung_Defender", "normal"),
        ["White Defender"] = new("Dung_Defender", "dream"),
        ["Broken Vessel"] = new("Broken_Vessel", "normal"),
        ["Lost Kin"] = new("Broken_Vessel", "dream"),
        ["Watcher Knight"] = new("Watcher_Knights", "normal"),
        ["Watcher Knights"] = new("Watcher_Knights", "normal"),
        ["Uumuu"] = new("Uumuu", "normal"),
        ["Traitor Lord"] = new("Traitor_Lord", "normal"),
        ["Hive Knight"] = new("Hive_Knight", "normal"),
        ["The Collector"] = new("Collector", "normal"),
        ["Collector"] = new("Collector", "normal"),
        ["God Tamer"] = new("God_Tamer", "normal"),
        ["Pure Vessel"] = new("Hollow_Knight", "Godhome"),
        ["Hollow Knight"] = new("Hollow_Knight", "normal"),
        ["The Radiance"] = new("Radiance", "normal"),
        ["Absolute Radiance"] = new("Radiance", "Godhome"),
        ["Nightmare King Grimm"] = new("Grimm", "Nightmare_King"),
        ["Troupe Master Grimm"] = new("Grimm", "Troupe_Master"),
        ["Grimm"] = new("Grimm", "Troupe_Master"),
        ["Brooding Mawlek"] = new("Brooding_Mawlek", "normal"),
        ["Massive Moss Charger"] = new("Moss_Charger", "normal"),
    };

    public static bool TryGet(string? enemyName, out ConfirmedBoss boss)
    {
        if (string.IsNullOrWhiteSpace(enemyName))
        {
            boss = default;
            return false;
        }
        string normalized = EpisodeLifecycleState.NormalizeEnemyType(enemyName!);
        return Entries.TryGetValue(normalized, out boss);
    }
}
