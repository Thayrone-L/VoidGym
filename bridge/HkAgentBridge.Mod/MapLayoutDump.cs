using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using Modding;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace HkAgentBridge;

internal static class MapLayoutDump
{
    private static string? _requestPath;
    private static bool _finished;

    internal static void Register()
    {
        string? request = Environment.GetEnvironmentVariable("HK_AGENT_MAP_LAYOUT_DUMP_REQUEST");
        if (string.IsNullOrWhiteSpace(request)) return;
        _requestPath = Path.GetFullPath(request);
        SetStatus("bridge-loaded");
        ModHooks.HeroUpdateHook += OnHeroUpdate;
        TryDump();
    }

    private static void OnHeroUpdate()
    {
        TryDump();
    }

    private static void TryDump()
    {
        if (_finished || _requestPath == null || !File.Exists(_requestPath)) return;
        JObject result;
        try
        {
            result = Collect();
        }
        catch (InvalidOperationException error) when (error.Message.Contains("not ready"))
        {
            SetStatus("waiting-for-loaded-GameMap");
            return;
        }
        _finished = true;
        try
        {
            string output = Path.Combine(Path.GetDirectoryName(_requestPath)!, "game-map-layout.json");
            string temporary = output + ".tmp";
            File.WriteAllText(temporary, result.ToString(Formatting.Indented), new UTF8Encoding(false));
            if (File.Exists(output)) File.Delete(output);
            File.Move(temporary, output);
            File.Delete(_requestPath);
            SetStatus("dump-complete");
        }
        catch (Exception error)
        {
            File.WriteAllText(_requestPath + ".error.txt", error.ToString(), new UTF8Encoding(false));
            SetStatus("failed: " + error.GetType().Name);
            try { File.Delete(_requestPath); } catch (IOException) { }
        }
        Application.Quit();
    }

    private static void SetStatus(string status)
    {
        if (_requestPath == null) return;
        try { File.WriteAllText(_requestPath + ".status.txt", status, new UTF8Encoding(false)); }
        catch (IOException) { }
    }

    private static JObject Collect()
    {
        GameObject? map = Resources.FindObjectsOfTypeAll<GameObject>()
            .Where(item => item.name == "GameMap")
            .OrderByDescending(item => item.scene.IsValid())
            .FirstOrDefault();
        if (map == null)
            throw new InvalidOperationException("GameMap is not ready; no prefab was instantiated.");
        var scenes = new Dictionary<string, (string area, Bounds bounds, bool hasSprite)>();
        foreach (Transform area in map.transform)
        {
            foreach (Transform sceneObject in area)
            {
                SpriteRenderer? sprite = sceneObject.GetComponent<SpriteRenderer>();
                Bounds bounds = default;
                bool hasSprite = sprite != null && sprite.sprite != null;
                if (hasSprite) bounds = sprite!.bounds;
                if (scenes.TryGetValue(sceneObject.name, out var previous))
                {
                    if (hasSprite && previous.hasSprite)
                    {
                        previous.bounds.Encapsulate(bounds);
                        scenes[sceneObject.name] = (previous.area, previous.bounds, true);
                    }
                    continue;
                }
                scenes[sceneObject.name] = (area.name, bounds, hasSprite);
            }
        }
        if (scenes.Count == 0)
            throw new InvalidOperationException("GameMap is not ready; no direct area/scene children were found.");
        string assemblyPath = Path.Combine(Application.dataPath, "Managed", "Assembly-CSharp.dll");
        string assemblyHash;
        using (var stream = File.OpenRead(assemblyPath))
        using (var sha = SHA256.Create()) assemblyHash = BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        var rows = new JArray(scenes.OrderBy(item => item.Key, StringComparer.Ordinal).Select(item =>
        {
            var row = new JObject { ["scene"] = item.Key, ["area"] = item.Value.area, ["has_sprite"] = item.Value.hasSprite };
            if (item.Value.hasSprite)
            {
                Bounds b = item.Value.bounds;
                row["rect"] = new JObject { ["x"] = b.min.x, ["y"] = b.min.y,
                    ["width"] = b.size.x, ["height"] = b.size.y };
            }
            else row["rect"] = JValue.CreateNull();
            return row;
        }));
        return new JObject { ["schema_version"] = 1, ["game_version"] = Application.version,
            ["game_assembly_sha256"] = assemblyHash, ["map_object"] = map.name, ["scenes"] = rows };
    }
}
