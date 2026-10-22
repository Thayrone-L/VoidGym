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
using UnityEngine.SceneManagement;

namespace HkAgentBridge;

internal static class MapLayoutDump
{
    private const float RequestCheckIntervalSeconds = 1f;
    private static string? _requestPath;
    private static bool _finished;
    private static float _nextRequestCheck;

    internal static void Register()
    {
        string? request = Environment.GetEnvironmentVariable("HK_AGENT_MAP_LAYOUT_DUMP_REQUEST");
        if (string.IsNullOrWhiteSpace(request)) return;
        _requestPath = Path.GetFullPath(request);
        SetStatus("bridge-loaded");
        ModHooks.HeroUpdateHook += OnHeroUpdate;
        TryDump();
    }

    private static void OnHeroUpdate() => TryDump();

    private static void TryDump()
    {
        if (_finished || _requestPath == null) return;
        if (Time.realtimeSinceStartup < _nextRequestCheck) return;
        _nextRequestCheck = Time.realtimeSinceStartup + RequestCheckIntervalSeconds;
        try
        {
            if (!File.Exists(_requestPath)) return;
            JObject result = Collect();
            string output = Path.Combine(Path.GetDirectoryName(_requestPath)!, "game-map-layout.json");
            string temporary = output + ".tmp";
            File.WriteAllText(temporary, result.ToString(Formatting.Indented), new UTF8Encoding(false));
            if (File.Exists(output)) File.Delete(output);
            File.Move(temporary, output);
            _finished = true;
            Unsubscribe();
            File.Delete(_requestPath);
            SetStatus("dump-complete");
            Application.Quit();
        }
        catch (InvalidOperationException error) when (error.Message.StartsWith("not ready:", StringComparison.Ordinal))
        {
            SetStatus(error.Message);
        }
        catch (Exception error)
        {
            Fail(error);
        }
    }

    private static void Fail(Exception error)
    {
        _finished = true;
        Unsubscribe();
        try
        {
            if (_requestPath != null)
                File.WriteAllText(_requestPath + ".error.txt", error.ToString(), new UTF8Encoding(false));
        }
        catch (Exception) { }
        SetStatus("failed: " + error.GetType().Name);
        try
        {
            if (_requestPath != null && File.Exists(_requestPath)) File.Delete(_requestPath);
        }
        catch (Exception) { }
        Application.Quit();
    }

    private static void Unsubscribe()
    {
        ModHooks.HeroUpdateHook -= OnHeroUpdate;
    }

    private static void SetStatus(string status)
    {
        if (_requestPath == null) return;
        try
        {
            File.WriteAllText(_requestPath + ".status.txt", status, new UTF8Encoding(false));
        }
        catch (Exception) { }
    }

    private static GameObject? FindInstantiatedGameMap(out string diagnostic)
    {
        GameManager? manager = GameManager.instance;
        GameObject? managerMap = manager == null ? null : manager.gameMap;
        GameMap? managerComponent = managerMap == null ? null : managerMap.GetComponent<GameMap>();
        if (IsInstantiatedMap(managerMap, managerComponent))
        {
            diagnostic = "GameManager.gameMap";
            return managerMap;
        }

        GameMap[] candidates = Resources.FindObjectsOfTypeAll<GameMap>()
            .Where(item => item != null && item.gameObject.scene.IsValid())
            .OrderByDescending(item => item.gameObject.activeInHierarchy)
            .ToArray();
        GameMap? candidate = candidates.FirstOrDefault();
        diagnostic = string.Join(
            ", ",
            candidates.Take(8).Select(item => $"{item.gameObject.name}@{item.gameObject.scene.name}")
        );
        return candidate?.gameObject;
    }

    private static bool IsInstantiatedMap(GameObject? map, GameMap? component)
    {
        return map != null && component != null && map.scene.IsValid();
    }

    private static JObject Collect()
    {
        Scene activeScene = UnityEngine.SceneManagement.SceneManager.GetActiveScene();
        GameManager? manager = GameManager.instance;
        GameObject? map = FindInstantiatedGameMap(out string candidates);
        if (map == null)
        {
            throw new InvalidOperationException(
                $"not ready: cena={activeScene.name}; GameManager={(manager == null ? "ausente" : "presente")}; " +
                $"candidatos GameMap={candidates}"
            );
        }

        GameMap? component = map.GetComponent<GameMap>();
        if (component == null || !map.scene.IsValid())
        {
            throw new InvalidOperationException(
                $"not ready: cena={activeScene.name}; GameManager={(manager == null ? "ausente" : "presente")}; " +
                $"mapa={map.name}; componente GameMap ou cena instanciada ausente"
            );
        }

        var scenes = new Dictionary<string, (string area, Bounds bounds, bool hasSprite)>();
        int areaCount = 0;
        int duplicateSceneEntries = 0;
        foreach (Transform area in map.transform)
        {
            areaCount++;
            foreach (Transform sceneObject in area)
            {
                SpriteRenderer? sprite = sceneObject.GetComponent<SpriteRenderer>();
                Bounds bounds = default;
                bool hasSprite = sprite != null && sprite.sprite != null;
                if (hasSprite) bounds = sprite!.bounds;
                if (scenes.TryGetValue(sceneObject.name, out var previous))
                {
                    duplicateSceneEntries++;
                    if (hasSprite && previous.hasSprite)
                    {
                        previous.bounds.Encapsulate(bounds);
                        scenes[sceneObject.name] = (previous.area, previous.bounds, true);
                    }
                    else if (hasSprite)
                    {
                        scenes[sceneObject.name] = (previous.area, bounds, true);
                    }
                    continue;
                }
                scenes[sceneObject.name] = (area.name, bounds, hasSprite);
            }
        }
        if (scenes.Count == 0)
        {
            throw new InvalidOperationException(
                $"not ready: cena={activeScene.name}; GameManager=presente; mapa={map.name}; " +
                "hierarquia área -> cena sem salas diretas"
            );
        }

        string assemblyPath = Path.Combine(Application.dataPath, "Managed", "Assembly-CSharp.dll");
        string assemblyHash;
        using (var stream = File.OpenRead(assemblyPath))
        using (var sha = SHA256.Create())
            assemblyHash = BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();

        int roomsWithSprite = scenes.Values.Count(item => item.hasSprite);
        int roomsWithoutSprite = scenes.Count - roomsWithSprite;
        var rows = new JArray(scenes.OrderBy(item => item.Key, StringComparer.Ordinal).Select(item =>
        {
            var row = new JObject
            {
                ["scene"] = item.Key,
                ["area"] = item.Value.area,
                ["has_sprite"] = item.Value.hasSprite,
            };
            if (item.Value.hasSprite)
            {
                Bounds bounds = item.Value.bounds;
                row["rect"] = new JObject
                {
                    ["x"] = bounds.min.x,
                    ["y"] = bounds.min.y,
                    ["width"] = bounds.size.x,
                    ["height"] = bounds.size.y,
                };
            }
            else
            {
                row["rect"] = JValue.CreateNull();
            }
            return row;
        }));
        return new JObject
        {
            ["schema_version"] = 1,
            ["game_version"] = Application.version,
            ["game_assembly_sha256"] = assemblyHash,
            ["map_object"] = map.name,
            ["map_component"] = nameof(GameMap),
            ["map_reference"] = "GameManager.gameMap",
            ["active_scene"] = activeScene.name,
            ["hierarchy"] = "GameMap -> area -> scene",
            ["area_count"] = areaCount,
            ["scene_count"] = scenes.Count,
            ["rooms_with_sprite"] = roomsWithSprite,
            ["rooms_without_sprite"] = roomsWithoutSprite,
            ["duplicate_scene_entries"] = duplicateSceneEntries,
            ["scenes"] = rows,
        };
    }
}
