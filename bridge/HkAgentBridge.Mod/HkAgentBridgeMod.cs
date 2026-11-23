using System;
using System.Reflection;
using System.IO;
using System.Globalization;
using System.Linq;
using System.Threading;
using Modding;
using UnityEngine;

namespace HkAgentBridge;

public sealed class HkAgentBridgeMod : Mod
{
    private BridgeClient? _client;
    private AgentInputBackend? _input;
    private VisualCaptureBackend? _capture;
    private readonly EpisodeLifecycleState _lifecycle = new();
    private readonly OperatorTelemetryState _operatorTelemetry = new();
    private readonly object _operatorSnapshotLock = new();
    private System.Collections.Generic.Dictionary<string, object?> _operatorSnapshot = new();
    private readonly EnemyObservationState _enemyObservations = new();
    private int _lastObservedHealth = -1;
    private int _episodeActive;
    private int _operatorTelemetryDiagnosticLogged;
    private int _enemyObservationDiagnosticLogged;
    private float _nextGeoRockScan;
    private string? _geoRockScene;
    private readonly System.Collections.Generic.Dictionary<string, int> _geoRockHits = new(StringComparer.Ordinal);
    private readonly System.Collections.Generic.HashSet<string> _brokenGeoRocks = new(StringComparer.Ordinal);
    private int _boundsSceneHandle = int.MinValue;
    private float[]? _cachedSceneBounds;
    private float _lastDetailedOperatorSnapshot = float.NegativeInfinity;
    private bool? _shadePending;
    private static readonly MemberInfo[] ProgressFlagMembers = ReadProgressFlagMembers();
    private static Type? _geoRockComponentType;
    private static bool _geoRockTypeLookupDone;

    public override string GetVersion() =>
        Assembly.GetExecutingAssembly().GetName().Version?.ToString() ?? "0.0.0";

    public override void Initialize()
    {
        MapLayoutDump.Register();
        BridgeConfiguration configuration;
        try
        {
            configuration = BridgeConfiguration.FromEnvironment();
        }
        catch (BridgeConfigurationException exception)
        {
            if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("HK_AGENT_HOST")))
            {
                LogWarn($"Bridge disabled: {exception.Message}");
                return;
            }
            LogError($"Bridge startup failed closed: {exception.Message}");
            Application.Quit();
            return;
        }

        try
        {
            if (configuration.InstanceMode)
            {
                BridgeConfiguration.ValidateInstanceRuntime(
                    configuration.ExpectedProductName!,
                    configuration.ExpectedDataRoot!,
                    Application.productName,
                    Application.persistentDataPath,
                    Path.Combine(
                        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                        "Low",
                        "Team Cherry",
                        "Hollow Knight"
                    )
                );
            }
            else
            {
                ApplySaveRoot(configuration.SaveRoot!);
            }
            Application.runInBackground = true;
        }
        catch (Exception exception)
        {
            LogError($"Bridge save isolation failed closed: {exception}");
            Application.Quit();
            return;
        }

        var input = new AgentInputBackend(message => Log(message));
        var capture = new VisualCaptureBackend(message => Log(message), input.ReadHumanSnapshot);
        _input = input;
        _capture = capture;
        _client = new BridgeClient(
            configuration,
            message => Log(message),
            message => LogError(message),
            ReleaseAll,
            input.SetHumanMode,
            (buttons, leaseMilliseconds) =>
                input.ApplyAction(buttons, leaseMilliseconds),
            (buttons, leaseMilliseconds) =>
                input.ApplyMenuAction(buttons, leaseMilliseconds),
            (encoding, semanticLabels, _combatEvents, cancellationToken) =>
                capture.CaptureOnceAsync(encoding, semanticLabels, cancellationToken),
            SnapshotRewardSignals,
            SnapshotOperatorTelemetry,
            BeginEpisodeLifecycle
        );
        ModHooks.HeroUpdateHook += OnHeroUpdate;
        ModHooks.BeforePlayerDeadHook += OnPlayerDying;
        ModHooks.AfterPlayerDeadHook += OnPlayerDead;
        ModHooks.SetPlayerBoolHook += OnSetPlayerBool;
        ModHooks.OnReceiveDeathEventHook += OnReceiveDeathEvent;
        ModHooks.AfterTakeDamageHook += OnAfterTakeDamage;
        On.HealthManager.TakeDamage += OnHealthManagerTakeDamage;
        On.GeoControl.OnTriggerEnter2D += OnGeoPickupTrigger;
        On.HeroController.TakeGeo += OnHeroTakeGeo;
        ModHooks.ApplicationQuitHook += OnApplicationQuit;
        input.AttachWhenReady();
        _client.Start();
    }

    private static void ApplySaveRoot(string requestedRoot)
    {
        Platform platform = Platform.Current;
        if (!(platform is DesktopPlatform desktop))
        {
            throw new InvalidOperationException("DesktopPlatform is unavailable");
        }
        FieldInfo? field = typeof(DesktopPlatform).GetField(
            "saveDirPath",
            BindingFlags.Instance | BindingFlags.NonPublic
        );
        if (field == null || field.FieldType != typeof(string))
        {
            throw new MissingFieldException(typeof(DesktopPlatform).FullName, "saveDirPath");
        }
        string canonical = Path.GetFullPath(requestedRoot).TrimEnd(
            Path.DirectorySeparatorChar,
            Path.AltDirectorySeparatorChar
        );
        field.SetValue(desktop, canonical);
        string? effective = field.GetValue(desktop) as string;
        if (effective == null || !string.Equals(
            Path.GetFullPath(effective).TrimEnd(
                Path.DirectorySeparatorChar,
                Path.AltDirectorySeparatorChar
            ),
            canonical,
            StringComparison.OrdinalIgnoreCase
        ))
        {
            throw new InvalidOperationException("DesktopPlatform save root verification failed");
        }
    }

    private void OnHeroUpdate()
    {
        _input?.AttachWhenReady();
        bool? shadePending = ReadShadePendingForClassicMode();
        if (shadePending.HasValue)
        {
            _lifecycle.ObserveShadePresence(shadePending.Value);
        }
        PlayerData? playerData = PlayerData.instance;
        if (playerData != null)
        {
            int current = playerData.GetInt("health");
            int previous = Interlocked.Exchange(ref _lastObservedHealth, current);
            bool atBench = playerData.GetBool("atBench");
            int restored = EpisodeLifecycleState.CountHealthRestored(
                previous, current, Volatile.Read(ref _episodeActive) != 0,
                _lifecycle.IsDeathStarted, atBench
            );
            _lifecycle.RecordHealthRestored(restored);
            ObserveOperatorTelemetry(playerData);
            RefreshOperatorSnapshotIfDue();
        }
        ObserveGeoRocks();
    }

    private void ObserveOperatorTelemetry(PlayerData playerData)
    {
        HeroController? hero = HeroController.instance;
        var scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene();
        string? zone = ReadStringMember(playerData, "mapZone");
        float? x = hero == null ? null : hero.transform.position.x;
        float? y = hero == null ? null : hero.transform.position.y;
        Camera? camera = Camera.main;
        float? cameraX = camera == null ? null : camera.transform.position.x;
        float? cameraY = camera == null ? null : camera.transform.position.y;
        if (_boundsSceneHandle != scene.handle)
        {
            _cachedSceneBounds = ReadSceneBounds(scene);
            _boundsSceneHandle = scene.handle;
        }
        float[]? bounds = _cachedSceneBounds;
        int grubs = SafeGetInt(playerData, "grubsCollected");
        int geo = SafeGetInt(playerData, "geo");
        var flags = new System.Collections.Generic.List<System.Collections.Generic.KeyValuePair<string, bool>>();
        foreach (MemberInfo member in ProgressFlagMembers)
        {
            try
            {
                bool value = member is FieldInfo field ? (bool)field.GetValue(playerData)! : (bool)((PropertyInfo)member).GetValue(playerData, null)!;
                flags.Add(new System.Collections.Generic.KeyValuePair<string, bool>(member.Name, value));
            }
            catch (Exception) { }
        }
        var charmCosts = new System.Collections.Generic.Dictionary<string, int>(StringComparer.Ordinal);
        var equippedCharms = new System.Collections.Generic.Dictionary<string, bool>(StringComparer.Ordinal);
        for (int charmId = 1; charmId <= 40; charmId++)
        {
            string flag = "gotCharm_" + charmId.ToString(CultureInfo.InvariantCulture);
            if (SafeProgressFlag(playerData, flag))
                charmCosts[flag] = Math.Min(6, SafeGetInt(playerData, "charmCost_" + charmId.ToString(CultureInfo.InvariantCulture)));
            equippedCharms["equippedCharm_" + charmId.ToString(CultureInfo.InvariantCulture)] =
                SafeProgressFlag(playerData, "equippedCharm_" + charmId.ToString(CultureInfo.InvariantCulture));
        }
        BridgeUiState ui = VisualCaptureBackend.ReadUiState();
        _operatorTelemetry.Observe(scene.name, zone, x, y, cameraX, cameraY, bounds, grubs, geo, flags,
            playerData.GetBool("atBench"), charmCosts, equippedCharms,
            SafeGetInt(playerData, "charmSlotsFilled"), SafeGetInt(playerData, "charmSlots"),
            ui.CharmMenuOpen, ui.CharmCursor);
        if (!scene.name.StartsWith("Menu_", StringComparison.Ordinal) && hero != null && camera != null &&
            Interlocked.CompareExchange(ref _operatorTelemetryDiagnosticLogged, 1, 0) == 0)
        {
            Log($"Operator telemetry gameplay sample: scene={scene.name}; x={x?.ToString("R", CultureInfo.InvariantCulture) ?? "null"}; y={y?.ToString("R", CultureInfo.InvariantCulture) ?? "null"}; camera_x={cameraX?.ToString("R", CultureInfo.InvariantCulture) ?? "null"}; camera_y={cameraY?.ToString("R", CultureInfo.InvariantCulture) ?? "null"}; geo={geo}; bounds={(bounds != null)}");
        }
    }

    private static bool IsProgressFlag(string name)
    {
        string lower = name.ToLowerInvariant();
        return lower.StartsWith("has", StringComparison.Ordinal) || lower.StartsWith("got", StringComparison.Ordinal)
            || lower.StartsWith("met", StringComparison.Ordinal) || lower.StartsWith("opened", StringComparison.Ordinal)
            || lower.StartsWith("entered", StringComparison.Ordinal)
            || lower.Contains("boss") || lower.Contains("defeated") || lower.Contains("killed");
    }

    private static MemberInfo[] ReadProgressFlagMembers()
    {
        const BindingFlags members = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        var result = new System.Collections.Generic.List<MemberInfo>();
        foreach (FieldInfo field in typeof(PlayerData).GetFields(members))
            if (field.FieldType == typeof(bool) && IsProgressFlag(field.Name)) result.Add(field);
        foreach (PropertyInfo property in typeof(PlayerData).GetProperties(members))
            if (property.PropertyType == typeof(bool) && property.GetIndexParameters().Length == 0 && property.GetMethod != null && IsProgressFlag(property.Name)) result.Add(property);
        return result.ToArray();
    }

    private static string? ReadStringMember(object source, string name)
    {
        const BindingFlags members = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        try
        {
            object? value = source.GetType().GetField(name, members)?.GetValue(source)
                ?? source.GetType().GetProperty(name, members)?.GetValue(source, null);
            return value as string;
        }
        catch (Exception) { return null; }
    }

    private static int SafeGetInt(PlayerData playerData, string name)
    {
        try { return Math.Max(0, playerData.GetInt(name)); }
        catch (Exception) { return 0; }
    }

    private static bool SafeProgressFlag(PlayerData playerData, string name)
    {
        try { return playerData.GetBool(name); }
        catch (Exception) { return false; }
    }

    private static float[]? ReadSceneBounds(UnityEngine.SceneManagement.Scene scene)
    {
        try
        {
            bool found = false;
            float minX = 0, minY = 0, maxX = 0, maxY = 0;
            foreach (GameObject root in scene.GetRootGameObjects())
            foreach (Collider2D collider in root.GetComponentsInChildren<Collider2D>(true))
            {
                if (collider == null || !collider.enabled || !collider.gameObject.activeInHierarchy) continue;
                Bounds b = collider.bounds;
                if (!found) { minX = b.min.x; minY = b.min.y; maxX = b.max.x; maxY = b.max.y; found = true; }
                else { minX = Math.Min(minX, b.min.x); minY = Math.Min(minY, b.min.y); maxX = Math.Max(maxX, b.max.x); maxY = Math.Max(maxY, b.max.y); }
            }
            return found ? new[] { minX, minY, maxX, maxY } : null;
        }
        catch (Exception) { return null; }
    }

    private System.Collections.Generic.Dictionary<string, object?> SnapshotOperatorTelemetry()
    {
        lock (_operatorSnapshotLock)
            return new System.Collections.Generic.Dictionary<string, object?>(_operatorSnapshot);
    }

    private void RefreshOperatorSnapshotIfDue()
    {
        if (Time.unscaledTime - _lastDetailedOperatorSnapshot < 0.25f)
            return;
        _lastDetailedOperatorSnapshot = Time.unscaledTime;
        var snapshot = _operatorTelemetry.Snapshot();
        BridgeUiState ui = VisualCaptureBackend.ReadUiState();
        snapshot["dialog_open"] = ui.DialogOpen;
        snapshot["menu_open"] = ui.MenuOpen;
        snapshot["charm_menu_open"] = ui.CharmMenuOpen;
        snapshot["charm_cursor"] = ui.CharmCursor;
        snapshot["dialogue_id"] = ui.DialogueId;
        snapshot["visible_places"] = ReadVisiblePlaces();
        snapshot["visible_enemies"] = ReadVisibleEnemies();
        snapshot["shop_prices_visible"] = ReadVisibleShopPrices(ui.MenuOpen || ui.DialogOpen);
        lock (_operatorSnapshotLock)
            _operatorSnapshot = snapshot;
    }

    private System.Collections.Generic.List<System.Collections.Generic.Dictionary<string, object?>> ReadVisibleEnemies()
    {
        var result = new System.Collections.Generic.List<System.Collections.Generic.Dictionary<string, object?>>();
        Camera? camera = Camera.main;
        HeroController? hero = HeroController.instance;
        if (camera == null || hero == null) return result;
        var seenRoots = new System.Collections.Generic.HashSet<int>();
        HealthManager[] managers;
        try { managers = UnityEngine.Object.FindObjectsOfType<HealthManager>(true); }
        catch (Exception error)
        {
            if (Interlocked.CompareExchange(ref _enemyObservationDiagnosticLogged, 1, 0) == 0)
                LogDebug($"Enemy visibility scan failed: {error.GetType().Name}");
            return result;
        }
        foreach (HealthManager manager in managers)
        {
            try
            {
                if (manager == null || !manager.gameObject.activeInHierarchy) continue;
                Transform root = manager.transform;
                if (root == null || root.root == hero.transform.root) continue;
                string name = EpisodeLifecycleState.NormalizeEnemyType(manager.gameObject.name);
                if (name.Length == 0 || string.Equals(name, "Hollow Shade", StringComparison.OrdinalIgnoreCase)) continue;
                int instanceId = manager.gameObject.GetInstanceID();
                if (!seenRoots.Add(instanceId)) continue;
                bool bossConfirmed = BossCatalog.TryGet(name, out ConfirmedBoss boss);
                Vector3 point = camera.WorldToViewportPoint(manager.transform.position);
                if (point.z <= 0 || point.x < 0 || point.x > 1 || point.y < 0 || point.y > 1) continue;
                result.Add(new System.Collections.Generic.Dictionary<string, object?>
                {
                    ["enemy_type_id"] = EnemyObservationState.TypeId(name),
                    ["instance_id"] = instanceId,
                    ["screen_x"] = point.x,
                    ["screen_y"] = point.y,
                    ["struck_this_encounter"] = _enemyObservations.Hits(instanceId),
                    ["boss_confirmed"] = bossConfirmed,
                    ["boss_key"] = bossConfirmed ? boss.Key : null,
                    ["boss_version"] = bossConfirmed ? boss.Version : null,
                });
            }
            catch (Exception error)
            {
                if (Interlocked.CompareExchange(ref _enemyObservationDiagnosticLogged, 1, 0) == 0)
                    LogDebug($"Enemy visibility entry skipped: {error.GetType().Name}");
            }
        }
        if (Interlocked.CompareExchange(ref _enemyObservationDiagnosticLogged, 1, 0) == 0)
            LogDebug($"Enemy visibility scan found {managers.Length} health managers and {result.Count} visible enemies");
        result.Sort((left, right) =>
        {
            float lx = (float)left["screen_x"]! - 0.5f, ly = (float)left["screen_y"]! - 0.5f;
            float rx = (float)right["screen_x"]! - 0.5f, ry = (float)right["screen_y"]! - 0.5f;
            return (lx * lx + ly * ly).CompareTo(rx * rx + ry * ry);
        });
        if (result.Count > 6) result.RemoveRange(6, result.Count - 6);
        return result;
    }

    private static System.Collections.Generic.List<System.Collections.Generic.Dictionary<string, object?>> ReadVisiblePlaces()
    {
        var places = new System.Collections.Generic.List<(float Distance, System.Collections.Generic.Dictionary<string, object?> Value)>();
        Camera? camera = Camera.main;
        if (camera == null) return new System.Collections.Generic.List<System.Collections.Generic.Dictionary<string, object?>>();
        void AddPlace(string kind, Transform transform)
        {
            Vector3 point = camera.WorldToViewportPoint(transform.position);
            if (point.z <= 0 || point.x < 0 || point.x > 1 || point.y < 0 || point.y > 1) return;
            float dx = point.x - 0.5f, dy = point.y - 0.5f;
            places.Add((dx * dx + dy * dy, new System.Collections.Generic.Dictionary<string, object?>
            { ["kind"] = kind, ["x"] = point.x, ["y"] = point.y,
              ["world_x"] = transform.position.x, ["world_y"] = transform.position.y }));
        }
        try
        {
            foreach (RestBench bench in UnityEngine.Object.FindObjectsOfType<RestBench>())
                if (bench != null && bench.isActiveAndEnabled) AddPlace("bench", bench.transform);
            foreach (SpawnStagMenu station in UnityEngine.Object.FindObjectsOfType<SpawnStagMenu>())
                if (station != null && station.isActiveAndEnabled) AddPlace("stag_station", station.transform);
            foreach (ShopMenuStock shop in UnityEngine.Object.FindObjectsOfType<ShopMenuStock>())
                if (shop != null && shop.isActiveAndEnabled)
                {
                    bool mapShop = PlayerData.instance != null &&
                        (SafeProgressFlag(PlayerData.instance, "metCornifer") || SafeProgressFlag(PlayerData.instance, "openedMapperShop"));
                    AddPlace(mapShop ? "map_shop" : "shop", shop.transform);
                }
        }
        catch (Exception) { }
        places.Sort((left, right) => left.Distance.CompareTo(right.Distance));
        var result = new System.Collections.Generic.List<System.Collections.Generic.Dictionary<string, object?>>();
        foreach (var place in places)
        {
            if (result.Count >= 8) break;
            result.Add(place.Value);
        }
        return result;
    }

    private static System.Collections.Generic.List<System.Collections.Generic.Dictionary<string, object?>> ReadVisibleShopPrices(bool menuOpen)
    {
        var result = new System.Collections.Generic.List<System.Collections.Generic.Dictionary<string, object?>>();
        if (!menuOpen) return result;
        try
        {
            foreach (ShopItemStats item in UnityEngine.Object.FindObjectsOfType<ShopItemStats>())
            {
                if (item == null || !item.isActiveAndEnabled || result.Count >= 64) continue;
                result.Add(new System.Collections.Generic.Dictionary<string, object?>
                { ["item_id"] = item.itemNumber.ToString(CultureInfo.InvariantCulture), ["geo"] = Math.Max(0, item.cost) });
            }
        }
        catch (Exception) { }
        return result;
    }

    private void OnHealthManagerTakeDamage(
        On.HealthManager.orig_TakeDamage original,
        HealthManager self,
        HitInstance hitInstance
    )
    {
        if (self == null)
        {
            original(self, hitInstance);
            return;
        }
        GameObject? source = hitInstance.Source;
        HeroController? hero = HeroController.instance;
        if (hero != null && source != null && self != null &&
            source.transform.root == hero.transform.root &&
            self.transform.root != hero.transform.root)
        {
            _enemyObservations.RecordHit(self.gameObject.GetInstanceID());
            string scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
            _lifecycle.RecordCombatEvent("hit_enemy", scene, StableObjectId(self.transform));
        }
        string objectName = self!.gameObject!.name!.Replace("(Clone)", "").Trim();
        if (hero != null && source != null && objectName == "Hollow Shade" &&
            source.transform.root == hero.transform.root)
        {
            _lifecycle.RecordShadeHit();
        }
        original(self, hitInstance);
    }

    private EpisodeLifecycleSnapshot SnapshotRewardSignals(bool includeCombatEvents)
    {
        string scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
        float? distance = null;
        HeroController? hero = HeroController.instance;
        if (hero != null && _lifecycle.ShadePending)
        {
            GameObject[] objects = UnityEngine.Object.FindObjectsOfType<GameObject>();
            foreach (GameObject item in objects)
            {
                if (item != null && item.activeInHierarchy &&
                    item.name.Replace("(Clone)", "").Trim() == "Hollow Shade" &&
                    item.scene.handle == hero.gameObject.scene.handle)
                {
                    distance = Vector2.Distance(hero.transform.position, item.transform.position);
                    break;
                }
            }
        }
        return _lifecycle.Snapshot(scene, distance, includeCombatEvents);
    }

    private void OnGeoPickupTrigger(
        On.GeoControl.orig_OnTriggerEnter2D original,
        GeoControl self,
        Collider2D collision
    )
    {
        PlayerData? playerData = PlayerData.instance;
        int before = playerData?.GetInt("geo") ?? 0;
        original(self, collision);
        int after = playerData?.GetInt("geo") ?? before;
        int gained = Math.Max(0, after - before);
        _lifecycle.RecordGeoGained(gained);
        if (gained > 0 && collision != null)
        {
            _lifecycle.RecordCombatEvent(
                "geo_collected",
                UnityEngine.SceneManagement.SceneManager.GetActiveScene().name,
                StableObjectId(collision.transform),
                gained
            );
        }
    }

    private void OnHeroTakeGeo(
        On.HeroController.orig_TakeGeo original,
        HeroController self,
        int amount
    )
    {
        PlayerData? playerData = PlayerData.instance;
        int before = playerData?.GetInt("geo") ?? 0;
        original(self, amount);
        int after = playerData?.GetInt("geo") ?? before;
        _lifecycle.RecordGeoSpent(Math.Max(0, before - after));
    }

    private void OnReceiveDeathEvent(
        EnemyDeathEffects enemyDeathEffects,
        bool eventAlreadyReceived,
        ref float? attackDirection,
        ref bool resetDeathEvent,
        ref bool spellBurn,
        ref bool isWatery
    )
    {
        if (!eventAlreadyReceived && enemyDeathEffects != null)
        {
            HealthManager? healthManager = enemyDeathEffects.gameObject.GetComponentInParent<HealthManager>();
            string enemyType = EpisodeLifecycleState.NormalizeEnemyType(
                healthManager == null ? enemyDeathEffects.gameObject.name : healthManager.gameObject.name
            );
            _lifecycle.RecordConfirmedEnemyKill(
                enemyType,
                UnityEngine.SceneManagement.SceneManager.GetActiveScene().name,
                healthManager == null ? 0 : _enemyObservations.Hits(healthManager.gameObject.GetInstanceID())
            );
            if (healthManager != null)
            {
                _lifecycle.RecordCombatEvent(
                    "kill_enemy",
                    UnityEngine.SceneManagement.SceneManager.GetActiveScene().name,
                    StableObjectId(healthManager.transform)
                );
            }
        }
    }

    private void ObserveGeoRocks()
    {
        if (Time.realtimeSinceStartup < _nextGeoRockScan) return;
        _nextGeoRockScan = Time.realtimeSinceStartup + 0.1f;
        string scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene().name;
        if (!string.Equals(scene, _geoRockScene, StringComparison.Ordinal))
        {
            _geoRockScene = scene;
            _geoRockHits.Clear();
            _brokenGeoRocks.Clear();
        }
        if (!_geoRockTypeLookupDone)
        {
            _geoRockComponentType = AppDomain.CurrentDomain.GetAssemblies()
                .Select(assembly => assembly.GetType("GeoRock", false))
                .FirstOrDefault(candidate => candidate != null);
            _geoRockTypeLookupDone = true;
        }
        Type? geoRockType = _geoRockComponentType;
        if (geoRockType == null) return;
        var seen = new System.Collections.Generic.HashSet<string>(StringComparer.Ordinal);
        try
        {
            foreach (UnityEngine.Object item in UnityEngine.Object.FindObjectsOfType(geoRockType))
            {
                if (!(item is Component component) || component == null ||
                    !component.gameObject.activeInHierarchy) continue;
                string objectId = StableObjectId(component.transform);
                int? hits = ReadGeoRockHits(component);
                if (!hits.HasValue) continue;
                seen.Add(objectId);
                if (_geoRockHits.TryGetValue(objectId, out int previous) && hits.Value < previous)
                {
                    int damage = previous - hits.Value;
                    for (int index = 0; index < damage; index++)
                        _lifecycle.RecordCombatEvent("hit_geo_rock", scene, objectId);
                }
                _geoRockHits[objectId] = hits.Value;
                if (hits.Value == 0 && _brokenGeoRocks.Add(objectId))
                    _lifecycle.RecordCombatEvent("geo_rock_broken", scene, objectId);
            }
            foreach (var old in _geoRockHits.ToArray())
            {
                if (seen.Contains(old.Key) || old.Value <= 0 || !_brokenGeoRocks.Add(old.Key)) continue;
                _lifecycle.RecordCombatEvent("hit_geo_rock", scene, old.Key);
                _lifecycle.RecordCombatEvent("geo_rock_broken", scene, old.Key);
            }
            foreach (string objectId in _brokenGeoRocks)
                _geoRockHits.Remove(objectId);
        }
        catch (Exception) { }
    }

    private static int? ReadGeoRockHits(object component)
    {
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        try
        {
            Type type = component.GetType();
            foreach (FieldInfo field in type.GetFields(flags))
            {
                object? value = field.GetValue(component);
                if (value == null) continue;
                if (field.Name.Equals("hitsLeft", StringComparison.OrdinalIgnoreCase) && value is int direct)
                    return Math.Max(0, direct);
                if (field.FieldType.Name == "GeoRockData")
                {
                    FieldInfo? hitsField = field.FieldType.GetField("hitsLeft", flags);
                    if (hitsField?.GetValue(value) is int hits) return Math.Max(0, hits);
                }
            }
            foreach (PropertyInfo property in type.GetProperties(flags))
            {
                if (property.GetIndexParameters().Length != 0 || property.GetMethod == null) continue;
                object? value = property.GetValue(component, null);
                if (property.Name.Equals("hitsLeft", StringComparison.OrdinalIgnoreCase) && value is int direct)
                    return Math.Max(0, direct);
                if (value != null && property.PropertyType.Name == "GeoRockData")
                {
                    FieldInfo? hitsField = property.PropertyType.GetField("hitsLeft", flags);
                    if (hitsField?.GetValue(value) is int hits) return Math.Max(0, hits);
                }
            }
        }
        catch (Exception) { }
        return null;
    }

    private static string StableObjectId(Transform transform)
    {
        var names = new System.Collections.Generic.List<string>();
        for (Transform? node = transform; node != null; node = node.parent)
        {
            string name = EpisodeLifecycleState.NormalizeEnemyType(node.name);
            names.Add(name + "[" + node.GetSiblingIndex().ToString(CultureInfo.InvariantCulture) + "]");
        }
        names.Reverse();
        return string.Join("/", names);
    }

    private int OnAfterTakeDamage(int hazardType, int damageAmount)
    {
        PlayerData? playerData = PlayerData.instance;
        if (playerData == null)
        {
            return damageAmount;
        }
        int currentHealth = playerData.GetInt("health");
        int previousHealth = Interlocked.Exchange(ref _lastObservedHealth, currentHealth);
        if (previousHealth >= 0)
        {
            _lifecycle.RecordHealthLost(Math.Max(0, previousHealth - currentHealth));
        }
        return damageAmount;
    }

    private bool OnSetPlayerBool(string name, bool value)
    {
        if (string.Equals(name, "soulLimited", StringComparison.Ordinal))
        {
            _lifecycle.ObserveShadePresence(value);
            if (_shadePending.HasValue && _shadePending.Value != value && Volatile.Read(ref _episodeActive) != 0)
            {
                HeroController? hero = HeroController.instance;
                _operatorTelemetry.RecordEvent(value ? "shade_created" : "shade_recovered", "shade",
                    hero == null ? null : hero.transform.position.x, hero == null ? null : hero.transform.position.y);
            }
            _shadePending = value;
        }
        return value;
    }

    private void OnPlayerDead()
    {
        HeroController? hero = HeroController.instance;
        _operatorTelemetry.RecordEvent("death", "player_death",
            hero == null ? null : hero.transform.position.x, hero == null ? null : hero.transform.position.y);
        if (_shadePending == true)
            _operatorTelemetry.RecordEvent("shade_lost", "shade_lost",
                hero == null ? null : hero.transform.position.x, hero == null ? null : hero.transform.position.y);
        _lifecycle.MarkPlayerDead();
        ReleaseAll();
        LogDebug("Player death marked as episode terminal");
    }

    private void OnPlayerDying()
    {
        _lifecycle.MarkPlayerDying();
        ReleaseAll();
    }

    private void ReleaseAll()
    {
        _input?.ReleaseAll();
        LogDebug("RELEASE_ALL completed");
    }

    private void OnApplicationQuit()
    {
        ModHooks.ApplicationQuitHook -= OnApplicationQuit;
        ModHooks.HeroUpdateHook -= OnHeroUpdate;
        ModHooks.BeforePlayerDeadHook -= OnPlayerDying;
        ModHooks.AfterPlayerDeadHook -= OnPlayerDead;
        ModHooks.SetPlayerBoolHook -= OnSetPlayerBool;
        ModHooks.OnReceiveDeathEventHook -= OnReceiveDeathEvent;
        ModHooks.AfterTakeDamageHook -= OnAfterTakeDamage;
        On.GeoControl.OnTriggerEnter2D -= OnGeoPickupTrigger;
        On.HeroController.TakeGeo -= OnHeroTakeGeo;
        On.HealthManager.TakeDamage -= OnHealthManagerTakeDamage;
        BridgeClient? client = Interlocked.Exchange(ref _client, null);
        client?.Dispose();
        AgentInputBackend? input = Interlocked.Exchange(ref _input, null);
        input?.Dispose();
        VisualCaptureBackend? capture = Interlocked.Exchange(ref _capture, null);
        capture?.Dispose();
    }

    private void BeginEpisodeLifecycle()
    {
        PlayerData? playerData = PlayerData.instance;
        if (playerData == null)
        {
            throw new InvalidOperationException("PlayerData is unavailable");
        }
        if (playerData.GetInt("permadeathMode") != 0)
        {
            throw new InvalidOperationException(
                "Shade run lifecycle supports Classic mode only"
            );
        }
        _lifecycle.BeginEpisode(playerData.GetBool("soulLimited"));
        _shadePending = playerData.GetBool("soulLimited");
        Interlocked.Exchange(ref _episodeActive, 1);
        Interlocked.Exchange(ref _lastObservedHealth, playerData.GetInt("health"));
    }

    private static bool? ReadShadePendingForClassicMode()
    {
        PlayerData? playerData = PlayerData.instance;
        if (playerData == null || playerData.GetInt("permadeathMode") != 0)
        {
            return null;
        }
        return playerData.GetBool("soulLimited");
    }
}
