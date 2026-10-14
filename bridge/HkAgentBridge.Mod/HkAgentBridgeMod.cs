using System;
using System.Reflection;
using System.IO;
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
    private int _lastObservedHealth = -1;

    public override string GetVersion() =>
        Assembly.GetExecutingAssembly().GetName().Version?.ToString() ?? "0.0.0";

    public override void Initialize()
    {
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
            ApplySaveRoot(configuration.SaveRoot);
            Application.runInBackground = true;
        }
        catch (Exception exception)
        {
            LogError($"Bridge save isolation failed closed: {exception}");
            Application.Quit();
            return;
        }

        var input = new AgentInputBackend(message => Log(message));
        var capture = new VisualCaptureBackend(message => Log(message));
        _input = input;
        _capture = capture;
        _client = new BridgeClient(
            configuration,
            message => Log(message),
            message => LogError(message),
            ReleaseAll,
            (buttons, leaseMilliseconds) =>
                input.ApplyAction(buttons, leaseMilliseconds),
            (buttons, leaseMilliseconds) =>
                input.ApplyMenuAction(buttons, leaseMilliseconds),
            cancellationToken => capture.CaptureOnceAsync(cancellationToken),
            _lifecycle.Snapshot,
            BeginEpisodeLifecycle
        );
        ModHooks.HeroUpdateHook += OnHeroUpdate;
        ModHooks.BeforePlayerDeadHook += OnPlayerDying;
        ModHooks.AfterPlayerDeadHook += OnPlayerDead;
        ModHooks.SetPlayerBoolHook += OnSetPlayerBool;
        ModHooks.OnReceiveDeathEventHook += OnReceiveDeathEvent;
        ModHooks.AfterTakeDamageHook += OnAfterTakeDamage;
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
            Interlocked.Exchange(ref _lastObservedHealth, playerData.GetInt("health"));
        }
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
        _lifecycle.RecordGeoGained(Math.Max(0, after - before));
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
            _lifecycle.RecordConfirmedEnemyKill();
        }
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
        }
        return value;
    }

    private void OnPlayerDead()
    {
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
