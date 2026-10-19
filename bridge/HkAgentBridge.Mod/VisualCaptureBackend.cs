using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace HkAgentBridge;

internal sealed class VisualCaptureBackend : IDisposable
{
    private const int MaximumWidth = 600;
    private const int MaximumHeight = 338;
    private const int MaximumPngBytes = 720_000;
    private const int MaximumRawRgb8Bytes = MaximumWidth * MaximumHeight * 3;

    private readonly ConcurrentQueue<CaptureRequest> _requests = new();
    private readonly Action<string> _log;
    private readonly GameObject _pumpObject;
    private long _nextFrameId = -1;
    private int _outstanding;
    private bool _disposed;
    private readonly Queue<float> _frameDeltas = new();
    private float _frameDeltaTotal;

    public VisualCaptureBackend(Action<string> log)
    {
        _log = log;
        _pumpObject = new GameObject("HK Agent Visual Capture");
        UnityEngine.Object.DontDestroyOnLoad(_pumpObject);
        CapturePump pump = _pumpObject.AddComponent<CapturePump>();
        pump.Initialize(this);
    }

    public Task<CapturedFrame> CaptureOnceAsync(
        string encoding,
        CancellationToken cancellationToken
    )
    {
        if (encoding != "png" && encoding != "raw_rgb8")
        {
            throw new ArgumentException("encoding must be png or raw_rgb8", nameof(encoding));
        }
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(VisualCaptureBackend));
        }
        if (Interlocked.CompareExchange(ref _outstanding, 1, 0) != 0)
        {
            throw new InvalidOperationException("A capture is already pending");
        }

        var request = new CaptureRequest(encoding, cancellationToken);
        request.Completion.Task.ContinueWith(
            _ => Interlocked.Exchange(ref _outstanding, 0),
            CancellationToken.None,
            TaskContinuationOptions.ExecuteSynchronously,
            TaskScheduler.Default
        );
        _requests.Enqueue(request);
        return request.Completion.Task;
    }

    private bool TryTakeRequest(out CaptureRequest request) =>
        _requests.TryDequeue(out request!);

    private IEnumerator CaptureAtEndOfFrame(CaptureRequest request, double queuedToStartMs)
    {
        yield return new WaitForEndOfFrame();
        if (request.IsCompleted)
        {
            yield break;
        }

        Texture2D? screenshot = null;
        Texture2D? output = null;
        RenderTexture? scaled = null;
        RenderTexture? previousRenderTarget = null;
        bool changedRenderTarget = false;
        double captureMs = 0;
        double encodeMs = 0;
        byte[]? frameBytes = null;
        int? pngBytes = null;
        try
        {
            long captureStarted = Stopwatch.GetTimestamp();
            screenshot = ScreenCapture.CaptureScreenshotAsTexture();
            if (screenshot == null || screenshot.width <= 0 || screenshot.height <= 0)
            {
                throw new InvalidOperationException("Unity returned an empty screenshot");
            }

            int width;
            int height;
            CalculateOutputSize(screenshot.width, screenshot.height, out width, out height);
            scaled = RenderTexture.GetTemporary(
                width,
                height,
                0,
                RenderTextureFormat.ARGB32,
                RenderTextureReadWrite.Default
            );
            Graphics.Blit(screenshot, scaled);
            previousRenderTarget = RenderTexture.active;
            RenderTexture.active = scaled;
            changedRenderTarget = true;
            output = new Texture2D(width, height, TextureFormat.RGB24, false);
            output.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
            output.Apply(false, false);
            captureMs = ElapsedMilliseconds(captureStarted);

            long encodeStarted = Stopwatch.GetTimestamp();
            if (request.Encoding == "png")
            {
                byte[] png = output.EncodeToPNG();
                if (png == null || png.Length == 0)
                {
                    throw new InvalidOperationException("Unity returned an empty PNG");
                }
                if (png.Length > MaximumPngBytes)
                {
                    throw new InvalidOperationException(
                        $"Captured PNG exceeds the {MaximumPngBytes}-byte safety limit"
                    );
                }
                frameBytes = png;
                pngBytes = png.Length;
            }
            else
            {
                byte[] bottomUp = output.GetRawTextureData();
                frameBytes = FrameEncoding.FlipVerticalRgb24(
                    bottomUp,
                    output.width,
                    output.height
                );
                if (frameBytes.Length > MaximumRawRgb8Bytes)
                {
                    throw new InvalidOperationException(
                        $"Captured RGB8 frame exceeds the {MaximumRawRgb8Bytes}-byte safety limit"
                    );
                }
            }
            encodeMs = ElapsedMilliseconds(encodeStarted);
            if (frameBytes == null || frameBytes.Length == 0)
                throw new InvalidOperationException("Unity returned an empty frame");

            long frameId = Interlocked.Increment(ref _nextFrameId);
            ActionAvailability? availability = ReadActionAvailability();
            long logStarted = Stopwatch.GetTimestamp();
            _log($"Captured diagnostic frame {frameId}: {output.width}x{output.height}, {frameBytes.Length} bytes ({request.Encoding})");
            double logMs = ElapsedMilliseconds(logStarted);
            request.TrySetResult(
                new CapturedFrame(
                    frameId, MonotonicNanoseconds(), output.width, output.height,
                    request.Encoding, frameBytes, frameBytes.Length, pngBytes,
                    queuedToStartMs, captureMs, encodeMs, logMs,
                    AverageFps(), Application.targetFrameRate, QualitySettings.vSyncCount,
                    availability
                )
            );
        }
        catch (Exception exception)
        {
            request.TrySetException(exception);
        }
        finally
        {
            if (changedRenderTarget)
            {
                RenderTexture.active = previousRenderTarget;
            }
            if (scaled != null)
            {
                RenderTexture.ReleaseTemporary(scaled);
            }
            if (output != null)
            {
                UnityEngine.Object.Destroy(output);
            }
            if (screenshot != null)
            {
                UnityEngine.Object.Destroy(screenshot);
            }
        }
    }

    private static ActionAvailability? ReadActionAvailability()
    {
        PlayerData? data = PlayerData.instance;
        HeroController? hero = HeroController.instance;
        if (data == null || hero == null) return null;
        BridgeUiState ui = ReadUiState();
        long equippedMask = 0;
        int notchUsed = SafeGetInt(data, "charmSlotsFilled");
        int notchTotal = SafeGetInt(data, "charmSlots");
        for (int charmId = 1; charmId <= 40; charmId++)
            if (SafeProgressFlag(data, "equippedCharm_" + charmId.ToString(System.Globalization.CultureInfo.InvariantCulture)))
                equippedMask |= 1L << (charmId - 1);
        return new ActionAvailability(data.MPCharge, data.equippedCharm_33 ? 24 : 33, 33,
            data.fireballLevel, data.quakeLevel, data.screamLevel, data.hasNailArt,
            data.hasCyclone, data.hasDashSlash, data.hasUpwardSlash,
            SafeProgressFlag(data, "hasDash"), SafeProgressFlag(data, "hasWallJump"),
            SafeProgressFlag(data, "hasDoubleJump"), SafeProgressFlag(data, "hasSuperDash"),
            SafeProgressFlag(data, "hasDreamNail"), SafeProgressFlag(data, "gotCharm_31"),
            ui.DialogOpen, ui.MenuOpen, SafeProgressFlag(data, "atBench"), ui.CharmMenuOpen,
            ui.CharmCursor, equippedMask, notchUsed, notchTotal);
    }

    internal static BridgeUiState ReadUiState()
    {
        bool dialogOpen = false;
        string? dialogueId = null;
        try
        {
            foreach (DialogueBox box in UnityEngine.Object.FindObjectsOfType<DialogueBox>())
                if (box != null && box.isActiveAndEnabled && !IsDialogueHidden(box) &&
                    !string.IsNullOrWhiteSpace(box.currentConversation))
                {
                    dialogOpen = true;
                    dialogueId = box.currentConversation;
                    break;
                }
        }
        catch (Exception) { }
        bool menuOpen = false;
        bool charmMenuOpen = false;
        int charmCursor = 0;
        try
        {
            GameManager? manager = GameManager.instance;
            object? dynamicMenu = manager?.ui?.currentDynamicMenu;
            menuOpen = manager != null && (manager.isPaused || dynamicMenu != null);
            if (dynamicMenu != null)
            {
                string menuType = dynamicMenu.GetType().Name;
                charmMenuOpen = menuType.IndexOf("charm", StringComparison.OrdinalIgnoreCase) >= 0;
                charmCursor = ReadNonNegativeIntMember(dynamicMenu, "currentIndex", "cursor", "selectedIndex");
            }
        }
        catch (Exception) { }
        return new BridgeUiState(dialogOpen, menuOpen, dialogueId, charmMenuOpen, charmCursor);
    }

    private static bool IsDialogueHidden(DialogueBox box)
    {
        try
        {
            FieldInfo? hidden = typeof(DialogueBox).GetField("hidden", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            return hidden?.GetValue(box) is bool value && value;
        }
        catch (Exception) { return false; }
    }

    private static bool SafeProgressFlag(PlayerData data, string name)
    {
        try { return data.GetBool(name); }
        catch (Exception) { return false; }
    }

    private static int ReadNonNegativeIntMember(object source, params string[] names)
    {
        const BindingFlags members = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        foreach (string name in names)
        {
            try
            {
                object? value = source.GetType().GetField(name, members)?.GetValue(source)
                    ?? source.GetType().GetProperty(name, members)?.GetValue(source, null);
                if (value is int index && index >= 0) return index;
            }
            catch (Exception) { }
        }
        return 0;
    }

    private static int SafeGetInt(PlayerData data, string name)
    {
        try { return Math.Max(0, data.GetInt(name)); }
        catch (Exception) { return 0; }
    }

    private static void CalculateOutputSize(
        int sourceWidth,
        int sourceHeight,
        out int width,
        out int height
    )
    {
        double scale = Math.Min(
            1.0,
            Math.Min((double)MaximumWidth / sourceWidth, (double)MaximumHeight / sourceHeight)
        );
        width = Math.Max(1, (int)Math.Round(sourceWidth * scale));
        height = Math.Max(1, (int)Math.Round(sourceHeight * scale));
    }

    private static long MonotonicNanoseconds()
    {
        long timestamp = Stopwatch.GetTimestamp();
        return checked((long)(timestamp * (1_000_000_000.0 / Stopwatch.Frequency)));
    }

    private static double ElapsedMilliseconds(long start) =>
        (Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency;

    private float AverageFps() =>
        _frameDeltaTotal > 0 ? _frameDeltas.Count / _frameDeltaTotal : 0;

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        while (_requests.TryDequeue(out CaptureRequest request))
        {
            request.TrySetException(new ObjectDisposedException(nameof(VisualCaptureBackend)));
        }
        UnityEngine.Object.Destroy(_pumpObject);
    }

    private sealed class CapturePump : MonoBehaviour
    {
        private VisualCaptureBackend? _owner;
        private bool _captureInProgress;

        public void Initialize(VisualCaptureBackend owner) => _owner = owner;

        private void Update()
        {
            float delta = Time.unscaledDeltaTime;
            _owner?._frameDeltas.Enqueue(delta);
            if (_owner != null)
            {
                _owner._frameDeltaTotal += delta;
                if (_owner._frameDeltas.Count > 60)
                {
                    _owner._frameDeltaTotal -= _owner._frameDeltas.Dequeue();
                }
            }
            if (_captureInProgress || _owner == null || !_owner.TryTakeRequest(out CaptureRequest request))
            {
                return;
            }
            if (request.IsCompleted)
            {
                return;
            }
            _captureInProgress = true;
            double queuedToStartMs = (Stopwatch.GetTimestamp() - request.QueuedTimestamp)
                * 1000.0 / Stopwatch.Frequency;
            StartCoroutine(RunCapture(request, queuedToStartMs));
        }

        private IEnumerator RunCapture(CaptureRequest request, double queuedToStartMs)
        {
            yield return _owner!.CaptureAtEndOfFrame(request, queuedToStartMs);
            _captureInProgress = false;
        }
    }

    private sealed class CaptureRequest
    {
        private readonly CancellationTokenRegistration _registration;

        public CaptureRequest(string encoding, CancellationToken cancellationToken)
        {
            Encoding = encoding;
            Completion = new TaskCompletionSource<CapturedFrame>(
                TaskCreationOptions.RunContinuationsAsynchronously
            );
            _registration = cancellationToken.Register(() => Completion.TrySetCanceled());
            Completion.Task.ContinueWith(
                _ => _registration.Dispose(),
                CancellationToken.None,
                TaskContinuationOptions.ExecuteSynchronously,
                TaskScheduler.Default
            );
        }

        public TaskCompletionSource<CapturedFrame> Completion { get; }
        public string Encoding { get; }
        public long QueuedTimestamp { get; } = Stopwatch.GetTimestamp();
        public bool IsCompleted => Completion.Task.IsCompleted;
        public void TrySetResult(CapturedFrame frame) => Completion.TrySetResult(frame);
        public void TrySetException(Exception exception) => Completion.TrySetException(exception);
    }
}

internal sealed class CapturedFrame
{
    public CapturedFrame(
        long frameId,
        long captureMonotonicNanoseconds,
        int width,
        int height,
        string encoding,
        byte[] data,
        int payloadByteCount,
        int? pngByteCount,
        double queuedToStartMs,
        double captureMs,
        double encodeMs,
        double logMs,
        float unityFps,
        int targetFrameRate,
        int vSyncCount,
        ActionAvailability? actionAvailability
    )
    {
        FrameId = frameId;
        CaptureMonotonicNanoseconds = captureMonotonicNanoseconds;
        Width = width;
        Height = height;
        Encoding = encoding;
        Data = data;
        PayloadByteCount = payloadByteCount;
        QueuedToStartMs = queuedToStartMs;
        CaptureMs = captureMs;
        EncodeMs = encodeMs;
        PngByteCount = pngByteCount;
        LogMs = logMs;
        UnityFps = unityFps;
        TargetFrameRate = targetFrameRate;
        VSyncCount = vSyncCount;
        ActionAvailability = actionAvailability;
    }

    public long FrameId { get; }
    public long CaptureMonotonicNanoseconds { get; }
    public int Width { get; }
    public int Height { get; }
    public string Encoding { get; }
    public byte[] Data { get; }
    public int PayloadByteCount { get; }
    public double QueuedToStartMs { get; }
    public double CaptureMs { get; }
    public double EncodeMs { get; }
    public int? PngByteCount { get; }
    public double LogMs { get; }
    public float UnityFps { get; }
    public int TargetFrameRate { get; }
    public int VSyncCount { get; }
    public ActionAvailability? ActionAvailability { get; }
}

internal sealed record BridgeUiState(bool DialogOpen, bool MenuOpen, string? DialogueId,
    bool CharmMenuOpen, int CharmCursor);

internal sealed record ActionAvailability(int mp_charge, int spell_cost, int focus_cost,
    int fireball_level, int quake_level, int scream_level, bool has_nail_art,
    bool has_cyclone, bool has_dash_slash, bool has_upward_slash, bool has_dash,
    bool has_walljump, bool has_double_jump, bool has_super_dash, bool has_dream_nail,
    bool has_dashmaster, bool dialog_open, bool menu_open, bool at_bench,
    bool charm_menu_open, int charm_cursor, long equipped_charm_mask,
    int charm_notches_used, int charm_notches_total);
