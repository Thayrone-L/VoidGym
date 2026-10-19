using System;
using System.Collections.Generic;
using System.Globalization;

namespace HkAgentBridge;

/// <summary>Pure delta tracker for operator telemetry; none of these fields belong to policy observations.</summary>
internal sealed class OperatorTelemetryState
{
    private readonly object _gate = new();
    private readonly Dictionary<string, bool> _flags = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _charmCosts = new(StringComparer.Ordinal);
    private readonly List<Dictionary<string, object?>> _events = new();
    private string? _scene;
    private string? _zone;
    private int? _grubs;
    private int? _geo;
    private float? _x, _y, _cameraX, _cameraY;
    private float[]? _bounds;
    private Dictionary<string, object?>? _pendingTransition;
    private int _stableEntryFrames;
    private float? _entryCandidateX, _entryCandidateY;
    private long _recentDeathUnixMs;
    private bool? _atBench;

    public void Observe(string? scene, string? zone, float? x, float? y, float? cameraX, float? cameraY,
        float[]? bounds, int? grubs, int? geo, IEnumerable<KeyValuePair<string, bool>> flags,
        bool atBench = false, IReadOnlyDictionary<string, int>? charmCosts = null)
    {
        lock (_gate)
        {
            long now = UtcNow();
            float? nextX = Finite(x), nextY = Finite(y);
            float[]? nextBounds = bounds != null && bounds.Length == 4 ? bounds : null;
            if (scene != null && _scene != null && scene != _scene)
            {
                FlushPendingTransition();
                string? reason = ClassifyNonSpatialTransition(_scene, scene, now);
                _pendingTransition = new Dictionary<string, object?>
                {
                    ["category"] = "scene_transition", ["confirmed"] = true,
                    ["from"] = _scene, ["to"] = scene, ["unix_ms"] = now,
                    ["exit_x"] = _x, ["exit_y"] = _y,
                    ["from_map_zone"] = _zone, ["to_map_zone"] = zone,
                    ["exit_position_valid"] = IsInsideScene(_x, _y, _bounds),
                    ["spatial"] = false,
                };
                if (reason != null) _pendingTransition["transition_kind"] = reason;
                _stableEntryFrames = 0;
                _entryCandidateX = _entryCandidateY = null;
            }

            if (_pendingTransition != null && string.Equals(scene, _pendingTransition["to"] as string, StringComparison.Ordinal))
            {
                bool valid = IsInsideScene(nextX, nextY, nextBounds);
                _stableEntryFrames = valid ? _stableEntryFrames + 1 : 0;
                if (valid && _stableEntryFrames == 1)
                {
                    _entryCandidateX = nextX;
                    _entryCandidateY = nextY;
                }
                if (_stableEntryFrames >= 2)
                {
                    _pendingTransition["entry_x"] = _entryCandidateX;
                    _pendingTransition["entry_y"] = _entryCandidateY;
                    bool hasReason = _pendingTransition.TryGetValue("transition_kind", out object? transitionKind)
                        && transitionKind != null;
                    _pendingTransition["spatial"] =
                        !hasReason &&
                        (bool)_pendingTransition["exit_position_valid"]! &&
                        _entryCandidateX.HasValue && _entryCandidateY.HasValue;
                    _events.Add(_pendingTransition);
                    _pendingTransition = null;
                    _stableEntryFrames = 0;
                }
            }
            _scene = scene;
            _zone = zone;
            _x = nextX; _y = nextY; _cameraX = Finite(cameraX); _cameraY = Finite(cameraY);
            _bounds = nextBounds;
            if (grubs.HasValue)
            {
                if (_grubs.HasValue && grubs.Value > _grubs.Value)
                    for (int i = _grubs.Value + 1; i <= Math.Min(grubs.Value, _grubs.Value + 128); i++)
                        _events.Add(Event("grub", "grub_" + i.ToString(CultureInfo.InvariantCulture)));
                _grubs = grubs;
            }
            _geo = geo;
            if (atBench && _atBench != true)
                _events.Add(Event("checkpoint", "bench_" + (scene ?? "unknown")));
            _atBench = atBench;
            _charmCosts.Clear();
            if (charmCosts != null)
                foreach (KeyValuePair<string, int> charm in charmCosts)
                    if (charm.Value >= 0 && charm.Value <= 6) _charmCosts[charm.Key] = charm.Value;
            foreach (KeyValuePair<string, bool> flag in flags)
            {
                bool prior;
                if (_flags.TryGetValue(flag.Key, out prior) && !prior && flag.Value)
                    _events.Add(Event(Classify(flag.Key), flag.Key));
                _flags[flag.Key] = flag.Value;
            }
        }
    }

    public Dictionary<string, object?> Snapshot()
    {
        lock (_gate)
        {
            var events = _events.ToArray();
            _events.Clear();
            return new Dictionary<string, object?>
            {
                ["timestamp_unix_ms"] = UtcNow(), ["scene_name"] = _scene, ["map_zone"] = _zone,
                ["x"] = _x, ["y"] = _y, ["camera_x"] = _cameraX, ["camera_y"] = _cameraY,
                ["scene_bounds"] = _bounds, ["geo_current"] = _geo,
                ["at_bench"] = _atBench == true,
                ["progress_flags"] = new Dictionary<string, bool>(_flags, StringComparer.Ordinal),
                ["charm_costs"] = new Dictionary<string, int>(_charmCosts, StringComparer.Ordinal),
                ["milestone_events_since_last_frame"] = events,
            };
        }
    }

    public void RecordEvent(string category, string key, float? x = null, float? y = null)
    {
        lock (_gate)
        {
            if (string.Equals(category, "death", StringComparison.Ordinal)) _recentDeathUnixMs = UtcNow();
            var item = Event(category, key);
            item["scene"] = _scene;
            item["x"] = Finite(x) ?? _x;
            item["y"] = Finite(y) ?? _y;
            _events.Add(item);
        }
    }

    private void FlushPendingTransition()
    {
        if (_pendingTransition == null) return;
        _pendingTransition["transition_kind"] ??= "unstable_scene_entry";
        _pendingTransition["spatial"] = false;
        _events.Add(_pendingTransition);
        _pendingTransition = null;
        _stableEntryFrames = 0;
    }

    private string? ClassifyNonSpatialTransition(string from, string to, long now)
    {
        if (_recentDeathUnixMs > 0 && now - _recentDeathUnixMs <= 8_000) return "death_respawn";
        string pair = (from + " " + to).ToLowerInvariant();
        foreach (string token in new[] { "dream", "stag", "elevator", "station", "menu", "title" })
            if (pair.IndexOf(token, StringComparison.Ordinal) >= 0) return token;
        return null;
    }

    private static bool IsInsideScene(float? x, float? y, float[]? bounds)
    {
        if (!x.HasValue || !y.HasValue) return false;
        if (bounds == null || bounds.Length != 4) return true;
        return x.Value >= bounds[0] && x.Value <= bounds[2] && y.Value >= bounds[1] && y.Value <= bounds[3];
    }

    private static float? Finite(float? value) => value.HasValue && !float.IsNaN(value.Value) && !float.IsInfinity(value.Value) ? value : null;
    private static long UtcNow() => DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
    private static Dictionary<string, object?> Event(string category, string key) => new()
    { ["category"] = category, ["key"] = key, ["confirmed"] = true, ["unix_ms"] = UtcNow() };
    private static string Classify(string flag)
    {
        string name = flag.ToLowerInvariant();
        if (name.Contains("boss") || name.Contains("defeated") || name.Contains("killed")) return "boss";
        return name.StartsWith("gotcharm_", StringComparison.Ordinal) || name.StartsWith("has", StringComparison.Ordinal) || name.StartsWith("got", StringComparison.Ordinal) ? "item" : "flag";
    }
}
