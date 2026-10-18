using System;
using System.Collections.Generic;
using System.Globalization;

namespace HkAgentBridge;

/// <summary>Pure delta tracker for operator telemetry; none of these fields belong to policy observations.</summary>
internal sealed class OperatorTelemetryState
{
    private readonly object _gate = new();
    private readonly Dictionary<string, bool> _flags = new(StringComparer.Ordinal);
    private readonly List<Dictionary<string, object?>> _events = new();
    private string? _scene;
    private string? _previousScene;
    private string? _zone;
    private int? _grubs;
    private int? _geo;
    private float? _x, _y, _cameraX, _cameraY;
    private float[]? _bounds;

    public void Observe(string? scene, string? zone, float? x, float? y, float? cameraX, float? cameraY,
        float[]? bounds, int? grubs, int? geo, IEnumerable<KeyValuePair<string, bool>> flags)
    {
        lock (_gate)
        {
            _scene = scene;
            _zone = zone;
            _x = Finite(x); _y = Finite(y); _cameraX = Finite(cameraX); _cameraY = Finite(cameraY);
            _bounds = bounds != null && bounds.Length == 4 ? bounds : null;
            if (scene != null && _previousScene != null && scene != _previousScene)
                _events.Add(new Dictionary<string, object?> { ["category"] = "scene_transition", ["from"] = _previousScene, ["to"] = scene, ["unix_ms"] = UtcNow() });
            if (scene != null) _previousScene = scene;
            if (grubs.HasValue)
            {
                if (_grubs.HasValue && grubs.Value > _grubs.Value)
                    for (int i = _grubs.Value + 1; i <= Math.Min(grubs.Value, _grubs.Value + 128); i++)
                        _events.Add(Event("grub", "grub_" + i.ToString(CultureInfo.InvariantCulture)));
                _grubs = grubs;
            }
            _geo = geo;
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
                ["milestone_events_since_last_frame"] = events,
            };
        }
    }

    public void RecordEvent(string category, string key, float? x = null, float? y = null)
    {
        lock (_gate)
        {
            var item = Event(category, key);
            item["scene"] = _scene;
            item["x"] = Finite(x) ?? _x;
            item["y"] = Finite(y) ?? _y;
            _events.Add(item);
        }
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
