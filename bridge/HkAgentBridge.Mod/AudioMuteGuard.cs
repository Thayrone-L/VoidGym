using System;
using System.Reflection;

namespace HkAgentBridge;

/// <summary>
/// Silences training instances so several games can run side by side without noise.
/// Sets the global UnityEngine.AudioListener.volume to 0 (independent of the game's own
/// volume sliders, which are AudioMixer settings, so no save or PlayerPrefs changes).
/// Reasserted every frame because nothing else should be audible. Disable by setting the
/// environment variable HK_AGENT_MUTE=0. Reflection avoids a compile-time reference to
/// UnityEngine.AudioModule.
/// </summary>
internal sealed class AudioMuteGuard
{
    private const string ListenerTypeName = "UnityEngine.AudioListener, UnityEngine.AudioModule";

    private readonly Action<string> _log;
    private readonly bool _enabled;
    private PropertyInfo? _volume;
    private bool _resolved;
    private bool _reported;

    public AudioMuteGuard(Action<string> log)
    {
        _log = log;
        _enabled = Environment.GetEnvironmentVariable("HK_AGENT_MUTE") != "0";
    }

    public void Enforce()
    {
        if (!_enabled)
        {
            return;
        }
        if (!_resolved)
        {
            _resolved = true;
            Type? listener = Type.GetType(ListenerTypeName, throwOnError: false);
            PropertyInfo? volume = listener?.GetProperty(
                "volume",
                BindingFlags.Public | BindingFlags.Static
            );
            if (volume != null && volume.PropertyType == typeof(float) && volume.CanWrite)
            {
                _volume = volume;
            }
            else
            {
                _log("Audio mute unavailable: UnityEngine.AudioListener.volume not found");
            }
        }
        if (_volume == null)
        {
            return;
        }
        if (_volume.GetValue(null) is float current && current != 0f)
        {
            _volume.SetValue(null, 0f);
            if (!_reported)
            {
                _reported = true;
                _log("Audio muted for training instance (HK_AGENT_MUTE=0 to disable)");
            }
        }
    }
}
