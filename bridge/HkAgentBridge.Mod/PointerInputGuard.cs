using System;
using System.Reflection;
using UnityEngine;

namespace HkAgentBridge;

/// <summary>
/// Keeps the physical mouse from steering Hollow Knight menus while the agent drives them.
/// The menu UI is processed by InControl.HollowKnightInputModule; with mouse input enabled,
/// a host pointer resting over the game window selects whatever it hovers (for example save
/// slot 3 instead of slot 1), which derails the menu bootstrap. The game may re-enable mouse
/// input when it notices a mouse, so the flags are reasserted every frame.
/// Reflection is used so the bridge does not need a compile-time UnityEngine.UI reference.
/// </summary>
internal sealed class PointerInputGuard
{
    private const string ModuleTypeName = "InControl.HollowKnightInputModule";
    private static readonly string[] DisabledFlags = { "allowMouseInput", "focusOnMouseHover" };
    private const float RescanIntervalSeconds = 1.0f;

    private readonly Action<string> _log;
    private readonly FieldInfo?[] _fields = new FieldInfo?[DisabledFlags.Length];
    private Type? _moduleType;
    private bool _typeResolved;
    private bool _unsupportedReported;
    private UnityEngine.Object? _module;
    private int _reportedModuleId;
    private float _nextRescan;

    public PointerInputGuard(Action<string> log) => _log = log;

    public void Enforce()
    {
        if (!ResolveType())
        {
            return;
        }

        UnityEngine.Object? module = _module;
        if (module == null)
        {
            if (Time.unscaledTime < _nextRescan)
            {
                return;
            }
            _nextRescan = Time.unscaledTime + RescanIntervalSeconds;
            module = UnityEngine.Object.FindObjectOfType(_moduleType!);
            _module = module;
            if (module == null)
            {
                return;
            }
        }

        bool changed = false;
        for (int index = 0; index < _fields.Length; index++)
        {
            FieldInfo? field = _fields[index];
            if (field != null && field.GetValue(module!) is true)
            {
                field.SetValue(module!, false);
                changed = true;
            }
        }

        int moduleId = module!.GetInstanceID();
        if (moduleId != _reportedModuleId)
        {
            _reportedModuleId = moduleId;
            _log($"Pointer input disabled on menu input module {moduleId} (changed={changed})");
        }
    }

    private bool ResolveType()
    {
        if (_typeResolved)
        {
            return _moduleType != null;
        }
        _typeResolved = true;
        _moduleType = typeof(InputHandler).Assembly.GetType(ModuleTypeName, throwOnError: false);
        int found = 0;
        if (_moduleType != null)
        {
            for (int index = 0; index < DisabledFlags.Length; index++)
            {
                FieldInfo? field = _moduleType.GetField(
                    DisabledFlags[index],
                    BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic
                );
                if (field != null && field.FieldType == typeof(bool))
                {
                    _fields[index] = field;
                    found++;
                }
            }
        }
        if (_moduleType == null || found == 0)
        {
            _moduleType = null;
            if (!_unsupportedReported)
            {
                _unsupportedReported = true;
                _log($"Pointer input guard unavailable: {ModuleTypeName} mouse flags not found");
            }
            return false;
        }
        return true;
    }
}
