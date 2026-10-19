using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using InControl;
using UnityEngine;

namespace HkAgentBridge;

internal sealed class AgentInputBackend : IDisposable
{
    // Long enough for an uncharmed Focus heal, still bounded by the input
    // watchdog and followed by RELEASE_ALL on the Python side.
    private const int MaximumLeaseMilliseconds = 2000;

    private static readonly IReadOnlyDictionary<string, int> ButtonMasks =
        new Dictionary<string, int>(StringComparer.Ordinal)
        {
            ["LEFT"] = 1 << 0,
            ["RIGHT"] = 1 << 1,
            ["UP"] = 1 << 2,
            ["DOWN"] = 1 << 3,
            ["JUMP"] = 1 << 4,
            ["ATTACK"] = 1 << 5,
            ["DASH"] = 1 << 6,
            ["CAST"] = 1 << 7,
            ["SUPER_DASH"] = 1 << 8,
            ["DREAM_NAIL"] = 1 << 9,
            ["PAUSE"] = 1 << 10,
            ["MENU_SUBMIT"] = 1 << 11,
            ["MENU_CANCEL"] = 1 << 12,
        };

    private readonly Action<string> _log;
    private readonly List<ActionBinding> _addedBindings = new();
    private readonly object _stateLock = new();
    private readonly GameObject _pumpObject;
    private readonly PointerInputGuard _pointerGuard;
    private readonly AudioMuteGuard _audioMute;
    private AgentInputDevice? _device;
    private HeroActions? _boundActions;
    private InputDevice? _previousDevice;
    private int _desiredMask;
    private long _leaseExpiresAt;
    private int _desiredMenuMask;
    private long _menuLeaseExpiresAt;
    private bool _disposed;

    public AgentInputBackend(Action<string> log)
    {
        _log = log;
        _pointerGuard = new PointerInputGuard(log);
        _audioMute = new AudioMuteGuard(log);
        _pumpObject = new GameObject("HK Agent Input Pump");
        UnityEngine.Object.DontDestroyOnLoad(_pumpObject);
        InputPump pump = _pumpObject.AddComponent<InputPump>();
        pump.Initialize(this);
    }

    public void AttachWhenReady()
    {
        if (_disposed || InputHandler.Instance == null)
        {
            return;
        }

        HeroActions actions = InputHandler.Instance.inputActions;
        if (_device == null)
        {
            _device = new AgentInputDevice(ReadInputState, _log);
            InputManager.AttachDevice(_device);
            _boundActions = actions;
            _previousDevice = actions.Device;
            _log("Virtual in-process input device attached");
        }
        else if (!ReferenceEquals(_boundActions, actions))
        {
            RemoveAddedBindings();
            _boundActions = actions;
            _previousDevice = actions.Device;
        }

        bool deviceRestored = !ReferenceEquals(actions.Device, _device);
        if (deviceRestored)
        {
            actions.Device = _device;
        }

        int bindingsRestored = EnsureBindings(actions);
        if (deviceRestored || bindingsRestored > 0)
        {
            _log(
                $"Virtual input route restored: device={deviceRestored}, " +
                $"bindings={bindingsRestored}"
            );
        }
    }

    private int EnsureBindings(HeroActions actions)
    {
        int restored = 0;
        restored += EnsureBinding(actions.left, InputControlType.Button0);
        restored += EnsureBinding(actions.right, InputControlType.Button1);
        restored += EnsureBinding(actions.up, InputControlType.Button2);
        restored += EnsureBinding(actions.down, InputControlType.Button3);
        restored += EnsureBinding(actions.jump, InputControlType.Button4);
        restored += EnsureBinding(actions.attack, InputControlType.Button5);
        restored += EnsureBinding(actions.dash, InputControlType.Button6);
        restored += EnsureBinding(actions.cast, InputControlType.Button7);
        restored += EnsureBinding(actions.superDash, InputControlType.Button8);
        restored += EnsureBinding(actions.dreamNail, InputControlType.Button9);
        restored += EnsureBinding(actions.left, InputControlType.Button10);
        restored += EnsureBinding(actions.right, InputControlType.Button11);
        restored += EnsureBinding(actions.up, InputControlType.Button12);
        restored += EnsureBinding(actions.down, InputControlType.Button13);
        restored += EnsureBinding(actions.menuSubmit, InputControlType.Button14);
        restored += EnsureBinding(actions.menuCancel, InputControlType.Button15);
        restored += EnsureBinding(actions.pause, InputControlType.Button16);
        return restored;
    }

    public string[] ApplyAction(IEnumerable<string> buttons, int leaseMilliseconds)
    {
        if (leaseMilliseconds < 1 || leaseMilliseconds > MaximumLeaseMilliseconds)
        {
            throw new ArgumentOutOfRangeException(
                nameof(leaseMilliseconds),
                $"lease_ms must be between 1 and {MaximumLeaseMilliseconds}"
            );
        }

        int mask = 0;
        var normalized = new List<string>();
        foreach (string button in buttons)
        {
            if (!ButtonMasks.TryGetValue(button, out int buttonMask))
            {
                throw new ArgumentException($"Unsupported button: {button}", nameof(buttons));
            }
            if ((mask & buttonMask) == 0)
            {
                mask |= buttonMask;
                normalized.Add(button);
            }
        }
        long leaseTicks = checked(
            (long)(leaseMilliseconds * (Stopwatch.Frequency / 1000.0))
        );
        lock (_stateLock)
        {
            ReleaseMenuUnsafe();
            _desiredMask = mask;
            _leaseExpiresAt = Stopwatch.GetTimestamp() + leaseTicks;
        }
        return normalized.ToArray();
    }

    public string[] ApplyMenuAction(IEnumerable<string> buttons, int leaseMilliseconds)
    {
        MenuActionSelection selection = MenuActionPolicy.Validate(
            buttons,
            leaseMilliseconds
        );

        long leaseTicks = checked(
            (long)(leaseMilliseconds * (Stopwatch.Frequency / 1000.0))
        );
        lock (_stateLock)
        {
            ReleaseGameplayUnsafe();
            _desiredMenuMask = selection.Mask;
            _menuLeaseExpiresAt = Stopwatch.GetTimestamp() + leaseTicks;
        }
        _device?.BeginMenuObservation();
        HeroActions? actions = InputHandler.Instance?.inputActions;
        InputControlType control = (InputControlType)(
            (int)InputControlType.Button10 + MenuBitIndex(selection.Mask)
        );
        bool bindingPresent = actions != null && MenuAction(actions, selection.Button)
            .HasBinding(new DeviceBindingSource(control));
        _log(
            $"Menu action armed: button={selection.Button}, binding_present={bindingPresent}, " +
            $"device_bound={actions != null && ReferenceEquals(actions.Device, _device)}, " +
            $"input_enabled={InputManager.Enabled}, suspend_background={InputManager.SuspendInBackground}"
        );
        return new[] { selection.Button };
    }

    private static int MenuBitIndex(int mask)
    {
        int index = 0;
        while ((mask >>= 1) != 0) index++;
        return index;
    }

    private static PlayerAction MenuAction(HeroActions actions, string button)
    {
        switch (button)
        {
            case "LEFT": return actions.left;
            case "RIGHT": return actions.right;
            case "UP": return actions.up;
            case "DOWN": return actions.down;
            case "SUBMIT": return actions.menuSubmit;
            case "CANCEL": return actions.menuCancel;
            case "PAUSE": return actions.pause;
            default: throw new ArgumentOutOfRangeException(nameof(button));
        }
    }

    public void ReleaseAll()
    {
        lock (_stateLock)
        {
            ReleaseGameplayUnsafe();
            ReleaseMenuUnsafe();
        }
    }

    private void ReleaseGameplayUnsafe()
    {
        _desiredMask = 0;
        _leaseExpiresAt = 0;
    }

    private void ReleaseMenuUnsafe()
    {
        _desiredMenuMask = 0;
        _menuLeaseExpiresAt = 0;
    }

    private InputStateSnapshot ReadInputState()
    {
        lock (_stateLock)
        {
            long now = Stopwatch.GetTimestamp();
            if (_leaseExpiresAt == 0 || now >= _leaseExpiresAt)
            {
                ReleaseGameplayUnsafe();
            }
            if (_menuLeaseExpiresAt == 0 || now >= _menuLeaseExpiresAt)
            {
                ReleaseMenuUnsafe();
            }

            return new InputStateSnapshot(_desiredMask, _desiredMenuMask);
        }
    }

    private int EnsureBinding(PlayerAction action, InputControlType control)
    {
        var source = new DeviceBindingSource(control);
        if (action.HasBinding(source))
        {
            return 0;
        }
        if (action.AddBinding(source))
        {
            _addedBindings.Add(new ActionBinding(action, source));
            return 1;
        }
        return 0;
    }

    private void RemoveAddedBindings()
    {
        foreach (ActionBinding binding in _addedBindings)
        {
            binding.Action.RemoveBinding(binding.Source);
        }
        _addedBindings.Clear();
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        ReleaseAll();

        RemoveAddedBindings();

        if (_device != null)
        {
            if (InputHandler.Instance != null &&
                ReferenceEquals(InputHandler.Instance.inputActions.Device, _device))
            {
                InputHandler.Instance.inputActions.Device = _previousDevice;
            }
            InputManager.DetachDevice(_device);
            _device = null;
        }
        UnityEngine.Object.Destroy(_pumpObject);
        _log("Virtual in-process input device detached");
    }

    private sealed class InputPump : MonoBehaviour
    {
        private AgentInputBackend? _owner;

        public void Initialize(AgentInputBackend owner) => _owner = owner;

        private void Update()
        {
            if (_owner == null)
            {
                return;
            }
            _owner.AttachWhenReady();
            if (!_owner._disposed)
            {
                _owner._pointerGuard.Enforce();
                _owner._audioMute.Enforce();
            }
        }
    }

    private sealed class ActionBinding
    {
        public ActionBinding(PlayerAction action, DeviceBindingSource source)
        {
            Action = action;
            Source = source;
        }

        public PlayerAction Action { get; }
        public DeviceBindingSource Source { get; }
    }

    private sealed class AgentInputDevice : InputDevice
    {
        private readonly Func<InputStateSnapshot> _readInputState;
        private readonly Action<string> _log;
        private int _menuObservationPending;

        public AgentInputDevice(Func<InputStateSnapshot> readInputState, Action<string> log)
            : base("HK Agent Virtual Controller")
        {
            _readInputState = readInputState;
            _log = log;
            // The action set is pinned to this device explicitly. Keeping it passive
            // prevents InControl from promoting it to the global active controller;
            // Hollow Knight handles that promotion by resetting/remapping inputActions,
            // which would remove the agent's menu bindings before EventSystem reads them.
            Passive = true;
            int controlCount = Math.Max(ButtonMasks.Count, 10 + MenuActionPolicy.ButtonCount);
            for (int index = 0; index < controlCount; index++)
            {
                InputControlType control =
                    (InputControlType)((int)InputControlType.Button0 + index);
                AddControl(control, $"Agent Button {index}");
            }
        }

        public void BeginMenuObservation() =>
            Interlocked.Exchange(ref _menuObservationPending, 1);

        public override void Update(ulong updateTick, float deltaTime)
        {
            InputStateSnapshot state = _readInputState();
            if (state.MenuMask != 0 &&
                Interlocked.Exchange(ref _menuObservationPending, 0) == 1)
            {
                _log($"Virtual device observed menu mask on update tick {updateTick}");
            }
            for (int index = 0; index < ButtonMasks.Count; index++)
            {
                // The gameplay pause token shares bit 10 in the wire mask, but
                // must target Hollow Knight's dedicated pause binding at Button16.
                InputControlType control = index switch
                {
                    10 => InputControlType.Button16,
                    11 => InputControlType.Button14,
                    12 => InputControlType.Button15,
                    _ => (InputControlType)((int)InputControlType.Button0 + index),
                };
                UpdateWithState(
                    control,
                    (state.GameplayMask & (1 << index)) != 0,
                    updateTick,
                    deltaTime
                );
            }

            for (int index = 0; index < MenuActionPolicy.ButtonCount; index++)
            {
                InputControlType control =
                    (InputControlType)((int)InputControlType.Button10 + index);
                UpdateWithState(
                    control,
                    (state.MenuMask & (1 << index)) != 0,
                    updateTick,
                    deltaTime
                );
            }
        }
    }

    private readonly struct InputStateSnapshot
    {
        public InputStateSnapshot(int gameplayMask, int menuMask)
        {
            GameplayMask = gameplayMask;
            MenuMask = menuMask;
        }

        public int GameplayMask { get; }
        public int MenuMask { get; }
    }
}
