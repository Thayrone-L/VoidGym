using System;
using System.Collections.Generic;

namespace HkAgentBridge;

internal static class MenuActionPolicy
{
    public const int ButtonCount = 7;
    public const int MaximumLeaseMilliseconds = 100;

    public static MenuActionSelection Validate(
        IEnumerable<string> buttons,
        int leaseMilliseconds
    )
    {
        if (buttons == null) throw new ArgumentNullException(nameof(buttons));
        if (leaseMilliseconds < 1 || leaseMilliseconds > MaximumLeaseMilliseconds)
        {
            throw new ArgumentOutOfRangeException(
                nameof(leaseMilliseconds),
                $"lease_ms must be between 1 and {MaximumLeaseMilliseconds}"
            );
        }

        int count = 0;
        int mask = 0;
        string? normalizedButton = null;
        foreach (string button in buttons)
        {
            count++;
            if (count > 1)
            {
                throw new ArgumentException(
                    "Exactly one menu button is required per command",
                    nameof(buttons)
                );
            }

            mask = ButtonMask(button);
            normalizedButton = button;
        }

        if (count != 1 || normalizedButton == null)
        {
            throw new ArgumentException(
                "Exactly one menu button is required per command",
                nameof(buttons)
            );
        }

        return new MenuActionSelection(normalizedButton, mask);
    }

    private static int ButtonMask(string button)
    {
        switch (button)
        {
            case "LEFT":
                return 1 << 0;
            case "RIGHT":
                return 1 << 1;
            case "UP":
                return 1 << 2;
            case "DOWN":
                return 1 << 3;
            case "SUBMIT":
                return 1 << 4;
            case "CANCEL":
                return 1 << 5;
            case "PAUSE":
                return 1 << 6;
            default:
                throw new ArgumentException(
                    $"Unsupported menu button: {button}",
                    nameof(button)
                );
        }
    }
}

internal readonly struct MenuActionSelection
{
    public MenuActionSelection(string button, int mask)
    {
        Button = button;
        Mask = mask;
    }

    public string Button { get; }
    public int Mask { get; }
}
