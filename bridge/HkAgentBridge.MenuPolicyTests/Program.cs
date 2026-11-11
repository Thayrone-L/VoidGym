using System;
using HkAgentBridge;

AssertAccepted("UP", 1);
AssertAccepted("CANCEL", MenuActionPolicy.MaximumLeaseMilliseconds);
AssertRejected(Array.Empty<string>(), 50);
AssertRejected(new[] { "UP", "DOWN" }, 50);
AssertRejected(new[] { "LEFT", "RIGHT" }, 50);
AssertRejected(new[] { "RIGHT", "SUBMIT" }, 50);
AssertRejected(new[] { "SUBMIT", "CANCEL" }, 50);
AssertAccepted("PAUSE", 50);
AssertRejected(new[] { "QUIT" }, 50);
AssertRejected(new[] { "UP" }, 0);
AssertRejected(new[] { "UP" }, MenuActionPolicy.MaximumLeaseMilliseconds + 1);

Console.WriteLine("Menu action policy tests passed");
return 0;

static void AssertAccepted(string button, int leaseMilliseconds)
{
    MenuActionSelection selection = MenuActionPolicy.Validate(
        new[] { button },
        leaseMilliseconds
    );
    if (!string.Equals(selection.Button, button, StringComparison.Ordinal) || selection.Mask == 0)
    {
        throw new InvalidOperationException($"Expected {button} to be accepted");
    }
}

static void AssertRejected(string[] buttons, int leaseMilliseconds)
{
    try
    {
        MenuActionPolicy.Validate(buttons, leaseMilliseconds);
    }
    catch (ArgumentException)
    {
        return;
    }

    throw new InvalidOperationException(
        $"Expected [{string.Join(",", buttons)}] at {leaseMilliseconds} ms to be rejected"
    );
}
