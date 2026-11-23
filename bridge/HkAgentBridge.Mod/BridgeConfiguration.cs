using System;
using System.IO;

namespace HkAgentBridge;

internal sealed class BridgeConfiguration
{
    public string Host { get; }
    public int Port { get; }
    public string RunId { get; }
    public string InstanceId { get; }
    public string Token { get; }
    public string GameProfile { get; }
    public string? SaveRoot { get; }
    public bool InstanceMode { get; }
    public string? ExpectedProductName { get; }
    public string? ExpectedDataRoot { get; }

    private BridgeConfiguration(
        string host,
        int port,
        string runId,
        string instanceId,
        string token,
        string gameProfile,
        string? saveRoot,
        bool instanceMode,
        string? expectedProductName,
        string? expectedDataRoot
    )
    {
        Host = host;
        Port = port;
        RunId = runId;
        InstanceId = instanceId;
        Token = token;
        GameProfile = gameProfile;
        SaveRoot = saveRoot;
        InstanceMode = instanceMode;
        ExpectedProductName = expectedProductName;
        ExpectedDataRoot = expectedDataRoot;
    }

    public static BridgeConfiguration FromEnvironment()
    {
        string? mode = Environment.GetEnvironmentVariable("HK_AGENT_INSTANCE_MODE");
        bool instanceMode = mode == "1";
        if (mode != null && !instanceMode)
        {
            throw new BridgeConfigurationException("HK_AGENT_INSTANCE_MODE must be 1 when set");
        }

        string? saveRootValue = Environment.GetEnvironmentVariable("HK_AGENT_SAVE_ROOT");
        string? saveRoot;
        string? expectedProductName = null;
        string? expectedDataRoot = null;
        if (instanceMode)
        {
            if (saveRootValue != null)
            {
                throw new BridgeConfigurationException("HK_AGENT_SAVE_ROOT cannot be set in instance mode");
            }
            expectedProductName = Required("HK_AGENT_PRODUCT_NAME");
            expectedDataRoot = CanonicalDirectory("HK_AGENT_INSTANCE_DATA_ROOT");
            saveRoot = null;
        }
        else
        {
            saveRoot = CanonicalDirectory("HK_AGENT_SAVE_ROOT");
        }

        string host = Required("HK_AGENT_HOST");
        string portText = Required("HK_AGENT_PORT");
        if (!int.TryParse(portText, out int port) || port is < 1 or > 65535)
        {
            throw new BridgeConfigurationException("HK_AGENT_PORT must be between 1 and 65535");
        }

        return new BridgeConfiguration(
            host,
            port,
            Required("HK_AGENT_RUN_ID"),
            Required("HK_AGENT_INSTANCE_ID"),
            Required("HK_AGENT_TOKEN"),
            Required("HK_AGENT_GAME_PROFILE"),
            saveRoot,
            instanceMode,
            expectedProductName,
            expectedDataRoot
        );
    }

    public static void ValidateInstanceRuntime(
        string expectedProductName,
        string expectedDataRoot,
        string actualProductName,
        string actualDataRoot,
        string personalDataRoot
    )
    {
        if (!string.Equals(actualProductName, expectedProductName, StringComparison.Ordinal))
        {
            throw new BridgeConfigurationException("Unity productName does not match the expected instance");
        }

        string canonicalExpected = CanonicalPath(expectedDataRoot, "expected data root");
        string canonicalActual = CanonicalPath(actualDataRoot, "persistent data path");
        string canonicalPersonal = CanonicalPath(personalDataRoot, "personal data root");
        if (!string.Equals(canonicalActual, canonicalExpected, StringComparison.OrdinalIgnoreCase))
        {
            throw new BridgeConfigurationException("Unity persistentDataPath does not match the expected instance");
        }
        string personalSuffix = Path.Combine("Team Cherry", "Hollow Knight");
        if (string.Equals(canonicalActual, canonicalPersonal, StringComparison.OrdinalIgnoreCase) ||
            canonicalActual.EndsWith(personalSuffix, StringComparison.OrdinalIgnoreCase))
        {
            throw new BridgeConfigurationException("Unity persistentDataPath resolves to personal saves");
        }
    }

    private static string CanonicalDirectory(string name)
    {
        string value = Required(name);
        string path = CanonicalPath(value, name);
        if (!Directory.Exists(path))
        {
            throw new BridgeConfigurationException($"{name} must be an existing absolute directory");
        }
        return path;
    }

    private static string CanonicalPath(string value, string name)
    {
        string path;
        try
        {
            path = Path.GetFullPath(value).TrimEnd(
                Path.DirectorySeparatorChar,
                Path.AltDirectorySeparatorChar
            );
        }
        catch (Exception exception) when (
            exception is ArgumentException ||
            exception is NotSupportedException ||
            exception is PathTooLongException
        )
        {
            throw new BridgeConfigurationException($"{name} is not a valid absolute path");
        }
        if (!Path.IsPathRooted(path))
        {
            throw new BridgeConfigurationException($"{name} must be an absolute path");
        }
        return path;
    }

    private static string Required(string name)
    {
        string? value = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new BridgeConfigurationException($"{name} is missing");
        }
        return value;
    }
}

internal sealed class BridgeConfigurationException : Exception
{
    public BridgeConfigurationException(string message) : base(message) { }
}
