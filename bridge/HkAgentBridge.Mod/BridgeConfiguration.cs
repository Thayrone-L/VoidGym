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
    public string SaveRoot { get; }

    private BridgeConfiguration(
        string host,
        int port,
        string runId,
        string instanceId,
        string token,
        string gameProfile,
        string saveRoot
    )
    {
        Host = host;
        Port = port;
        RunId = runId;
        InstanceId = instanceId;
        Token = token;
        GameProfile = gameProfile;
        SaveRoot = saveRoot;
    }

    public static BridgeConfiguration FromEnvironment()
    {
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
            CanonicalDirectory("HK_AGENT_SAVE_ROOT")
        );
    }

    private static string CanonicalDirectory(string name)
    {
        string value = Required(name);
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
        if (!Path.IsPathRooted(path) || !Directory.Exists(path))
        {
            throw new BridgeConfigurationException($"{name} must be an existing absolute directory");
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
