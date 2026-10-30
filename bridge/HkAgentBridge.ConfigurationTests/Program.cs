using HkAgentBridge;
using System.Text.Json;

if (args.Length == 2 && args[0] == "--environment-json")
{
    foreach (Dictionary<string, string> environment in JsonSerializer.Deserialize<List<Dictionary<string, string>>>(File.ReadAllText(args[1]))!)
    {
        ClearEnvironment();
        foreach ((string name, string value) in environment)
            Environment.SetEnvironmentVariable(name, value);
        BridgeConfiguration configuration = BridgeConfiguration.FromEnvironment();
        bool expectedInstance = environment.ContainsKey("HK_AGENT_INSTANCE_MODE");
        Assert(configuration.InstanceMode == expectedInstance, "Python mode matches DLL mode");
        Assert((configuration.SaveRoot is null) == expectedInstance, "Python save-root contract matches DLL");
    }
    Console.WriteLine("PASS: Python-produced legacy and instance environments accepted");
    return;
}

string root = Path.Combine(Path.GetTempPath(), "hk-agent-bridge-tests-" + Guid.NewGuid());
string dataRoot = Path.Combine(root, "instance-data");
string saveRoot = Path.Combine(root, "save-root");
Directory.CreateDirectory(dataRoot);
Directory.CreateDirectory(saveRoot);

try
{
    SetCommonEnvironment();
    Environment.SetEnvironmentVariable("HK_AGENT_INSTANCE_MODE", "1");
    Environment.SetEnvironmentVariable("HK_AGENT_PRODUCT_NAME", "Hollow Knight AI 1");
    Environment.SetEnvironmentVariable("HK_AGENT_INSTANCE_DATA_ROOT", dataRoot);
    BridgeConfiguration instance = BridgeConfiguration.FromEnvironment();
    Assert(instance.InstanceMode && instance.SaveRoot is null, "instance without SaveRoot accepted");

    Environment.SetEnvironmentVariable("HK_AGENT_SAVE_ROOT", saveRoot);
    AssertThrows<BridgeConfigurationException>(() => _ = BridgeConfiguration.FromEnvironment(), "instance with SaveRoot rejected");

    Environment.SetEnvironmentVariable("HK_AGENT_INSTANCE_MODE", null);
    Environment.SetEnvironmentVariable("HK_AGENT_PRODUCT_NAME", null);
    Environment.SetEnvironmentVariable("HK_AGENT_INSTANCE_DATA_ROOT", null);
    Environment.SetEnvironmentVariable("HK_AGENT_SAVE_ROOT", null);
    AssertThrows<BridgeConfigurationException>(() => _ = BridgeConfiguration.FromEnvironment(), "legacy without SaveRoot rejected");

    Environment.SetEnvironmentVariable("HK_AGENT_SAVE_ROOT", saveRoot);
    BridgeConfiguration legacy = BridgeConfiguration.FromEnvironment();
    Assert(!legacy.InstanceMode && legacy.SaveRoot == Path.GetFullPath(saveRoot), "legacy with SaveRoot accepted");

    BridgeConfiguration.ValidateInstanceRuntime(
        "Hollow Knight AI 1", dataRoot, "Hollow Knight AI 1", dataRoot,
        Path.Combine(root, "Team Cherry", "Hollow Knight")
    );
    AssertThrows<BridgeConfigurationException>(() => BridgeConfiguration.ValidateInstanceRuntime(
        "Hollow Knight AI 1", dataRoot, "Hollow Knight AI 2", dataRoot,
        Path.Combine(root, "Team Cherry", "Hollow Knight")
    ), "divergent productName rejected");
    AssertThrows<BridgeConfigurationException>(() => BridgeConfiguration.ValidateInstanceRuntime(
        "Hollow Knight AI 1", dataRoot, "Hollow Knight AI 1", saveRoot,
        Path.Combine(root, "Team Cherry", "Hollow Knight")
    ), "divergent persistentDataPath rejected");
    string personalRoot = Path.Combine(root, "Team Cherry", "Hollow Knight");
    AssertThrows<BridgeConfigurationException>(() => BridgeConfiguration.ValidateInstanceRuntime(
        "Hollow Knight AI 1", personalRoot, "Hollow Knight AI 1", personalRoot, personalRoot
    ), "personal save path rejected");

    byte[] bottomUp = {
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18,
    };
    byte[] expectedTopDown = {
        10, 11, 12, 13, 14, 15, 16, 17, 18,
        1, 2, 3, 4, 5, 6, 7, 8, 9,
    };
    byte[] actualTopDown = FrameEncoding.FlipVerticalRgb24(bottomUp, 3, 2);
    Assert(actualTopDown.SequenceEqual(expectedTopDown), "3x2 RGB24 rows flipped to top-down");
    AssertThrows<ArgumentException>(
        () => FrameEncoding.FlipVerticalRgb24(new byte[17], 3, 2),
        "RGB24 row flip rejects truncated payload"
    );

    Console.WriteLine("PASS: 10 bridge configuration/runtime/frame assertions");
}
finally
{
    Directory.Delete(root, recursive: true);
    ClearEnvironment();
}

static void SetCommonEnvironment()
{
    Environment.SetEnvironmentVariable("HK_AGENT_HOST", "127.0.0.1");
    Environment.SetEnvironmentVariable("HK_AGENT_PORT", "43123");
    Environment.SetEnvironmentVariable("HK_AGENT_RUN_ID", "run-test");
    Environment.SetEnvironmentVariable("HK_AGENT_INSTANCE_ID", "instance-test");
    Environment.SetEnvironmentVariable("HK_AGENT_TOKEN", "token-test");
    Environment.SetEnvironmentVariable("HK_AGENT_GAME_PROFILE", "development");
}

static void ClearEnvironment()
{
    foreach (string name in new[] {
        "HK_AGENT_HOST", "HK_AGENT_PORT", "HK_AGENT_RUN_ID", "HK_AGENT_INSTANCE_ID",
        "HK_AGENT_TOKEN", "HK_AGENT_GAME_PROFILE", "HK_AGENT_INSTANCE_MODE",
        "HK_AGENT_PRODUCT_NAME", "HK_AGENT_INSTANCE_DATA_ROOT", "HK_AGENT_SAVE_ROOT"
    }) Environment.SetEnvironmentVariable(name, null);
}

static void Assert(bool condition, string message)
{
    if (!condition) throw new InvalidOperationException("FAIL: " + message);
}

static void AssertThrows<TException>(Action action, string message) where TException : Exception
{
    try { action(); }
    catch (TException) { return; }
    throw new InvalidOperationException("FAIL: " + message);
}
