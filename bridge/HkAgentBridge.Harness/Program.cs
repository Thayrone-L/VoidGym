using System.Buffers.Binary;
using System.Net.Sockets;
using System.Text.Json;

const int ProtocolVersion = 1;
const int MaxMessageBytes = 1_048_576;

if (args.Length != 5)
{
    Console.Error.WriteLine(
        "Usage: HkAgentBridge.Harness <host> <port> <run-id> <instance-id> <token>"
    );
    return 2;
}

string host = args[0];
int port = int.Parse(args[1]);
string runId = args[2];
string instanceId = args[3];
string token = args[4];
string sessionId = Guid.NewGuid().ToString();
string traceId = Guid.NewGuid().ToString();

using var client = new TcpClient();
await client.ConnectAsync(host, port);
await using NetworkStream stream = client.GetStream();

var hello = new
{
    protocol_version = ProtocolVersion,
    message_type = "HELLO",
    run_id = runId,
    instance_id = instanceId,
    session_id = sessionId,
    seq = 0,
    trace_id = traceId,
    sent_monotonic_ns = MonotonicNanoseconds(),
    payload = new
    {
        token,
        game_profile = "development",
        game_version = "bridge-harness",
        bridge_version = "0.0.0-dev",
        capabilities = Array.Empty<string>(),
    },
};

await WriteMessageAsync(stream, hello);
using JsonDocument response = await ReadMessageAsync(stream);
JsonElement root = response.RootElement;
if (root.GetProperty("message_type").GetString() != "WELCOME")
{
    throw new InvalidDataException("Supervisor did not return WELCOME");
}
if (root.GetProperty("run_id").GetString() != runId)
{
    throw new InvalidDataException("WELCOME run_id mismatch");
}

Console.WriteLine($"WELCOME instance={instanceId} session={sessionId}");
return 0;

static async Task WriteMessageAsync<T>(NetworkStream stream, T value)
{
    byte[] body = JsonSerializer.SerializeToUtf8Bytes(value);
    if (body.Length is 0 or > MaxMessageBytes)
    {
        throw new InvalidDataException($"Invalid message length: {body.Length}");
    }

    byte[] header = new byte[sizeof(uint)];
    BinaryPrimitives.WriteUInt32BigEndian(header, checked((uint)body.Length));
    await stream.WriteAsync(header);
    await stream.WriteAsync(body);
    await stream.FlushAsync();
}

static long MonotonicNanoseconds()
{
    long timestamp = System.Diagnostics.Stopwatch.GetTimestamp();
    return checked((long)(timestamp * (1_000_000_000.0 / System.Diagnostics.Stopwatch.Frequency)));
}

static async Task<JsonDocument> ReadMessageAsync(NetworkStream stream)
{
    byte[] header = new byte[sizeof(uint)];
    await stream.ReadExactlyAsync(header);
    uint length = BinaryPrimitives.ReadUInt32BigEndian(header);
    if (length is 0 or > MaxMessageBytes)
    {
        throw new InvalidDataException($"Invalid message length: {length}");
    }

    byte[] body = new byte[checked((int)length)];
    await stream.ReadExactlyAsync(body);
    return JsonDocument.Parse(body);
}
