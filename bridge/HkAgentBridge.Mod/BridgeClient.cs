using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace HkAgentBridge;

internal sealed class BridgeClient : IDisposable
{
    private const int ProtocolVersion = 1;
    private const int MaxMessageBytes = 1_048_576;
    private const int CaptureTimeoutMilliseconds = 8000;

    private readonly BridgeConfiguration _configuration;
    private readonly Action<string> _log;
    private readonly Action<string> _logError;
    private readonly Action _releaseAll;
    private readonly Func<IEnumerable<string>, int, string[]> _applyAction;
    private readonly Func<IEnumerable<string>, int, string[]> _applyMenuAction;
    private readonly Func<CancellationToken, Task<CapturedFrame>> _captureOnce;
    private readonly Func<EpisodeLifecycleSnapshot> _lifecycleSnapshot;
    private readonly Action _beginEpisodeLifecycle;
    private readonly CancellationTokenSource _cancellation = new();
    private readonly SemaphoreSlim _writeLock = new(1, 1);
    private readonly string _sessionId = Guid.NewGuid().ToString();
    private long _outboundSequence;
    private long _lastInboundSequence;
    private readonly EpisodeCorrelationState _correlation = new();
    private Task? _worker;
    private TcpClient? _tcpClient;

    public BridgeClient(
        BridgeConfiguration configuration,
        Action<string> log,
        Action<string> logError,
        Action releaseAll,
        Func<IEnumerable<string>, int, string[]> applyAction,
        Func<IEnumerable<string>, int, string[]> applyMenuAction,
        Func<CancellationToken, Task<CapturedFrame>> captureOnce,
        Func<EpisodeLifecycleSnapshot> lifecycleSnapshot,
        Action beginEpisodeLifecycle
    )
    {
        _configuration = configuration;
        _log = log;
        _logError = logError;
        _releaseAll = releaseAll;
        _applyAction = applyAction;
        _applyMenuAction = applyMenuAction;
        _captureOnce = captureOnce;
        _lifecycleSnapshot = lifecycleSnapshot;
        _beginEpisodeLifecycle = beginEpisodeLifecycle;
    }

    public void Start()
    {
        if (_worker != null) throw new InvalidOperationException("Bridge client already started");
        _worker = Task.Run(() => RunAsync(_cancellation.Token));
    }

    private async Task RunAsync(CancellationToken cancellationToken)
    {
        try
        {
            var client = new TcpClient();
            _tcpClient = client;
            using (cancellationToken.Register(client.Dispose))
            {
                await client.ConnectAsync(_configuration.Host, _configuration.Port).ConfigureAwait(false);
                NetworkStream stream = client.GetStream();
                JObject hello = CreateEnvelope(
                    "HELLO",
                    0,
                    new Dictionary<string, object>
                    {
                        ["token"] = _configuration.Token,
                        ["game_profile"] = _configuration.GameProfile,
                        ["game_version"] = Constants.GAME_VERSION,
                        ["effective_save_root"] = _configuration.SaveRoot,
                        ["bridge_version"] = "0.0.0-dev",
                        ["capabilities"] = new[]
                        {
                            "capture_once",
                            "frame_png_rgb8",
                            "begin_episode",
                            "apply_action",
                            "apply_menu_action",
                            "player_death_terminal",
                            "shade_run_lifecycle",
                        },
                    }
                );
                await WriteMessageAsync(stream, hello, cancellationToken).ConfigureAwait(false);
                JObject welcome = await ReadMessageAsync(stream, cancellationToken).ConfigureAwait(false);
                if (!string.Equals((string?)welcome["message_type"], "WELCOME", StringComparison.Ordinal))
                {
                    throw new InvalidDataException("Supervisor did not return WELCOME");
                }
                if (!string.Equals((string?)welcome["run_id"], _configuration.RunId, StringComparison.Ordinal))
                {
                    throw new InvalidDataException("WELCOME run_id mismatch");
                }
                _log($"Supervisor handshake complete: instance={_configuration.InstanceId}");

                int heartbeatIntervalMs = (int?)welcome["payload"]?["heartbeat_interval_ms"] ?? 500;
                heartbeatIntervalMs = Math.Max(100, Math.Min(heartbeatIntervalMs, 5000));
                using (var connectionCancellation =
                    CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
                {
                    Task heartbeat = HeartbeatLoopAsync(
                        stream,
                        heartbeatIntervalMs,
                        connectionCancellation.Token
                    );
                    Task commands = CommandLoopAsync(stream, connectionCancellation.Token);
                    try
                    {
                        Task completed = await Task.WhenAny(heartbeat, commands).ConfigureAwait(false);
                        await completed.ConfigureAwait(false);
                    }
                    finally
                    {
                        connectionCancellation.Cancel();
                        try
                        {
                            await Task.WhenAll(heartbeat, commands).ConfigureAwait(false);
                        }
                        catch (OperationCanceledException)
                            when (connectionCancellation.IsCancellationRequested)
                        {
                        }
                    }
                }
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            _log("Bridge connection stopped");
        }
        catch (ObjectDisposedException) when (cancellationToken.IsCancellationRequested)
        {
            _log("Bridge connection stopped");
        }
        catch (Exception exception)
        {
            _logError($"Bridge connection failed: {exception}");
        }
        finally
        {
            _releaseAll();
        }
    }

    private async Task HeartbeatLoopAsync(
        NetworkStream stream,
        int heartbeatIntervalMs,
        CancellationToken cancellationToken
    )
    {
        while (true)
        {
            await Task.Delay(heartbeatIntervalMs, cancellationToken).ConfigureAwait(false);
            await WriteSequencedMessageAsync(
                stream,
                "HEARTBEAT",
                new
                {
                    state = "READY",
                    process_id = Process.GetCurrentProcess().Id,
                },
                null,
                cancellationToken
            ).ConfigureAwait(false);
        }
    }

    private async Task CommandLoopAsync(
        NetworkStream stream,
        CancellationToken cancellationToken
    )
    {
        while (true)
        {
            JObject command = await ReadMessageAsync(stream, cancellationToken).ConfigureAwait(false);
            ValidateCommand(command);
            string messageType = (string)command["message_type"]!;
            string traceId = (string)command["trace_id"]!;
            string responseType = "ACK";
            object payload;
            switch (messageType)
            {
                case "BEGIN_EPISODE":
                    try
                    {
                        string? episodeId = (string?)command["payload"]?["episode_id"];
                        if (string.IsNullOrWhiteSpace(episodeId))
                        {
                            throw new InvalidDataException("BEGIN_EPISODE requires episode_id");
                        }
                        _releaseAll();
                        _beginEpisodeLifecycle();
                        _correlation.BeginEpisode(episodeId!);
                        payload = new
                        {
                            command_type = messageType,
                            status = "accepted",
                            episode_id = episodeId,
                        };
                    }
                    catch (Exception exception)
                        when (exception is InvalidDataException ||
                              exception is ArgumentException ||
                              exception is InvalidOperationException)
                    {
                        responseType = "ERROR";
                        payload = new
                        {
                            command_type = messageType,
                            error_code = "invalid_episode",
                            recoverable = true,
                            detail = exception.Message,
                        };
                    }
                    break;
                case "PING":
                    payload = new { command_type = messageType, status = "ok" };
                    break;
                case "RELEASE_ALL":
                    _releaseAll();
                    payload = new { command_type = messageType, status = "released" };
                    break;
                case "APPLY_ACTION":
                    try
                    {
                        JToken? actionPayload = command["payload"];
                        JArray? buttonValues = actionPayload?["buttons"] as JArray;
                        int? leaseMilliseconds = (int?)actionPayload?["lease_ms"];
                        long? actionId = (long?)actionPayload?["action_id"];
                        string? episodeId = (string?)actionPayload?["episode_id"];
                        if (buttonValues == null || leaseMilliseconds == null || actionId == null ||
                            string.IsNullOrWhiteSpace(episodeId))
                        {
                            throw new InvalidDataException(
                                "APPLY_ACTION requires episode_id, action_id, buttons and lease_ms"
                            );
                        }
                        if (actionId.Value < 0)
                        {
                            throw new InvalidDataException("action_id must be non-negative");
                        }
                        var buttons = new List<string>();
                        foreach (JToken value in buttonValues)
                        {
                            string? button = (string?)value;
                            if (string.IsNullOrWhiteSpace(button))
                            {
                                throw new InvalidDataException("Button names must be strings");
                            }
                            buttons.Add(button!);
                        }
                        _correlation.Snapshot(episodeId!);
                        string[] normalized = _applyAction(buttons, leaseMilliseconds.Value);
                        _correlation.RecordAction(actionId.Value, episodeId!);
                        payload = new
                        {
                            command_type = messageType,
                            status = "accepted",
                            episode_id = episodeId,
                            action_id = actionId.Value,
                            buttons = normalized,
                            lease_ms = leaseMilliseconds.Value,
                        };
                    }
                    catch (Exception exception)
                        when (exception is InvalidDataException ||
                              exception is ArgumentException ||
                              exception is InvalidOperationException ||
                              exception is OverflowException)
                    {
                        responseType = "ERROR";
                        payload = new
                        {
                            command_type = messageType,
                            error_code = "invalid_action",
                            recoverable = true,
                            detail = exception.Message,
                        };
                    }
                    break;
                case "APPLY_MENU_ACTION":
                    try
                    {
                        JToken? actionPayload = command["payload"];
                        JArray? buttonValues = actionPayload?["buttons"] as JArray;
                        int? leaseMilliseconds = (int?)actionPayload?["lease_ms"];
                        if (buttonValues == null || leaseMilliseconds == null)
                        {
                            throw new InvalidDataException(
                                "APPLY_MENU_ACTION requires buttons and lease_ms"
                            );
                        }
                        var buttons = new List<string>();
                        foreach (JToken value in buttonValues)
                        {
                            string? button = (string?)value;
                            if (string.IsNullOrWhiteSpace(button))
                            {
                                throw new InvalidDataException("Menu button names must be strings");
                            }
                            buttons.Add(button!);
                        }
                        string[] normalized = _applyMenuAction(
                            buttons,
                            leaseMilliseconds.Value
                        );
                        payload = new
                        {
                            command_type = messageType,
                            status = "accepted",
                            buttons = normalized,
                            lease_ms = leaseMilliseconds.Value,
                        };
                    }
                    catch (Exception exception)
                        when (exception is InvalidDataException ||
                              exception is ArgumentException ||
                              exception is OverflowException)
                    {
                        responseType = "ERROR";
                        payload = new
                        {
                            command_type = messageType,
                            error_code = "invalid_menu_action",
                            recoverable = true,
                            detail = exception.Message,
                        };
                    }
                    break;
                case "CAPTURE_ONCE":
                    try
                    {
                        string? requestedEpisodeId =
                            (string?)command["payload"]?["episode_id"];
                        CorrelationSnapshot correlation =
                            _correlation.Snapshot(requestedEpisodeId);
                        using (var captureCancellation =
                            CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
                        {
                            captureCancellation.CancelAfter(CaptureTimeoutMilliseconds);
                            CapturedFrame frame = await _captureOnce(captureCancellation.Token)
                                .ConfigureAwait(false);
                            EpisodeLifecycleSnapshot lifecycle = _lifecycleSnapshot();
                            responseType = "FRAME";
                            payload = new
                            {
                                frame_id = frame.FrameId,
                                episode_id = correlation.EpisodeId,
                                capture_monotonic_ns = frame.CaptureMonotonicNanoseconds,
                                width = frame.Width,
                                height = frame.Height,
                                channels = 3,
                                pixel_format = "RGB8",
                                encoding = "png",
                                data_base64 = Convert.ToBase64String(frame.PngBytes),
                                scene_transition = false,
                                player_dead = lifecycle.PlayerDead,
                                death_persisted = lifecycle.PlayerDead,
                                shade_pending = lifecycle.ShadePending,
                                shade_recovered = lifecycle.ShadeRecovered,
                                run_failed = lifecycle.RunFailed,
                                enemy_kills_since_last_frame = lifecycle.EnemyKillsSinceLastFrame,
                                hp_lost_since_last_frame = lifecycle.HealthLostSinceLastFrame,
                                geo_gained_since_last_frame = lifecycle.GeoGainedSinceLastFrame,
                                geo_spent_since_last_frame = lifecycle.GeoSpentSinceLastFrame,
                                duplicate_of = (long?)null,
                                last_applied_action_id = correlation.LastAppliedActionId,
                            };
                        }
                    }
                    catch (OperationCanceledException)
                        when (!cancellationToken.IsCancellationRequested)
                    {
                        responseType = "ERROR";
                        payload = new
                        {
                            command_type = messageType,
                            error_code = "capture_timeout",
                            recoverable = true,
                            detail = $"Capture did not complete within {CaptureTimeoutMilliseconds} ms",
                        };
                    }
                    catch (Exception exception)
                    {
                        responseType = "ERROR";
                        payload = new
                        {
                            command_type = messageType,
                            error_code = "capture_failed",
                            recoverable = true,
                            detail = exception.Message,
                        };
                    }
                    break;
                default:
                    responseType = "ERROR";
                    payload = new
                    {
                        command_type = messageType,
                        error_code = "unsupported_command",
                        recoverable = true,
                    };
                    break;
            }
            await WriteSequencedMessageAsync(
                stream,
                responseType,
                payload,
                traceId,
                cancellationToken
            ).ConfigureAwait(false);
        }
    }

    private void ValidateCommand(JObject command)
    {
        if ((int?)command["protocol_version"] != ProtocolVersion)
        {
            throw new InvalidDataException("Command protocol version mismatch");
        }
        if (!string.Equals((string?)command["run_id"], _configuration.RunId, StringComparison.Ordinal))
        {
            throw new InvalidDataException("Command run_id mismatch");
        }
        if (!string.Equals(
                (string?)command["instance_id"],
                _configuration.InstanceId,
                StringComparison.Ordinal
            ))
        {
            throw new InvalidDataException("Command instance_id mismatch");
        }
        if (!string.Equals((string?)command["session_id"], _sessionId, StringComparison.Ordinal))
        {
            throw new InvalidDataException("Command session_id mismatch");
        }
        if (string.IsNullOrWhiteSpace((string?)command["message_type"]) ||
            string.IsNullOrWhiteSpace((string?)command["trace_id"]))
        {
            throw new InvalidDataException("Command identity is incomplete");
        }
        long? sequence = (long?)command["seq"];
        if (sequence == null || sequence <= _lastInboundSequence)
        {
            throw new InvalidDataException(
                $"Command sequence must be newer than {_lastInboundSequence}"
            );
        }
        _lastInboundSequence = sequence.Value;
    }

    private long NextSequence() => Interlocked.Increment(ref _outboundSequence);

    private JObject CreateEnvelope(
        string messageType,
        long sequence,
        object payload,
        string? traceId = null
    ) =>
        JObject.FromObject(
            new
            {
                protocol_version = ProtocolVersion,
                message_type = messageType,
                run_id = _configuration.RunId,
                instance_id = _configuration.InstanceId,
                session_id = _sessionId,
                seq = sequence,
                trace_id = traceId ?? Guid.NewGuid().ToString(),
                sent_monotonic_ns = MonotonicNanoseconds(),
                payload,
            }
        );

    private static long MonotonicNanoseconds()
    {
        long timestamp = Stopwatch.GetTimestamp();
        return checked((long)(timestamp * (1_000_000_000.0 / Stopwatch.Frequency)));
    }

    private async Task WriteMessageAsync(
        NetworkStream stream,
        JObject value,
        CancellationToken cancellationToken
    )
    {
        await _writeLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await WriteMessageUnlockedAsync(stream, value, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _writeLock.Release();
        }
    }

    private async Task WriteSequencedMessageAsync(
        NetworkStream stream,
        string messageType,
        object payload,
        string? traceId,
        CancellationToken cancellationToken
    )
    {
        await _writeLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            JObject envelope = CreateEnvelope(messageType, NextSequence(), payload, traceId);
            await WriteMessageUnlockedAsync(stream, envelope, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _writeLock.Release();
        }
    }

    private static async Task WriteMessageUnlockedAsync(
        NetworkStream stream,
        JObject value,
        CancellationToken cancellationToken
    )
    {
        byte[] body = Encoding.UTF8.GetBytes(value.ToString(Formatting.None));
        if (body.Length is 0 or > MaxMessageBytes)
        {
            throw new InvalidDataException($"Invalid message length: {body.Length}");
        }
        byte[] header = new byte[sizeof(uint)];
        uint length = checked((uint)body.Length);
        header[0] = (byte)(length >> 24);
        header[1] = (byte)(length >> 16);
        header[2] = (byte)(length >> 8);
        header[3] = (byte)length;
        await stream.WriteAsync(header, 0, header.Length, cancellationToken).ConfigureAwait(false);
        await stream.WriteAsync(body, 0, body.Length, cancellationToken).ConfigureAwait(false);
        await stream.FlushAsync(cancellationToken).ConfigureAwait(false);
    }

    private static async Task<JObject> ReadMessageAsync(
        NetworkStream stream,
        CancellationToken cancellationToken
    )
    {
        byte[] header = new byte[sizeof(uint)];
        await ReadExactlyAsync(stream, header, cancellationToken).ConfigureAwait(false);
        uint length =
            ((uint)header[0] << 24)
            | ((uint)header[1] << 16)
            | ((uint)header[2] << 8)
            | header[3];
        if (length is 0 or > MaxMessageBytes)
        {
            throw new InvalidDataException($"Invalid message length: {length}");
        }
        byte[] body = new byte[checked((int)length)];
        await ReadExactlyAsync(stream, body, cancellationToken).ConfigureAwait(false);
        string json = Encoding.UTF8.GetString(body);
        return JObject.Parse(json);
    }

    private static async Task ReadExactlyAsync(
        Stream stream,
        byte[] buffer,
        CancellationToken cancellationToken
    )
    {
        int offset = 0;
        while (offset < buffer.Length)
        {
            int read = await stream
                .ReadAsync(buffer, offset, buffer.Length - offset, cancellationToken)
                .ConfigureAwait(false);
            if (read == 0) throw new EndOfStreamException();
            offset += read;
        }
    }

    public void Dispose()
    {
        _cancellation.Cancel();
        _tcpClient?.Dispose();
        try
        {
            _worker?.Wait(TimeSpan.FromSeconds(2));
        }
        catch (AggregateException aggregate)
            when (aggregate.InnerExceptions.Count == 1 &&
                  aggregate.InnerException is OperationCanceledException)
        {
        }
        _cancellation.Dispose();
        _writeLock.Dispose();
    }
}
