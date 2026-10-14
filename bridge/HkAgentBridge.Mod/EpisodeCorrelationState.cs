using System;

namespace HkAgentBridge;

internal sealed class EpisodeCorrelationState
{
    private readonly object _gate = new object();
    private string? _episodeId;
    private long? _lastAppliedActionId;

    public void BeginEpisode(string episodeId)
    {
        if (!Guid.TryParse(episodeId, out _))
        {
            throw new ArgumentException("episode_id must be a UUID", nameof(episodeId));
        }
        lock (_gate)
        {
            _episodeId = episodeId;
            _lastAppliedActionId = null;
        }
    }

    public void RecordAction(long actionId, string episodeId)
    {
        if (actionId < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(actionId));
        }
        lock (_gate)
        {
            if (string.IsNullOrWhiteSpace(episodeId) || _episodeId == null)
            {
                throw new InvalidOperationException("an active episode_id is required");
            }
            ValidateRequestedEpisodeLocked(episodeId);
            _lastAppliedActionId = actionId;
        }
    }

    public CorrelationSnapshot Snapshot(string? requestedEpisodeId)
    {
        lock (_gate)
        {
            ValidateRequestedEpisodeLocked(requestedEpisodeId);
            return new CorrelationSnapshot(_episodeId, _lastAppliedActionId);
        }
    }

    private void ValidateRequestedEpisodeLocked(string? requestedEpisodeId)
    {
        if (requestedEpisodeId == null)
        {
            return;
        }
        if (!string.Equals(requestedEpisodeId, _episodeId, StringComparison.Ordinal))
        {
            throw new InvalidOperationException("episode_id does not match the active episode");
        }
    }
}

internal sealed class CorrelationSnapshot
{
    public CorrelationSnapshot(string? episodeId, long? lastAppliedActionId)
    {
        EpisodeId = episodeId;
        LastAppliedActionId = lastAppliedActionId;
    }

    public string? EpisodeId { get; }
    public long? LastAppliedActionId { get; }
}
