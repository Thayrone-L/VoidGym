using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;

namespace HkAgentBridge;

internal sealed class EnemyObservationState
{
    private readonly object _gate = new();
    private readonly Dictionary<int, int> _hitsByInstance = new();

    public void RecordHit(int instanceId)
    {
        lock (_gate)
        {
            if (_hitsByInstance.TryGetValue(instanceId, out int hits))
                _hitsByInstance[instanceId] = Math.Min(100_000, hits + 1);
            else
                _hitsByInstance[instanceId] = 1;
        }
    }

    public int Hits(int instanceId)
    {
        lock (_gate) return _hitsByInstance.TryGetValue(instanceId, out int hits) ? hits : 0;
    }

    public static string TypeId(string name)
    {
        string canonical = EpisodeLifecycleState.NormalizeEnemyType(name).ToLowerInvariant();
        using SHA256 sha = SHA256.Create();
        byte[] digest = sha.ComputeHash(Encoding.UTF8.GetBytes(canonical));
        var result = new StringBuilder("e_");
        for (int index = 0; index < 5; index++) result.Append(digest[index].ToString("x2"));
        return result.ToString();
    }
}
