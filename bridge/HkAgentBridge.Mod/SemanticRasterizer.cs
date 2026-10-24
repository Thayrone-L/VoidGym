using System;

namespace HkAgentBridge;

internal static class SemanticRasterizer
{
    public const int Width = 48;
    public const int Height = 27;
    public const byte Ignore = 255;

    public static byte[] NewGrid() => new byte[Width * Height];

    public static byte[] NewPriorityGrid() => new byte[Width * Height];

    public static void StampViewportRect(
        byte[] labels,
        byte[] priorities,
        float minX,
        float minY,
        float maxX,
        float maxY,
        byte label,
        byte priority
    )
    {
        if (labels == null || priorities == null || labels.Length != Width * Height ||
            priorities.Length != Width * Height)
        {
            throw new ArgumentException("semantic grids must be 27x48");
        }
        if (label > 9 && label != Ignore)
            throw new ArgumentOutOfRangeException(nameof(label));
        if (float.IsNaN(minX) || float.IsInfinity(minX) || float.IsNaN(minY) ||
            float.IsInfinity(minY) || float.IsNaN(maxX) || float.IsInfinity(maxX) ||
            float.IsNaN(maxY) || float.IsInfinity(maxY) || maxX < minX || maxY < minY)
        {
            throw new ArgumentException("viewport rectangle is invalid");
        }
        if (label == Ignore)
            return;

        int x0 = Math.Max(0, Math.Min(Width, (int)Math.Floor(minX * Width)));
        int x1 = Math.Max(0, Math.Min(Width, (int)Math.Ceiling(maxX * Width)));
        int y0 = Math.Max(0, Math.Min(Height, (int)Math.Floor((1f - maxY) * Height)));
        int y1 = Math.Max(0, Math.Min(Height, (int)Math.Ceiling((1f - minY) * Height)));
        for (int y = y0; y < y1; y++)
        for (int x = x0; x < x1; x++)
        {
            int index = y * Width + x;
            if (priority >= priorities[index])
            {
                labels[index] = label;
                priorities[index] = priority;
            }
        }
    }

    public static byte[] Pack4Bit(byte[] labels)
    {
        if (labels == null || labels.Length != Width * Height)
            throw new ArgumentException("semantic labels must be a 27x48 grid", nameof(labels));
        var packed = new byte[(labels.Length + 1) / 2];
        for (int index = 0; index < labels.Length; index += 2)
        {
            byte first = ToNibble(labels[index]);
            byte second = ToNibble(labels[index + 1]);
            packed[index / 2] = (byte)((first << 4) | second);
        }
        return packed;
    }

    private static byte ToNibble(byte label)
    {
        if (label <= 9) return label;
        if (label == Ignore) return 15;
        throw new ArgumentOutOfRangeException(nameof(label), "unknown semantic class");
    }
}
