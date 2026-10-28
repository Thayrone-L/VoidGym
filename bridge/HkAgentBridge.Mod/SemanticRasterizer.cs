using System;

namespace HkAgentBridge;

internal static class SemanticRasterizer
{
    public const int Width = 48;
    public const int Height = 27;
    public const byte Ignore = 255;

    public static byte[] NewGrid() => new byte[Width * Height];

    public static byte[] NewPriorityGrid() => new byte[Width * Height];

    public static void StampCell(byte[] labels, byte[] priorities, int x, int y,
        byte label, byte priority)
    {
        if (labels == null || priorities == null || labels.Length != Width * Height ||
            priorities.Length != Width * Height)
            throw new ArgumentException("semantic grids must be 27x48");
        if ((uint)x >= Width || (uint)y >= Height)
            throw new ArgumentOutOfRangeException(nameof(x), "cell is outside the semantic grid");
        if (label > 9 && label != Ignore)
            throw new ArgumentOutOfRangeException(nameof(label));
        if (label == Ignore) return;
        int index = y * Width + x;
        if (priority >= priorities[index])
        {
            labels[index] = label;
            priorities[index] = priority;
        }
    }

    public static void StampIgnoreCell(byte[] labels, byte[] priorities, int x, int y,
        byte maximumOverwritablePriority)
    {
        if (labels == null || priorities == null || labels.Length != Width * Height ||
            priorities.Length != Width * Height)
            throw new ArgumentException("semantic grids must be 27x48");
        if ((uint)x >= Width || (uint)y >= Height)
            throw new ArgumentOutOfRangeException(nameof(x), "cell is outside the semantic grid");
        int index = y * Width + x;
        if ((labels[index] == 0 || labels[index] == 1) &&
            priorities[index] <= maximumOverwritablePriority)
        {
            labels[index] = Ignore;
            priorities[index] = 1;
        }
    }

    public static void StampOpenHorizontalBacking(byte[] labels, byte[] priorities,
        float minX, float maxX, float y, bool terrainBelow, byte solidPriority)
    {
        ValidateGrid(labels, priorities);
        if (float.IsNaN(minX) || float.IsInfinity(minX) || float.IsNaN(maxX) ||
            float.IsInfinity(maxX) || float.IsNaN(y) || float.IsInfinity(y) ||
            minX < 0f || maxX > 1f || maxX < minX || y < 0f || y > 1f)
            throw new ArgumentException("open terrain line is outside the viewport");
        if (maxX <= 0f || minX >= 1f) return;
        int x0 = Math.Max(0, Math.Min(Width - 1, (int)Math.Floor(minX * Width)));
        int x1 = Math.Max(x0 + 1, Math.Min(Width, (int)Math.Ceiling(maxX * Width)));
        int lineRow = Math.Max(0, Math.Min(Height - 1,
            (int)Math.Floor((1f - y) * Height)));
        int direction = terrainBelow ? 1 : -1;
        for (int x = x0; x < x1; x++)
        for (int row = lineRow + direction; row >= 0 && row < Height; row += direction)
        {
            int index = row * Width + x;
            if (labels[index] == 1 && priorities[index] >= solidPriority)
                break;
            StampIgnoreCell(labels, priorities, x, row, solidPriority);
        }
    }

    public static void StampClosedContour(byte[] labels, byte[] priorities, float[] points,
        byte label, byte priority)
    {
        ValidateGrid(labels, priorities);
        if (points == null || points.Length < 6 || points.Length % 2 != 0)
            throw new ArgumentException("closed contour must contain at least three points");
        for (int index = 0; index < points.Length; index++)
            if (float.IsNaN(points[index]) || float.IsInfinity(points[index]))
                throw new ArgumentException("closed contour contains a non-finite point");
        if (label > 9 && label != Ignore)
            throw new ArgumentOutOfRangeException(nameof(label));
        if (label == Ignore) return;
        for (int y = 0; y < Height; y++)
        for (int x = 0; x < Width; x++)
        {
            float px = (x + .5f) / Width;
            float py = 1f - (y + .5f) / Height;
            bool inside = false;
            for (int i = 0, j = points.Length / 2 - 1; i < points.Length / 2; j = i++)
            {
                float xi = points[i * 2];
                float yi = points[i * 2 + 1];
                float xj = points[j * 2];
                float yj = points[j * 2 + 1];
                if ((yi > py) != (yj > py) &&
                    px < (xj - xi) * (py - yi) / (yj - yi) + xi)
                    inside = !inside;
            }
            if (inside) StampCell(labels, priorities, x, y, label, priority);
        }
    }

    private static void ValidateGrid(byte[] labels, byte[] priorities)
    {
        if (labels == null || priorities == null || labels.Length != Width * Height ||
            priorities.Length != Width * Height)
            throw new ArgumentException("semantic grids must be 27x48");
    }

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
