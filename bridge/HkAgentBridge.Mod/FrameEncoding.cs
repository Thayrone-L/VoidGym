using System;

namespace HkAgentBridge;

internal static class FrameEncoding
{
    public static byte[] FlipVerticalRgb24(byte[] bottomUpPixels, int width, int height)
    {
        if (bottomUpPixels == null)
        {
            throw new ArgumentNullException(nameof(bottomUpPixels));
        }
        if (width <= 0 || height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width), "Frame dimensions must be positive");
        }

        int rowBytes = checked(width * 3);
        int expectedBytes = checked(rowBytes * height);
        if (bottomUpPixels.Length != expectedBytes)
        {
            throw new ArgumentException("RGB24 payload length does not match frame dimensions", nameof(bottomUpPixels));
        }

        var topDownPixels = new byte[expectedBytes];
        for (int row = 0; row < height; row++)
        {
            Buffer.BlockCopy(
                bottomUpPixels,
                checked((height - row - 1) * rowBytes),
                topDownPixels,
                checked(row * rowBytes),
                rowBytes
            );
        }
        return topDownPixels;
    }
}
