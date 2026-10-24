using HkAgentBridge;

byte[] labels = SemanticRasterizer.NewGrid();
byte[] priorities = SemanticRasterizer.NewPriorityGrid();
SemanticRasterizer.StampViewportRect(labels, priorities,
    0.25f, 0.5f, 0.5f, 0.75f, 2, 11);

for (int y = 0; y < SemanticRasterizer.Height; y++)
for (int x = 0; x < SemanticRasterizer.Width; x++)
{
    bool expected = x >= 12 && x < 24 && y >= 6 && y < 14;
    byte expectedLabel = expected ? (byte)2 : (byte)0;
    if (labels[y * SemanticRasterizer.Width + x] != expectedLabel)
        throw new InvalidOperationException($"rasterized rectangle differs at ({x}, {y})");
}

SemanticRasterizer.StampViewportRect(labels, priorities,
    0.25f, 0.5f, 0.5f, 0.75f, SemanticRasterizer.Ignore, 50);
if (labels[6 * SemanticRasterizer.Width + 12] != 2)
    throw new InvalidOperationException("ignore label erased an overlapping class");
if (labels[0] != 0)
    throw new InvalidOperationException("ignore label changed an unclassified cell");

SemanticRasterizer.StampViewportRect(labels, priorities,
    0.25f, 0.5f, 0.5f, 0.75f, 3, 60);
if (labels[6 * SemanticRasterizer.Width + 12] != 3)
    throw new InvalidOperationException("hazard priority did not replace platform");

byte[] packed = SemanticRasterizer.Pack4Bit(labels);
if (packed.Length != SemanticRasterizer.Width * SemanticRasterizer.Height / 2)
    throw new InvalidOperationException("4-bit payload length is invalid");

Console.WriteLine("semantic rasterizer rectangle and priority checks passed");
