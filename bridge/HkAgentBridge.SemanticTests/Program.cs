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

TestLShapedColliderDoesNotFillItsBoundingBox();
TestDiagonalEdgeOnlyStampsCrossedCells();
TestOpenFloorEdgeLeavesSolidLineAndIgnoredBacking();
TestClosedTerrainContourFillsInterior();
Console.WriteLine("semantic rasterizer shape and priority checks passed");

static void TestLShapedColliderDoesNotFillItsBoundingBox()
{
    byte[] grid = SemanticRasterizer.NewGrid();
    byte[] priorities = SemanticRasterizer.NewPriorityGrid();
    // L-shaped polygon along the viewport's left and bottom edges.
    StampPolygon(grid, priorities, [(0f, 0f), (.12f, 0f), (.12f, .88f), (1f, .88f),
        (1f, 1f), (0f, 1f)], 1, 10);
    int center = (SemanticRasterizer.Height / 2) * SemanticRasterizer.Width +
                 SemanticRasterizer.Width / 2;
    if (grid[center] != 0)
        throw new InvalidOperationException("L-shaped collider filled the center cell");
    if (grid[SemanticRasterizer.Width - 1] != 1 ||
        grid[(SemanticRasterizer.Height - 1) * SemanticRasterizer.Width] != 1)
        throw new InvalidOperationException("L-shaped collider failed to stamp its arms");
}

static void TestDiagonalEdgeOnlyStampsCrossedCells()
{
    byte[] grid = SemanticRasterizer.NewGrid();
    byte[] priorities = SemanticRasterizer.NewPriorityGrid();
    // A zero-width diagonal EdgeCollider2D, clipped against each grid-cell box.
    for (int y = 0; y < SemanticRasterizer.Height; y++)
    for (int x = 0; x < SemanticRasterizer.Width; x++)
    {
        float left = (float)x / SemanticRasterizer.Width;
        float right = (float)(x + 1) / SemanticRasterizer.Width;
        float bottom = 1f - (float)(y + 1) / SemanticRasterizer.Height;
        float top = 1f - (float)y / SemanticRasterizer.Height;
        if (SegmentIntersectsRect((0f, 0f), (1f, 1f), left, bottom, right, top))
            SemanticRasterizer.StampCell(grid, priorities, x, y, 3, 60);
    }
    int painted = grid.Count(value => value == 3);
    if (painted != SemanticRasterizer.Width + SemanticRasterizer.Height - 1)
        throw new InvalidOperationException($"diagonal edge stamped {painted} cells");
    if (grid[SemanticRasterizer.Width - 1] != 3 ||
        grid[(SemanticRasterizer.Height - 1) * SemanticRasterizer.Width] != 3)
        throw new InvalidOperationException("diagonal edge endpoints were not stamped");
    if (grid[SemanticRasterizer.Width / 2] != 0)
        throw new InvalidOperationException("diagonal edge stamped a non-crossed cell");
}

static void TestOpenFloorEdgeLeavesSolidLineAndIgnoredBacking()
{
    byte[] grid = SemanticRasterizer.NewGrid();
    byte[] priorities = SemanticRasterizer.NewPriorityGrid();
    int lineRow = SemanticRasterizer.Height / 2;
    for (int x = 0; x < SemanticRasterizer.Width; x++)
        SemanticRasterizer.StampCell(grid, priorities, x, lineRow, 1, 10);
    SemanticRasterizer.StampOpenHorizontalBacking(grid, priorities, 0f, 1f, .5f,
        terrainBelow: true, solidPriority: 10);
    for (int x = 0; x < SemanticRasterizer.Width; x++)
    {
        if (grid[lineRow * SemanticRasterizer.Width + x] != 1)
            throw new InvalidOperationException("open floor edge was not kept SOLID");
        for (int y = lineRow + 1; y < SemanticRasterizer.Height; y++)
            if (grid[y * SemanticRasterizer.Width + x] != SemanticRasterizer.Ignore)
                throw new InvalidOperationException("cells below the open floor were left EMPTY");
    }
    int hazard = (lineRow + 1) * SemanticRasterizer.Width;
    SemanticRasterizer.StampCell(grid, priorities, 0, lineRow + 1, 3, 60);
    SemanticRasterizer.StampIgnoreCell(grid, priorities, 0, lineRow + 1, 10);
    if (grid[hazard] != 3)
        throw new InvalidOperationException("terrain IGNORE erased an overlapping hazard");
}

static void TestClosedTerrainContourFillsInterior()
{
    byte[] grid = SemanticRasterizer.NewGrid();
    byte[] priorities = SemanticRasterizer.NewPriorityGrid();
    SemanticRasterizer.StampClosedContour(grid, priorities,
        [.25f, .25f, .75f, .25f, .75f, .75f, .25f, .75f], 1, 10);
    byte center = grid[(SemanticRasterizer.Height / 2) * SemanticRasterizer.Width +
                       SemanticRasterizer.Width / 2];
    byte outside = grid[0];
    if (center != 1 || outside != 0)
        throw new InvalidOperationException("closed terrain contour was not filled precisely");
}

static void StampPolygon(byte[] grid, byte[] priorities, (float X, float Y)[] vertices,
    byte label, byte priority)
{
    for (int y = 0; y < SemanticRasterizer.Height; y++)
    for (int x = 0; x < SemanticRasterizer.Width; x++)
    {
        float left = (float)x / SemanticRasterizer.Width;
        float right = (float)(x + 1) / SemanticRasterizer.Width;
        float bottom = 1f - (float)(y + 1) / SemanticRasterizer.Height;
        float top = 1f - (float)y / SemanticRasterizer.Height;
        bool intersects = vertices.Any(point => point.X >= left && point.X <= right &&
                                                    point.Y >= bottom && point.Y <= top) ||
            PointInPolygon((left + right) / 2f, (bottom + top) / 2f, vertices);
        for (int index = 0; index < vertices.Length && !intersects; index++)
        {
            var first = vertices[index];
            var second = vertices[(index + 1) % vertices.Length];
            intersects = SegmentIntersectsRect(first, second, left, bottom, right, top);
        }
        if (intersects)
            SemanticRasterizer.StampCell(grid, priorities, x, y, label, priority);
    }
}

static bool PointInPolygon(float x, float y, (float X, float Y)[] vertices)
{
    bool inside = false;
    for (int i = 0, j = vertices.Length - 1; i < vertices.Length; j = i++)
    {
        var a = vertices[i];
        var b = vertices[j];
        if ((a.Y > y) != (b.Y > y) && x < (b.X - a.X) * (y - a.Y) / (b.Y - a.Y) + a.X)
            inside = !inside;
    }
    return inside;
}

static bool SegmentIntersectsRect((float X, float Y) first, (float X, float Y) second,
    float left, float bottom, float right, float top)
{
    float dx = second.X - first.X;
    float dy = second.Y - first.Y;
    float enter = 0f;
    float leave = 1f;
    return Clip(-dx, first.X - left, ref enter, ref leave) &&
           Clip(dx, right - first.X, ref enter, ref leave) &&
           Clip(-dy, first.Y - bottom, ref enter, ref leave) &&
           Clip(dy, top - first.Y, ref enter, ref leave);
}

static bool Clip(float p, float q, ref float enter, ref float leave)
{
    if (p == 0f) return q >= 0f;
    float ratio = q / p;
    if (p < 0f)
    {
        if (ratio > leave) return false;
        if (ratio > enter) enter = ratio;
    }
    else
    {
        if (ratio < enter) return false;
        if (ratio < leave) leave = ratio;
    }
    return true;
}
