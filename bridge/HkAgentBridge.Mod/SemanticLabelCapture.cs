using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;

namespace HkAgentBridge;

internal sealed class SemanticLabelFrame
{
    public SemanticLabelFrame(byte[] packedGrid, float vx, float vy, bool onGround)
    {
        PackedGrid = packedGrid;
        Vx = vx;
        Vy = vy;
        OnGround = onGround;
    }

    public byte[] PackedGrid { get; }
    public float Vx { get; }
    public float Vy { get; }
    public bool OnGround { get; }
}

internal static class SemanticLabelCapture
{
    private static readonly Collider2D[] OverlapBuffer = new Collider2D[256];
    private static readonly HashSet<int> LoggedSemanticColliders = new HashSet<int>();
    private const int SemanticDiagnosticColliderLimit = 256;
    private const byte Empty = 0;
    private const byte Solid = 1;
    private const byte Platform = 2;
    private const byte Hazard = 3;
    private const byte Enemy = 4;
    private const byte Player = 5;
    private const byte Breakable = 6;
    private const byte Pickup = 7;
    private const byte Interactive = 8;
    private const byte Exit = 9;
    private const byte PrioritySolid = 10;
    private const byte PriorityPlatform = 11;
    private const byte PriorityExit = 20;
    private const byte PriorityInteractive = 30;
    private const byte PriorityPickup = 40;
    private const byte PriorityBreakable = 50;
    private const byte PriorityHazard = 60;
    private const byte PriorityEnemy = 70;
    private const byte PriorityPlayer = 80;

    private static bool LabelBreakables => string.Equals(
        Environment.GetEnvironmentVariable("HK_LABEL_BREAKABLES"), "true",
        StringComparison.OrdinalIgnoreCase
    );

    public static SemanticLabelFrame Capture(Action<string>? log = null)
    {
        Camera? camera = Camera.main;
        if (camera == null)
            throw new InvalidOperationException("semantic labels requested without a main camera");

        byte[] labels = SemanticRasterizer.NewGrid();
        byte[] priorities = SemanticRasterizer.NewPriorityGrid();
        UnityEngine.SceneManagement.Scene scene =
            UnityEngine.SceneManagement.SceneManager.GetActiveScene();
        if (!scene.IsValid())
            throw new InvalidOperationException("semantic labels requested without an active scene");
        PhysicsScene2D physicsScene = scene.GetPhysicsScene2D();
        ContactFilter2D contactFilter = new ContactFilter2D();
        contactFilter.useTriggers = true;
        HeroController? hero = HeroController.instance;
        float gameplayPlaneDepth = camera.WorldToViewportPoint(Vector3.zero).z;
        Vector3 viewBottomLeft = camera.ViewportToWorldPoint(
            new Vector3(0f, 0f, gameplayPlaneDepth));
        Vector3 viewTopRight = camera.ViewportToWorldPoint(
            new Vector3(1f, 1f, gameplayPlaneDepth));
        Vector2 viewSize = new Vector2(Mathf.Abs(viewTopRight.x - viewBottomLeft.x),
            Mathf.Abs(viewTopRight.y - viewBottomLeft.y));
        Vector2 viewCenter = new Vector2((viewBottomLeft.x + viewTopRight.x) * 0.5f,
            (viewBottomLeft.y + viewTopRight.y) * 0.5f);
        int visibleCount = physicsScene.OverlapBox(viewCenter, viewSize, 0f, contactFilter,
            OverlapBuffer);
        bool[] candidateCells = new bool[SemanticRasterizer.Width * SemanticRasterizer.Height];
        List<EdgeCollider2D> terrainEdges = new List<EdgeCollider2D>();
        for (int index = 0; index < visibleCount; index++)
        {
            Collider2D collider = OverlapBuffer[index];
            if (collider == null || !collider.enabled || !collider.gameObject.activeInHierarchy ||
                (collider.gameObject.scene != scene &&
                 (hero == null || collider.GetComponentInParent<HeroController>() != hero)))
                continue;

            LogSemanticCollider(collider, log);
            if (!TryClassify(collider, hero, out byte label, out byte priority))
                continue;
            if (collider is EdgeCollider2D edge && label == Solid)
            {
                terrainEdges.Add(edge);
                continue;
            }
            if (!StampColliderShape(camera, collider, label, priority, labels, priorities))
                MarkCandidateCells(camera, collider.bounds, candidateCells);
        }

        bool[] terrainLineCells = new bool[SemanticRasterizer.Width * SemanticRasterizer.Height];
        foreach (EdgeCollider2D edge in terrainEdges)
            StampTerrainEdge(camera, edge, labels, priorities, terrainLineCells);
        foreach (EdgeCollider2D edge in terrainEdges)
            StampTerrainBacking(camera, edge, labels, priorities, terrainLineCells);

        for (int y = 0; y < SemanticRasterizer.Height; y++)
        for (int x = 0; x < SemanticRasterizer.Width; x++)
        {
            int cellIndex = y * SemanticRasterizer.Width + x;
            if (!candidateCells[cellIndex])
                continue;
            Vector3 lowerLeft = camera.ViewportToWorldPoint(new Vector3(
                (float)x / SemanticRasterizer.Width,
                1f - (float)(y + 1) / SemanticRasterizer.Height,
                gameplayPlaneDepth));
            Vector3 upperRight = camera.ViewportToWorldPoint(new Vector3(
                (float)(x + 1) / SemanticRasterizer.Width,
                1f - (float)y / SemanticRasterizer.Height,
                gameplayPlaneDepth));
            Vector2 size = new Vector2(Mathf.Abs(upperRight.x - lowerLeft.x),
                Mathf.Abs(upperRight.y - lowerLeft.y));
            Vector2 center = new Vector2((upperRight.x + lowerLeft.x) * 0.5f,
                (upperRight.y + lowerLeft.y) * 0.5f);
            int count = physicsScene.OverlapBox(center, size, 0f, contactFilter, OverlapBuffer);
            for (int index = 0; index < count; index++)
            {
                Collider2D collider = OverlapBuffer[index];
                OverlapBuffer[index] = null!;
                if (collider == null || !collider.enabled || !collider.gameObject.activeInHierarchy ||
                    (collider.gameObject.scene != scene &&
                     (hero == null || collider.GetComponentInParent<HeroController>() != hero)))
                    continue;
                if (!TryClassify(collider, hero, out byte label, out byte priority))
                    continue;
                SemanticRasterizer.StampCell(labels, priorities, x, y, label, priority);
            }
        }

        float vx = 0f;
        float vy = 0f;
        bool onGround = false;
        if (hero != null && hero.isActiveAndEnabled)
        {
            float runSpeed = Mathf.Abs(hero.RUN_SPEED);
            if (runSpeed > 0.001f)
            {
                vx = hero.current_velocity.x / runSpeed;
                vy = hero.current_velocity.y / runSpeed;
            }
            onGround = hero.cState != null && hero.cState.onGround;
        }

        return new SemanticLabelFrame(SemanticRasterizer.Pack4Bit(labels), vx, vy, onGround);
    }

    private static void MarkCandidateCells(Camera camera, Bounds bounds, bool[] candidates)
    {
        float depth = bounds.center.z;
        Vector3[] corners =
        {
            new(bounds.min.x, bounds.min.y, depth),
            new(bounds.min.x, bounds.max.y, depth),
            new(bounds.max.x, bounds.min.y, depth),
            new(bounds.max.x, bounds.max.y, depth),
        };
        float minX = float.PositiveInfinity;
        float minY = float.PositiveInfinity;
        float maxX = float.NegativeInfinity;
        float maxY = float.NegativeInfinity;
        foreach (Vector3 corner in corners)
        {
            Vector3 viewport = camera.WorldToViewportPoint(corner);
            if (viewport.z <= 0f) continue;
            minX = Mathf.Min(minX, viewport.x);
            minY = Mathf.Min(minY, viewport.y);
            maxX = Mathf.Max(maxX, viewport.x);
            maxY = Mathf.Max(maxY, viewport.y);
        }
        if (maxX < 0f || minX > 1f || maxY < 0f || minY > 1f ||
            float.IsInfinity(minX) || float.IsInfinity(minY))
            return;

        int x0 = Mathf.Clamp(Mathf.FloorToInt(minX * SemanticRasterizer.Width),
            0, SemanticRasterizer.Width - 1);
        int x1 = Mathf.Clamp(Mathf.CeilToInt(maxX * SemanticRasterizer.Width),
            0, SemanticRasterizer.Width);
        int y0 = Mathf.Clamp(Mathf.FloorToInt((1f - maxY) * SemanticRasterizer.Height),
            0, SemanticRasterizer.Height - 1);
        int y1 = Mathf.Clamp(Mathf.CeilToInt((1f - minY) * SemanticRasterizer.Height),
            0, SemanticRasterizer.Height);
        for (int y = y0; y < y1; y++)
        for (int x = x0; x < x1; x++)
            candidates[y * SemanticRasterizer.Width + x] = true;
    }

    private static bool StampColliderShape(Camera camera, Collider2D collider, byte label,
        byte priority, byte[] labels, byte[] priorities)
    {
        if (collider is BoxCollider2D box)
        {
            Vector2 half = box.size * 0.5f;
            Vector2[] local =
            {
                box.offset + new Vector2(-half.x, -half.y),
                box.offset + new Vector2(-half.x, half.y),
                box.offset + new Vector2(half.x, half.y),
                box.offset + new Vector2(half.x, -half.y),
            };
            StampLocalPolygon(camera, box.transform, local, label, priority, labels, priorities);
            return true;
        }
        if (collider is PolygonCollider2D polygon)
        {
            for (int path = 0; path < polygon.pathCount; path++)
            {
                List<Vector2> local = new List<Vector2>();
                polygon.GetPath(path, local);
                StampLocalPolygon(camera, polygon.transform, local.ToArray(), label, priority,
                    labels, priorities);
            }
            return true;
        }
        if (collider is EdgeCollider2D edge)
        {
            Vector2[] points = edge.points;
            for (int index = 1; index < points.Length; index++)
            {
                Vector3 first = edge.transform.TransformPoint(edge.offset + points[index - 1]);
                Vector3 second = edge.transform.TransformPoint(edge.offset + points[index]);
                StampProjectedSegment(camera.WorldToViewportPoint(first),
                    camera.WorldToViewportPoint(second), label, priority, labels, priorities);
            }
            return true;
        }
        if (collider is CompositeCollider2D composite)
        {
            for (int path = 0; path < composite.pathCount; path++)
            {
                Vector2[] local = new Vector2[composite.GetPathPointCount(path)];
                composite.GetPath(path, local);
                StampLocalPolygon(camera, composite.transform, local, label, priority,
                    labels, priorities);
            }
            return true;
        }
        if (collider is CircleCollider2D circle)
        {
            const int samples = 32;
            Vector2[] local = new Vector2[samples];
            for (int index = 0; index < samples; index++)
            {
                float angle = 2f * Mathf.PI * index / samples;
                local[index] = circle.offset + circle.radius *
                    new Vector2(Mathf.Cos(angle), Mathf.Sin(angle));
            }
            StampLocalPolygon(camera, circle.transform, local, label, priority,
                labels, priorities);
            return true;
        }
        return false;
    }

    private static void StampTerrainEdge(Camera camera, EdgeCollider2D edge, byte[] labels,
        byte[] priorities, bool[] lineCells)
    {
        Vector2[] points = edge.points;
        if (points.Length < 2) return;
        bool closed = points.Length >= 4 && Vector2.Distance(points[0], points[points.Length - 1]) < .001f;
        if (closed)
        {
            float[] contour = new float[(points.Length - 1) * 2];
            for (int index = 0; index < points.Length - 1; index++)
            {
                Vector3 world = edge.transform.TransformPoint(edge.offset + points[index]);
                Vector3 viewport = camera.WorldToViewportPoint(world);
                contour[index * 2] = viewport.x;
                contour[index * 2 + 1] = viewport.y;
            }
            SemanticRasterizer.StampClosedContour(labels, priorities, contour, Solid, PrioritySolid);
        }
        for (int index = 1; index < points.Length; index++)
        {
            Vector3 firstWorld = edge.transform.TransformPoint(edge.offset + points[index - 1]);
            Vector3 secondWorld = edge.transform.TransformPoint(edge.offset + points[index]);
            Vector3 first = camera.WorldToViewportPoint(firstWorld);
            Vector3 second = camera.WorldToViewportPoint(secondWorld);
            StampProjectedSegment(first, second, Solid, PrioritySolid, labels, priorities);
            MarkTerrainLineCells(first, second, lineCells);
        }
    }

    private static void StampTerrainBacking(Camera camera, EdgeCollider2D edge, byte[] labels,
        byte[] priorities, bool[] lineCells)
    {
        Vector2[] points = edge.points;
        if (points.Length < 2 || (points.Length >= 4 &&
            Vector2.Distance(points[0], points[points.Length - 1]) < .001f))
            return;
        for (int index = 1; index < points.Length; index++)
        {
            Vector3 firstWorld = edge.transform.TransformPoint(edge.offset + points[index - 1]);
            Vector3 secondWorld = edge.transform.TransformPoint(edge.offset + points[index]);
            Vector3 first = camera.WorldToViewportPoint(firstWorld);
            Vector3 second = camera.WorldToViewportPoint(secondWorld);
            if (first.z <= 0f || second.z <= 0f) continue;
            float dx = second.x - first.x;
            float dy = second.y - first.y;
            if (Mathf.Abs(dx) >= Mathf.Abs(dy) * 2f)
            {
                float y = (first.y + second.y) * .5f;
                if (y >= 0f && y <= 1f)
                    SemanticRasterizer.StampOpenHorizontalBacking(labels, priorities,
                        Mathf.Clamp01(Mathf.Min(first.x, second.x)),
                        Mathf.Clamp01(Mathf.Max(first.x, second.x)), y, y < .5f,
                        PrioritySolid);
                continue;
            }
            StampNormalBacking(first, second, labels, priorities, lineCells);
        }
    }

    private static void MarkTerrainLineCells(Vector3 first, Vector3 second, bool[] lineCells)
    {
        if (first.z <= 0f || second.z <= 0f) return;
        Vector2 a = new Vector2(first.x, first.y);
        Vector2 b = new Vector2(second.x, second.y);
        GetSegmentGridBounds(a, b, out int x0, out int x1, out int y0, out int y1);
        for (int y = y0; y < y1; y++)
        for (int x = x0; x < x1; x++)
        {
            float left = (float)x / SemanticRasterizer.Width;
            float right = (float)(x + 1) / SemanticRasterizer.Width;
            float bottom = 1f - (float)(y + 1) / SemanticRasterizer.Height;
            float top = 1f - (float)y / SemanticRasterizer.Height;
            if (SegmentIntersectsRect(a, b, left, bottom, right, top))
                lineCells[y * SemanticRasterizer.Width + x] = true;
        }
    }

    private static void GetSegmentGridBounds(Vector2 first, Vector2 second, out int x0,
        out int x1, out int y0, out int y1)
    {
        x0 = Mathf.Clamp(Mathf.FloorToInt(Mathf.Min(first.x, second.x) * SemanticRasterizer.Width),
            0, SemanticRasterizer.Width - 1);
        x1 = Mathf.Clamp(Mathf.CeilToInt(Mathf.Max(first.x, second.x) * SemanticRasterizer.Width),
            0, SemanticRasterizer.Width);
        y0 = Mathf.Clamp(Mathf.FloorToInt((1f - Mathf.Max(first.y, second.y)) * SemanticRasterizer.Height),
            0, SemanticRasterizer.Height - 1);
        y1 = Mathf.Clamp(Mathf.CeilToInt((1f - Mathf.Min(first.y, second.y)) * SemanticRasterizer.Height),
            0, SemanticRasterizer.Height);
    }

    private static void StampNormalBacking(Vector3 first, Vector3 second, byte[] labels,
        byte[] priorities, bool[] lineCells)
    {
        Vector2 tangent = new Vector2(second.x - first.x, second.y - first.y).normalized;
        Vector2 terrainNormal = new Vector2(-tangent.y, tangent.x);
        const float cellStep = .5f;
        int maxSteps = SemanticRasterizer.Width + SemanticRasterizer.Height;
        GetSegmentGridBounds(new Vector2(first.x, first.y), new Vector2(second.x, second.y),
            out int x0, out int x1, out int y0, out int y1);
        for (int y = y0; y < y1; y++)
        for (int x = x0; x < x1; x++)
        {
            float left = (float)x / SemanticRasterizer.Width;
            float right = (float)(x + 1) / SemanticRasterizer.Width;
            float bottom = 1f - (float)(y + 1) / SemanticRasterizer.Height;
            float top = 1f - (float)y / SemanticRasterizer.Height;
            if (!SegmentIntersectsRect(new Vector2(first.x, first.y),
                new Vector2(second.x, second.y), left, bottom, right, top))
                continue;
            for (int step = 1; step <= maxSteps; step++)
            {
                float px = (x + .5f) / SemanticRasterizer.Width +
                           terrainNormal.x * cellStep * step / SemanticRasterizer.Width;
                float py = 1f - (y + .5f) / SemanticRasterizer.Height +
                           terrainNormal.y * cellStep * step / SemanticRasterizer.Height;
                int targetX = Mathf.FloorToInt(px * SemanticRasterizer.Width);
                int targetY = Mathf.FloorToInt((1f - py) * SemanticRasterizer.Height);
                if ((uint)targetX >= SemanticRasterizer.Width ||
                    (uint)targetY >= SemanticRasterizer.Height)
                    break;
                int targetIndex = targetY * SemanticRasterizer.Width + targetX;
                if (lineCells[targetIndex]) break;
                SemanticRasterizer.StampIgnoreCell(labels, priorities, targetX, targetY,
                    PrioritySolid);
            }
        }
    }

    private static void StampLocalPolygon(Camera camera, Transform transform, Vector2[] local,
        byte label, byte priority, byte[] labels, byte[] priorities)
    {
        if (local.Length < 3) return;
        Vector2[] viewport = new Vector2[local.Length];
        for (int index = 0; index < local.Length; index++)
        {
            Vector3 world = transform.TransformPoint(local[index]);
            Vector3 point = camera.WorldToViewportPoint(world);
            viewport[index] = new Vector2(point.x, point.y);
        }
        StampViewportPolygon(viewport, label, priority, labels, priorities);
    }

    private static void StampViewportPolygon(Vector2[] polygon, byte label, byte priority,
        byte[] labels, byte[] priorities)
    {
        float minX = float.PositiveInfinity;
        float minY = float.PositiveInfinity;
        float maxX = float.NegativeInfinity;
        float maxY = float.NegativeInfinity;
        foreach (Vector2 point in polygon)
        {
            minX = Mathf.Min(minX, point.x);
            minY = Mathf.Min(minY, point.y);
            maxX = Mathf.Max(maxX, point.x);
            maxY = Mathf.Max(maxY, point.y);
        }
        if (maxX < 0f || minX > 1f || maxY < 0f || minY > 1f) return;
        int x0 = Mathf.Clamp(Mathf.FloorToInt(minX * SemanticRasterizer.Width),
            0, SemanticRasterizer.Width - 1);
        int x1 = Mathf.Clamp(Mathf.CeilToInt(maxX * SemanticRasterizer.Width),
            0, SemanticRasterizer.Width);
        int y0 = Mathf.Clamp(Mathf.FloorToInt((1f - maxY) * SemanticRasterizer.Height),
            0, SemanticRasterizer.Height - 1);
        int y1 = Mathf.Clamp(Mathf.CeilToInt((1f - minY) * SemanticRasterizer.Height),
            0, SemanticRasterizer.Height);
        for (int y = y0; y < y1; y++)
        for (int x = x0; x < x1; x++)
        {
            float left = (float)x / SemanticRasterizer.Width;
            float right = (float)(x + 1) / SemanticRasterizer.Width;
            float bottom = 1f - (float)(y + 1) / SemanticRasterizer.Height;
            float top = 1f - (float)y / SemanticRasterizer.Height;
            if (PolygonIntersectsRect(polygon, left, bottom, right, top))
                SemanticRasterizer.StampCell(labels, priorities, x, y, label, priority);
        }
    }

    private static void StampProjectedSegment(Vector3 first, Vector3 second, byte label,
        byte priority, byte[] labels, byte[] priorities)
    {
        if (first.z <= 0f || second.z <= 0f) return;
        Vector2[] line = { new(first.x, first.y), new(second.x, second.y) };
        float minX = Mathf.Min(first.x, second.x);
        float maxX = Mathf.Max(first.x, second.x);
        float minY = Mathf.Min(first.y, second.y);
        float maxY = Mathf.Max(first.y, second.y);
        if (maxX < 0f || minX > 1f || maxY < 0f || minY > 1f) return;
        int x0 = Mathf.Clamp(Mathf.FloorToInt(minX * SemanticRasterizer.Width), 0,
            SemanticRasterizer.Width - 1);
        int x1 = Mathf.Clamp(Mathf.CeilToInt(maxX * SemanticRasterizer.Width), 0,
            SemanticRasterizer.Width);
        int y0 = Mathf.Clamp(Mathf.FloorToInt((1f - maxY) * SemanticRasterizer.Height), 0,
            SemanticRasterizer.Height - 1);
        int y1 = Mathf.Clamp(Mathf.CeilToInt((1f - minY) * SemanticRasterizer.Height), 0,
            SemanticRasterizer.Height);
        for (int y = y0; y < y1; y++)
        for (int x = x0; x < x1; x++)
        {
            float left = (float)x / SemanticRasterizer.Width;
            float right = (float)(x + 1) / SemanticRasterizer.Width;
            float bottom = 1f - (float)(y + 1) / SemanticRasterizer.Height;
            float top = 1f - (float)y / SemanticRasterizer.Height;
            if (SegmentIntersectsRect(line[0], line[1], left, bottom, right, top))
                SemanticRasterizer.StampCell(labels, priorities, x, y, label, priority);
        }
    }

    private static bool PolygonIntersectsRect(Vector2[] polygon, float left, float bottom,
        float right, float top)
    {
        if (PointInPolygon(new Vector2((left + right) * 0.5f, (bottom + top) * 0.5f), polygon))
            return true;
        foreach (Vector2 point in polygon)
            if (point.x >= left && point.x <= right && point.y >= bottom && point.y <= top)
                return true;
        for (int index = 0; index < polygon.Length; index++)
        {
            Vector2 first = polygon[index];
            Vector2 second = polygon[(index + 1) % polygon.Length];
            if (SegmentIntersectsRect(first, second, left, bottom, right, top)) return true;
        }
        return false;
    }

    private static bool PointInPolygon(Vector2 point, Vector2[] polygon)
    {
        bool inside = false;
        for (int i = 0, j = polygon.Length - 1; i < polygon.Length; j = i++)
        {
            Vector2 a = polygon[i];
            Vector2 b = polygon[j];
            if ((a.y > point.y) != (b.y > point.y) &&
                point.x < (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x)
                inside = !inside;
        }
        return inside;
    }

    private static bool SegmentIntersectsRect(Vector2 first, Vector2 second, float left,
        float bottom, float right, float top)
    {
        if ((first.x < left && second.x < left) || (first.x > right && second.x > right) ||
            (first.y < bottom && second.y < bottom) || (first.y > top && second.y > top))
            return false;
        if ((first.x >= left && first.x <= right && first.y >= bottom && first.y <= top) ||
            (second.x >= left && second.x <= right && second.y >= bottom && second.y <= top))
            return true;
        return SegmentsIntersect(first, second, new Vector2(left, bottom), new Vector2(right, bottom)) ||
               SegmentsIntersect(first, second, new Vector2(right, bottom), new Vector2(right, top)) ||
               SegmentsIntersect(first, second, new Vector2(right, top), new Vector2(left, top)) ||
               SegmentsIntersect(first, second, new Vector2(left, top), new Vector2(left, bottom));
    }

    private static bool SegmentsIntersect(Vector2 a, Vector2 b, Vector2 c, Vector2 d)
    {
        float abC = Cross(b - a, c - a);
        float abD = Cross(b - a, d - a);
        float cdA = Cross(d - c, a - c);
        float cdB = Cross(d - c, b - c);
        return abC * abD <= 0f && cdA * cdB <= 0f;
    }

    private static float Cross(Vector2 a, Vector2 b) => a.x * b.y - a.y * b.x;

    private static bool TryClassify(
        Collider2D collider,
        HeroController? hero,
        out byte label,
        out byte priority
    )
    {
        label = Empty;
        priority = 0;
        int layer = collider.gameObject.layer;
        if (layer == (int)GlobalEnums.PhysLayers.TERRAIN)
            SetCandidate(Solid, PrioritySolid, ref label, ref priority);

        if (collider.GetComponentInParent<DropPlatform>() != null ||
            collider.GetComponentInParent<DreamPlatform>() != null ||
            collider.GetComponentInParent<LiftPlatform>() != null ||
            collider.GetComponentInParent<VinePlatform>() != null ||
            collider.GetComponentInParent<FlipPlatform>() != null)
            SetCandidate(Platform, PriorityPlatform, ref label, ref priority);

        if (collider.GetComponentInParent<TransitionPoint>() != null)
            SetCandidate(Exit, PriorityExit, ref label, ref priority);

        if (IsGrassOrCuttable(collider) &&
            collider.GetComponentInParent<DamageHero>() == null &&
            collider.GetComponentInParent<HealthManager>() == null &&
            (hero == null || collider.GetComponentInParent<HeroController>() != hero))
            return false;

        if (collider.GetComponentInParent<RestBench>() != null ||
            collider.GetComponentInParent<BridgeLever>() != null ||
            collider.GetComponentInParent<BossStatueLever>() != null ||
            collider.GetComponentInParent<StagTravel>() != null)
            SetCandidate(Interactive, PriorityInteractive, ref label, ref priority);

        if (collider.GetComponentInParent<GeoControl>() != null)
            SetCandidate(Pickup, PriorityPickup, ref label, ref priority);

        if (IsBreakable(collider))
            SetCandidate(LabelBreakables ? Breakable : SemanticRasterizer.Ignore,
                PriorityBreakable, ref label, ref priority);

        DamageHero? damageHero = collider.GetComponentInParent<DamageHero>();
        DamageHero? contactDamage = collider.GetComponent<DamageHero>();
        HealthManager? health = collider.GetComponentInParent<HealthManager>();
        if (SemanticColliderRules.IsEnvironmentalHazard(
                damageHero != null, damageHero?.hazardType ?? 0, health != null))
            SetCandidate(Hazard, PriorityHazard, ref label, ref priority);

        if (health != null && !health.isDead && health.isActiveAndEnabled &&
            SemanticColliderRules.IsEnemyBodyCollider(
                collider.isTrigger, collider.gameObject == health.gameObject, contactDamage != null))
            SetCandidate(Enemy, PriorityEnemy, ref label, ref priority);

        if (hero != null && collider.GetComponentInParent<HeroController>() == hero)
            SetCandidate(Player, PriorityPlayer, ref label, ref priority);

        return priority != 0;
    }

    private static void LogSemanticCollider(Collider2D collider, Action<string>? log)
    {
        if (!string.Equals(Environment.GetEnvironmentVariable("HK_LABEL_DIAGNOSTICS"), "true",
                StringComparison.OrdinalIgnoreCase))
            return;
        HealthManager? health = collider.GetComponentInParent<HealthManager>();
        DamageHero? damageHero = collider.GetComponentInParent<DamageHero>();
        DamageHero? contactDamage = collider.GetComponent<DamageHero>();
        bool hazardCandidate = damageHero != null && damageHero.hazardType > 0 && health == null;
        bool namedHazard = collider.gameObject.name.IndexOf("spike", StringComparison.OrdinalIgnoreCase) >= 0 ||
                           collider.gameObject.name.IndexOf("acid", StringComparison.OrdinalIgnoreCase) >= 0 ||
                           collider.gameObject.name.IndexOf("hazard", StringComparison.OrdinalIgnoreCase) >= 0;
        if ((health == null && damageHero == null && !namedHazard) ||
            LoggedSemanticColliders.Count >= SemanticDiagnosticColliderLimit ||
            !LoggedSemanticColliders.Add(collider.GetInstanceID()))
            return;

        string owner = health != null ? health.gameObject.name : damageHero!.gameObject.name;
        string hazardType = damageHero != null ? damageHero.hazardType.ToString() : "none";
        string layer = LayerMask.LayerToName(collider.gameObject.layer);
        bool enemyBody = health != null && SemanticColliderRules.IsEnemyBodyCollider(
            collider.isTrigger, collider.gameObject == health.gameObject, contactDamage != null);
        bool environmentalHazard = SemanticColliderRules.IsEnvironmentalHazard(
            damageHero != null, damageHero?.hazardType ?? 0, health != null);
        StringBuilder hierarchy = new StringBuilder();
        Transform? current = collider.transform;
        for (int depth = 0; current != null && depth < 8; depth++, current = current.parent)
        {
            if (hierarchy.Length != 0) hierarchy.Append('/');
            hierarchy.Append(current.name);
        }
        Component[] components = collider.gameObject.GetComponents<Component>();
        List<string> componentNames = new List<string>(components.Length);
        foreach (Component component in components)
            if (component != null) componentNames.Add(component.GetType().Name);
        log?.Invoke($"HK_LABEL_COLLIDER owner={owner} hierarchy={hierarchy} " +
                  $"collider={collider.gameObject.name} components={string.Join(",", componentNames)} " +
                  $"type={collider.GetType().Name} trigger={collider.isTrigger} layer={layer} " +
                  $"healthManager={(health != null)} damageHero={(damageHero != null)} " +
                  $"contactDamage={(contactDamage != null)} hazardCandidate={hazardCandidate} " +
                  $"hazardType={hazardType} enemyBody={enemyBody} hazard={environmentalHazard}");
    }

    private static bool IsBreakable(Collider2D collider) =>
        collider.GetComponentInParent<Breakable>() != null ||
        collider.GetComponentInParent<BreakableObject>() != null ||
        collider.GetComponentInParent<BreakableInfectedVine>() != null ||
        collider.GetComponentInParent<BreakablePole>() != null ||
        collider.GetComponentInParent<BreakablePoleSimple>() != null ||
        collider.GetComponentInParent<BreakablePoleTopLand>() != null ||
        collider.GetComponentInParent<DestructibleProp>() != null;

    private static bool IsGrassOrCuttable(Collider2D collider) =>
        collider.GetComponentInParent<GrassBehaviour>() != null ||
        collider.GetComponentInParent<GrassCut>() != null ||
        collider.GetComponentInParent<GrassSpriteBehaviour>() != null ||
        collider.GetComponentInParent<Grass>() != null ||
        collider.GetComponentInParent<VinePlatformCut>() != null;

    private static void SetCandidate(byte candidate, byte candidatePriority,
        ref byte label, ref byte priority)
    {
        // IGNORE means “this object supplies no target”; it must not erase a
        // useful class already stamped by an overlapping collider.
        if (candidate == SemanticRasterizer.Ignore)
            return;
        if (candidatePriority >= priority)
        {
            label = candidate;
            priority = candidatePriority;
        }
    }

}
