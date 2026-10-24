using System;
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

    public static SemanticLabelFrame Capture()
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

        HeroController? hero = HeroController.instance;
        foreach (GameObject root in scene.GetRootGameObjects())
        foreach (Collider2D collider in root.GetComponentsInChildren<Collider2D>(true))
        {
            if (collider == null || !collider.enabled || !collider.gameObject.activeInHierarchy)
                continue;
            if (!TryClassify(collider, hero, out byte label, out byte priority))
                continue;
            StampCollider(camera, collider, labels, priorities, label, priority);
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

        if (layer == (int)GlobalEnums.PhysLayers.INTERACTIVE_OBJECT ||
            collider.GetComponentInParent<RestBench>() != null)
            SetCandidate(Interactive, PriorityInteractive, ref label, ref priority);

        if (collider.GetComponentInParent<GeoControl>() != null)
            SetCandidate(Pickup, PriorityPickup, ref label, ref priority);

        if (IsBreakable(collider))
            SetCandidate(LabelBreakables ? Breakable : SemanticRasterizer.Ignore,
                PriorityBreakable, ref label, ref priority);

        if (collider.GetComponentInParent<DamageHero>() != null)
            SetCandidate(Hazard, PriorityHazard, ref label, ref priority);

        HealthManager? health = collider.GetComponentInParent<HealthManager>();
        if (health != null && !health.isDead && health.isActiveAndEnabled)
            SetCandidate(Enemy, PriorityEnemy, ref label, ref priority);

        if (hero != null && collider.GetComponentInParent<HeroController>() == hero)
            SetCandidate(Player, PriorityPlayer, ref label, ref priority);

        return priority != 0;
    }

    private static bool IsBreakable(Collider2D collider) =>
        collider.GetComponentInParent<Breakable>() != null ||
        collider.GetComponentInParent<BreakableObject>() != null ||
        collider.GetComponentInParent<BreakableInfectedVine>() != null ||
        collider.GetComponentInParent<BreakablePole>() != null ||
        collider.GetComponentInParent<BreakablePoleSimple>() != null ||
        collider.GetComponentInParent<BreakablePoleTopLand>() != null ||
        collider.GetComponentInParent<DestructibleProp>() != null;

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

    private static void StampCollider(Camera camera, Collider2D collider, byte[] labels,
        byte[] priorities, byte label, byte priority)
    {
        Bounds bounds = collider.bounds;
        float z = bounds.center.z;
        Vector3 first = camera.WorldToViewportPoint(new Vector3(bounds.min.x, bounds.min.y, z));
        Vector3 second = camera.WorldToViewportPoint(new Vector3(bounds.min.x, bounds.max.y, z));
        Vector3 third = camera.WorldToViewportPoint(new Vector3(bounds.max.x, bounds.min.y, z));
        Vector3 fourth = camera.WorldToViewportPoint(new Vector3(bounds.max.x, bounds.max.y, z));
        if (first.z <= 0 || second.z <= 0 || third.z <= 0 || fourth.z <= 0)
            return;
        float minX = Mathf.Min(first.x, second.x, third.x, fourth.x);
        float maxX = Mathf.Max(first.x, second.x, third.x, fourth.x);
        float minY = Mathf.Min(first.y, second.y, third.y, fourth.y);
        float maxY = Mathf.Max(first.y, second.y, third.y, fourth.y);
        SemanticRasterizer.StampViewportRect(labels, priorities, minX, minY, maxX, maxY,
            label, priority);
    }
}
