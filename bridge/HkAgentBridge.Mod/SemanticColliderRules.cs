namespace HkAgentBridge;

internal static class SemanticColliderRules
{
    public static bool IsEnemyBodyCollider(bool isTrigger, bool isHealthManagerObject,
        bool hasContactDamage) =>
        !isTrigger && (isHealthManagerObject || hasContactDamage);

    public static bool IsEnvironmentalHazard(bool hasContactDamage, int hazardType,
        bool hasHealthManager, bool isTrigger, float screenAreaFraction, bool hasHazardVisual) =>
        hasContactDamage && hazardType > 0 && !hasHealthManager &&
        (!isTrigger || screenAreaFraction <= 0.08f || hasHazardVisual);

    public static bool IsInteractiveCollider(float screenAreaFraction) =>
        screenAreaFraction <= 0.08f;
}
