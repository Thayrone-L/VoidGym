namespace HkAgentBridge;

internal static class SemanticColliderRules
{
    public static bool IsEnemyBodyCollider(bool isTrigger, bool isHealthManagerObject,
        bool hasContactDamage) =>
        !isTrigger && (isHealthManagerObject || hasContactDamage);

    public static bool IsEnvironmentalHazard(bool hasDamageHero, int hazardType,
        bool hasHealthManager, bool isTrigger, float screenAreaFraction, bool hasHazardVisual) =>
        hasDamageHero && hazardType > 0 && !hasHealthManager &&
        (!isTrigger || screenAreaFraction <= 0.08f || hasHazardVisual);

    public static bool IsInteractiveCollider(bool isTrigger, float screenAreaFraction) =>
        !isTrigger && screenAreaFraction <= 0.08f;
}
