namespace HkAgentBridge;

internal static class SemanticColliderRules
{
    public static bool IsEnemyBodyCollider(bool isTrigger, bool isHealthManagerObject,
        bool hasContactDamage) =>
        !isTrigger && (isHealthManagerObject || hasContactDamage);

    public static bool IsEnvironmentalHazard(bool hasDamageHero, int hazardType,
        bool hasHealthManager) =>
        hasDamageHero && hazardType > 0 && !hasHealthManager;
}
