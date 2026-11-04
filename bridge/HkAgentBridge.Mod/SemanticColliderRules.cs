namespace HkAgentBridge;

internal static class SemanticColliderRules
{
    public static byte? BreakableLabel(bool isGeoRock, bool hasOtherBreakable,
        bool diagnosticLabelsEnabled)
    {
        if (isGeoRock) return 6;
        if (!hasOtherBreakable) return null;
        return diagnosticLabelsEnabled ? (byte)6 : SemanticRasterizer.Ignore;
    }

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
