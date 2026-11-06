(function (root) {
  function worldToGameMap(scene, x, y, bounds, layout) {
    if (!Array.isArray(bounds) || bounds.length !== 4 || !Array.isArray(layout?.scenes)) return null;
    const row = layout.scenes.find(item => item.scene === scene && item.has_sprite === true && item.rect);
    if (!row) return null;
    const [wx1, wy1, wx2, wy2] = bounds;
    const rect = row.rect;
    if (![x, y, wx1, wy1, wx2, wy2, rect.x, rect.y, rect.width, rect.height].every(Number.isFinite)
        || wx2 <= wx1 || wy2 <= wy1) return null;
    return [rect.x + (x - wx1) * rect.width / (wx2 - wx1),
      rect.y + (y - wy1) * rect.height / (wy2 - wy1)];
  }
  if (typeof module !== "undefined" && module.exports) module.exports = worldToGameMap;
  if (root) root.worldToGameMap = worldToGameMap;
})(typeof globalThis === "undefined" ? this : globalThis);
