const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const worldToGameMap = require("../src/hk_agent/control_plane_web/map-transform.js");

const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures", "map_transform_cases.json"), "utf8"));
for (const item of fixture.cases) {
  test(`shared transform: ${item.name}`, () => {
    assert.deepEqual(worldToGameMap(item.scene, ...item.point, item.world_bounds, item.layout), item.expected);
  });
}
