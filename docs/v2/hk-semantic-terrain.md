# Hollow Knight semantic terrain and interactives

`EdgeCollider2D` terrain is rasterized in two passes. Closed paths whose last
point returns to the first are filled as polygons, then their boundary segments
are stamped SOLID. Open paths keep their segments SOLID and mark cells behind
the line IGNORE until another terrain line or the screen edge. Horizontal lines
in the lower half of the viewport are treated as floor (terrain below); lines in
the upper half are treated as ceiling (terrain above). For non-horizontal open
segments, the directed edge's left normal is used as the terrain side, opposite
the outward-facing right normal. IGNORE only replaces EMPTY or low-priority
SOLID, so hazards, enemies, the player, and other higher-priority classes remain
visible.

Only component-bearing objects receive INTERACTIVE. The isolated install's
`Assembly-CSharp.dll` contains the supported `RestBench`, `BridgeLever`,
`BossStatueLever`, and `StagTravel` types. `DialogueBox` is a UI dialogue surface,
not a world collider, and `LookAnimNPC` describes animation rather than a
conversation contract; neither is used as a proxy for interactability. The
assembly contains no `InspectRegion`, `Conversation`, sign, or general NPC
interaction component type. Such objects therefore remain unlabelled until a
concrete world component can be identified.

`GrassBehaviour`, `GrassCut`, `GrassSpriteBehaviour`, `Grass`, and
`VinePlatformCut` are excluded from semantic collision classes. In the absence
of a blocking or damaging component they remain EMPTY; damaging components are
still classified HAZARD by their `DamageHero` component. The
`INTERACTIVE_OBJECT` layer alone no longer implies INTERACTIVE.
