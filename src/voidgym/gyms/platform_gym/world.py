"""Invented room geometry and mutable entities for the procedural platform gym."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from voidgym.gyms.platform_gym.params import PhysicsParams


@dataclass(frozen=True, slots=True)
class Rect:
    """Axis-aligned rectangle in world units; y increases upward."""

    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        if self.w <= 0.0 or self.h <= 0.0:
            raise ValueError("rectangle dimensions must be positive")

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def top(self) -> float:
        return self.y + self.h

    def intersects(self, other: Rect) -> bool:
        """Return true only when rectangle interiors overlap."""
        return self.x < other.right and self.right > other.x and self.y < other.top and self.top > other.y

    def contains_point(self, x: float, y: float) -> bool:
        return self.x <= x <= self.right and self.y <= y <= self.top


@dataclass(slots=True)
class Solid:
    rect: Rect
    kind: Literal["solid", "platform"] = "solid"

    def __post_init__(self) -> None:
        if self.kind not in {"solid", "platform"}:
            raise ValueError("solid kind must be solid or platform")


@dataclass(slots=True)
class Hazard:
    rect: Rect
    kind: Literal["spikes", "thorns", "acid"] = "spikes"

    def __post_init__(self) -> None:
        if self.kind not in {"spikes", "thorns", "acid"}:
            raise ValueError("hazard kind must be spikes, thorns or acid")


@dataclass(slots=True)
class Breakable:
    rect: Rect
    hp: int = 3
    reveals: Literal["passage", "pickup", "nothing"] = "nothing"
    cue: bool = True
    broken: bool = False

    def __post_init__(self) -> None:
        if self.hp < 1:
            raise ValueError("breakable hp must be positive")
        if self.reveals not in {"passage", "pickup", "nothing"}:
            raise ValueError("breakable reveals value is invalid")


@dataclass(slots=True)
class Enemy:
    rect: Rect
    kind: str = "walker"
    hp: int = 2
    patrol_left: float = 0.0
    patrol_right: float = 0.0
    speed: float = 1.5
    alive: bool = True
    blocks_path: bool = False
    reward_hit_paid: bool = False

    def __post_init__(self) -> None:
        if not self.kind or self.hp < 1 or self.speed < 0.0:
            raise ValueError("enemy kind, hp or speed is invalid")
        if self.patrol_right < self.patrol_left:
            raise ValueError("enemy patrol_right must not be left of patrol_left")


@dataclass(slots=True)
class Pickup:
    rect: Rect
    kind: Literal["coin", "ability_double_jump", "ability_dash", "heal"] = "coin"
    velocity_y: float = 0.0
    falling: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"coin", "ability_double_jump", "ability_dash", "heal"}:
            raise ValueError("pickup kind is invalid")


@dataclass(slots=True)
class Interactive:
    rect: Rect
    kind: Literal["bench"] = "bench"

    def __post_init__(self) -> None:
        if self.kind != "bench":
            raise ValueError("interactive kind must be bench")


@dataclass(slots=True)
class Exit:
    rect: Rect
    to_room: int
    to_entry: str = "left"


@dataclass(slots=True)
class Door:
    rect: Rect
    requires: str
    opened: bool = False

    def __post_init__(self) -> None:
        if self.requires not in {"ability_double_jump", "ability_dash"}:
            raise ValueError("door requires an available ability")


@dataclass(slots=True)
class Room:
    id: int
    width: float
    height: float
    solids: list[Solid]
    hazards: list[Hazard]
    breakables: list[Breakable]
    enemies: list[Enemy]
    pickups: list[Pickup]
    interactives: list[Interactive]
    exits: list[Exit]
    doors: list[Door]
    spawn: tuple[float, float]
    physics: PhysicsParams
    style_seed: int
    treasure_rocks: list[TreasureRock] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.id < 0 or self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("room id and dimensions are invalid")
        if not 0.0 <= self.spawn[0] <= self.width or not 0.0 <= self.spawn[1] <= self.height:
            raise ValueError("room spawn must be inside the room bounds")


@dataclass(slots=True)
class TreasureRock:
    rect: Rect
    hp: int = 3
    broken: bool = False
    visual_variant: int = 0

    def __post_init__(self) -> None:
        if self.hp < 2 or self.hp > 4:
            raise ValueError("treasure rocks must take two to four hits")
        if self.visual_variant < 0:
            raise ValueError("visual variant must be non-negative")


@dataclass(slots=True)
class Player:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    on_ground: bool = False
    facing: int = 1
    hp: int = 3
    max_hp: int = 3
    invuln_timer: float = 0.0
    jump_hold_time: float = 0.0
    jumping: bool = False
    abilities: set[str] = field(default_factory=set)
    last_safe_pos: tuple[float, float] = (0.0, 0.0)
    jump_was_pressed: bool = False
    coyote_timer: float = 0.0
    jump_buffer_timer: float = 0.0
    air_jump_available: bool = True
    dash_timer: float = 0.0
    dash_cooldown: float = 0.0
    attack_cooldown: float = 0.0
    attack_timer: float = 0.0
    attack_was_pressed: bool = False
    interact_was_pressed: bool = False
    dash_was_pressed: bool = False
    attack_targets_hit: set[tuple[str, int]] = field(default_factory=set)
    bench_pos: tuple[float, float] | None = None
    coins: int = 0
    score: int = 0

    def __post_init__(self) -> None:
        if self.facing not in {-1, 1}:
            raise ValueError("player facing must be -1 or +1")
        if self.max_hp < 1 or not 0 <= self.hp <= self.max_hp:
            raise ValueError("player hp must be within max_hp")
