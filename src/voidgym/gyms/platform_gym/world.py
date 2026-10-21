"""Modelo geométrico genérico do ginásio."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    def intersects(self, other: "Rect") -> bool:
        return self.x < other.x + other.w and self.x + self.w > other.x and self.y < other.y + other.h and self.y + self.h > other.y

    def contains_point(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h

    def expanded(self, amount: float) -> "Rect":
        return Rect(self.x - amount, self.y - amount, self.w + 2 * amount, self.h + 2 * amount)


@dataclass
class Solid:
    rect: Rect
    kind: str = "solid"


@dataclass
class Hazard:
    rect: Rect
    kind: str = "spikes"


@dataclass
class Breakable:
    rect: Rect
    hp: int = 3
    reveals: str = "nothing"
    cue: bool = True
    broken: bool = False


@dataclass
class Enemy:
    rect: Rect
    kind: str = "walker"
    hp: int = 2
    patrol_left: float = 0.0
    patrol_right: float = 4.0
    speed: float = 1.5
    alive: bool = True


@dataclass
class Pickup:
    rect: Rect
    kind: str = "coin"


@dataclass
class Interactive:
    rect: Rect
    kind: str = "bench"


@dataclass
class Exit:
    rect: Rect
    to_room: int = -1
    to_entry: str = "left"


@dataclass
class Door:
    rect: Rect
    requires: str = "ability_double_jump"
    opened: bool = False


@dataclass
class Room:
    id: int
    width: float
    height: float
    solids: list[Solid] = field(default_factory=list)
    hazards: list[Hazard] = field(default_factory=list)
    breakables: list[Breakable] = field(default_factory=list)
    enemies: list[Enemy] = field(default_factory=list)
    pickups: list[Pickup] = field(default_factory=list)
    interactives: list[Interactive] = field(default_factory=list)
    exits: list[Exit] = field(default_factory=list)
    doors: list[Door] = field(default_factory=list)
    spawn: tuple[float, float] = (1.0, 1.0)
    physics: object | None = None
    style_seed: int = 0


@dataclass
class Player:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    on_ground: bool = True
    facing: int = 1
    hp: int = 5
    max_hp: int = 5
    invuln_timer: float = 0.0
    jump_hold_time: float = 0.0
    jumping: bool = False
    abilities: set[str] = field(default_factory=set)
    last_safe_pos: tuple[float, float] = (1.0, 1.0)
    coyote_time: float = 0.0
    jump_buffer: float = 0.0
    double_jump_used: bool = False
    dash_time: float = 0.0
    dash_ready: bool = True
    attack_cooldown: float = 0.0
    respawn: tuple[float, float] | None = None

    def rect(self, room: Room) -> Rect:
        physics = room.physics
        width = getattr(physics, "player_width", .6)
        height = getattr(physics, "player_height", 1.2)
        return Rect(self.x, self.y, width, height)
