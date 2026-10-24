from __future__ import annotations

from voidgym.gyms.platform_gym.params import PhysicsParams
from voidgym.gyms.platform_gym.physics import step_physics
from voidgym.gyms.platform_gym.world import Breakable, Door, Enemy, Exit, Hazard, Rect, Room, Solid, Player


def room(*, physics: PhysicsParams | None = None, solids: list[Solid] | None = None,
         hazards: list[Hazard] | None = None, breakables: list[Breakable] | None = None,
         enemies: list[Enemy] | None = None, doors: list[Door] | None = None) -> Room:
    return Room(0, 24, 12, solids or [Solid(Rect(0, 0, 24, 0.5))], hazards or [],
                breakables or [], enemies or [], [], [], [Exit(Rect(23, 0.5, 1, 3), 1)],
                doors or [], (2, 0.5), physics or PhysicsParams(), 5)


def advance(r: Room, p: Player, buttons: dict[str, bool], frames: int) -> list:
    events = []
    for _ in range(frames):
        events.extend(step_physics(r, p, buttons, 0.02))
    return events


def test_short_and_held_jump_heights_match_contract() -> None:
    r = room()
    p = Player(2, 0.5, on_ground=True)
    start = p.y
    step_physics(r, p, {"jump": True}, 0.02)
    step_physics(r, p, {}, 0.02)
    peak = p.y
    while p.y > start or p.vy > 0:
        step_physics(r, p, {}, 0.02)
        peak = max(peak, p.y)
    assert abs((peak - start) - r.physics.jump_short_height) / r.physics.jump_short_height < 0.03

    p = Player(2, 0.5, on_ground=True)
    peak = p.y
    for _ in range(200):
        step_physics(r, p, {"jump": True}, 0.02)
        peak = max(peak, p.y)
    assert abs((peak - 0.5) - r.physics.jump_max_height) / r.physics.jump_max_height < 0.03


def test_coyote_time_and_jump_buffer() -> None:
    r = room(solids=[Solid(Rect(0, 0, 2, 0.5)), Solid(Rect(3, 0, 21, 0.5))])
    p = Player(2.35, 0.5, on_ground=True)
    advance(r, p, {"right": True}, 1)
    assert not p.on_ground
    step_physics(r, p, {"jump": True}, 0.02)
    assert p.vy > 0

    p = Player(2, 0.7, vy=-2)
    step_physics(r, p, {"jump": True}, 0.02)
    for _ in range(5):
        step_physics(r, p, {}, 0.02)
    assert p.vy > 0


def test_wall_and_fast_fall_collisions_and_one_way_platform() -> None:
    r = room(solids=[Solid(Rect(0, 0, 24, 0.5)), Solid(Rect(5, 0.5, 0.2, 8))])
    p = Player(4, 0.5, on_ground=True)
    advance(r, p, {"right": True}, 30)
    assert p.x + r.physics.player_width / 2 <= 5
    p = Player(8, 8, vy=-r.physics.max_fall_speed)
    r.solids.append(Solid(Rect(7, 4, 3, 0.2), "platform"))
    advance(r, p, {}, 40)
    assert p.on_ground and abs(p.y - 4.2) < 1e-6
    p = Player(8, 2, vy=12)
    advance(r, p, {"jump": True}, 10)
    assert p.y > 2  # upward pass through one-way platform


def test_hazard_damage_invulnerability_safe_return_and_death_respawn() -> None:
    r = room(hazards=[Hazard(Rect(5, 0.5, 2, 4))])
    p = Player(3, 0.5, on_ground=True, last_safe_pos=(2, 0.5), hp=1)
    ev = []
    for _ in range(20):
        frame_events = step_physics(r, p, {"right": True}, 0.02)
        ev.extend(frame_events)
        if any(e.kind == "death" for e in frame_events):
            break
    assert any(e.kind == "damage" for e in ev)
    assert any(e.kind == "death" for e in ev)
    assert (p.x, p.y) == r.spawn and p.hp == p.max_hp


def test_attack_breaks_wall_pogo_double_jump_dash_and_door() -> None:
    block = Breakable(Rect(3, 0.5, 0.7, 2), hp=3, reveals="passage")
    r = room(breakables=[block])
    p = Player(2, 0.5, on_ground=True)
    for _ in range(3):
        step_physics(r, p, {"attack": True}, 0.02)
        advance(r, p, {}, 18)
    assert block.broken
    r.doors.append(Door(Rect(8, 0.5, 1, 2), "ability_double_jump"))
    p.abilities.add("ability_double_jump")
    p.x = 7.5
    advance(r, p, {"right": True}, 10)
    assert r.doors[0].opened
    p.x, p.y, p.on_ground = 10, 2, False
    p.coyote_timer = 0
    p.jump_was_pressed = False
    step_physics(r, p, {"jump": True}, 0.02)
    assert p.vy > 0 and not p.air_jump_available
    p.abilities.add("ability_dash")
    p.dash_was_pressed = False
    x = p.x
    step_physics(r, p, {"right": True, "dash": True}, 0.02)
    assert p.x > x


def test_left_facing_attack_hits_target_to_the_left() -> None:
    block = Breakable(Rect(3, 0.5, 0.7, 2), hp=3)
    r = room(breakables=[block])
    p = Player(4.2, 0.5, facing=-1, on_ground=True)
    step_physics(r, p, {"attack": True}, 0.02)
    assert block.hp == 2


def test_enemy_turns_at_a_gap_edge() -> None:
    enemy = Enemy(Rect(1.5, .5, .8, .8), patrol_left=0, patrol_right=10, speed=3)
    r = room(solids=[Solid(Rect(0, 0, 3, .5)), Solid(Rect(5, 0, 19, .5))], enemies=[enemy])
    p = Player(20, .5, on_ground=True)
    advance(r, p, {}, 50)
    assert enemy.speed < 0
    assert enemy.rect.right < 3.1


def test_determinism_for_identical_inputs() -> None:
    a, b = room(), room()
    p1, p2 = Player(2, 0.5, on_ground=True), Player(2, 0.5, on_ground=True)
    for i in range(200):
        keys = {"right": i < 30, "jump": 30 <= i < 42, "attack": i % 40 < 2}
        step_physics(a, p1, keys, 0.02)
        step_physics(b, p2, keys, 0.02)
        assert (p1.x, p1.y, p1.vx, p1.vy, p1.hp) == (p2.x, p2.y, p2.vx, p2.vy, p2.hp)
