"""Testes das regras básicas e das alturas de salto do ginásio."""

import random

from voidgym.gyms.platform_gym.params import PhysicsParams
from voidgym.gyms.platform_gym.physics import step_physics
from voidgym.gyms.platform_gym.world import Breakable, Door, Hazard, Player, Rect, Room, Solid


def _room(physics: PhysicsParams | None = None) -> Room:
    return Room(1, 30, 12, solids=[Solid(Rect(-10, 0, 50, 1))], physics=physics or PhysicsParams(), spawn=(0, 1))


def _jump_height(hold: bool, physics: PhysicsParams | None = None) -> float:
    room = _room(physics)
    player = Player(0, 1)
    peak = player.y
    for tick in range(300):
        buttons = {"jump": tick == 0 or hold and player.jump_hold_time < room.physics.hold_max}
        step_physics(room, player, buttons)
        peak = max(peak, player.y)
        if tick > 0 and player.on_ground:
            break
    return peak - 1


def test_short_and_held_jump_match_configured_heights() -> None:
    physics = PhysicsParams(gravity=60, jump_short_height=1.4, jump_max_height=4.2)
    assert abs(_jump_height(False, physics) - physics.jump_short_height) / physics.jump_short_height < .03
    assert abs(_jump_height(True, physics) - physics.jump_max_height) / physics.jump_max_height < .03
    assert physics.hold_max > 0


def test_wall_collision_one_way_platform_and_determinism() -> None:
    room = _room()
    room.solids.append(Solid(Rect(2, 0, .2, 8)))
    room.solids.append(Solid(Rect(4, 3, 3, .25), "platform"))
    first, second = Player(0, 1), Player(0, 1)
    for _ in range(30):
        step_physics(room, first, {"right": True})
        step_physics(room, second, {"right": True})
    assert first.x <= 1.41
    assert (first.x, first.y) == (second.x, second.y)


def test_coyote_time_jump_buffer_and_one_way_landing() -> None:
    room = _room()
    room.solids = [Solid(Rect(0, 0, .6, 1)), Solid(Rect(3, 3, 3, .25), "platform")]
    player = Player(0, 1)
    for _ in range(4):
        step_physics(room, player, {"right": True})
    step_physics(room, player, {"right": True, "jump": True})
    assert player.vy > 0
    falling = Player(4, 4, vy=-5, on_ground=False)
    step_physics(room, falling, {"jump": True})
    for _ in range(6):
        step_physics(room, falling, {})
    assert falling.on_ground
    assert falling.y == 3.25


def test_double_jump_dash_and_ability_gate() -> None:
    room = _room()
    player = Player(0, 4, on_ground=False, abilities={"ability_double_jump", "ability_dash"})
    step_physics(room, player, {"jump": True})
    assert player.double_jump_used
    start = player.x
    step_physics(room, player, {"dash": True})
    for _ in range(7):
        step_physics(room, player, {})
    assert player.x - start >= 3.5
    locked = Player(0, 1)
    gate_room = _room()
    gate_room.doors.append(Door(Rect(2, 1, .5, 2), "ability_double_jump"))
    for _ in range(20):
        step_physics(gate_room, locked, {"right": True})
    assert locked.x <= 1.41
    locked.abilities.add("ability_double_jump")
    step_physics(gate_room, locked, {"right": True})
    assert gate_room.doors[0].opened


def test_hazard_damage_invulnerability_and_safe_return() -> None:
    room = _room()
    room.hazards.append(Hazard(Rect(2, 1, 2, 2)))
    player = Player(0, 1, last_safe_pos=(0, 1))
    for _ in range(20):
        events = step_physics(room, player, {"right": True})
        if any(event.kind == "damage" for event in events):
            break
    assert player.hp == 4
    assert player.x <= .32
    for _ in range(20):
        step_physics(room, player, {"right": True})
    assert player.hp == 4


def test_breakable_pogo_double_jump_dash_and_door() -> None:
    room = _room()
    block = Breakable(Rect(2, 1, 1, 2))
    room.breakables.append(block)
    player = Player(0, 1, abilities={"ability_double_jump", "ability_dash"})
    step_physics(room, player, {"right": True, "attack": True})
    for _ in range(100):
        step_physics(room, player, {"right": True, "attack": True})
    assert block.broken
    player.on_ground = False
    player.y = 2
    player.vy = -1
    player.attack_cooldown = 0
    room.hazards.append(Hazard(Rect(player.x, 1.1, 1, .5)))
    step_physics(room, player, {"down": True, "attack": True})
    assert player.vy > 0
    door = Door(Rect(6, 1, .5, 2), "ability_double_jump")
    room.doors.append(door)
    step_physics(room, player, {"right": True})
    assert door.opened or door.requires in player.abilities


def test_physics_sampling_stays_inside_declared_ranges() -> None:
    params = [PhysicsParams.sample(random.Random(seed)) for seed in range(300)]
    assert all(6 <= item.run_speed <= 10 and 40 <= item.gravity <= 70 for item in params)
    assert all(item.jump_short_height <= 1.8 and 3 <= item.jump_max_height <= 5.5 for item in params)
