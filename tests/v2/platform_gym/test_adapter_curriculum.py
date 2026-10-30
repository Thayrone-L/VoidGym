import numpy as np
from gymnasium.utils.env_checker import check_env

from voidgym.adapters.hollow_knight.adapter import HollowKnightAdapter
from voidgym.envs.adapter_env import AdapterEnv
from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter
from voidgym.gyms.platform_gym.curriculum import Curriculum
from voidgym.gyms.platform_gym.reward import CombinedPlatformReward


def test_action_names_match_hollow_knight_adapter_order() -> None:
    hk_names = HollowKnightAdapter.__new__(HollowKnightAdapter).action_table()
    adapter = PlatformGymAdapter()
    assert tuple(adapter.action_table()) == tuple(hk_names)
    assert len(adapter.action_mask()) == len(hk_names)


def test_mask_and_adapter_env_contract() -> None:
    adapter = PlatformGymAdapter()
    env = AdapterEnv(adapter, CombinedPlatformReward())
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=91)
    assert "sem" not in obs and "training_labels" in info
    assert info["training_labels"]["sem"].shape == (27, 48)
    mask = adapter.action_mask()
    assert not mask[20] and not mask[33] and not mask[35]
    for _ in range(10_000):
        valid = np.flatnonzero(adapter.action_mask())
        action = int(np.random.default_rng(_).choice(valid))
        obs, _, terminated, truncated, info = env.step(action)
        assert obs["pixels"].shape == (12, 108, 192)
        assert not terminated
        if truncated:
            env.reset()


def test_level_eight_advances_rooms_and_terminates_after_key_gate() -> None:
    adapter = PlatformGymAdapter(level=8, seed=45)
    assert 3 <= len(adapter._rooms) <= 6
    ability = next(p for p in adapter.room.pickups if p.kind == "ability_double_jump")
    adapter.player.x = ability.rect.x + ability.rect.w / 2
    adapter.player.y = ability.rect.y
    adapter.apply(0)
    assert "ability_double_jump" in adapter.player.abilities

    for index in range(len(adapter._rooms)):
        if index == len(adapter._rooms) - 1:
            door = adapter.room.doors[0]
            adapter.player.x = door.rect.x - adapter.room.physics.player_width / 2
            adapter.player.y = adapter.room.spawn[1]
            adapter.player.on_ground = True
            adapter.apply(2)
            assert door.opened
        exit_ = adapter.room.exits[0]
        adapter.player.x = exit_.rect.x + exit_.rect.w / 2
        adapter.player.y = exit_.rect.y
        adapter.player.on_ground = True
        adapter.apply(0)
        if index < len(adapter._rooms) - 1:
            assert adapter._room_index == index + 1
            assert "ability_double_jump" in adapter.player.abilities
            assert not adapter.signals()["terminated"]
    assert adapter.signals()["terminated"]


def test_curriculum_progress_and_save_resume(tmp_path) -> None:
    curriculum = Curriculum(seed=11)
    for _ in range(800):
        curriculum.record(True)
    assert curriculum.level == 8
    path = tmp_path / "curriculum.json"
    curriculum.save(path)
    resumed = Curriculum.load(path)
    assert resumed.level == 8 and resumed.episodes == 800
    assert resumed.style_split(10) == "test"


def test_curriculum_samples_next_level_with_sixty_percent_current_level_success() -> None:
    curriculum = Curriculum(seed=11)
    curriculum.level = 2
    for index in range(100):
        curriculum.record(index < 60, level=2)
    sampled = {curriculum.sample_level(seed) for seed in range(10_000)}
    assert curriculum.level == 2
    assert 3 in sampled
    assert curriculum.success_rate(2) == .6


def test_curriculum_load_migrates_older_current_window_to_level_history(tmp_path) -> None:
    path = tmp_path / "old-curriculum.json"
    path.write_text('{"level":2,"seed":4,"episodes":3,"current":[true,false,true]}',
                    encoding="utf-8")
    curriculum = Curriculum.load(path)
    assert list(curriculum.history[2]) == [True, False, True]
    assert curriculum.success_rate(2) == 2 / 3


def test_curriculum_target_uses_last_100_episodes_per_level() -> None:
    curriculum = Curriculum(seed=2)
    for level in range(9):
        curriculum.history[level].extend(
            [True] * (70 if level < 8 else 30) + [False] * (30 if level < 8 else 70)
        )
    assert curriculum.training_goal_met()
    curriculum.history[3].clear()
    curriculum.history[3].extend([True] * 69 + [False] * 31)
    assert not curriculum.training_goal_met()


def test_curriculum_stagnation_window_stops_only_unmet_level() -> None:
    curriculum = Curriculum(seed=3)
    curriculum.history[2].extend([True, False, False])
    assert curriculum.update_progress_watch(100, window=3, no_improvement_steps=5) is None
    assert curriculum.update_progress_watch(104, window=3, no_improvement_steps=5) is None
    assert curriculum.update_progress_watch(105, window=3, no_improvement_steps=5) == 2
    curriculum.history[2].clear()
    curriculum.history[2].extend([True, True, True])
    assert curriculum.update_progress_watch(106, window=3, no_improvement_steps=5) is None
    assert 2 not in curriculum.last_improvement_step


def test_curriculum_progress_watch_round_trips(tmp_path) -> None:
    curriculum = Curriculum(seed=4)
    curriculum.history[2].extend([True, False, False])
    curriculum.update_progress_watch(123, window=3, no_improvement_steps=5)
    path = tmp_path / "curriculum.json"
    curriculum.save(path)
    resumed = Curriculum.load(path)
    assert resumed.best_window_rate == {2: 1 / 3}
    assert resumed.last_improvement_step == {2: 123}
