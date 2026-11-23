import numpy as np

from voidgym.core.experience_store import ExperienceReader, ExperienceWriter, Retention
from voidgym.core.reward import UniversalReward


def test_experience_round_trip_filters_and_background_flush(tmp_path) -> None:
    root = tmp_path / "experience"
    writer = ExperienceWriter(root, "fake", "campaign", source="test", block_steps=2,
                              action_table=("NOOP", "RIGHT"))
    for index in range(3):
        writer.append({"frame": np.zeros((108, 192, 3), dtype=np.uint8), "action": index,
                       "action_mask": np.array([True, False]), "reward_universal": 1.0,
                       "done": index == 2, "scene": "room-a", "tags": ["record_progress"]})
    writer.close()
    blocks = list(ExperienceReader(root, game_id="fake", campaign_id="campaign", scene="room-a"))
    assert [block["record"]["steps"] for block in blocks] == [2, 1]
    assert blocks[0]["arrays"]["frame"].shape == (2, 108, 192, 3)
    assert blocks[0]["arrays"]["action"].dtype == np.int16
    assert blocks[0]["arrays"]["action_mask"].dtype == np.uint8
    assert blocks[0]["meta"]["action_table"] == ["NOOP", "RIGHT"]
    assert blocks[0]["meta"]["scene_runs"] == [["room-a", 0, 2]]
    stacked = next(ExperienceReader(root).iter_stacked(frame_stack=4))
    assert stacked["arrays"]["frames_stacked"].shape == (2, 4, 108, 192, 3)


def test_retention_preserves_demonstrations_and_help_resolved(tmp_path) -> None:
    root = tmp_path / "experience"
    for source, tags in (("actor", []), ("demonstration", []), ("actor", ["help_resolved"])):
        with ExperienceWriter(root, "fake", "campaign", source=source, block_steps=1) as writer:
            writer.append({"frame": np.full((108, 192, 3), 1, dtype=np.uint8), "action": 0,
                           "scene": "r", "tags": tags})
    removed = Retention(0.0000001).apply(root)
    blocks = list(ExperienceReader(root))
    assert removed >= 1
    assert {block["record"]["source"] for block in blocks} == {"demonstration", "actor"}
    assert any("help_resolved" in block["record"]["tags"] for block in blocks)


def test_retention_keeps_ten_percent_sample_per_day_and_scene(tmp_path) -> None:
    root = tmp_path / "experience"
    for index in range(20):
        scene = "room-a" if index < 10 else "room-b"
        with ExperienceWriter(root, "fake", "campaign", source="actor", block_steps=1) as writer:
            writer.append({"frame": np.zeros((108, 192, 3), dtype=np.uint8),
                           "action": index, "scene": scene})
    removed = Retention(0.000001).apply(root)
    remaining = list(ExperienceReader(root))
    assert removed >= 1
    assert len(remaining) == 2
    assert {scene for item in remaining for scene in item["record"]["scenes"]} == {"room-a", "room-b"}


def test_universal_reward_is_nonzero_for_idle_steps() -> None:
    reward = UniversalReward()
    total = sum(reward({}, {"room_id": "a", "player_pos_room": (0.5, 0.5)}) for _ in range(20_000))
    assert total < 0
    assert reward.zero_reward_fraction() < 0.9
