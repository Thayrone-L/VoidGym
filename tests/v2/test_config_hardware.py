import json
from dataclasses import asdict
from pathlib import Path

import pytest

from voidgym.core.config import config_from_dict, config_from_toml, config_hash, migrate
from voidgym.core.hardware import HardwareProfile, plan_resources


def test_config_round_trip_and_stable_hash() -> None:
    path = Path(__file__).parents[2] / "configs/v2/exemplos/pretrain_platform.toml"
    config = config_from_toml(path)
    raw = json.loads(json.dumps(asdict(config)))
    restored = config_from_dict(raw)
    assert config == restored
    assert config_hash(config) == config_hash(restored)


def test_config_unknown_key_suggests_correction() -> None:
    with pytest.raises(ValueError, match="help_stuck_count"):
        config_from_dict({"help": {"help_stuck_cont": 10}})


def test_config_range_has_bounds() -> None:
    with pytest.raises(ValueError, match="entre 1 e 100"):
        config_from_dict({"ppo": {"epochs": 0}})
    with pytest.raises(TypeError, match="inteiro"):
        config_from_dict({"ppo": {"epochs": 1.5}})
    with pytest.raises(ValueError, match="checkpoint é obrigatório"):
        config_from_dict({"start_from": {"mode": "resume"}})


def test_migrate_initial_schema() -> None:
    assert migrate({"campaign_id": "x"}) == {"campaign_id": "x", "schema_version": 1}


@pytest.mark.parametrize(
    "profile,expected_device,expected_instances",
    [
        (HardwareProfile(8, 16, 16, 12, "RTX", 12, 10, "8.6", 100, "Windows"), "cuda", 2),
        (HardwareProfile(4, 8, 8, 5, None, None, None, None, 80, "Windows"), "cpu", 1),
        (HardwareProfile(16, 32, 64, 48, "GPU", 24, 20, "9.0", 500, "Linux"), "cuda", 3),
        (HardwareProfile(None, None, None, None, None, None, None, None, None, "Unknown"), "cpu", 1),
    ],
)
def test_resource_plan_for_profiles(profile, expected_device, expected_instances) -> None:
    plan = plan_resources(profile)
    assert plan.train_device == expected_device
    assert plan.game_instances == expected_instances
    assert plan.actor_device == "cpu"


def test_overrides_warn_when_overcommitted() -> None:
    profile = HardwareProfile(4, 8, 8, 7, None, None, None, None, 40, "Windows")
    plan = plan_resources(profile, overrides={"game_instances": 2})
    assert any("excede" in warning for warning in plan.warnings)


def test_platform_env_count_respects_commit_headroom_and_worker_measurement() -> None:
    profile = HardwareProfile(
        8, 16, 32, 20, "RTX", 12, 10, "8.6", 100, "Windows",
        commit_used_gb=20, commit_limit_gb=30, commit_available_gb=10,
        paging_file_total_gb=16, worker_private_gb=.5,
    )
    plan = plan_resources(profile, overrides={"gym_envs": 20})
    assert plan.gym_envs == 9
    assert plan.commit_headroom_gb >= 4
    assert any("reduzidos" in warning for warning in plan.warnings)


def test_platform_plan_refuses_to_consume_three_gib_commit_reserve() -> None:
    profile = HardwareProfile(
        8, 16, 32, 20, "RTX", 12, 10, "8.6", 100, "Windows",
        commit_available_gb=4.2, worker_private_gb=.5,
    )
    with pytest.raises(ValueError, match="3 GiB"):
        plan_resources(profile)


def test_pretrain_plan_reserves_learner_and_worker_growth() -> None:
    profile = HardwareProfile(
        8, 16, 16, 8, "RTX", 12, 10, "8.6", 100, "Windows",
        commit_available_gb=12.8, worker_private_gb=.1,
    )
    plan = plan_resources(profile, "pretrain_platform", {"gym_envs": 20})
    assert plan.gym_envs == 11
    assert plan.commit_headroom_gb is not None and plan.commit_headroom_gb >= 3
    with pytest.raises(ValueError, match="3 GiB"):
        plan_resources(profile.__class__(
            8, 16, 16, 8, "RTX", 12, 10, "8.6", 100, "Windows",
            commit_available_gb=8.39, worker_private_gb=.1,
        ), "pretrain_platform")
