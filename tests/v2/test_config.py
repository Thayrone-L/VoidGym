"""Validação, migração, TOML/JSON e hash estável da configuração."""

from pathlib import Path

import pytest

from voidgym.core.config import (
    CampaignConfig,
    campaign_from_dict,
    campaign_from_json,
    campaign_from_toml,
    campaign_to_json,
    config_hash,
    migrate,
    save_config,
)
from voidgym.core.errors import ConfigError


def test_toml_json_round_trip(tmp_path: Path) -> None:
    path = Path("configs/v2/exemplos/pretrain_platform.toml")
    config = campaign_from_toml(path)
    json_path = tmp_path / "config.json"
    save_config(config, json_path)
    assert campaign_from_json(json_path.read_text(encoding="utf-8")) == config
    assert campaign_from_json(campaign_to_json(config)) == config


def test_unknown_key_has_suggestion() -> None:
    with pytest.raises(ConfigError, match="help_stuck_count"):
        campaign_from_dict({"help": {"help_stuck_cont": 5}})


def test_range_migration_hash_and_defaults() -> None:
    migrated = migrate({"schema_version": 0, "campaign_id": "old"})
    assert migrated["schema_version"] == 1
    first = campaign_from_dict(migrated)
    assert first.ppo.batch_size == 2048
    assert config_hash(first) == config_hash(campaign_from_dict(migrated))
    with pytest.raises(ConfigError, match="learning_rate.*entre"):
        campaign_from_dict({"ppo": {"learning_rate": 2.0}})
    assert campaign_from_dict({"campaign_id": "fresh"}) == CampaignConfig(campaign_id="fresh")
