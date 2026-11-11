"""Configuração imutável e serializável de uma campanha VoidGym."""

import difflib
import hashlib
import json
import math
import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from voidgym.core.errors import VoidGymTypeError as TypeError
from voidgym.core.errors import VoidGymValueError as ValueError
from voidgym.core.observation import MODEL_CONTRACT_VERSION


def process_environment() -> dict[str, str]:
    """Return a snapshot of process environment for OS integration adapters."""
    return dict(os.environ)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    contract: str = MODEL_CONTRACT_VERSION
    hidden_size: int = 256
    recurrent_layers: int = 1
    specialists: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PPOConfig:
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    epochs: int = 4
    batch_size: int = 512
    sequence_length: int = 32


@dataclass(frozen=True, slots=True)
class AuxConfig:
    c_sem: float = 0.0
    c_inv: float = 0.0
    c_mot: float = 0.0
    augment_random_shift: bool = False


@dataclass(frozen=True, slots=True)
class RewardConfig:
    novelty: float = 0.1
    progress: float = 1.0
    damage: float = 1.0
    death: float = 2.0
    time_cost: float = 0.001
    room_change: float = 0.2
    game_reward: str = "default"


@dataclass(frozen=True, slots=True)
class ResourceOverrides:
    game_instances: int | None = None
    gym_envs: int | None = None
    train_device: str | None = None
    storage_budget_gb: float = 100.0


@dataclass(frozen=True, slots=True)
class StorageConfig:
    budget_gb: float = 100.0
    block_steps: int = 1024
    root: str = "runtime/v2/experience"


@dataclass(frozen=True, slots=True)
class HelpConfig:
    help_stuck_count: int = 10
    help_stuck_minutes: int = 60


@dataclass(frozen=True, slots=True)
class EvalConfig:
    interval_rounds: int = 5
    episodes: int = 4


@dataclass(frozen=True, slots=True)
class StartFrom:
    mode: str = "fresh"
    checkpoint: str | None = None


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    schema_version: int = 1
    campaign_id: str = "campaign"
    kind: str = "pretrain_platform"
    game_id: str = "platform_gym"
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    aux: AuxConfig = field(default_factory=AuxConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    resources: ResourceOverrides = field(default_factory=ResourceOverrides)
    storage: StorageConfig = field(default_factory=StorageConfig)
    help: HelpConfig = field(default_factory=HelpConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    start_from: StartFrom | None = None


_SECTIONS = {"model": ModelConfig, "ppo": PPOConfig, "aux": AuxConfig,
             "reward": RewardConfig, "resources": ResourceOverrides,
             "storage": StorageConfig, "help": HelpConfig, "evaluation": EvalConfig,
             "start_from": StartFrom}


def _reject_unknown(name: str, fields: Any, raw: dict[str, Any]) -> None:
    unknown = set(raw) - set(fields)
    if unknown:
        key = min(unknown)
        hint = difflib.get_close_matches(key, fields, n=1)
        suggestion = f"; você quis dizer '{hint[0]}'?" if hint else ""
        raise ValueError(f"chave '{key}' não existe em [{name}]{suggestion}")


def _section_bounds(name: str) -> dict[str, tuple[float, float]]:
    integer_ranges: dict[str, dict[str, tuple[float, float]]] = {
        "ppo": {"epochs": (1, 100), "batch_size": (1, 65536),
                "sequence_length": (1, 4096)},
        "help": {"help_stuck_count": (1, 1000), "help_stuck_minutes": (1, 10080)},
        "model": {"hidden_size": (16, 4096), "recurrent_layers": (1, 8)},
        "storage": {"block_steps": (1, 1024)},
        "evaluation": {"interval_rounds": (1, 10000), "episodes": (1, 10000)},
        "resources": {"game_instances": (1, 256), "gym_envs": (1, 4096)},
    }
    ranges: dict[str, tuple[float, float]] = dict(integer_ranges.get(name, {}))
    if name == "ppo":
        ranges.update({"learning_rate": (1e-7, 1.0), "gamma": (0.0, 1.0),
                       "gae_lambda": (0.0, 1.0), "clip_range": (0.0, 1.0),
                       "entropy_coef": (0.0, 1.0)})
    elif name == "aux":
        ranges.update({key: (0.0, 10.0) for key in ("c_sem", "c_inv", "c_mot")})
    elif name == "reward":
        ranges.update({key: (0.0, 1000.0) for key in
                       ("novelty", "progress", "damage", "death", "time_cost", "room_change")})
    elif name == "storage":
        ranges["budget_gb"] = (0.1, 1_000_000)
    elif name == "resources":
        ranges["storage_budget_gb"] = (0.1, 1_000_000)
    return ranges


def _validate_ranges(name: str, value: Any) -> None:
    integer_fields = {
        "ppo": {"epochs", "batch_size", "sequence_length"},
        "model": {"hidden_size", "recurrent_layers"},
        "resources": {"game_instances", "gym_envs"},
        "storage": {"block_steps"},
        "help": {"help_stuck_count", "help_stuck_minutes"},
        "evaluation": {"interval_rounds", "episodes"},
    }.get(name, set())
    for key, (low, high) in _section_bounds(name).items():
        number = getattr(value, key)
        if number is None:
            continue
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise TypeError(f"[{name}].{key} deve ser numérico")
        if not math.isfinite(number):
            raise ValueError(f"[{name}].{key} deve ser finito")
        if key in integer_fields and not isinstance(number, int):
            raise TypeError(f"[{name}].{key} deve ser inteiro")
        if not low <= number <= high:
            raise ValueError(f"[{name}].{key} deve estar entre {low} e {high}")


def _validate_section_types(name: str, value: Any) -> None:
    if name == "aux" and not isinstance(value.augment_random_shift, bool):
        raise TypeError("[aux].augment_random_shift deve ser booleano")
    if name == "model":
        if not isinstance(value.contract, str) or not value.contract:
            raise ValueError("[model].contract deve ser uma string não vazia")
        if not all(isinstance(item, str) and item for item in value.specialists):
            raise ValueError("[model].specialists deve conter nomes não vazios")
    if name == "resources" and value.train_device not in {None, "cpu", "cuda"}:
        raise ValueError("[resources].train_device deve ser cpu ou cuda")
    if name == "storage" and (not isinstance(value.root, str) or not value.root):
        raise ValueError("[storage].root deve ser uma string não vazia")
    if name == "reward" and (not isinstance(value.game_reward, str) or not value.game_reward):
        raise ValueError("[reward].game_reward deve ser uma string não vazia")
    if name == "start_from":
        if value.mode not in {"fresh", "checkpoint", "resume"}:
            raise ValueError("[start_from].mode deve ser fresh, checkpoint ou resume")
        if value.checkpoint is not None and not isinstance(value.checkpoint, str):
            raise TypeError("[start_from].checkpoint deve ser string ou ausente")
        if value.mode != "fresh" and not value.checkpoint:
            raise ValueError("[start_from].checkpoint é obrigatório para checkpoint/resume")


def _make_section(name: str, cls: Any, raw: dict[str, Any]) -> Any:
    fields = cls.__dataclass_fields__
    _reject_unknown(name, fields, raw)
    normalized = dict(raw)
    if cls is ModelConfig and isinstance(normalized.get("specialists"), list):
        normalized["specialists"] = tuple(normalized["specialists"])
    value = cls(**normalized)
    _validate_ranges(name, value)
    _validate_section_types(name, value)
    return value


def config_from_dict(raw: dict[str, Any]) -> CampaignConfig:
    """Valida chaves e faixas, preenchendo os padrões do esquema."""
    raw = dict(raw)
    raw.pop("config_hash", None)
    top_fields = CampaignConfig.__dataclass_fields__
    unknown = set(raw) - set(top_fields)
    if unknown:
        key = min(unknown)
        hint = difflib.get_close_matches(key, top_fields, n=1)
        suggestion = f"; você quis dizer '{hint[0]}'?" if hint else ""
        raise ValueError(f"chave '{key}' não existe na campanha{suggestion}")
    values = dict(raw)
    for section, cls in _SECTIONS.items():
        section_value = values.get(section)
        if section_value is None:
            continue
        if not isinstance(section_value, dict):
            raise TypeError(f"[{section}] deve ser um objeto")
        values[section] = _make_section(section, cls, section_value)
    config = CampaignConfig(**values)
    if isinstance(config.schema_version, bool) or not isinstance(config.schema_version, int):
        raise TypeError("schema_version deve ser inteiro")
    if config.schema_version != 1:
        raise ValueError("schema_version deve ser 1; migre a configuração antes de carregar")
    for name, value in (("campaign_id", config.campaign_id), ("game_id", config.game_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} deve ser uma string não vazia")
    if config.kind not in {"pretrain_platform", "game_training", "evaluation", "e2e"}:
        raise ValueError("kind deve ser pretrain_platform, game_training, evaluation ou e2e")
    if config.resources.train_device not in {None, "cpu", "cuda"}:
        raise ValueError("resources.train_device deve ser cpu ou cuda")
    return config


def config_from_toml(path: Path) -> CampaignConfig:
    with path.open("rb") as stream:
        return config_from_dict(tomllib.load(stream))


def migrate(raw: dict[str, Any]) -> dict[str, Any]:
    """Migra o esquema inicial sem alterar dados reconhecidos."""
    result = dict(raw)
    version = result.get("schema_version", 0)
    if version == 0:
        result["schema_version"] = 1
        return result
    if version == 1:
        return result
    raise ValueError(f"Não há migração disponível do schema_version {version}")


def config_hash(config: CampaignConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_campaign_config(config: CampaignConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data = asdict(config)
    data["config_hash"] = config_hash(config)
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2), encoding="utf-8")
    temporary.replace(path)
