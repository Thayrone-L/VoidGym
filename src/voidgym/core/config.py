"""Configuração tipada, migração e serialização reproduzível da v2."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, cast

from voidgym.core.errors import ConfigError


@dataclass(frozen=True)
class ModelConfig:
    contract_version: str = "2.0"
    image_height: int = 108
    image_width: int = 192
    frame_stack: int = 4
    sem_height: int = 27
    sem_width: int = 48
    experts: tuple[str, ...] = ("explore",)


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 2.5e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    n_epochs: int = 4
    batch_size: int = 2048
    seq_len: int = 32
    rollout_steps: int = 256


@dataclass(frozen=True)
class AuxConfig:
    c_sem: float = 0.5
    c_inv: float = 0.1
    c_mot: float = 0.1
    augment: bool = True


@dataclass(frozen=True)
class RewardConfig:
    novelty: float = 0.02
    progress: float = 1.0
    damage: float = -0.3
    death: float = -1.0
    time: float = -0.001
    game_reward_name: str = "platform_gym"
    episode_slice_steps: int = 256
    episode_slice_budget: float = 4.0
    rolling_window_steps: int = 1000
    rolling_window_budget: float = 12.0


@dataclass(frozen=True)
class ResourceOverrides:
    game_instances: int | None = None
    gym_envs: int | None = None
    train_device: str | None = None
    torch_threads_learner: int | None = None
    batch_size: int | None = None
    storage_budget_gb: float | None = None


@dataclass(frozen=True)
class StorageConfig:
    budget_gb: float = 100.0
    block_steps: int = 1024
    experience_root: str | None = None


@dataclass(frozen=True)
class HelpConfig:
    help_stuck_count: int = 10
    help_no_progress_minutes: int = 60


@dataclass(frozen=True)
class EvalConfig:
    episodes: int = 100
    deterministic: bool = True
    interval_steps: int = 1_000_000


@dataclass(frozen=True)
class StartFrom:
    kind: str = "fresh"
    path: str | None = None


@dataclass(frozen=True)
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

    def validate(self) -> CampaignConfig:
        """Valida faixas e invariantes do contrato antes de iniciar a campanha."""
        if self.schema_version != 1:
            raise ConfigError("schema_version deve ser 1 após a migração")
        if self.kind not in {"pretrain_platform", "game_training", "evaluation"}:
            raise ConfigError("kind deve ser pretrain_platform, game_training ou evaluation")
        if not self.campaign_id or not self.game_id:
            raise ConfigError("campaign_id e game_id não podem ser vazios")
        _range("ppo.learning_rate", self.ppo.learning_rate, 1e-8, 1.0)
        _range("ppo.gamma", self.ppo.gamma, 0.0, 1.0)
        _range("ppo.gae_lambda", self.ppo.gae_lambda, 0.0, 1.0)
        _range("ppo.clip_range", self.ppo.clip_range, 0.0, 1.0)
        _range("ppo.n_epochs", self.ppo.n_epochs, 1, 100)
        _range("ppo.batch_size", self.ppo.batch_size, 1, 1_000_000)
        _range("ppo.seq_len", self.ppo.seq_len, 1, 2048)
        _range("storage.budget_gb", self.storage.budget_gb, 0.1, 1_000_000)
        if self.storage.block_steps != 1024:
            raise ConfigError("storage.block_steps deve ser 1024 no contrato v2")
        if (self.model.image_height, self.model.image_width, self.model.frame_stack) != (
            108,
            192,
            4,
        ):
            raise ConfigError("model deve usar imagem 192×108 e pilha de quatro quadros")
        if (self.model.sem_height, self.model.sem_width) != (27, 48):
            raise ConfigError("model deve usar mapa semântico 27×48")
        _range("ppo.ent_coef", self.ppo.ent_coef, 0.0, 10.0)
        _range("ppo.rollout_steps", self.ppo.rollout_steps, 1, 1_000_000)
        _range("aux.c_sem", self.aux.c_sem, 0.0, 10.0)
        _range("aux.c_inv", self.aux.c_inv, 0.0, 10.0)
        _range("aux.c_mot", self.aux.c_mot, 0.0, 10.0)
        _range("help.help_stuck_count", self.help.help_stuck_count, 1, 10_000)
        _range("help.help_no_progress_minutes", self.help.help_no_progress_minutes, 1, 1440)
        _range("evaluation.episodes", self.evaluation.episodes, 1, 100_000)
        _range("evaluation.interval_steps", self.evaluation.interval_steps, 1, 1_000_000_000)
        _range("reward.episode_slice_steps", self.reward.episode_slice_steps, 1, 1_000_000)
        _range("reward.episode_slice_budget", self.reward.episode_slice_budget, 0.0, 1_000_000)
        _range("reward.rolling_window_steps", self.reward.rolling_window_steps, 1, 1_000_000)
        _range("reward.rolling_window_budget", self.reward.rolling_window_budget, 0.0, 1_000_000)
        if self.resources.train_device not in {None, "cpu", "cuda"}:
            raise ConfigError("resources.train_device deve ser cpu ou cuda")
        for name in ("game_instances", "gym_envs", "torch_threads_learner", "batch_size"):
            value = getattr(self.resources, name)
            if value is not None:
                _range(f"resources.{name}", value, 1, 1_000_000)
        if self.resources.storage_budget_gb is not None:
            _range("resources.storage_budget_gb", self.resources.storage_budget_gb, 0.1, 1_000_000)
        if self.start_from is not None and self.start_from.kind not in {
            "fresh",
            "universal",
            "resume",
        }:
            raise ConfigError("start_from.kind deve ser fresh, universal ou resume")
        return self


def _range(name: str, value: float, minimum: float, maximum: float) -> None:
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} deve estar entre {minimum} e {maximum}")


def _section[T](cls: type[T], payload: object, name: str) -> T:
    if not isinstance(payload, dict):
        raise ConfigError(f"A seção [{name}] deve ser um objeto")
    allowed = {item.name for item in fields(cast(Any, cls))}
    unknown = set(payload) - allowed
    if unknown:
        key = min(unknown)
        match = difflib.get_close_matches(key, sorted(allowed), n=1)
        suggestion = f"; você quis dizer '{match[0]}'?" if match else ""
        raise ConfigError(f"Chave '{key}' não existe em [{name}]{suggestion}")
    values = dict(payload)
    if cls is ModelConfig and "experts" in values:
        values["experts"] = tuple(values["experts"])
    return cls(**values)


def migrate(raw: dict[str, Any]) -> dict[str, Any]:
    """Migra a configuração legada sem apagar campos desconhecidos."""
    version = raw.get("schema_version", 0)
    if not isinstance(version, int) or version < 0 or version > 1:
        raise ConfigError("Não há migração disponível para essa schema_version")
    result = dict(raw)
    result["schema_version"] = 1
    return result


def campaign_from_dict(raw: dict[str, Any]) -> CampaignConfig:
    migrated = migrate(raw)
    allowed = {item.name for item in fields(CampaignConfig)}
    unknown = set(migrated) - allowed
    if unknown:
        key = min(unknown)
        match = difflib.get_close_matches(key, sorted(allowed), n=1)
        suggestion = f"; você quis dizer '{match[0]}'?" if match else ""
        raise ConfigError(f"Chave '{key}' não existe na configuração{suggestion}")
    types = {
        "model": ModelConfig,
        "ppo": PPOConfig,
        "aux": AuxConfig,
        "reward": RewardConfig,
        "resources": ResourceOverrides,
        "storage": StorageConfig,
        "help": HelpConfig,
        "evaluation": EvalConfig,
    }
    values = dict(migrated)
    for name, cls in types.items():
        if name in values:
            values[name] = _section(cls, values[name], name)
    if values.get("start_from") is not None:
        values["start_from"] = _section(StartFrom, values["start_from"], "start_from")
    return CampaignConfig(**values).validate()


def campaign_from_toml(path: Path) -> CampaignConfig:
    with path.open("rb") as stream:
        return campaign_from_dict(tomllib.load(stream))


def to_payload(config: CampaignConfig) -> dict[str, Any]:
    return asdict(config.validate())


def campaign_to_json(config: CampaignConfig) -> str:
    return json.dumps(to_payload(config), ensure_ascii=False, indent=2, sort_keys=True)


def campaign_from_json(data: str) -> CampaignConfig:
    raw = json.loads(data)
    if not isinstance(raw, dict):
        raise ConfigError("O conteúdo JSON da campanha precisa ser um objeto")
    return campaign_from_dict(raw)


def save_config(config: CampaignConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(campaign_to_json(config) + "\n", encoding="utf-8")


def config_hash(config: CampaignConfig) -> str:
    canonical = json.dumps(
        to_payload(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def runtime_root_from_environment() -> Path | None:
    """Lê a única variável de ambiente prevista para substituir o runtime."""
    configured = os.environ.get("VOIDGYM_RUNTIME_ROOT")
    return Path(configured).expanduser() if configured else None
