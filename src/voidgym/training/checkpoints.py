"""Atomic, component-wise universal policy checkpoints."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import torch

from voidgym.core.observation import FRAME_STACK, MODEL_CONTRACT_VERSION, UNIVERSAL_IMAGE_HW
from voidgym.core.semantics import SEM_GRID
from voidgym.models.policy import UniversalPolicy


def save_universal(path: Path, model: UniversalPolicy, action_table: tuple[str, ...] | list[str],
                   *, steps: int, metrics: dict[str, float] | None = None,
                   optimizer: torch.optim.Optimizer | None = None,
                   curriculum: dict[str, Any] | None = None) -> Path:
    """Write model components to a temporary directory and atomically publish it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    try:
        _save_state(temporary / "encoder.pt", model.encoder)
        _save_state(temporary / "perception.pt", model.perception)
        _save_state(temporary / "aux_heads.pt", {
            "inverse_dynamics": model.inverse_dynamics.state_dict(),
            "hero_motion": model.hero_motion.state_dict(),
        })
        _save_state(temporary / "core.pt", {
            "history_embedding": model.history_embedding.state_dict(),
            "history_projection": model.history_projection.state_dict(),
            "core": model.core.state_dict(),
        })
        (temporary / "actor").mkdir()
        for name, actor in model.actor.items():
            _save_state(temporary / "actor" / f"{name}.pt", actor)
        _save_state(temporary / "critic.pt", {
            "privileged_projection": model.privileged_projection.state_dict(),
            "critic": model.critic.state_dict(),
        })
        if model.game_extras:
            (temporary / "game_extras").mkdir()
            for name, adapter in model.game_extras.items():
                _save_state(temporary / "game_extras" / f"{name}.pt", adapter)
        if optimizer is not None:
            _save_state(temporary / "optimizer.pt", optimizer.state_dict())
        if curriculum is not None:
            (temporary / "curriculum.json").write_text(json.dumps(curriculum, indent=2) + "\n",
                                                        encoding="utf-8")
        manifest = {
            "contract_version": MODEL_CONTRACT_VERSION, "action_table": list(action_table),
            "semantic_grid": list(SEM_GRID), "frame_stack": FRAME_STACK,
            "image_hw": list(UNIVERSAL_IMAGE_HW), "feature_map": [32, 14, 24],
            "feature_vector": 256, "trained_steps": steps, "metrics": metrics or {},
            "experts": list(model.actor.keys()), "game_ids": list(model.game_extras.keys()),
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                  encoding="utf-8")
        backup = path.with_name(f".{path.name}.old")
        if backup.exists():
            shutil.rmtree(backup)
        if path.exists():
            path.replace(backup)
        temporary.replace(path)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return path


def load_universal(path: Path, model: UniversalPolicy,
                   action_table: tuple[str, ...] | list[str], *,
                   optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    """Load a checkpoint only when its ordered action names match exactly."""
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("action_table") != list(action_table):
        raise ValueError("checkpoint action table differs from current adapter, including order")
    if manifest.get("experts") != list(model.actor.keys()):
        raise ValueError("checkpoint actor experts differ from the initialized policy")
    _load_state(path / "encoder.pt", model.encoder)
    _load_state(path / "perception.pt", model.perception)
    aux = _load_state(path / "aux_heads.pt")
    model.inverse_dynamics.load_state_dict(aux["inverse_dynamics"])
    model.hero_motion.load_state_dict(aux["hero_motion"])
    core = _load_state(path / "core.pt")
    model.history_embedding.load_state_dict(core["history_embedding"])
    model.history_projection.load_state_dict(core["history_projection"])
    model.core.load_state_dict(core["core"])
    for name, actor in model.actor.items():
        _load_state(path / "actor" / f"{name}.pt", actor)
    critic = _load_state(path / "critic.pt")
    model.privileged_projection.load_state_dict(critic["privileged_projection"])
    model.critic.load_state_dict(critic["critic"])
    for name in manifest.get("game_ids", []):
        if name not in model.game_extras:
            state = _load_state(path / "game_extras" / f"{name}.pt")
            input_dim = int(state["0.weight"].shape[1])
            model.add_game_extras(name, input_dim)
        _load_state(path / "game_extras" / f"{name}.pt", model.game_extras[name])
    if optimizer is not None and (path / "optimizer.pt").is_file():
        optimizer.load_state_dict(_compatible_optimizer_state(
            optimizer, _load_state(path / "optimizer.pt")
        ))
    return cast(dict[str, Any], manifest)


def _compatible_optimizer_state(optimizer: torch.optim.Optimizer,
                                saved_state: dict[str, Any]) -> dict[str, Any]:
    """Split an older single optimizer group while retaining Adam moments."""
    current_groups = optimizer.state_dict()["param_groups"]
    saved_groups = saved_state["param_groups"]
    if len(saved_groups) == len(current_groups):
        return saved_state
    current_total = sum(len(group["params"]) for group in current_groups)
    saved_total = sum(len(group["params"]) for group in saved_groups)
    if len(saved_groups) != 1 or saved_total != current_total:
        raise ValueError("checkpoint optimizer parameter groups are incompatible")

    saved_ids = list(saved_groups[0]["params"])
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    for saved_id, parameter in zip(saved_ids, parameters, strict=True):
        state = saved_state["state"].get(saved_id, {})
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            moment = state.get(name)
            if moment is not None and tuple(moment.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"checkpoint optimizer {name} shape is incompatible with current parameters"
                )

    offset = 0
    migrated_groups = []
    for group in current_groups:
        count = len(group["params"])
        migrated_groups.append({**saved_groups[0], "params": saved_ids[offset:offset + count]})
        offset += count
    return {**saved_state, "param_groups": migrated_groups}


def prune_checkpoints(root: Path, *, keep: int = 5) -> None:
    """Keep the newest five rolling checkpoints and all million-step milestones."""
    checkpoints = sorted((item for item in root.glob("universal-*") if item.is_dir()),
                         key=lambda item: item.stat().st_mtime, reverse=True)
    rolling = [item for item in checkpoints if "permanent" not in item.name]
    for item in rolling[keep:]:
        shutil.rmtree(item)


def _save_state(path: Path, value: Any) -> None:
    if isinstance(value, torch.nn.Module):
        value = value.state_dict()
    torch.save(value, path)


def _load_state(path: Path, module: torch.nn.Module | None = None) -> Any:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if module is not None:
        module.load_state_dict(value)
    return value
