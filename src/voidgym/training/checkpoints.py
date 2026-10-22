"""Checkpoint universal em peças com publicação atômica."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch

from voidgym.core.actions import UNIVERSAL_ACTIONS
from voidgym.models.policy import UniversalPolicy


def save_universal(model: UniversalPolicy, destination: str | Path, steps: int,
                   curriculum: dict[str, object] | None = None,
                   metrics: dict[str, object] | None = None) -> Path:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        groups = {
            "encoder.pt": model.encoder.state_dict(), "perception.pt": model.perception.state_dict(),
            "aux_heads.pt": {"inverse": model.inverse_dynamics.state_dict(), "motion": model.hero_motion.state_dict()},
            "core.pt": {"core": model.core.state_dict(), "history": model.action_history.state_dict()},
            "actor/explore.pt": model.actors["explore"].state_dict(), "critic.pt": model.critics.state_dict(),
        }
        for name, state in groups.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, path)
        (staging / "manifest.json").write_text(json.dumps({
            "version": "2.0", "game_ids": ["platform_gym"], "actions": [a.name for a in UNIVERSAL_ACTIONS],
            "sem_grid": [27, 48], "image": [12, 108, 192], "steps": steps,
            "metrics": metrics or {},
        }, indent=2), encoding="utf-8")
        (staging / "curriculum.json").write_text(json.dumps(curriculum or {}), encoding="utf-8")
        backup = target.with_name(target.name + ".old")
        if backup.exists():
            shutil.rmtree(backup)
        if target.exists():
            os.replace(target, backup)
        os.replace(staging, target)
        if backup.exists():
            shutil.rmtree(backup)
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_universal(path: str | Path, action_table: tuple[str, ...], device: str = "cpu") -> UniversalPolicy:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if tuple(manifest["actions"]) != tuple(action_table):
        raise ValueError("Checkpoint action table differs name-by-name from the adapter")
    model = UniversalPolicy(len(action_table))
    model.encoder.load_state_dict(torch.load(root / "encoder.pt", map_location=device, weights_only=True))
    model.perception.load_state_dict(torch.load(root / "perception.pt", map_location=device, weights_only=True))
    auxiliary = torch.load(root / "aux_heads.pt", map_location=device, weights_only=True)
    model.inverse_dynamics.load_state_dict(auxiliary["inverse"])
    model.hero_motion.load_state_dict(auxiliary["motion"])
    model.core.load_state_dict(torch.load(root / "core.pt", map_location=device, weights_only=True)["core"])
    model.action_history.load_state_dict(torch.load(root / "core.pt", map_location=device, weights_only=True)["history"])
    model.actors["explore"].load_state_dict(torch.load(root / "actor/explore.pt", map_location=device, weights_only=True))
    model.critics.load_state_dict(torch.load(root / "critic.pt", map_location=device, weights_only=True))
    return model.to(device)
