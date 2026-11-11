"""Persistent, bounded self-imitation replay for shared recurrent PPO."""
from __future__ import annotations

import json
import math
import os
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


class SelfImitationBank:
    """Atomically persisted sequence bank; independent of checkpoint retention."""

    def __init__(self, root: Path, *, max_steps: int = 50_000, max_bytes: int = 2_000_000_000) -> None:
        self.root = root
        self.max_steps = max_steps
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def _manifest(self) -> dict[str, Any]:
        try:
            value = json.loads((self.root / "manifest.json").read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"schema_version": 1, "samples": []}
        return value if isinstance(value, dict) and isinstance(value.get("samples"), list) else {"schema_version": 1, "samples": []}

    def _save(self, manifest: Mapping[str, Any]) -> None:
        path = self.root / "manifest.json"
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(5)}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(dict(manifest), stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def ingest(self, rollouts: Sequence[Mapping[str, np.ndarray]], *, round_number: int, gamma: float, chunk: int = 32) -> int:  # noqa: C901 - applies bounded replay retention
        manifest = self._manifest()
        samples = [row for row in manifest["samples"] if isinstance(row, dict)]
        added = 0
        recurrent = {"state_h", "state_c"}
        for slot, rollout in enumerate(rollouts):
            if not recurrent <= set(rollout):
                continue
            count = len(rollout["returns"])
            returns = np.asarray(rollout["returns"], dtype=np.float32).reshape(-1)
            values = np.asarray(rollout["values"], dtype=np.float32).reshape(-1)
            for start in range(0, count - chunk + 1, chunk):
                stop = start + chunk
                advantage = returns[start:stop] - values[start:stop]
                if float(np.max(advantage)) <= 0:
                    continue
                # Keep only above-expectation sequences; R is the discounted return at each step.
                payload = {key: np.asarray(value[start:stop]).copy() for key, value in rollout.items() if key.startswith("obs__") or key in {"actions", "action_masks", "episode_starts", "returns", "log_probs", "privileged"}}
                scene_values = np.asarray(rollout.get("meta__scene", np.asarray([], dtype="U1")))[start:stop]
                valid_scenes = [str(value) for value in scene_values if str(value)]
                scene = max(set(valid_scenes), key=valid_scenes.count) if valid_scenes else "unknown"
                event_values = np.asarray(rollout.get("meta__events", np.asarray([], dtype="U1")))[start:stop]
                confirmed_events: list[str] = []
                for raw_events in event_values:
                    try:
                        decoded = json.loads(str(raw_events))
                    except json.JSONDecodeError:
                        continue
                    if isinstance(decoded, list):
                        confirmed_events.extend(value for value in decoded if value in {
                            "scene_change", "shade_created", "shade_recovered", "shade_lost",
                            "item", "bench", "help_solo",
                        })
                # `max(set(...))` makes ties depend on hash iteration order, so
                # the same rollout could be filed under different event buckets.
                event = max(confirmed_events, key=confirmed_events.count) if confirmed_events else "high_return"
                for name in ("meta__scene", "meta__progress", "meta__events"):
                    if name in rollout:
                        payload[name] = np.asarray(rollout[name][start:stop]).copy()
                for key in recurrent:
                    payload[key] = np.asarray(rollout[key][start]).copy()
                filename = f"r{round_number:08d}-s{slot}-{start:08d}-{secrets.token_hex(3)}.npz"
                pending = self.root / f".{filename}.tmp"
                with pending.open("wb") as stream:
                    np.savez_compressed(stream, **payload)  # type: ignore[arg-type]
                    stream.flush()
                    os.fsync(stream.fileno())
                final = self.root / filename
                os.replace(pending, final)
                priority = max(0.0, float(np.max(advantage)))
                progress_values = np.asarray(rollout.get("meta__progress", np.asarray([], dtype=np.float32)))[start:stop]
                best_progress = float(np.max(progress_values)) if len(progress_values) else 0.0
                samples.append({"file": filename, "steps": chunk, "bytes": final.stat().st_size, "priority": priority, "scene": scene, "event": event, "best_progress": best_progress, "round": round_number, "slot": slot, "start": start})
                added += 1
        # Evict lowest-priority items first while preserving diverse event/scene buckets.
        samples.sort(key=lambda item: (float(item.get("priority", 0)), str(item.get("scene")), str(item.get("event"))))
        while sum(int(row.get("steps", 0)) for row in samples) > self.max_steps or sum(int(row.get("bytes", 0)) for row in samples) > self.max_bytes:
            victim = samples.pop(0)
            try:
                (self.root / str(victim["file"])).unlink()
            except OSError:
                pass
        self._save({"schema_version": 1, "samples": samples})
        return added

    def load(self, limit: int = 8) -> list[dict[str, np.ndarray]]:
        rows = sorted(self._manifest()["samples"], key=lambda item: float(item.get("priority", 0)), reverse=True)
        result = []
        for row in rows[:limit]:
            try:
                with np.load(self.root / str(row["file"]), allow_pickle=False) as archive:
                    result.append({name: archive[name].copy() for name in archive.files})
            except (OSError, ValueError, KeyError):
                continue
        return result

    def summary(self) -> dict[str, Any]:
        rows = self._manifest()["samples"]
        by_event: dict[str, int] = {}
        by_scene: dict[str, int] = {}
        for row in rows:
            event, scene = str(row.get("event", "high_return")), str(row.get("scene", "unknown"))
            by_event[event] = by_event.get(event, 0) + 1
            by_scene[scene] = by_scene.get(scene, 0) + 1
        progress = [float(row["best_progress"]) for row in rows if isinstance(row.get("best_progress"), (int, float))]
        return {"sample_count": len(rows), "step_count": sum(int(row.get("steps", 0)) for row in rows), "by_event": by_event, "by_scene": by_scene, "best_progress": max(progress) if progress else None}


def sil_positive_advantage(returns: Any, values: Any) -> Any:
    """SIL ignores trajectories that no longer outperform the current critic."""
    return (returns - values).clamp_min(0)


def sil_guard_reason(entropy: float, kl: float, *, entropy_floor: float, kl_limit: float) -> str | None:
    if not math.isfinite(entropy) or entropy < entropy_floor:
        return "entropy_below_floor"
    if not math.isfinite(kl) or kl > kl_limit:
        return "kl_above_limit"
    return None


def sequence_rollout(sequence: Any) -> dict[str, np.ndarray]:
    """Convert one universal-policy rollout sequence to the bounded bank schema."""
    row = {f"obs__{key}": value.detach().cpu().numpy()[0]
           for key, value in sequence.observations.items()}
    for key in ("actions", "action_masks", "episode_starts", "returns", "old_log_probs"):
        name = "log_probs" if key == "old_log_probs" else key
        row[name] = getattr(sequence, key).detach().cpu().numpy()[0]
    row["values"] = sequence.old_values.detach().cpu().numpy()[0]
    state_h, state_c = (value.detach().cpu().numpy()[:, 0]
                        for value in sequence.initial_state)
    count = len(row["returns"])
    row["state_h"] = np.repeat(state_h[None], count, axis=0)
    row["state_c"] = np.repeat(state_c[None], count, axis=0)
    row["privileged"] = sequence.privileged.detach().cpu().numpy()[0]
    return row


def train_sil_update(
    policy: Any, optimizer: Any, samples: Sequence[Mapping[str, np.ndarray]], *,
    game_id: str | None = None, actor_coef: float = 0.1, value_coef: float = 0.01,
    entropy_floor: float = 0.02, kl_limit: float = 0.05,
    max_grad_norm: float = 0.5,
) -> dict[str, Any]:
    """Apply SIL to recurrent UniversalPolicy sequences without the v1/SB3 API."""
    import torch
    from torch.distributions import Categorical

    if not samples:
        return {"active": False, "reason": "empty_bank", "weight": 0.0, "positive_samples": 0}
    device = next(policy.parameters()).device
    names = set.intersection(*(set(sample) for sample in samples))
    obs_names = sorted(name.removeprefix("obs__") for name in names if name.startswith("obs__"))
    if not {"pixels", "action_history"}.issubset(obs_names):
        return {"active": False, "reason": "invalid_observation", "weight": 0.0, "positive_samples": 0}
    observations = {name: torch.as_tensor(np.stack([sample[f"obs__{name}"] for sample in samples]), device=device) for name in obs_names}
    actions = torch.as_tensor(np.stack([sample["actions"] for sample in samples]), dtype=torch.long, device=device)
    masks = torch.as_tensor(np.stack([sample["action_masks"] for sample in samples]), dtype=torch.bool, device=device)
    returns = torch.as_tensor(np.stack([sample["returns"] for sample in samples]), dtype=torch.float32, device=device)
    starts = torch.as_tensor(np.stack([sample["episode_starts"] for sample in samples]), dtype=torch.bool, device=device)
    if not masks.any(dim=-1).all() or not masks.gather(-1, actions.unsqueeze(-1)).all():
        return {"active": False, "reason": "invalid_action_mask", "weight": 0.0, "positive_samples": 0}
    state = tuple(torch.as_tensor(np.stack([sample[key] for sample in samples], axis=1), device=device, dtype=torch.float32) for key in ("state_h", "state_c"))
    privileged = torch.as_tensor(np.stack([sample["privileged"] for sample in samples]), device=device, dtype=torch.float32) if "privileged" in names else None
    extras = observations.get("game_extras")
    output = policy(observations, state, starts, masks, privileged=privileged,
                    game_id=game_id if extras is not None else None, game_extras=extras)
    logits, values = output["logits"], output["values"]
    distribution = Categorical(logits=logits)
    log_prob = distribution.log_prob(actions)  # type: ignore[no-untyped-call]
    entropy = distribution.entropy()  # type: ignore[no-untyped-call]
    advantage = sil_positive_advantage(returns.reshape(-1), values.reshape(-1))
    positive = int((advantage > 0).sum().item())
    entropy_value = float(entropy.detach().mean().cpu())
    old_log = torch.as_tensor(np.stack([sample["log_probs"] for sample in samples]), dtype=torch.float32, device=device)
    approx_kl = float((old_log.reshape(-1) - log_prob.detach().reshape(-1)).mean().cpu())
    guard = sil_guard_reason(entropy_value, approx_kl, entropy_floor=entropy_floor, kl_limit=kl_limit)
    if guard:
        return {"active": False, "reason": guard, "weight": 0.0, "positive_samples": positive, "kl": approx_kl, "entropy": entropy_value}
    actor_loss = -(log_prob.reshape(-1) * advantage.detach()).mean()
    value_loss = 0.5 * advantage.square().mean()
    loss = actor_coef * actor_loss + value_coef * value_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()
    return {"active": positive > 0, "reason": "updated" if positive else "no_positive_advantage", "weight": float(loss.detach().cpu()), "positive_samples": positive, "entropy": entropy_value, "kl": approx_kl, "actor_loss": float(actor_loss.detach().cpu()), "value_loss": float(value_loss.detach().cpu())}
