from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def _worker_pythonpath(current: str | None = None) -> str:
    """Ensure the spawned episode worker can import the checkout from any cwd."""
    value = os.environ.get("PYTHONPATH", "") if current is None else current
    entries = [item for item in value.split(os.pathsep) if item]
    normalized = {os.path.normcase(os.path.abspath(item)) for item in entries}
    for path in (ROOT / "src", ROOT):
        resolved = str(path.resolve(strict=True))
        key = os.path.normcase(os.path.abspath(resolved))
        if key not in normalized:
            entries.insert(0, resolved)
            normalized.add(key)
    return os.pathsep.join(entries)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_inventory(root: Path) -> dict[str, str]:
    pattern = re.compile(r"user\d+(?:_[\d.]+)?\.(?:dat|modded\.json)(?:\.bak\d*)?$", re.IGNORECASE)
    return {path.name: _file_sha256(path) for path in sorted(root.iterdir())
            if path.is_file() and not path.is_symlink() and pattern.fullmatch(path.name)}


def _restore_save_files(root: Path, snapshot: Path,
                        before: dict[str, str]) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    after = _save_inventory(root)
    changed = sorted(name for name, digest in before.items() if after.get(name) != digest)
    added = sorted(set(after) - set(before))
    for name in changed:
        shutil.copy2(snapshot / name, root / name)
    for name in added:
        (root / name).unlink()
    restored = _save_inventory(root)
    return {"instance_data_root": str(root), "before_sha256": before,
            "after_sha256": restored, "restored": restored == before,
            "restored_files": changed, "removed_generated_save_files": added}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inference-only HK policy probe")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("duration must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    # The owner launches ``python -m hk_agent.episode_worker`` in a child
    # process. The parent-only sys.path inserts above are not inherited there.
    os.environ["PYTHONPATH"] = _worker_pythonpath()
    os.environ["HK_AGENT_SLOT_ID"] = "0"
    os.environ["HK_LABEL_DIAGNOSTICS"] = "true"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames_dir = output / "frames"
    labels_dir = output / "labels"
    inputs_dir = output / "model_inputs"
    for directory in (frames_dir, labels_dir, inputs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    from hk_agent.player_knowledge import PlayerKnowledge
    from tools.run_first_training import parse_args as parse_hk_args
    from voidgym.adapters.hollow_knight.runtime import create_hollow_knight_env
    from voidgym.core.observation import fit_to_universal
    from voidgym.models.policy import UniversalPolicy
    from voidgym.training.buffers import _privileged
    from voidgym.training.checkpoints import load_universal

    install = Path(r"C:\Users\thayr\Games\HollowKnight-AI\instances\hk-ai-1")
    save_root = Path(r"C:\Users\thayr\AppData\LocalLow\Team Cherry\Hollow Knight AI 1")
    save_snapshot = output / "instance-save-before"
    save_snapshot.mkdir(parents=True, exist_ok=True)
    before_save_hashes = _save_inventory(save_root)
    for name in before_save_hashes:
        shutil.copy2(save_root / name, save_snapshot / name)
        if _file_sha256(save_snapshot / name) != before_save_hashes[name]:
            raise RuntimeError(f"save snapshot verification failed for {name}")
    shared = output / "managed"
    hk_args = parse_hk_args([
        "--task", "exploration", "--reward-version", "explore_v4",
        "--game-install", str(install), "--instance-data-root", str(save_root),
        "--save-source", "existing_save", "--save-slot", "1", "--continuous",
        "--death-penalty", "-5.0", "--damage-penalty", "-0.5",
        "--run-failed-penalty", "-5.0", "--enemy-kill-reward-budget", "5.0",
        "--ent-coef", "0.01", "--gamma", "0.995", "--gae-lambda", "0.95",
        "--n-epochs", "4", "--target-kl", "0.03", "--learning-rate", "0.00025",
        "--clip-range", "0.2",
        "--lease-ms", "80", "--frame-stack", "4", "--width", "600", "--height", "338",
        "--output-root", str(output / "task"), "--log-file", str(output / "GameLog.txt"),
        "--horizon", "3000",
    ])
    env, owner = create_hollow_knight_env(
        hk_args, shared_root=shared, actor_id="probe-86j", slot_id=0,
        request_semantic_labels=True,
    )
    device = torch.device(args.device)
    policy = UniversalPolicy(action_count=len(env.actions))
    policy.add_game_extras("hollow_knight", int(PlayerKnowledge.space().shape[0]))
    manifest = load_universal(args.checkpoint.resolve(strict=True), policy, tuple(env.actions))
    policy.to(device).eval()
    action_names = tuple(env.actions)
    attack_indices = [i for i, name in enumerate(action_names) if "ATTACK" in name or "NAIL_ART" in name]
    if not attack_indices:
        raise RuntimeError("HK action table contains no attack actions")

    backend = env.adapter.env.environment.backend
    original_step = backend.step
    applied_rows: list[dict[str, Any]] = []

    def traced_step(*, action_id: int, buttons: tuple[str, ...], lease_ms: int) -> Any:
        result = original_step(action_id=action_id, buttons=buttons, lease_ms=lease_ms)
        applied = result.applied_action
        applied_rows.append({
            "action_id": applied.action_id,
            "requested_buttons": applied.requested_buttons,
            "applied_buttons": applied.applied_buttons,
            "lease_ms": applied.lease_ms,
            "combat_events": result.frame.combat_events,
        })
        return result

    backend.step = traced_step
    records_path = output / "decisions.jsonl"
    counts: Counter[str] = Counter()
    mask_counts: Counter[str] = Counter()
    decisions_with_attack_available = 0
    attack_action_slots_available = 0
    attack_probability_sum = 0.0
    rightward_advance = 0.0
    hit_enemy_events = 0
    position_records = 0
    movement_directions: Counter[str] = Counter()
    previous_position: tuple[str | None, float] | None = None
    positions_by_scene: dict[str, list[float]] = {}
    total_steps = 0
    episode_resets = 0
    sampled_gif_frames: list[Image.Image] = []
    input_comparisons: list[dict[str, Any]] = []
    started = 0.0
    observation: dict[str, np.ndarray] | None = None
    info: dict[str, Any] = {}
    state: tuple[torch.Tensor, torch.Tensor] | None = None
    episode_start = True
    step_in_episode = 0
    try:
        observation, info = env.reset(seed=8601)
        state = policy.initial_state(1, device)
        started = time.monotonic()
        with records_path.open("w", encoding="utf-8", buffering=1) as records:
            while time.monotonic() - started < args.duration:
                masks = np.asarray(env.action_masks(), dtype=np.bool_)
                obs_t = {key: torch.as_tensor(value[None], device=device)
                         for key, value in observation.items()}
                mask_t = torch.as_tensor(masks[None], device=device)
                privileged = _privileged(info)[None].to(device)
                starts = torch.tensor([episode_start], dtype=torch.bool, device=device)
                extras = obs_t.get("game_extras")
                with torch.inference_mode():
                    result = policy(
                        obs_t, state, starts, mask_t, privileged=privileged,
                        game_id="hollow_knight" if extras is not None else None,
                        game_extras=extras,
                    )
                    logits = result["logits"]
                    next_state = result["state"]
                    if not isinstance(logits, torch.Tensor) or not isinstance(next_state, tuple):
                        raise TypeError("policy output does not contain logits and recurrent state")
                    probabilities = Categorical(logits=logits).probs[0]
                    action_tensor = Categorical(logits=logits).sample()  # type: ignore[no-untyped-call]
                    if total_steps in {0, 1, 2, 3, 4, 20}:
                        stack = np.asarray(observation["pixels"], dtype=np.uint8)
                        channels = np.moveaxis(stack, 0, -1).reshape(108, 192, 4, 3)
                        if channels.shape != (108, 192, 4, 3):
                            raise RuntimeError(f"unexpected policy image stack shape: {channels.shape}")
                        input_step_dir = inputs_dir / f"hk_step_{total_steps:06d}"
                        input_step_dir.mkdir(parents=True, exist_ok=True)
                        sheet = Image.new("RGB", (384, 216))
                        frame_hashes = []
                        for frame_index in range(4):
                            pixels = channels[:, :, frame_index, :].copy()
                            frame = Image.fromarray(pixels, mode="RGB")
                            frame.save(input_step_dir / f"frame_{frame_index}.png")
                            sheet.paste(frame, ((frame_index % 2) * 192,
                                                (frame_index // 2) * 108))
                            frame_hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
                        sheet.save(input_step_dir / "stack.png")
                        action_history = np.asarray(observation["action_history"], dtype=np.int64)
                        np.savez_compressed(input_step_dir / "model_input.npz",
                                            pixels=stack, action_history=action_history)
                        zero_observation = dict(obs_t)
                        zero_observation["action_history"] = torch.zeros_like(
                            obs_t["action_history"]
                        )
                        zero_result = policy(
                            zero_observation, state, starts, mask_t,
                            privileged=privileged,
                            game_id="hollow_knight" if extras is not None else None,
                            game_extras=extras,
                        )
                        zero_logits = zero_result["logits"]
                        if not isinstance(zero_logits, torch.Tensor):
                            raise TypeError("null-history policy output has invalid logits")
                        null_probabilities = torch.softmax(zero_logits, dim=-1)[0]
                        real_probabilities = probabilities
                        real_top = int(real_probabilities.argmax().item())
                        null_top = int(null_probabilities.argmax().item())
                        real_attack_probability = float(real_probabilities[attack_indices].sum().item())
                        null_attack_probability = float(null_probabilities[attack_indices].sum().item())
                        comparison = {
                            "step": total_steps,
                            "action_history": action_history.tolist(),
                            "frame_sha256": frame_hashes,
                            "distinct_input_frames": len(set(frame_hashes)),
                            "pixel_shape_chw": list(stack.shape),
                            "pixel_dtype": str(stack.dtype),
                            "pixel_min": int(stack.min()),
                            "pixel_max": int(stack.max()),
                            "real_history_top_action": action_names[real_top],
                            "null_history_top_action": action_names[null_top],
                            "top_action_changed": real_top != null_top,
                            "total_variation_distance": float(
                                0.5 * torch.abs(real_probabilities - null_probabilities).sum().item()
                            ),
                            "real_history_attack_probability": real_attack_probability,
                            "null_history_attack_probability": null_attack_probability,
                        }
                        input_comparisons.append(comparison)
                action = int(action_tensor.item())
                probs = probabilities.detach().cpu().numpy()
                attack_prob = float(probs[attack_indices].sum())
                attack_probability_sum += attack_prob
                counts[action_names[action]] += 1
                for index, allowed in enumerate(masks):
                    if allowed:
                        mask_counts[action_names[index]] += 1
                available_attack_indices = [index for index in attack_indices if masks[index]]
                decisions_with_attack_available += int(bool(available_attack_indices))
                attack_action_slots_available += len(available_attack_indices)
                before_applied = len(applied_rows)
                next_observation, reward, terminated, truncated, next_info = env.step(action)
                if len(applied_rows) != before_applied + 1:
                    raise RuntimeError("backend did not report exactly one applied action")
                applied = applied_rows[-1]
                signals = next_info.get("signals", {})
                combat_events = signals.get("combat_events", applied["combat_events"])
                telemetry = signals.get("operator_telemetry", {})
                telemetry = telemetry if isinstance(telemetry, dict) else {}
                x = signals.get("x", telemetry.get("x"))
                y = signals.get("y", telemetry.get("y"))
                scene_name = signals.get("scene_name")
                position_x = float(x) if isinstance(x, (int, float)) else None
                position_y = float(y) if isinstance(y, (int, float)) else None
                delta_x = None
                if position_x is not None and position_y is not None:
                    position_records += 1
                    scene_key = str(scene_name or "unknown")
                    if scene_key not in positions_by_scene:
                        positions_by_scene[scene_key] = [position_x, position_x]
                    else:
                        positions_by_scene[scene_key][1] = position_x
                    if previous_position is not None and previous_position[0] == scene_name:
                        delta_x = position_x - previous_position[1]
                        rightward_advance += max(0.0, delta_x)
                    previous_position = (scene_name if isinstance(scene_name, str) else None,
                                         position_x)
                labels = env.adapter.training_labels() or {}
                velocity = labels.get("hero_velocity")
                vx = (float(velocity[0]) if isinstance(velocity, (tuple, list, np.ndarray))
                      and len(velocity) >= 1 else None)
                movement_direction = ("right" if vx is not None and vx > .05 else
                                      "left" if vx is not None and vx < -.05 else "stationary")
                movement_directions[movement_direction] += 1
                if isinstance(combat_events, (tuple, list)):
                    hit_enemy_events += sum(
                        isinstance(event, dict) and event.get("kind") == "hit_enemy"
                        for event in combat_events
                    )
                frame_rgb = env.adapter.frame_rgb().copy()
                if total_steps % 5 == 0:
                    universal = fit_to_universal(frame_rgb)
                    image = Image.fromarray(universal, mode="RGB")
                    sampled_gif_frames.append(image)
                    frame_path = frames_dir / f"frame-{total_steps:06d}.jpg"
                    image.save(frame_path, quality=92)
                    labels = env.adapter.training_labels()
                    if labels and isinstance(labels.get("sem"), np.ndarray):
                        np.savez_compressed(
                            labels_dir / f"frame-{total_steps:06d}.npz",
                            image=universal, sem=labels["sem"],
                        )
                records.write(json.dumps({
                    "step": total_steps,
                    "elapsed_seconds": time.monotonic() - started,
                    "episode_step": step_in_episode,
                    "action_index": action,
                    "action_name": action_names[action],
                    "action_mask": masks.astype(bool).tolist(),
                    "action_history": observation["action_history"],
                    "input_history_comparison": (
                        input_comparisons[-1] if input_comparisons
                        and input_comparisons[-1]["step"] == total_steps else None
                    ),
                    "allowed_action_names": [action_names[i] for i, allowed in enumerate(masks) if allowed],
                    "attack_probabilities": {action_names[i]: float(probs[i]) for i in attack_indices},
                    "attack_probability_total": attack_prob,
                    "applied_buttons": applied["applied_buttons"],
                    "requested_buttons": applied["requested_buttons"],
                    "combat_events": combat_events,
                    "position_x": position_x,
                    "position_y": position_y,
                    "delta_x_since_previous_decision": delta_x,
                    "hero_velocity_x": vx,
                    "hero_movement_direction": movement_direction,
                    "applied_direction": ("right" if "RIGHT" in applied["applied_buttons"] else
                                          "left" if "LEFT" in applied["applied_buttons"] else None),
                    "scene_name": scene_name,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }, default=_json_default, sort_keys=True) + "\n")
                total_steps += 1
                step_in_episode += 1
                if terminated or truncated:
                    observation, info = env.reset(seed=8601 + total_steps)
                    state = policy.initial_state(1, device)
                    episode_start = True
                    step_in_episode = 0
                    episode_resets += 1
                else:
                    observation, info = next_observation, next_info
                    state = next_state
                    episode_start = False
                if total_steps % 100 == 0:
                    records.flush()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
        if len(sampled_gif_frames) >= 300:
            gif_frames = sampled_gif_frames[-300:]
        else:
            gif_frames = sampled_gif_frames
        if gif_frames:
            gif_frames[0].save(
                output / "policy-30s.gif", save_all=True, append_images=gif_frames[1:],
                duration=max(10, round(30_000 / len(gif_frames))), loop=0, optimize=True,
            )
        (inputs_dir / "history_comparison.json").write_text(
            json.dumps(input_comparisons, ensure_ascii=False, indent=2, default=_json_default)
            + "\n", encoding="utf-8"
        )
    finally:
        backend.step = original_step
        try:
            env.close()
        finally:
            try:
                owner.close()
            finally:
                restoration = _restore_save_files(save_root, save_snapshot,
                                                 before_save_hashes)
                (output / "save-restoration.json").write_text(
                    json.dumps(restoration, indent=2) + "\n", encoding="utf-8"
                )

    summary = {
        "status": "COMPLETED",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_action_table": (manifest.get("action_table") or manifest.get("actions")),
        "adapter_action_table": action_names,
        "action_table_matches": tuple(manifest.get("action_table") or manifest.get("actions") or ()) == action_names,
        "device": str(device),
        "duration_seconds": time.monotonic() - started,
        "requested_duration_seconds": args.duration,
        "decisions": total_steps,
        "episode_resets": episode_resets,
        "chosen_actions": dict(counts),
        "attack_action_indices": attack_indices,
        "attack_action_names": [action_names[i] for i in attack_indices],
        "attack_mask_allowed_decisions": decisions_with_attack_available,
        "attack_mask_allowed_action_slots": attack_action_slots_available,
        "attack_mask_allowed_per_action": {name: mask_counts[name] for name in (action_names[i] for i in attack_indices)},
        "attack_probability_mean": attack_probability_sum / max(1, total_steps),
        "position_records": position_records,
        "position_coverage": position_records / max(1, total_steps),
        "rightward_advance_world_units": sum(
            max(0.0, last - first) for first, last in positions_by_scene.values()
        ),
        "cumulative_positive_x_motion_world_units": rightward_advance,
        "net_x_change_by_scene": {
            scene: last - first for scene, (first, last) in positions_by_scene.items()
        },
        "hit_enemy_events": hit_enemy_events,
        "hero_movement_directions": dict(movement_directions),
        "direction_measurement": (
            "movement direction inferred from semantic hero_velocity.vx; "
            "applied button direction is recorded separately"
        ),
        "records": str(records_path),
        "gif": str(output / "policy-30s.gif"),
        "save_restoration": restoration,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
