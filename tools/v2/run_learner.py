"""Learner de referência mínimo com checkpoint/resume para o harness E2E."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"registro inválido: {path}")
    return value


def _write(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def run(root: Path, actors: tuple[str, ...], checkpoint: Path,
        stop_after_round: int) -> None:
    rollout_dir = root / "rollouts"
    policy_dir = root / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    last_round = 0
    updates = 0
    policy_parameter = 0.0
    if checkpoint.is_file():
        saved = _read(checkpoint)
        last_round = int(saved["round"])
        updates = int(saved["updates"])
        policy_parameter = float(saved["policy_parameter"])
    for round_id in range(last_round + 1, stop_after_round + 1):
        deadline = time.monotonic() + 180.0
        rows: list[dict[str, Any]] = []
        while len(rows) < len(actors):
            for actor_id in actors:
                path = rollout_dir / f"round-{round_id}-{actor_id}.json"
                if path.is_file() and not any(row["actor_id"] == actor_id for row in rows):
                    row = _read(path)
                    if row.get("round") != round_id:
                        raise ValueError(f"rollout de {actor_id} tem rodada incorreta")
                    rows.append(row)
            if len(rows) == len(actors):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"atores não publicaram a rodada {round_id}")
            time.sleep(0.02)
        updates += 1
        positions = [float(row["signals"]["position"]) for row in rows]
        policy_parameter += sum(positions) / len(positions) / 1000.0
        saved = {"round": round_id, "updates": updates,
                 "actors": list(actors), "rollouts": rows,
                 "policy_parameter": policy_parameter}
        _write(checkpoint, saved)
        _write(policy_dir / f"round-{round_id}.json", {
            "round": round_id, "updates": updates,
            "policy_parameter": policy_parameter,
            "action": 2 if policy_parameter >= 0 else 1,
        })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fake", "universal"), default="fake")
    parser.add_argument("--shared-root", type=Path)
    parser.add_argument("--campaign-config", type=Path)
    parser.add_argument("--universal-checkpoint", type=Path)
    parser.add_argument("--actors")
    parser.add_argument("--campaign-id", default="v2-hollow-knight")
    parser.add_argument("--experience-root", type=Path, default=Path("runtime/v2/experience"))
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.00025)
    parser.add_argument("--lr-encoder-scale", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--c-sem", type=float, default=0.5)
    parser.add_argument("--c-inv", type=float, default=0.1)
    parser.add_argument("--c-mot", type=float, default=0.1)
    parser.add_argument("--freeze-encoder-rounds", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    v2_args, remaining = parser.parse_known_args()
    if v2_args.mode == "universal":
        if (v2_args.shared_root is None or v2_args.universal_checkpoint is None
                or v2_args.campaign_config is None):
            parser.error("universal mode requires --shared-root, --campaign-config and --universal-checkpoint")
        from hk_agent.player_knowledge import PlayerKnowledge
        from voidgym.core.config import config_from_dict
        from voidgym.core.hardware import detect_hardware, plan_resources
        from voidgym.training.shared_learner import run_shared_learner

        if remaining:
            parser.error(f"unexpected universal learner arguments: {' '.join(remaining)}")
        root = v2_args.shared_root.resolve()
        campaign_data = json.loads(v2_args.campaign_config.read_text(encoding="utf-8"))
        campaign = config_from_dict(campaign_data)
        if campaign.game_id != "hollow_knight" or campaign.campaign_id != v2_args.campaign_id:
            parser.error("CampaignConfig does not match this Hollow Knight campaign")
        start_from = campaign.start_from
        if (start_from is None or not start_from.checkpoint
                or Path(start_from.checkpoint).resolve() != v2_args.universal_checkpoint.resolve()):
            parser.error("CampaignConfig checkpoint does not match --universal-checkpoint")
        manifest = json.loads((v2_args.universal_checkpoint / "manifest.json").read_text(encoding="utf-8"))
        action_table = manifest.get("action_table")
        if not isinstance(action_table, list) or not action_table or not all(isinstance(x, str) for x in action_table):
            parser.error("universal checkpoint has no valid ordered action table")
        profile = detect_hardware(v2_args.experience_root)
        actor_names = v2_args.actors or "actor-slot-0"
        plan = plan_resources(profile, "game_training", {"game_instances": len(actor_names.split(","))})
        device = plan.train_device if v2_args.device == "auto" else v2_args.device
        if device == "cuda" and plan.train_device != "cuda":
            parser.error("CUDA was explicitly selected but ResourcePlan does not provide it")
        actors = tuple(item for item in actor_names.split(",") if item)
        result = run_shared_learner(
            shared_root=root, universal_checkpoint=v2_args.universal_checkpoint.resolve(strict=True),
            action_table=tuple(action_table), actors=actors, total_timesteps=v2_args.timesteps,
            rollout_steps=v2_args.rollout_steps, sequence_length=v2_args.sequence_length,
            game_id="hollow_knight", game_extras_dim=int(PlayerKnowledge.space().shape[0]),
            learning_rate=campaign.ppo.learning_rate, lr_encoder_scale=v2_args.lr_encoder_scale,
            gamma=campaign.ppo.gamma, gae_lambda=campaign.ppo.gae_lambda, clip_range=campaign.ppo.clip_range,
            entropy_coef=campaign.ppo.entropy_coef, epochs=campaign.ppo.epochs,
            batch_size=campaign.ppo.batch_size, c_sem=campaign.aux.c_sem, c_inv=campaign.aux.c_inv,
            c_mot=campaign.aux.c_mot, device_name=device, seed=v2_args.seed,
            continuous=v2_args.continuous,
            freeze_encoder_rounds=v2_args.freeze_encoder_rounds,
            stop_file=v2_args.stop_file,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stop-after-round", type=int, required=True)
    args = parser.parse_args(remaining)
    actors = tuple(item for item in (v2_args.actors or "actor-a,actor-b").split(",") if item)
    if not actors or args.stop_after_round < 1:
        parser.error("at least one actor and a positive target round are required")
    run(args.root, actors, args.checkpoint, args.stop_after_round)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
