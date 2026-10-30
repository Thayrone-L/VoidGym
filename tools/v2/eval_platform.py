"""Evaluate a universal checkpoint on platform-gym behavior and perception."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.distributions import Categorical

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from voidgym.envs.adapter_env import AdapterEnv
from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter
from voidgym.gyms.platform_gym.params import Split
from voidgym.models.policy import UniversalPolicy
from voidgym.training.checkpoints import load_universal


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--compare-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "relatorios" / "85_eval")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--level8-sets", type=int, default=30)
    parser.add_argument("--level7-sets", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--perception-frames", type=int, default=5625)
    parser.add_argument("--perception-max-steps", type=int, default=512,
                        help="maximum actions per episode for balanced perception sampling")
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=85_000)
    parser.add_argument("--perception-only", action="store_true",
                        help="skip behavior rollouts and run only balanced perception")
    return parser.parse_args()


def _evaluate_balanced(args: argparse.Namespace) -> dict[str, Any]:
    from eval_perception_platform import evaluate

    perception_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        compare_checkpoint=args.compare_checkpoint,
        output=args.output / f"perception_{args.split}",
        frames=args.perception_frames,
        split=args.split,
        seed=args.seed + 1_000_000,
        max_steps=args.perception_max_steps,
        episodes_per_level=100,
        max_episodes_per_level=500,
        stride=8,
    )
    return evaluate(perception_args)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.perception_only:
        result = _evaluate_balanced(args)
        print(json.dumps(result, indent=2))
        return {"perception": result, "output": args.output}
    device = torch.device("cpu")
    model = UniversalPolicy().to(device)
    action_table = tuple(PlatformGymAdapter(level=0, split=args.split).action_table())
    checkpoint_manifest = load_universal(args.checkpoint, model, action_table)
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "gifs").mkdir(exist_ok=True)
    progress_path = args.output / "behavior_progress.json"
    stats: list[dict[str, Any]] = []
    completed: set[tuple[str, int]] = set()
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        stats = progress["stats"]
        completed = {(row["mode"], row["level"]) for row in stats}
    for stochastic in (False, True):
        mode = "stochastic" if stochastic else "deterministic"
        for level in range(9):
            if (mode, level) in completed:
                continue
            level8_sets = args.level7_sets if args.level7_sets is not None else args.level8_sets
            episodes = level8_sets if level == 8 else args.episodes
            rows = [_episode(
                model, level, args.seed + level * 10_000 + index * 10,
                split=args.split, stochastic=stochastic, max_steps=args.max_steps,
                output_root=args.output,
                capture=(not stochastic and level in {3, 4, 6} and index == 0),
            ) for index in range(episodes)]
            stats.append({
                "mode": mode, "level": level, "episodes": episodes,
                "success_rate": float(np.mean([row["success"] for row in rows])),
                "mean_steps": float(np.mean([row["steps"] for row in rows])),
                "deaths": int(sum(row["deaths"] for row in rows)),
                "stuck": int(sum(row["stuck"] for row in rows)),
            })
            _save_progress(progress_path, stats)
            completed.add((mode, level))
            print(f"Avaliação: {mode}, nível {level} concluído.", flush=True)
    perception = _evaluate_balanced(args)
    report = _format_report(stats, perception)
    (args.output / "tabela.md").write_text(report, encoding="utf-8", newline="\n")
    (args.output / "manifest_eval.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "trained_steps": checkpoint_manifest.get("trained_steps"),
        "episodes_per_level": args.episodes,
        "level8_sets": args.level8_sets,
        "split": args.split,
        "perception_frames": perception["sampled_frames"],
        "perception_output": str(args.output / f"perception_{args.split}"),
        "perception_frames_by_level": perception["frames_by_level"],
    }, indent=2) + "\n", encoding="utf-8")
    return {"stats": stats, "perception": perception, "output": args.output}


def _save_progress(path: Path, stats: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"stats": stats}, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _episode(model: UniversalPolicy, level: int, seed: int, *, split: Split,
             stochastic: bool, max_steps: int, output_root: Path,
             capture: bool) -> dict[str, Any]:
    adapter = PlatformGymAdapter(level=level, seed=seed, split=split)
    env = AdapterEnv(adapter, lambda _before, _after: 0.0)
    observation, info = env.reset(seed=seed)
    state = model.initial_state(1, torch.device("cpu"))
    starts = torch.ones(1, dtype=torch.bool)
    deaths = 0
    stuck = False
    success = False
    frames: list[Image.Image] = []
    steps = 0
    for steps in range(1, max_steps + 1):
        tensors = {key: torch.as_tensor(value[None]) for key, value in observation.items()}
        mask = torch.as_tensor(env.action_masks()[None])
        with torch.no_grad():
            policy_output = model(tensors, state, starts, mask)
            logits = policy_output["logits"]
            next_state = policy_output["state"]
            assert isinstance(logits, Tensor) and isinstance(next_state, tuple)
            distribution = Categorical(logits=logits)
            action = (int(distribution.sample().item())  # type: ignore[no-untyped-call]
                      if stochastic else int(logits.argmax(-1).item()))
            state = next_state
        observation, _, terminated, truncated, info = env.step(action)
        starts = torch.tensor([terminated or truncated])
        signals = info.get("signals", {})
        deaths += sum(getattr(event, "kind", None) == "death"
                      for event in signals.get("events", ()))
        stuck |= bool(signals.get("stuck", False))
        success |= (terminated if level == 8 else
                    bool(signals.get("room_changed", False)))
        if capture and steps % 5 == 0:
            frames.append(Image.fromarray(adapter.frame_rgb()).resize(
                (384, 216), Image.Resampling.NEAREST))
        if success or terminated or truncated:
            break
    env.close()
    if capture and frames:
        frames[0].save(output_root / "gifs" / f"nivel_{level}.gif", save_all=True,
                       append_images=frames[1:], duration=60, loop=0)
    return {"success": success, "steps": steps, "deaths": deaths, "stuck": stuck}


def _format_report(stats: list[dict[str, Any]], perception: dict[str, Any]) -> str:
    lines = [
        "# Avaliação do pré-treino v2", "",
        "| Modo | Nível | Episódios | Sucesso | Passos médios | Mortes | Travadas |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stats:
        lines.append(
            f"| {row['mode']} | {row['level']} | {row['episodes']} | "
            f"{row['success_rate']:.1%} | {row['mean_steps']:.1f} | "
            f"{row['deaths']} | {row['stuck']} |"
        )
    lines.extend([
        "", "## Percepção balanceada", "",
        (f"Amostra: {perception['sampled_frames']} quadros ({perception['split']}), "
         "distribuída pelos níveis e coletada com rota fixa independente do checkpoint."),
        "Confusão total e por nível, suportes e IoU estão no diretório de percepção.",
        "GIFs: gifs/nivel_3.gif, gifs/nivel_4.gif, gifs/nivel_6.gif.", "",
    ])
    return "\n".join(lines)


def main() -> int:
    evaluate(_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
