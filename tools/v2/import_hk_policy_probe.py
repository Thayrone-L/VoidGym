"""Import the sampled frame/label pairs from a 86h HK policy probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hk_agent.temporal_actions import PLAYER_INTERACTION_ACTIONS, TEMPORAL_COMBAT_ACTIONS
from voidgym.core.experience_store import ExperienceWriter

ACTION_TABLE = tuple(item.name for item in TEMPORAL_COMBAT_ACTIONS + PLAYER_INTERACTION_ACTIONS)


def import_probe(probe_root: Path, output_root: Path, campaign_id: str,
                 count: int | None = None) -> dict[str, Any]:
    summary = json.loads((probe_root / "summary.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in
               (probe_root / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    candidates = []
    for record in records:
        step = record.get("step")
        if type(step) is int and step % 5 == 0:
            sidecar = probe_root / "labels" / f"frame-{step:06d}.npz"
            if sidecar.is_file():
                candidates.append((record, sidecar))
    if count is not None:
        if count < 1 or count > len(candidates):
            raise ValueError(f"count must be between 1 and {len(candidates)}")
        indices = np.linspace(0, len(candidates) - 1, count, dtype=np.int64)
        candidates = [candidates[int(index)] for index in indices]
    written = 0
    previous_scene: str | None = None
    with ExperienceWriter(output_root, "hollow_knight", campaign_id, source="test",
                          block_steps=512, action_table=ACTION_TABLE) as writer:
        for record, sidecar in candidates:
            step = int(record["step"])
            with np.load(sidecar, allow_pickle=False) as data:
                frame = np.asarray(data["image"], dtype=np.uint8).copy()
                sem = np.asarray(data["sem"], dtype=np.uint8).copy()
            if frame.shape != (108, 192, 3) or sem.shape != (27, 48):
                raise ValueError(f"unexpected frame/semantic shape at step {step}")
            scene = str(record.get("scene_name") or "")
            writer.append({
                "frame": frame,
                "sem": sem,
                "action": int(record["action_index"]),
                "done": bool(record["terminated"]),
                "truncated": bool(record["truncated"]),
                "discontinuity": written == 0 or scene != previous_scene,
                "scene": scene,
                "tags": ["86h_policy_probe", f"decision:{step}"],
                "priority": 0.0,
            })
            written += 1
            previous_scene = scene
    result = {"probe_root": str(probe_root), "experience_root": str(output_root),
              "campaign_id": campaign_id, "source": "test", "sampled_frames": written,
              "requested_count": count,
              "action_table_matches_checkpoint": summary.get("action_table_matches")}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "probe-import.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--campaign-id", default="86h-policy-probe")
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    print(json.dumps(import_probe(args.probe_root, args.output_root, args.campaign_id,
                                  args.count),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
