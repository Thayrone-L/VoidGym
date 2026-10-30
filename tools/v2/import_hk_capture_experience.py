"""Import moving semantic PNG sidecars into the standard experience format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hk_agent.semantic_labels import unpack_u4
from hk_agent.temporal_actions import (
    PLAYER_INTERACTION_ACTIONS,
    TEMPORAL_COMBAT_ACTIONS,
)
from voidgym.core.experience_store import ExperienceWriter

ACTION_TABLE = tuple(item.name for item in TEMPORAL_COMBAT_ACTIONS + PLAYER_INTERACTION_ACTIONS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--campaign", default="86c-moving-kings-pass")
    parser.add_argument("--count", type=int, default=1500)
    args = parser.parse_args()
    manifest = json.loads(args.capture_manifest.read_text(encoding="utf-8"))
    captures = manifest.get("captures")
    if not isinstance(captures, list):
        raise TypeError("capture manifest must contain a captures list")
    records = [row for row in captures if isinstance(row, dict)
               and row.get("duplicate_of") is None
               and isinstance(row.get("semantic_labels_path"), str)]
    if len(records) < args.count:
        raise ValueError(f"need {args.count} unique labeled captures, found {len(records)}")

    written = 0
    previous_scene: str | None = None
    with ExperienceWriter(args.output_root, "hollow_knight", args.campaign,
                          source="test", block_steps=512, action_table=ACTION_TABLE) as writer:
        for record in records[:args.count]:
            image_path = Path(str(record["path"]))
            labels_path = Path(str(record["semantic_labels_path"]))
            with Image.open(image_path) as source:
                frame = np.asarray(source.convert("RGB").resize(
                    (192, 108), Image.Resampling.BILINEAR), dtype=np.uint8).copy()
            labels = unpack_u4(labels_path.read_bytes(), width=48, height=27)
            scene = str(record.get("scene_name") or "")
            writer.append({
                "frame": frame,
                "sem": labels,
                "action": 0,
                "done": False,
                "truncated": False,
                "discontinuity": written == 0 or scene != previous_scene,
                "scene": scene,
                "tags": ["86c_moving_capture"],
                "priority": 0.0,
            })
            previous_scene = scene
            written += 1
    print(json.dumps({"written_frames": written, "scene_counts": {
        scene: sum(str(row.get("scene_name") or "") == scene for row in records[:written])
        for scene in sorted({str(row.get("scene_name") or "") for row in records[:written]})
    }, "experience_root": str(args.output_root), "campaign": args.campaign},
        ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
