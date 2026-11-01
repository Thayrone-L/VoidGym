"""Export one exact four-frame PlatformGym policy input and metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from voidgym.envs.adapter_env import AdapterEnv
from voidgym.gyms.platform_gym.adapter import PlatformGymAdapter


def export(output: Path, *, level: int, seed: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    adapter = PlatformGymAdapter(level=level, seed=seed, split="test")
    env = AdapterEnv(adapter, lambda _before, _after: 0.0)
    observation, _info = env.reset(seed=seed)
    route = ("RIGHT", "RIGHT", "RIGHT", "RIGHT+JUMP_TAP")
    try:
        for name in route:
            action = env.actions.index(name)
            if not bool(env.action_masks()[action]):
                raise RuntimeError(f"PlatformGym mask unexpectedly rejects {name}")
            observation, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                raise RuntimeError("PlatformGym episode ended before model input capture")
    finally:
        env.close()

    stack = np.asarray(observation["pixels"], dtype=np.uint8)
    channels = np.moveaxis(stack, 0, -1).reshape(108, 192, 4, 3)
    if stack.shape != (12, 108, 192) or channels.shape != (108, 192, 4, 3):
        raise ValueError(f"unexpected policy stack shape: {stack.shape}")
    sheet = Image.new("RGB", (384, 216))
    hashes = []
    for index in range(4):
        pixels = channels[:, :, index, :].copy()
        frame = Image.fromarray(pixels, mode="RGB")
        frame.save(output / f"gym_frame_{index}.png")
        sheet.paste(frame, ((index % 2) * 192, (index // 2) * 108))
        hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
    sheet.save(output / "gym_stack.png")
    np.savez_compressed(output / "gym_model_input.npz", pixels=stack,
                        action_history=np.asarray(observation["action_history"], dtype=np.int64))
    result = {
        "level": level, "seed": seed, "route": route,
        "pixel_shape_chw": list(stack.shape), "pixel_dtype": str(stack.dtype),
        "pixel_min": int(stack.min()), "pixel_max": int(stack.max()),
        "channel_order": "RGB per frame", "frame_sha256": hashes,
        "distinct_frames": len(set(hashes)),
        "action_history": np.asarray(observation["action_history"], dtype=np.int64).tolist(),
        "letterbox": "fit_to_universal() produced the 192x108 RGB frames used in this exact stack",
    }
    (output / "gym_input_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--seed", type=int, default=8601)
    args = parser.parse_args()
    print(json.dumps(export(args.output, level=args.level, seed=args.seed),
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
