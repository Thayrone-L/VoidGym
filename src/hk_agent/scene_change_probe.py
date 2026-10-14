from __future__ import annotations

import hashlib
import json
import os
import random
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class ProbeTask(Protocol):
    action_space: gym.Space[int]

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[dict[str, NDArray[np.generic]], dict[str, Any]]: ...

    def step(
        self, action: int
    ) -> tuple[dict[str, NDArray[np.generic]], float, bool, bool, dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class SceneChangeProbeConfig:
    seed: int = 23
    episodes: int = 3
    horizon_steps: int = 512
    action_repeat: int = 3

    def validate(self) -> None:
        if not 1 <= self.episodes <= 32:
            raise ValueError("episodes must be in [1, 32]")
        if not 1 <= self.horizon_steps <= 10_000:
            raise ValueError("horizon_steps must be in [1, 10000]")
        if not 1 <= self.action_repeat <= 16:
            raise ValueError("action_repeat must be in [1, 16]")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def run_scene_change_probe(
    task: ProbeTask,
    *,
    config: SceneChangeProbeConfig,
    artifact_root: Path,
    stop_file: Path | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Run bounded seeded random actions without constructing a learner."""

    config.validate()
    random_source = random.Random(config.seed)
    discrete_actions = cast(gym.spaces.Discrete, task.action_space)
    summaries: list[dict[str, Any]] = []
    total_positive_events = 0
    for episode_index in range(config.episodes):
        if stop_file is not None and stop_file.is_file():
            break
        observation, _ = task.reset(seed=config.seed + episode_index)
        del observation
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        reason: str | None = None
        positive_events = 0
        frames: list[dict[str, Any]] = []
        selected_action = 0
        repeat_remaining = 0
        while steps < config.horizon_steps and not (terminated or truncated):
            if stop_file is not None and stop_file.is_file():
                reason = "campaign_stop_requested"
                break
            if repeat_remaining == 0:
                selected_action = random_source.randrange(int(discrete_actions.n))
                repeat_remaining = config.action_repeat
            repeat_remaining -= 1
            observation, reward, terminated, truncated, info = task.step(selected_action)
            steps += 1
            total_reward += float(reward)
            reason_value = info.get("termination_reason")
            reason = str(reason_value) if reason_value is not None else None
            detector = cast(dict[str, Any], info.get("detector", {}))
            event = info.get("visual_regime_change_detected") is True
            dark_run = int(detector.get("dark_run", 0))
            stable_run = int(detector.get("stable_run", 0))
            if event:
                positive_events += 1
                total_positive_events += 1
            if event or dark_run > 0 or stable_run > 0:
                frame_path = (
                    artifact_root
                    / "frames"
                    / f"episode-{episode_index + 1:03d}"
                    / f"step-{steps:04d}.png"
                )
                write_rgb_png(frame_path, observation["pixels"][-1])
                frames.append(
                    {
                        "step": steps,
                        "path": str(frame_path),
                        "sha256": sha256_file(frame_path),
                        "event": event,
                        "dark_run": dark_run,
                        "stable_run": stable_run,
                    }
                )
        summaries.append(
            {
                "episode": episode_index + 1,
                "seed": config.seed + episode_index,
                "reset_started": True,
                "steps": steps,
                "total_reward": total_reward,
                "terminated": terminated,
                "truncated": truncated,
                "termination_reason": reason,
                "ended_by_death": reason == "player_death",
                "positive_events": positive_events,
                "captured_frames": frames,
            }
        )
        if stop_file is not None and stop_file.is_file():
            break
    return summaries, total_positive_events


def completed_probe_result(
    summaries: list[dict[str, Any]], positive_events: int
) -> dict[str, Any]:
    if positive_events < 0:
        raise ValueError("positive_events cannot be negative")
    deaths = sum(item.get("ended_by_death") is True for item in summaries)
    return {
        "status": "COMPLETED",
        "gate_passed": positive_events > 0,
        "positive_events": positive_events,
        "death_episodes": deaths,
        "episode_resets_started": len(summaries),
        "operational_failures": 0,
        "episodes": summaries,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rgb_png(path: Path, frame: NDArray[np.generic]) -> None:
    pixels = np.asarray(frame)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("probe frame must be an HxWx3 uint8 array")
    height, width, _ = pixels.shape
    raw = b"".join(b"\x00" + np.ascontiguousarray(row).tobytes() for row in pixels)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(png)
    os.replace(temporary, path)
