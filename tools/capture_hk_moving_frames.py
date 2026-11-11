"""Capture a short deterministic walk-and-jump route through an isolated HK instance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time
import sys
from collections.abc import Callable
from typing import Any, cast

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hk_agent.semantic_labels import unpack_u4
from voidgym.core.semantics import Sem


def _request(host: str, port: int, request: dict[str, object]) -> dict[str, Any]:
    with socket.create_connection((host, port), timeout=20) as connection:
        connection.settimeout(30)
        connection.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
        response = bytearray()
        while not response.endswith(b"\n"):
            chunk = connection.recv(8192)
            if not chunk:
                raise ConnectionError("game control server closed without a response")
            response.extend(chunk)
            if len(response) > 8192:
                raise ValueError("game control response exceeded 8 KiB")
    result = cast(dict[str, Any], json.loads(response))
    if result.get("ok") is not True:
        raise RuntimeError(f"game control command failed: {result}")
    return result


_TRANSIENT_ERRORS = (ConnectionError, OSError, RuntimeError, TimeoutError, TypeError, ValueError)


def _capture_with_retries(
    request: Callable[..., dict[str, Any]], *, retries: int = 3,
    retry_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reconnect and retry a capture without replaying the gameplay action."""
    for attempt in range(retries + 1):
        try:
            result = request("capture")
            capture = result.get("capture")
            if not isinstance(capture, dict):
                raise TypeError("capture response is missing its capture record")
            return capture
        except _TRANSIENT_ERRORS as error:
            if retry_events is not None:
                retry_events.append({
                    "command": "capture", "attempt": attempt + 1,
                    "error": f"{type(error).__name__}: {error}",
                })
            if attempt >= retries:
                raise
            time.sleep(min(0.25 * (attempt + 1), 0.75))
    raise AssertionError("unreachable capture retry state")


def _buttons(index: int, route: str = "walk-jump") -> list[str]:
    if route in {"semantic-guided-forward", "combat-guided-forward"}:
        return ["RIGHT"]
    if route == "guided-forward":
        # Sustained, periodic jumps are suitable for crossing the long King's
        # Pass gap; the original short pulse pattern can repeatedly stall there.
        return ["RIGHT", "JUMP"] if index % 6 in {0, 1} else ["RIGHT"]
    if route == "high-jump-forward":
        # Longer held jumps preserve forward motion while clearing the wider
        # gaps in King's Pass from a fresh isolated-save checkpoint.
        return ["RIGHT", "JUMP"] if index % 12 < 5 else ["RIGHT"]
    # Keep forward motion while adding repeated short jumps; every few hundred
    # frames, reverse briefly to revisit the hazard and enemy corridor.
    if index % 360 in range(150, 166):
        return ["LEFT", "JUMP"] if index % 4 == 0 else ["LEFT"]
    phase = index % 24
    return ["RIGHT", "JUMP"] if 14 <= phase < 19 else ["RIGHT"]


def _semantic_guided_buttons(labels: np.ndarray, jump_countdown: int, *,
                             attack_nearby: bool = False) -> tuple[list[str], int]:
    if jump_countdown > 0:
        return ["RIGHT", "JUMP"], jump_countdown - 1
    player = np.argwhere(labels == int(Sem.PLAYER))
    if not len(player):
        return ["RIGHT"], 0
    solid = labels == int(Sem.SOLID)
    player_right = int(player[:, 1].max())
    player_center = int(player[:, 1].mean())
    player_top = int(player[:, 0].min())
    player_bottom = int(player[:, 0].max())
    enemies = np.argwhere(labels == int(Sem.ENEMY))
    if attack_nearby and len(enemies):
        ahead = enemies[enemies[:, 1] > player_right]
        if len(ahead):
            nearest = ahead[np.argmin(ahead[:, 1])]
            vertical_distance = max(player_top - int(nearest[0]),
                                    int(nearest[0]) - player_bottom, 0)
            if int(nearest[1]) - player_right <= 8 and vertical_distance <= 5:
                return ["RIGHT", "ATTACK"], 0
    floor_row = int(player[:, 0].max())
    floor_end = min(solid.shape[0], floor_row + 3)
    support_row = min(solid.shape[0] - 1, floor_row + 1)
    current_support = bool(solid[support_row, player_center])
    for distance in range(1, 7):
        column = player_right + distance
        if column >= solid.shape[1] - 8:
            break
        has_floor = bool(solid[floor_row:floor_end, column].any())
        future_floor = bool(solid[floor_row:floor_end,
                                 column + 2:min(solid.shape[1], column + 10)].any())
        if current_support and not has_floor and future_floor:
            return ["RIGHT", "JUMP"], 7
        obstacle_top = max(0, floor_row - 6)
        if bool(solid[obstacle_top:floor_row, column].any()):
            return ["RIGHT", "JUMP"], 6
        if bool((labels[obstacle_top:floor_end, column] == int(Sem.HAZARD)).any()):
            return ["RIGHT", "JUMP"], 6
    return ["RIGHT"], 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=1500)
    parser.add_argument("--start-index", type=int, default=0,
                        help="route index to continue across capture chunks")
    parser.add_argument("--lease-ms", type=int, default=100)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--route", choices=("walk-jump", "guided-forward", "high-jump-forward",
                      "semantic-guided-forward", "combat-guided-forward"), default="walk-jump")
    parser.add_argument("--keep-open", action="store_true",
                        help="leave the isolated game session open after this chunk")
    args = parser.parse_args()
    if args.frames < 1 or args.start_index < 0:
        parser.error("--frames must be positive and --start-index non-negative")
    if not 1 <= args.lease_ms <= 500:
        parser.error("--lease-ms must be between 1 and 500")
    session = json.loads(args.session.read_text(encoding="utf-8"))
    if session.get("status") != "ready" or session.get("host") != "127.0.0.1":
        raise ValueError("control metadata is not a ready localhost session")
    token = session.get("token")
    port = session.get("port")
    if not isinstance(token, str) or not isinstance(port, int):
        raise TypeError("control metadata is incomplete")
    captures: list[dict[str, Any]] = []
    retry_events: list[dict[str, Any]] = []
    started = time.time()
    def request(command: str, **values: object) -> dict[str, Any]:
        return _request("127.0.0.1", port, {"token": token, "command": command, **values})

    request("release")
    first = _capture_with_retries(request, retry_events=retry_events)
    captures.append(first)
    jump_countdown = 0
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    for offset in range(args.frames):
        index = args.start_index + offset
        buttons = _buttons(index, args.route)
        if args.route in {"semantic-guided-forward", "combat-guided-forward"}:
            semantic_path = first.get("semantic_labels_path")
            if isinstance(semantic_path, str):
                labels = unpack_u4(Path(semantic_path).read_bytes(), width=48, height=27)
                buttons, jump_countdown = _semantic_guided_buttons(
                    labels, jump_countdown, attack_nearby=args.route == "combat-guided-forward")
        lease_ms = args.lease_ms
        if args.route in {"semantic-guided-forward", "combat-guided-forward"} and "JUMP" in buttons:
            # Keep the virtual jump input held through the frame capture; a
            # 100 ms lease expires while the screenshot and labels are prepared.
            lease_ms = min(500, max(250, args.lease_ms))
        try:
            action = request("action", buttons=buttons, lease_ms=lease_ms)
            captured = action["capture"]
        except _TRANSIENT_ERRORS as error:
            # The bridge may have applied the action before the follow-up
            # capture failed. Recover the current frame instead of sending the
            # action again and accidentally moving twice.
            retry_events.append({
                "command": "action", "attempt": 1,
                "error": f"{type(error).__name__}: {error}",
                "recovered_by_capture": True,
            })
            captured = _capture_with_retries(request, retry_events=retry_events)
        captures.append(captured)
        first = captured
        if (index + 1) % 50 == 0:
            temporary = args.progress.with_suffix(".tmp")
            temporary.write_text(json.dumps({
                "completed_actions": index + 1,
                "chunk_actions": offset + 1,
                "capture_count": len(captures),
                "latest_scene": captures[-1].get("scene_name"),
                "latest_visible_enemies": captures[-1].get("visible_enemies", []),
                "retry_events": retry_events,
            }, indent=2) + "\n", encoding="utf-8")
            temporary.replace(args.progress)
    request("release")
    if not args.keep_open:
        request("close")
    result = {
        "actions": args.frames, "start_index": args.start_index,
        "capture_count": len(captures), "session_left_open": args.keep_open,
        "started_unix": started, "finished_unix": time.time(),
        "duration_seconds": time.time() - started, "retry_events": retry_events,
        "captures": captures,
    }
    temporary = args.progress.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.progress)
    print(json.dumps({key: value for key, value in result.items() if key != "captures"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
