"""Capture a short deterministic walk-and-jump route through an isolated HK instance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time
from typing import Any, cast


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


def _buttons(index: int) -> list[str]:
    # Keep forward motion while adding repeated short jumps; every few hundred
    # frames, reverse briefly to revisit the hazard and enemy corridor.
    if index % 360 in range(150, 166):
        return ["LEFT", "JUMP"] if index % 4 == 0 else ["LEFT"]
    phase = index % 24
    return ["RIGHT", "JUMP"] if 14 <= phase < 19 else ["RIGHT"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=1500)
    parser.add_argument("--lease-ms", type=int, default=100)
    parser.add_argument("--progress", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1500:
        parser.error("--frames must be at least 1500")
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
    started = time.time()
    def request(command: str, **values: object) -> dict[str, Any]:
        return _request("127.0.0.1", port, {"token": token, "command": command, **values})

    request("release")
    first = request("capture")["capture"]
    captures.append(first)
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    for index in range(args.frames):
        action = request("action", buttons=_buttons(index), lease_ms=args.lease_ms)
        captures.append(action["capture"])
        if (index + 1) % 50 == 0:
            temporary = args.progress.with_suffix(".tmp")
            temporary.write_text(json.dumps({
                "completed_actions": index + 1,
                "capture_count": len(captures),
                "latest_scene": captures[-1].get("scene_name"),
                "latest_visible_enemies": captures[-1].get("visible_enemies", []),
            }, indent=2) + "\n", encoding="utf-8")
            temporary.replace(args.progress)
    request("release")
    request("close")
    result = {
        "actions": args.frames, "capture_count": len(captures),
        "started_unix": started, "finished_unix": time.time(),
        "duration_seconds": time.time() - started, "captures": captures,
    }
    temporary = args.progress.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.progress)
    print(json.dumps({key: value for key, value in result.items() if key != "captures"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
