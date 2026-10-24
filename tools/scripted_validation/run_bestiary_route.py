from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hk_agent.bestiary import BestiaryMemory
from hk_agent.gameplay_control import MAXIMUM_CONTROL_MESSAGE_BYTES


def request(metadata: dict[str, Any], command: str, *, buttons: list[str] | None = None, lease_ms: int | None = None) -> dict[str, Any]:
    payload: dict[str, object] = {"token": metadata["token"], "command": command}
    if command == "action":
        payload.update({"buttons": buttons, "lease_ms": lease_ms})
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
    with socket.create_connection((metadata["host"], metadata["port"]), timeout=20) as client:
        client.sendall(encoded)
        client.settimeout(20)
        response = bytearray()
        while not response.endswith(b"\n"):
            chunk = client.recv(4096)
            if not chunk:
                raise ConnectionError("control server closed without a response")
            response.extend(chunk)
            if len(response) > MAXIMUM_CONTROL_MESSAGE_BYTES:
                raise ValueError("control response is too large")
    result = json.loads(response)
    if result.get("ok") is not True:
        raise RuntimeError(str(result.get("error", result)))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Fixed Hollow Knight route for stage 60")
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    metadata = json.loads(args.session.read_text(encoding="utf-8"))
    if metadata.get("status") != "ready" or metadata.get("host") != "127.0.0.1":
        raise ValueError("interactive isolated session is not ready")
    args.output.mkdir(parents=True, exist_ok=True)
    memory = BestiaryMemory()
    captures: list[dict[str, object]] = []

    def observe(result: dict[str, Any]) -> bool:
        capture = result.get("capture")
        if not isinstance(capture, dict):
            raise TypeError("control response has no capture")
        visible = capture.get("visible_enemies", [])
        telemetry = {"visible_enemies": visible}
        if isinstance(capture.get("scene_name"), str):
            telemetry["scene_name"] = capture["scene_name"]
        events = capture.get("bestiary_events", [])
        memory.update(
            telemetry,
            hp_lost=capture.get("hp_lost_since_last_frame", 0),
            events=events,
        )
        record = {
            "index": len(captures) + 1,
            "action": result.get("buttons", ["CAPTURE"]),
            "frame": capture.get("path"),
            "scene_name": capture.get("scene_name"),
            "visible_enemies": visible,
            "bestiary_events": events,
        }
        captures.append(record)
        payload = {
            "schema_version": 1,
            "bestiary": memory.to_payload(),
            "latest_visible_enemies": visible,
            "captures": captures,
        }
        temporary = args.output / "player-knowledge.json.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        temporary.replace(args.output / "player-knowledge.json")
        return bool(visible)

    try:
        enemy_seen = observe(request(metadata, "capture"))
        route = [( ["RIGHT"], 650) for _ in range(8)] + [(["RIGHT", "JUMP"], 550) for _ in range(8)]
        for buttons, lease in route:
            if enemy_seen:
                break
            enemy_seen = observe(request(metadata, "action", buttons=buttons, lease_ms=lease))
        fight_steps = 0
        for _ in range(32):
            if not enemy_seen and fight_steps >= 8:
                break
            enemy_seen = observe(request(metadata, "action", buttons=["ATTACK"], lease_ms=350))
            fight_steps += 1
        for _ in range(10):
            observe(request(metadata, "capture"))
        result = {
            "route_id": "kings-pass-first-crawlid-v1",
            "started_unix": time.time(),
            "enemy_slot_seen": any(bool(row["visible_enemies"]) for row in captures),
            "fight_actions": fight_steps,
            "enemy_type_count": len(memory.experience),
            "bestiary": memory.to_payload(),
            "capture_count": len(captures),
        }
        (args.output / "route-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["enemy_slot_seen"] else 2
    finally:
        try:
            request(metadata, "close")
        except (ConnectionError, OSError, RuntimeError):
            pass


if __name__ == "__main__":
    raise SystemExit(main())