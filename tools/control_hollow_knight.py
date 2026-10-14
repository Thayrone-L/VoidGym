from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import Any

from hk_agent.gameplay_control import GAMEPLAY_BUTTONS, MAXIMUM_CONTROL_MESSAGE_BYTES


def _metadata(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "ready":
        raise ValueError("control metadata does not describe a ready session")
    if value.get("host") != "127.0.0.1":
        raise ValueError("control metadata host must be 127.0.0.1")
    if not isinstance(value.get("port"), int) or not isinstance(value.get("token"), str):
        raise TypeError("control metadata is incomplete")
    return value


def run(args: argparse.Namespace) -> int:
    metadata = _metadata(args.session.resolve(strict=True))
    request: dict[str, object] = {"token": metadata["token"], "command": args.command}
    if args.command == "action":
        request.update({"buttons": args.button, "lease_ms": args.lease_ms})
    encoded = json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n"
    with socket.create_connection((metadata["host"], metadata["port"]), timeout=args.timeout) as client:
        client.sendall(encoded)
        client.settimeout(args.timeout)
        response = b""
        while not response.endswith(b"\n"):
            chunk = client.recv(4096)
            if not chunk:
                raise ConnectionError("control server closed without a response")
            response += chunk
            if len(response) > MAXIMUM_CONTROL_MESSAGE_BYTES:
                raise ValueError("control response is too large")
    decoded = json.loads(response)
    print(json.dumps(decoded, indent=2, ensure_ascii=False))
    return 0 if decoded.get("ok") is True else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control one live isolated Hollow Knight session")
    parser.add_argument("--session", required=True, type=Path, help="ready metadata JSON")
    parser.add_argument("--timeout", type=float, default=15.0)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("capture")
    action = subparsers.add_parser("action")
    action.add_argument("--button", action="append", required=True, choices=sorted(GAMEPLAY_BUTTONS))
    action.add_argument("--lease-ms", required=True, type=int)
    subparsers.add_parser("release")
    subparsers.add_parser("close")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except (ConnectionError, json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error
