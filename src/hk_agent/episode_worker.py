from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hk_agent.isolated_session import _close_process
from hk_agent.process_owner import EPISODE_REQUEST_ENV, _write_json_atomic

__all__ = ["asyncio"]


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    episode_id: str
    game_executable: Path
    game_arguments: tuple[str, ...]
    personal_root: Path
    experimental_root: Path
    stop_path: Path
    result_path: Path
    close_timeout_seconds: float
    maximum_episode_seconds: float | None
    endpoint: dict[str, str | int]
    redirected_save_mode: bool = False

    @classmethod
    def from_environment(cls) -> WorkerRequest:
        encoded = os.environ.pop(EPISODE_REQUEST_ENV, None)
        if not encoded:
            raise ValueError(f"{EPISODE_REQUEST_ENV} is missing")
        try:
            value = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")))
        except (ValueError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid episode request") from error
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("unsupported episode request")
        endpoint = value.get("endpoint")
        arguments = value.get("game_arguments")
        if not isinstance(endpoint, dict) or not isinstance(arguments, list):
            raise TypeError("invalid episode request fields")
        required_endpoint = {"host", "port", "run_id", "token", "game_profile"}
        if set(endpoint) != required_endpoint:
            raise ValueError("invalid supervisor endpoint")
        request = cls(
            episode_id=_string(value, "episode_id"),
            game_executable=Path(_string(value, "game_executable")),
            game_arguments=tuple(_string_item(item, "game_arguments") for item in arguments),
            personal_root=Path(_string(value, "personal_root")),
            experimental_root=Path(_string(value, "experimental_root")),
            redirected_save_mode=_boolean(value, "redirected_save_mode", default=False),
            stop_path=Path(_string(value, "stop_path")),
            result_path=Path(_string(value, "result_path")),
            close_timeout_seconds=_positive(value, "close_timeout_seconds", 60.0),
            maximum_episode_seconds=_optional_positive(
                value, "maximum_episode_seconds", 86400.0
            ),
            endpoint={
                "host": _string(endpoint, "host"),
                "port": _port(endpoint.get("port")),
                "run_id": _string(endpoint, "run_id"),
                "token": _string(endpoint, "token"),
                "game_profile": _string(endpoint, "game_profile"),
            },
        )
        if not request.game_executable.is_file():
            raise ValueError(f"game executable not found: {request.game_executable}")
        return request


def _string(value: dict[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item or "\0" in item:
        raise ValueError(f"{key} must be a non-empty string")
    return item


def _boolean(value: dict[str, Any], key: str, *, default: bool) -> bool:
    item = value.get(key, default)
    if not isinstance(item, bool):
        raise TypeError(f"{key} must be a boolean")
    return item


def _string_item(value: object, key: str) -> str:
    if not isinstance(value, str) or "\0" in value:
        raise ValueError(f"{key} must contain strings")
    return value


def _positive(value: dict[str, Any], key: str, maximum: float) -> float:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise TypeError(f"{key} must be numeric")
    converted = float(item)
    if not 0 < converted <= maximum:
        raise ValueError(f"{key} is outside its safe range")
    return converted


def _optional_positive(
    value: dict[str, Any], key: str, maximum: float
) -> float | None:
    if value.get(key) is None:
        return None
    return _positive(value, key, maximum)


def _port(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError("endpoint port is invalid")
    return value


def _assert_isolation(request: WorkerRequest) -> None:
    if not request.experimental_root.is_dir():
        raise RuntimeError("save-isolation roots do not exist")
    if request.redirected_save_mode:
        attributes = getattr(request.experimental_root.lstat(), "st_file_attributes", 0)
        if request.experimental_root.is_symlink() or (os.name == "nt" and attributes & 0x400):
            raise RuntimeError("redirected save root must be a real directory")
        return
    if not request.personal_root.is_dir():
        raise RuntimeError("save-isolation roots do not exist")
    if not os.path.samefile(request.personal_root, request.experimental_root):
        raise RuntimeError("personal saves are not redirected to experimental saves")
    if os.name == "nt":
        attributes = getattr(request.personal_root.lstat(), "st_file_attributes", 0)
        if not attributes & 0x400:
            raise RuntimeError("personal save root is not a Windows reparse point")


def _game_environment(request: WorkerRequest) -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("HK_OWNER_") or name == EPISODE_REQUEST_ENV:
            environment.pop(name, None)
    endpoint = request.endpoint
    environment.update(
        {
            "HK_AGENT_HOST": str(endpoint["host"]),
            "HK_AGENT_PORT": str(endpoint["port"]),
            "HK_AGENT_RUN_ID": str(endpoint["run_id"]),
            "HK_AGENT_INSTANCE_ID": f"episode-{request.episode_id}",
            "HK_AGENT_TOKEN": str(endpoint["token"]),
            "HK_AGENT_GAME_PROFILE": str(endpoint["game_profile"]),
            "HK_AGENT_SAVE_ROOT": str(request.experimental_root.resolve(strict=False)),
        }
    )
    return environment


async def run_worker(request: WorkerRequest) -> int:
    process: asyncio.subprocess.Process | None = None
    reason = "startup_failed"
    forced = False
    exit_code: int | None = None
    try:
        _assert_isolation(request)
        process = await asyncio.create_subprocess_exec(
            str(request.game_executable),
            *request.game_arguments,
            cwd=request.game_executable.parent,
            env=_game_environment(request),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = (
            None
            if request.maximum_episode_seconds is None
            else time.monotonic() + request.maximum_episode_seconds
        )
        while process.returncode is None:
            if request.stop_path.exists():
                reason = "stop_requested"
                break
            if deadline is not None and time.monotonic() >= deadline:
                reason = "episode_time_limit"
                break
            await asyncio.sleep(0.1)
        if process.returncode is not None:
            reason = "game_exited"
            exit_code = process.returncode
        else:
            exit_code, forced, _ = await _close_process(
                process, request.close_timeout_seconds
            )
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        reason = f"worker_failure:{type(error).__name__}:{error}"
        if process is not None and process.returncode is None:
            try:
                exit_code, forced, _ = await _close_process(
                    process, request.close_timeout_seconds
                )
            except (OSError, RuntimeError, TimeoutError):
                forced = True
    ok = reason in {"stop_requested", "episode_time_limit"} and not forced
    _write_json_atomic(
        request.result_path,
        {
            "schema_version": 1,
            "episode_id": request.episode_id,
            "ok": ok,
            "reason": reason,
            "game_exit_code": exit_code,
            "forced_termination": forced,
        },
    )
    return 0 if ok else 1


def main() -> int:
    try:
        request = WorkerRequest.from_environment()
    except (TypeError, ValueError):
        return 2
    return asyncio.run(run_worker(request))


if __name__ == "__main__":
    raise SystemExit(main())
