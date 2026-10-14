from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from hk_agent.supervisor import SupervisorServer


async def run() -> int:
    repository = Path(__file__).resolve().parents[1]
    harness = (
        repository
        / "bridge"
        / "HkAgentBridge.Harness"
        / "bin"
        / "Release"
        / "net8.0"
        / "HkAgentBridge.Harness.dll"
    )
    if not harness.is_file():
        raise FileNotFoundError(f"Build the bridge harness first: {harness}")

    supervisor = SupervisorServer(token="cross-language-test-token")
    await supervisor.start()
    assert supervisor.port is not None
    try:
        process = await asyncio.create_subprocess_exec(
            "dotnet",
            str(harness),
            "127.0.0.1",
            str(supervisor.port),
            supervisor.run_id,
            "hk-00",
            supervisor.token,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        accepted = await supervisor.wait_for_bridge(timeout=5.0)
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5.0)
        if process.returncode != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="replace"))

        print(
            json.dumps(
                {
                    "ok": True,
                    "instance_id": accepted.instance_id,
                    "game_profile": accepted.game_profile,
                    "game_version": accepted.game_version,
                    "harness": stdout.decode("utf-8").strip(),
                },
                indent=2,
            )
        )
        return 0
    finally:
        await supervisor.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
