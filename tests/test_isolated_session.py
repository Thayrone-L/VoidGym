from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from hk_agent.isolated_session import SessionRequest, _capture_menu_frame, _game_environment
from hk_agent.protocol import ProtocolError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "tools" / "run_isolated_session.py"


def test_instance_game_environment_uses_bridge_instance_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "game.exe"
    executable.touch()
    data_root = tmp_path / "instance-data"
    data_root.mkdir()
    request = SessionRequest.from_mapping(
        {
            "game_executable": str(executable),
            "game_profile": "development",
            "instance_id": "instance-1",
            "result_path": str(tmp_path / "result.json"),
            "personal_root": str(tmp_path / "personal"),
            "experimental_root": str(tmp_path / "experimental"),
            "instance_data_root": str(data_root),
            "expected_product_name": "Hollow Knight AI 1",
        }
    )
    monkeypatch.setenv("HK_AGENT_SAVE_ROOT", "stale-save")
    monkeypatch.setenv("HK_AGENT_INSTANCE_MODE", "stale-mode")
    monkeypatch.setenv("HK_AGENT_PRODUCT_NAME", "stale-product")
    monkeypatch.setenv("HK_AGENT_INSTANCE_DATA_ROOT", "stale-data")
    environment = _game_environment(
        SimpleNamespace(port=43123, run_id="run-1", token="token-1"), request
    )
    assert environment["HK_AGENT_INSTANCE_MODE"] == "1"
    assert environment["HK_AGENT_PRODUCT_NAME"] == "Hollow Knight AI 1"
    assert environment["HK_AGENT_INSTANCE_DATA_ROOT"] == str(data_root)
    assert "HK_AGENT_SAVE_ROOT" not in environment


def test_isolated_session_menu_writer_refuses_raw_frame(tmp_path: Path) -> None:
    class RawSupervisor:
        async def capture_once(self, **_: object) -> object:
            return SimpleNamespace(encoding="raw_rgb8")

    request = SimpleNamespace(menu_output_dir=tmp_path, timeout_seconds=1.0)
    result = SimpleNamespace(menu_captures=[])
    with pytest.raises(ProtocolError, match="requires FRAME encoding=png"):
        asyncio.run(
            _capture_menu_frame(request, result, RawSupervisor(), object(), 0, "before")  # type: ignore[arg-type]
        )
    assert not list(tmp_path.glob("*.png"))


def _inventory(root: Path) -> list[tuple[str, str]]:
    return [
        (str(path.relative_to(root)), sha256(path.read_bytes()).hexdigest())
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


@pytest.mark.skipif(os.name != "nt", reason="Windows junction behavior")
@pytest.mark.parametrize("keep_connection_open", [False, True])
def test_smoke_runs_inside_save_transaction_without_leaking_credentials(
    tmp_path: Path,
    keep_connection_open: bool,
) -> None:
    personal = tmp_path / "personal" / "Hollow Knight"
    experimental = tmp_path / "experimental"
    backups = tmp_path / "backups"
    capture = tmp_path / "capture.png"
    environment_audit = tmp_path / "game-environment.json"
    fake_game = tmp_path / "fake_game.py"
    fake_launcher_source = tmp_path / "fake_launcher.cs"
    fake_launcher = tmp_path / "hk_isolated_fake_game.exe"
    personal.mkdir(parents=True)
    experimental.mkdir()
    (personal / "user1.dat").write_bytes(b"personal")
    (experimental / "user1.dat").write_bytes(b"experimental")
    before = _inventory(personal)

    fake_game.write_text(
        textwrap.dedent(
            """
            import asyncio
            import base64
            import json
            import os
            import sys
            import time
            import uuid

            from hk_agent.protocol import Envelope, PROTOCOL_VERSION, read_message, write_message

            KEYS = (
                "HK_AGENT_HOST", "HK_AGENT_PORT", "HK_AGENT_RUN_ID",
                "HK_AGENT_INSTANCE_ID", "HK_AGENT_TOKEN", "HK_AGENT_GAME_PROFILE",
            )
            PNG = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
                "AScY42YAAAAASUVORK5CYII="
            )

            async def send(writer, kind, seq, trace, payload):
                await write_message(writer, Envelope(
                    protocol_version=PROTOCOL_VERSION,
                    message_type=kind,
                    run_id=os.environ["HK_AGENT_RUN_ID"],
                    instance_id=os.environ["HK_AGENT_INSTANCE_ID"],
                    session_id="fake-session",
                    seq=seq,
                    trace_id=trace,
                    sent_monotonic_ns=time.monotonic_ns(),
                    payload=payload,
                ))

            async def main():
                audit = {"present": sorted(key for key in KEYS if key in os.environ)}
                audit["request_present"] = "HK_ISOLATED_SESSION_REQUEST" in os.environ
                with open(sys.argv[1], "w", encoding="utf-8") as stream:
                    json.dump(audit, stream)
                reader, writer = await asyncio.open_connection(
                    os.environ["HK_AGENT_HOST"], int(os.environ["HK_AGENT_PORT"])
                )
                trace = str(uuid.uuid4())
                await send(writer, "HELLO", 0, trace, {
                    "token": os.environ["HK_AGENT_TOKEN"],
                    "game_profile": os.environ["HK_AGENT_GAME_PROFILE"],
                    "game_version": "fake-1.0",
                })
                await read_message(reader)
                await send(writer, "HEARTBEAT", 1, str(uuid.uuid4()), {})
                sequence = 2
                while True:
                    command = await read_message(reader)
                    if command.message_type == "PING":
                        await send(writer, "ACK", sequence, command.trace_id, {})
                    elif command.message_type == "CAPTURE_ONCE":
                        await send(writer, "FRAME", sequence, command.trace_id, {
                            "frame_id": 7,
                            "capture_monotonic_ns": time.monotonic_ns(),
                            "width": 1,
                            "height": 1,
                            "channels": 3,
                            "pixel_format": "RGB8",
                            "encoding": "png",
                            "data_base64": base64.b64encode(PNG).decode("ascii"),
                            "scene_transition": False,
                            "duplicate_of": None,
                            "last_applied_action_id": None,
                        })
                    elif command.message_type == "RELEASE_ALL":
                        await send(writer, "ACK", sequence, command.trace_id, {})
                        if len(sys.argv) > 2 and sys.argv[2] == "hold-open":
                            await reader.read()
                        writer.close()
                        await writer.wait_closed()
                        return
                    sequence += 1

            asyncio.run(main())
            """
        ),
        encoding="utf-8",
    )
    fake_launcher_source.write_text(
        textwrap.dedent(
            r'''
            using System;
            using System.Diagnostics;
            using System.Linq;

            internal static class FakeLauncher
            {
                private static string Quote(string value)
                {
                    return "\"" + value.Replace("\"", "\\\"") + "\"";
                }

                public static int Main(string[] args)
                {
                    if (args.Length < 3) return 64;
                    var start = new ProcessStartInfo
                    {
                        FileName = args[0],
                        Arguments = string.Join(" ", args.Skip(1).Select(Quote)),
                        UseShellExecute = false,
                    };
                    using (var process = Process.Start(start))
                    {
                        process.WaitForExit();
                        return process.ExitCode;
                    }
                }
            }
            '''
        ),
        encoding="utf-8",
    )
    compiler = (
        Path(os.environ["SystemRoot"])
        / "Microsoft.NET"
        / "Framework64"
        / "v4.0.30319"
        / "csc.exe"
    )
    if not compiler.is_file():
        pytest.skip(".NET Framework C# compiler is unavailable")
    compilation = subprocess.run(
        [str(compiler), "/nologo", f"/out:{fake_launcher}", str(fake_launcher_source)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compilation.returncode == 0, compilation.stdout + compilation.stderr

    inherited_environment = os.environ.copy()
    for key in tuple(inherited_environment):
        if key.startswith("HK_AGENT_") or key == "HK_ISOLATED_SESSION_REQUEST":
            inherited_environment.pop(key)
    command = [
        sys.executable,
        str(RUNNER),
        "--game-executable",
        str(fake_launcher),
        "--personal-root",
        str(personal),
        "--experimental-root",
        str(experimental),
        "--backup-root",
        str(backups),
        "--game-profile",
        "test-profile",
        "--game-argument",
        sys.executable,
        "--game-argument",
        str(fake_game),
        "--game-argument",
        str(environment_audit),
    ]
    if keep_connection_open:
        command.extend(("--game-argument", "hold-open"))
    command.extend(
        (
            "--capture-output",
            str(capture),
            "--timeout",
            "10",
            "--close-timeout",
            "0.5",
        )
    )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=inherited_environment,
        timeout=15,
    )

    assert completed.returncode == (1 if keep_connection_open else 0), (
        completed.stderr or completed.stdout
    )
    result = json.loads(completed.stdout)
    assert result["ok"] is (not keep_connection_open)
    assert result["handshake_ok"] is True
    assert result["capture_ok"] is True
    assert result["release_all_ok"] is True
    assert result["forced_termination"] is keep_connection_open
    assert result["transaction_exit_code"] == 0
    assert capture.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert _inventory(personal) == before
    assert not personal.is_symlink()

    audit = json.loads(environment_audit.read_text(encoding="utf-8"))
    assert audit["present"] == sorted(
        [
            "HK_AGENT_HOST",
            "HK_AGENT_INSTANCE_ID",
            "HK_AGENT_GAME_PROFILE",
            "HK_AGENT_PORT",
            "HK_AGENT_RUN_ID",
            "HK_AGENT_TOKEN",
        ]
    )
    assert audit["request_present"] is False
    manifests = list(backups.glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8-sig"))
    assert manifest["status"] == "COMPLETED"
