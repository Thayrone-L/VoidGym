from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hk_agent.config import ConfigError, load_config
from hk_agent.isolated_session import (
    MAXIMUM_MENU_LEASE_MILLISECONDS,
    MENU_BUTTONS,
    REQUEST_ENVIRONMENT_VARIABLE,
)

POWERSHELL = (
    Path(os.environ.get("SystemRoot", r"C:\Windows"))
    / "System32"
    / "WindowsPowerShell"
    / "v1.0"
    / "powershell.exe"
)
TRANSACTIONAL_LAUNCHER = PROJECT_ROOT / "tools" / "start-isolated-hollow-knight.ps1"


def _resolve_settings(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, str]:
    explicit = (
        args.game_executable,
        args.personal_root,
        args.experimental_root,
        args.backup_root,
        args.game_profile,
    )
    if any(value is not None for value in explicit):
        if not all(value is not None for value in explicit):
            raise ConfigError(
                "Explicit mode requires --game-executable, --personal-root, "
                "--experimental-root, --backup-root, and --game-profile"
            )
        return (
            args.game_executable.resolve(strict=True),
            args.personal_root.resolve(strict=False),
            args.experimental_root.resolve(strict=False),
            args.backup_root.resolve(strict=False),
            args.game_profile,
        )

    config_path = (args.config or (PROJECT_ROOT / "config.local.toml")).resolve(strict=False)
    config = load_config(config_path)
    return (
        config.game_executable,
        config.personal_save_root,
        config.experimental_save_root,
        config.backup_root,
        config.game_profile,
    )


def _encoded_request(document: dict[str, object]) -> str:
    content = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(content).decode("ascii")


def _parse_menu_step(value: str) -> dict[str, object]:
    button, separator, raw_lease = value.partition(":")
    if separator != ":" or button not in MENU_BUTTONS:
        allowed = ", ".join(sorted(MENU_BUTTONS))
        raise argparse.ArgumentTypeError(
            f"menu step must be BUTTON:LEASE_MS; BUTTON is one of {allowed}"
        )
    try:
        lease_ms = int(raw_lease)
    except ValueError as error:
        raise argparse.ArgumentTypeError("menu lease must be an integer") from error
    if lease_ms < 1 or lease_ms > MAXIMUM_MENU_LEASE_MILLISECONDS:
        raise argparse.ArgumentTypeError(
            f"menu lease must be between 1 and {MAXIMUM_MENU_LEASE_MILLISECONDS} ms"
        )
    return {"button": button, "lease_ms": lease_ms}


def run(args: argparse.Namespace) -> int:
    game, personal, experimental, backups, profile = _resolve_settings(args)
    if os.name != "nt" or not POWERSHELL.is_file():
        raise RuntimeError("The transactional save launcher requires Windows PowerShell")

    backups.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".hk-isolated-session-", dir=backups
    ) as temporary_directory:
        result_path = Path(temporary_directory) / "session-result.json"
        request: dict[str, object] = {
            "mode": args.mode,
            "game_executable": str(game),
            "game_arguments": args.game_argument,
            "game_profile": profile,
            "instance_id": args.instance_id,
            "result_path": str(result_path),
            "capture_output": (
                str(args.capture_output.resolve(strict=False))
                if args.capture_output is not None
                else None
            ),
            "personal_root": str(personal),
            "experimental_root": str(experimental),
            "timeout_seconds": args.timeout,
            "close_timeout_seconds": args.close_timeout,
        }
        if args.instance_data_root is not None:
            if args.expected_product_name is None:
                raise ConfigError("--expected-product-name is required with --instance-data-root")
            request.update({
                "instance_data_root": str(args.instance_data_root.resolve(strict=False)),
                "expected_product_name": args.expected_product_name,
            })
        elif args.expected_product_name is not None:
            raise ConfigError("--expected-product-name requires --instance-data-root")
        if args.mode == "menu":
            if args.capture_output is not None:
                raise ConfigError("--capture-output is only valid in smoke mode")
            if args.menu_output_dir is None or args.menu_report_output is None:
                raise ConfigError(
                    "menu mode requires --menu-output-dir and --menu-report-output"
                )
            request.update(
                {
                    "menu_steps": args.menu_step,
                    "menu_initial_wait_seconds": args.menu_initial_wait,
                    "menu_step_wait_seconds": args.menu_step_wait,
                    "menu_final_wait_seconds": args.menu_final_wait,
                    "menu_output_dir": str(args.menu_output_dir.resolve(strict=False)),
                    "menu_report_output": str(
                        args.menu_report_output.resolve(strict=False)
                    ),
                }
            )
        elif args.mode == "interactive":
            if args.capture_output is not None or args.menu_report_output is not None:
                raise ConfigError(
                    "interactive mode does not accept --capture-output or --menu-report-output"
                )
            if (
                args.menu_output_dir is None
                or args.control_metadata is None
                or args.control_output_dir is None
            ):
                raise ConfigError(
                    "interactive mode requires --menu-output-dir, --control-metadata, "
                    "and --control-output-dir"
                )
            request.update(
                {
                    "menu_steps": args.menu_step,
                    "menu_initial_wait_seconds": args.menu_initial_wait,
                    "menu_step_wait_seconds": args.menu_step_wait,
                    "menu_final_wait_seconds": args.menu_final_wait,
                    "menu_output_dir": str(args.menu_output_dir.resolve(strict=False)),
                    "control_metadata_path": str(args.control_metadata.resolve(strict=False)),
                    "control_output_dir": str(args.control_output_dir.resolve(strict=False)),
                    "control_idle_timeout_seconds": args.control_idle_timeout,
                    "control_settle_seconds": args.control_settle,
                }
            )
        elif args.menu_step:
            raise ConfigError("--menu-step is only valid in menu or interactive mode")
        environment = os.environ.copy()
        environment[REQUEST_ENVIRONMENT_VARIABLE] = _encoded_request(request)
        environment["HK_TX_LAUNCHER"] = str(TRANSACTIONAL_LAUNCHER)
        environment["HK_TX_PYTHON"] = sys.executable
        environment["HK_TX_PERSONAL"] = str(personal)
        environment["HK_TX_EXPERIMENTAL"] = str(experimental)
        environment["HK_TX_BACKUPS"] = str(backups)
        environment["HK_TX_GUARD"] = str(game)

        command = (
            "& $env:HK_TX_LAUNCHER "
            "-GameExecutable $env:HK_TX_PYTHON "
            "-PersonalRoot $env:HK_TX_PERSONAL "
            "-ExperimentalRoot $env:HK_TX_EXPERIMENTAL "
            "-BackupRoot $env:HK_TX_BACKUPS "
            "-GuardProcessExecutable $env:HK_TX_GUARD "
            "-GameArguments @('-m','hk_agent.isolated_session','--request-env')"
        )
        completed = subprocess.run(
            [
                str(POWERSHELL),
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = {
                "ok": False,
                "failure": "Session worker did not produce a result",
            }
        result["transaction_exit_code"] = completed.returncode
        if completed.returncode != 0:
            result["transaction_error"] = completed.stderr.strip()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("ok") is True and completed.returncode == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a smoke, menu, or interactive bridge session under save isolation"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--game-executable", type=Path)
    parser.add_argument("--personal-root", type=Path)
    parser.add_argument("--experimental-root", type=Path)
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--game-profile")
    parser.add_argument("--game-argument", action="append", default=[])
    parser.add_argument("--instance-id", default="isolated-smoke-01")
    parser.add_argument("--instance-data-root", type=Path)
    parser.add_argument("--expected-product-name")
    parser.add_argument("--mode", choices=("smoke", "menu", "interactive"), default="smoke")
    parser.add_argument("--capture-output", type=Path)
    parser.add_argument(
        "--menu-step",
        action="append",
        default=[],
        type=_parse_menu_step,
        metavar="BUTTON:LEASE_MS",
        help="one deterministic menu pulse; repeat for a fixed sequence",
    )
    parser.add_argument("--menu-initial-wait", type=float, default=5.0)
    parser.add_argument("--menu-step-wait", type=float, default=0.15)
    parser.add_argument("--menu-final-wait", type=float, default=0.0)
    parser.add_argument("--menu-output-dir", type=Path)
    parser.add_argument("--menu-report-output", type=Path)
    parser.add_argument("--control-metadata", type=Path)
    parser.add_argument("--control-output-dir", type=Path)
    parser.add_argument("--control-idle-timeout", type=float, default=300.0)
    parser.add_argument("--control-settle", type=float, default=0.15)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--close-timeout", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except (ConfigError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
