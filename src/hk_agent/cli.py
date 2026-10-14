from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from hk_agent.config import ConfigError, load_config
from hk_agent.distributed_knowledge import (
    KnowledgeValidationError,
    export_contribution,
    sync_from_git_ref,
    validate_knowledge_tree,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hk-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Validate local project configuration")
    doctor.add_argument("--config", type=Path, default=Path("config.local.toml"))

    knowledge = subparsers.add_parser("knowledge", help="Manage portable training knowledge")
    knowledge_commands = knowledge.add_subparsers(dest="knowledge_command", required=True)
    export = knowledge_commands.add_parser("export", help="Export a sanitized manifest summary")
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--root", type=Path, default=Path("knowledge"))
    validate = knowledge_commands.add_parser("validate", help="Validate the knowledge tree")
    validate.add_argument("--root", type=Path, default=Path("knowledge"))
    sync = knowledge_commands.add_parser("sync", help="Import knowledge via git show")
    sync.add_argument("--repo", type=Path, default=Path("."))
    sync.add_argument("--ref", default="origin/knowledge")
    sync.add_argument("--root", type=Path, default=Path("knowledge"))
    return parser


def _doctor(config_path: Path) -> int:
    try:
        config = load_config(config_path)
    except ConfigError as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False, indent=2))
        return 2

    report = {
        "ok": True,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "game_profile": config.game_profile,
        "game_executable": str(config.game_executable),
        "training_enabled": config.training_enabled,
        "parallel_enabled": config.parallel_enabled,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "doctor":
        return _doctor(arguments.config)
    if arguments.command == "knowledge":
        try:
            if arguments.knowledge_command == "export":
                report = export_contribution(arguments.manifest, arguments.root)
            elif arguments.knowledge_command == "validate":
                index = validate_knowledge_tree(arguments.root)
                report = {"ok": True, "contributions": len(index["contributions"])}
            elif arguments.knowledge_command == "sync":
                report = sync_from_git_ref(arguments.repo, arguments.ref, arguments.root)
            else:
                raise AssertionError(f"Unhandled knowledge command: {arguments.knowledge_command}")
        except (KnowledgeValidationError, OSError) as error:
            print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False, indent=2))
            return 2
        print(json.dumps({"ok": True, **report}, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(f"Unhandled command: {arguments.command}")
