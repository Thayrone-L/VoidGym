from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from hk_agent.config import ConfigError, load_config
from hk_agent.distributed_knowledge import (
    KnowledgeValidationError,
    export_candidate,
    export_contribution,
    promote_candidate_file,
    sync_from_git_ref,
    validate_champion_file,
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
    candidate = knowledge_commands.add_parser(
        "candidate-export", help="Export a sanitized candidate manifest"
    )
    candidate.add_argument("--contribution", type=Path, required=True)
    candidate.add_argument("--worker-id", required=True)
    candidate.add_argument("--artifact-uri", required=True)
    candidate.add_argument("--artifact-sha256", required=True)
    candidate.add_argument("--artifact-size-bytes", type=int, required=True)
    candidate.add_argument(
        "--artifact-format", choices=("sb3-zip", "safetensors"), default="sb3-zip"
    )
    candidate.add_argument(
        "--trust-tier",
        choices=("owned", "quarantined", "external-untrusted"),
        default="owned",
    )
    candidate.add_argument("--parent-generation", type=int, required=True)
    candidate.add_argument("--parent-sha256", required=True)
    candidate.add_argument("--root", type=Path, default=Path("knowledge"))
    validate = knowledge_commands.add_parser("validate", help="Validate the knowledge tree")
    validate.add_argument("--root", type=Path, default=Path("knowledge"))
    sync = knowledge_commands.add_parser("sync", help="Import knowledge via git show")
    sync.add_argument("--repo", type=Path, default=Path("."))
    sync.add_argument("--ref", default="origin/knowledge")
    sync.add_argument("--root", type=Path, default=Path("knowledge"))
    champion_validate = knowledge_commands.add_parser(
        "champion-validate", help="Validate the local champion registry record"
    )
    champion_validate.add_argument("--champion", type=Path, default=Path("knowledge/champion.json"))
    champion_promote = knowledge_commands.add_parser(
        "champion-promote", help="Promote a candidate using local compare-and-swap"
    )
    champion_promote.add_argument("--champion", type=Path, default=Path("knowledge/champion.json"))
    champion_promote.add_argument("--candidate", type=Path, required=True)
    champion_promote.add_argument("--expected-generation", type=int, required=True)
    champion_promote.add_argument("--expected-sha256", required=True)
    champion_promote.add_argument("--evaluation-report-sha256", required=True)
    champion_promote.add_argument("--evaluation-suite-sha256", required=True)
    champion_promote.add_argument("--evaluator-commit", required=True)
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
            elif arguments.knowledge_command == "candidate-export":
                report = export_candidate(
                    arguments.contribution,
                    arguments.root,
                    worker_id=arguments.worker_id,
                    artifact_uri=arguments.artifact_uri,
                    artifact_sha256=arguments.artifact_sha256,
                    artifact_size_bytes=arguments.artifact_size_bytes,
                    artifact_format=arguments.artifact_format,
                    trust_tier=arguments.trust_tier,
                    parent_generation=arguments.parent_generation,
                    parent_sha256=arguments.parent_sha256,
                )
            elif arguments.knowledge_command == "validate":
                index = validate_knowledge_tree(arguments.root)
                report = {"ok": True, "contributions": len(index["contributions"])}
            elif arguments.knowledge_command == "sync":
                report = sync_from_git_ref(arguments.repo, arguments.ref, arguments.root)
            elif arguments.knowledge_command == "champion-validate":
                report = validate_champion_file(arguments.champion)
            elif arguments.knowledge_command == "champion-promote":
                report = promote_candidate_file(
                    arguments.champion,
                    arguments.candidate,
                    expected_generation=arguments.expected_generation,
                    expected_champion_sha256=arguments.expected_sha256,
                    evaluation_report_sha256=arguments.evaluation_report_sha256,
                    evaluation_suite_sha256=arguments.evaluation_suite_sha256,
                    evaluator_commit=arguments.evaluator_commit,
                )
            else:
                raise AssertionError(f"Unhandled knowledge command: {arguments.knowledge_command}")
        except (KnowledgeValidationError, OSError) as error:
            print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False, indent=2))
            return 2
        print(json.dumps({"ok": True, **report}, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(f"Unhandled command: {arguments.command}")
