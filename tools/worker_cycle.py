from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hk_agent.distributed_knowledge import KnowledgeValidationError
from hk_agent.worker_cycle import (
    WorkerCycleError,
    finalize_candidate,
    load_worker_config,
    preflight_worker,
    sync_worker_knowledge,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Operate one local distributed-training worker")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config.worker.local.json"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    sync = commands.add_parser("sync", help="import an already-fetched knowledge Git ref")
    sync.add_argument("--repo", type=Path, default=PROJECT_ROOT)
    sync.add_argument("--ref", help="must equal trusted_ref from the local config")
    preflight = commands.add_parser(
        "preflight", help="validate and freeze the champion for one session"
    )
    preflight.add_argument("--session-id", required=True)
    finalize = commands.add_parser("finalize", help="export a finished session candidate")
    finalize.add_argument("--manifest", type=Path, required=True)
    finalize.add_argument("--session-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_worker_config(args.config)
        if args.command == "sync":
            report = sync_worker_knowledge(config, repo=args.repo, ref=args.ref)
        elif args.command == "preflight":
            report = preflight_worker(config, args.session_id)
        elif args.command == "finalize":
            report = finalize_candidate(config, args.manifest, args.session_id)
        else:
            raise AssertionError(f"unhandled worker command: {args.command}")
    except (KnowledgeValidationError, WorkerCycleError, OSError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"ok": True, **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
