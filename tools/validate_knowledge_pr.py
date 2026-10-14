from __future__ import annotations

import argparse
from pathlib import Path

from hk_agent.knowledge_ci import validate_knowledge_pull_request


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a knowledge PR without executing code from its candidate checkout."
    )
    parser.add_argument("--candidate-repository", type=Path, required=True)
    parser.add_argument("--trusted-knowledge", type=Path, required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--allow-champion-update", action="store_true")
    args = parser.parse_args()
    validate_knowledge_pull_request(
        args.candidate_repository,
        args.trusted_knowledge,
        base_sha=args.base_sha,
        candidate_sha=args.candidate_sha,
        allow_champion_update=args.allow_champion_update,
    )
    print("Knowledge pull request is valid.")


if __name__ == "__main__":
    main()
