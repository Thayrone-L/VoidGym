"""Caminhos canônicos locais para dados e campanhas da v2."""

from pathlib import Path

from voidgym.core.errors import VoidGymValueError as ValueError


def runtime_root(root: Path | None = None) -> Path:
    return (root if root is not None else Path("runtime/v2")).resolve()


def campaign_root(campaign_id: str, root: Path | None = None) -> Path:
    if not campaign_id or Path(campaign_id).name != campaign_id:
        raise ValueError("campaign_id precisa ser um nome de pasta simples")
    return runtime_root(root) / "campaigns" / campaign_id
