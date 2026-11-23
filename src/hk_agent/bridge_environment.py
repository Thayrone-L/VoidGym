"""Build child environments using the bridge's environment contract."""

from __future__ import annotations

from collections.abc import Mapping

BRIDGE_VARIABLES = frozenset(
    {
        "HK_AGENT_HOST",
        "HK_AGENT_PORT",
        "HK_AGENT_RUN_ID",
        "HK_AGENT_INSTANCE_ID",
        "HK_AGENT_TOKEN",
        "HK_AGENT_GAME_PROFILE",
        "HK_AGENT_SAVE_ROOT",
        "HK_AGENT_INSTANCE_MODE",
        "HK_AGENT_PRODUCT_NAME",
        "HK_AGENT_INSTANCE_DATA_ROOT",
    }
)


def clean_bridge_environment(base_env: Mapping[str, str]) -> dict[str, str]:
    """Copy an environment without any bridge-owned variables."""
    return {key: value for key, value in base_env.items() if key not in BRIDGE_VARIABLES}


def build_bridge_environment(
    base_env: Mapping[str, str],
    *,
    mode: str,
    host: str | None = None,
    port: str | None = None,
    run_id: str | None = None,
    instance_id: str | None = None,
    token: str | None = None,
    game_profile: str | None = None,
    save_root: str | None = None,
    product_name: str | None = None,
    instance_data_root: str | None = None,
    **unexpected: object,
) -> dict[str, str]:
    """Return a clean child environment configured for exactly one bridge mode."""
    if unexpected:
        raise ValueError(f"unexpected bridge parameters: {', '.join(sorted(unexpected))}")
    if mode not in {"legacy", "instance"}:
        raise ValueError("mode must be 'legacy' or 'instance'")
    host = _required(host, "host")
    port = _required(port, "port")
    run_id = _required(run_id, "run_id")
    instance_id = _required(instance_id, "instance_id")
    token = _required(token, "token")
    game_profile = _required(game_profile, "game_profile")
    if mode == "legacy":
        if save_root is None or not save_root.strip() or product_name is not None or instance_data_root is not None:
            raise ValueError("legacy mode requires save_root and forbids instance parameters")
    elif save_root is not None or not product_name or not product_name.strip() or not instance_data_root or not instance_data_root.strip():
        raise ValueError("instance mode requires product_name and instance_data_root and forbids save_root")

    environment = clean_bridge_environment(base_env)
    environment.update(
        {
            "HK_AGENT_HOST": host,
            "HK_AGENT_PORT": port,
            "HK_AGENT_RUN_ID": run_id,
            "HK_AGENT_INSTANCE_ID": instance_id,
            "HK_AGENT_TOKEN": token,
            "HK_AGENT_GAME_PROFILE": game_profile,
        }
    )
    if mode == "legacy":
        assert save_root is not None
        environment["HK_AGENT_SAVE_ROOT"] = save_root
    else:
        assert product_name is not None and instance_data_root is not None
        environment.update(
            {
                "HK_AGENT_INSTANCE_MODE": "1",
                "HK_AGENT_PRODUCT_NAME": product_name,
                "HK_AGENT_INSTANCE_DATA_ROOT": instance_data_root,
            }
        )
    return environment


def _required(value: str | None, name: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{name} is required")
    return value
