from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from hk_agent.bridge_environment import BRIDGE_VARIABLES, build_bridge_environment

ROOT = Path(__file__).resolve().parents[1]


def _base() -> dict[str, str]:
    return {name: f"stale-{name}" for name in BRIDGE_VARIABLES} | {"KEEP_ME": "yes"}


def _common() -> dict[str, str]:
    return {
        "host": "127.0.0.1",
        "port": "43123",
        "run_id": "run-1",
        "instance_id": "instance-1",
        "token": "token-1",
        "game_profile": "development",
    }


def test_legacy_mode_cleans_old_bridge_values(tmp_path: Path) -> None:
    environment = build_bridge_environment(
        _base(), mode="legacy", **_common(), save_root=str(tmp_path)
    )
    assert environment["HK_AGENT_SAVE_ROOT"] == str(tmp_path)
    assert not {"HK_AGENT_INSTANCE_MODE", "HK_AGENT_PRODUCT_NAME", "HK_AGENT_INSTANCE_DATA_ROOT"} & environment.keys()
    assert environment["KEEP_ME"] == "yes"


def test_instance_mode_cleans_old_bridge_values(tmp_path: Path) -> None:
    environment = build_bridge_environment(
        _base(),
        mode="instance",
        **_common(),
        product_name="Hollow Knight AI 1",
        instance_data_root=str(tmp_path),
    )
    assert environment["HK_AGENT_INSTANCE_MODE"] == "1"
    assert environment["HK_AGENT_PRODUCT_NAME"] == "Hollow Knight AI 1"
    assert environment["HK_AGENT_INSTANCE_DATA_ROOT"] == str(tmp_path)
    assert "HK_AGENT_SAVE_ROOT" not in environment


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "legacy", "product_name": "product", "instance_data_root": "data"},
        {"mode": "legacy"},
        {"mode": "instance", "save_root": "save"},
        {"mode": "instance", "product_name": "product"},
        {"mode": "unknown", "save_root": "save"},
    ],
)
def test_incompatible_mode_parameters_raise(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        build_bridge_environment(_base(), **_common(), **kwargs)  # type: ignore[arg-type]


def test_python_environments_are_accepted_by_bridge_configuration(tmp_path: Path) -> None:
    dotnet = shutil.which("dotnet")
    if dotnet is None:
        pytest.skip("dotnet is unavailable; Python-to-C# bridge contract was not run")
    data_root = tmp_path / "instance-data"
    save_root = tmp_path / "save-root"
    data_root.mkdir()
    save_root.mkdir()
    legacy = build_bridge_environment(
        _base(), mode="legacy", **_common(), save_root=str(save_root)
    )
    instance = build_bridge_environment(
        _base(),
        mode="instance",
        **_common(),
        product_name="Hollow Knight AI 1",
        instance_data_root=str(data_root),
    )
    environment_json = tmp_path / "environments.json"
    environment_json.write_text(json.dumps([legacy, instance]), encoding="utf-8")
    project = ROOT / "bridge" / "HkAgentBridge.ConfigurationTests" / "HkAgentBridge.ConfigurationTests.csproj"
    completed = subprocess.run(
        [dotnet, "run", "--project", str(project), "--configuration", "Release", "--", "--environment-json", str(environment_json)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Python-produced legacy and instance environments accepted" in completed.stdout
