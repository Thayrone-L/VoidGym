from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from hk_agent.config import ConfigError
from hk_agent.control_plane_server import build_control_plane, main


def test_dry_run_factory_does_not_require_local_config(tmp_path: Path) -> None:
    control_plane = build_control_plane(tmp_path, allow_execute=False)

    assert not control_plane.allow_execute
    assert control_plane.max_active_launches == 1
    assert not control_plane.parallel_enabled
    assert control_plane.runtime_root == (tmp_path / "artifacts" / "runtime").resolve()
    assert control_plane.training_device == "cpu"
    assert control_plane.torch_threads == 3
    assert control_plane.torch_interop_threads == 1


def test_allow_execute_without_valid_config_exits_with_short_message() -> None:
    error = ConfigError("Configuration not found: config.training.local.toml")
    with (
        patch.object(sys, "argv", ["control-plane", "--allow-execute"]),
        patch("hk_agent.control_plane_server.build_control_plane", side_effect=error),
        pytest.raises(SystemExit, match="Configuration not found"),
    ):
        main()
