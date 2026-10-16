from __future__ import annotations

import json
from pathlib import Path

import pytest

from hk_agent.instance_install import MARKER_NAME
from hk_agent.instance_marker import load_instance_marker


def test_marker_loader_rejects_unknown_and_missing_fields(tmp_path: Path) -> None:
    marker = tmp_path / MARKER_NAME
    marker.write_text(json.dumps({"schema_version": 1, "unexpected": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="closed schema"):
        load_instance_marker(tmp_path, expected_index=1)
