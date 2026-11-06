from __future__ import annotations

import json
from pathlib import Path

from tools.v2.eval_86i_hk_recapture import _capture_paths


def test_capture_manifest_excludes_only_explicit_menu_scene_frames(tmp_path: Path) -> None:
    menu = tmp_path / "00001-capture.png"
    game = tmp_path / "00002-capture.png"
    menu.write_bytes(b"menu")
    game.write_bytes(b"game")
    manifest = tmp_path / "capture-manifest.json"
    manifest.write_text(json.dumps({"captures": [
        {"path": str(menu), "scene_name": "Menu_Title"},
        {"path": str(game), "scene_name": "Tutorial_01"},
    ]}), encoding="utf-8")

    paths, total, excluded = _capture_paths(tmp_path, manifest)

    assert paths == [game.resolve()]
    assert total == 2
    assert excluded == 1
