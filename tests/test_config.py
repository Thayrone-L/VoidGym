from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hk_agent.config import ConfigError, load_config


def _document(game_root: Path, personal: Path, experimental: Path, backup: Path) -> str:
    return f'''
[game]
profile = "development"
install_path = "{game_root.as_posix()}"
executable = "hollow_knight.exe"

[saves]
personal_root = "{personal.as_posix()}"
experimental_root = "{experimental.as_posix()}"
backup_root = "{backup.as_posix()}"

[training]
enabled = false

[instances]
parallel_enabled = false
'''


class ConfigTests(unittest.TestCase):
    def test_accepts_separate_roots_with_existing_game(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game = root / "game"
            game.mkdir()
            (game / "hollow_knight.exe").touch()
            config_path = root / "config.toml"
            config_path.write_text(
                _document(game, root / "personal", root / "experimental", root / "backup"),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.game_profile, "development")
            self.assertFalse(config.training_enabled)
            self.assertFalse(config.parallel_enabled)

    def test_rejects_same_personal_and_experimental_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game = root / "game"
            game.mkdir()
            (game / "hollow_knight.exe").touch()
            shared = root / "shared"
            config_path = root / "config.toml"
            config_path.write_text(
                _document(game, shared, shared, root / "backup"),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigError, "must differ"):
                load_config(config_path)

    def test_rejects_parallel_execution_during_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game = root / "game"
            game.mkdir()
            (game / "hollow_knight.exe").touch()
            config_path = root / "config.toml"
            contents = _document(
                game, root / "personal", root / "experimental", root / "backup"
            ).replace("parallel_enabled = false", "parallel_enabled = true")
            config_path.write_text(contents, encoding="utf-8")

            with self.assertRaisesRegex(ConfigError, "Parallel execution"):
                load_config(config_path)

    def test_training_profile_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game = root / "game"
            game.mkdir()
            (game / "hollow_knight.exe").touch()
            config_path = root / "config.toml"
            contents = _document(
                game, root / "personal", root / "experimental", root / "backup"
            ).replace("enabled = false", "enabled = true", 1)
            config_path.write_text(contents, encoding="utf-8")

            with self.assertRaisesRegex(ConfigError, "Training must remain disabled"):
                load_config(config_path)
            config = load_config(config_path, allow_training=True)
            self.assertTrue(config.training_enabled)


if __name__ == "__main__":
    unittest.main()
