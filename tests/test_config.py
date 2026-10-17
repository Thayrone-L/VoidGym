from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
count = 1
parallel_enabled = false
'''


def _parallel_config(
    root: Path,
    *,
    install_roots: tuple[Path, Path] | None = None,
    data_roots: tuple[Path, Path] | None = None,
    products: tuple[str, str] = ("Hollow Knight AI 1", "Hollow Knight AI 2"),
    marker_error: bool = False,
) -> None:
    game = root / "game"
    game.mkdir()
    (game / "hollow_knight.exe").touch()
    installs = install_roots or (root / "hk-ai-1", root / "hk-ai-2")
    data = data_roots or (root / "data-1", root / "data-2")
    paths = ", ".join(f'"{path.as_posix()}"' for path in installs)
    config_path = root / "config.toml"
    config_path.write_text(
        _document(game, root / "personal", root / "experimental", root / "backup").replace(
            "count = 1\nparallel_enabled = false",
            f"count = 2\nparallel_enabled = true\ninstall_roots = [{paths}]",
        ),
        encoding="utf-8",
    )
    if marker_error:
        with patch("hk_agent.config.load_instance_marker", side_effect=ValueError("invalid marker")):
            load_config(config_path)
        return
    markers = [
        SimpleNamespace(product_name=products[index], persistent_data_path_expected=str(data[index]))
        for index in range(2)
    ]
    with patch("hk_agent.config.load_instance_marker", side_effect=markers):
        load_config(config_path)


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
            self.assertEqual(config.instance_count, 1)
            self.assertEqual(config.training_device, "cpu")
            self.assertEqual(config.torch_threads, 3)

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

    def test_accepts_governed_parallel_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game = root / "game"
            game.mkdir()
            (game / "hollow_knight.exe").touch()
            config_path = root / "config.toml"
            contents = _document(
                game, root / "personal", root / "experimental", root / "backup"
            ).replace(
                "count = 1\nparallel_enabled = false",
                "count = 2\nparallel_enabled = true\ntraining_device = 'cpu'\ntorch_threads = 3\ninstall_roots = ['" + str((root / 'hk-ai-1').as_posix()) + "', '" + str((root / 'hk-ai-2').as_posix()) + "']",
            )
            config_path.write_text(contents, encoding="utf-8")
            markers = [
                SimpleNamespace(product_name="Hollow Knight AI 1", persistent_data_path_expected=str(root / "data1")),
                SimpleNamespace(product_name="Hollow Knight AI 2", persistent_data_path_expected=str(root / "data2")),
            ]
            with patch("hk_agent.config.load_instance_marker", side_effect=markers):
                config = load_config(config_path)
            self.assertTrue(config.parallel_enabled)
            self.assertEqual(config.instance_count, 2)
            self.assertEqual(config.instance_product_names, ("Hollow Knight AI 1", "Hollow Knight AI 2"))

    def test_parallel_config_rejects_invalid_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
            ConfigError, "marker is missing or invalid"
        ):
            _parallel_config(Path(temporary), marker_error=True)

    def test_parallel_config_rejects_duplicate_products(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
            ConfigError, "productName values must be unique"
        ):
            _parallel_config(
                Path(temporary), products=("Hollow Knight AI", "hollow knight ai")
            )

    def test_parallel_config_rejects_install_copy_of_original(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ConfigError, "managed root must not overlap"):
                _parallel_config(root, install_roots=(root / "game", root / "hk-ai-2"))

    def test_parallel_config_rejects_data_root_over_personal_saves(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ConfigError, "managed root must not overlap"):
                _parallel_config(
                    root,
                    data_roots=(root / "personal", root / "data-2"),
                )

    def test_parallel_config_rejects_instance_roots_over_all_protected_roots(self) -> None:
        for protected in ("game", "experimental", "backup", "artifacts/runtime"):
            with self.subTest(protected=protected), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with self.assertRaisesRegex(ConfigError, "managed root must not overlap"):
                    _parallel_config(
                        root,
                        install_roots=(root / protected, root / "hk-ai-2"),
                    )
            with self.subTest(data_protected=protected), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with self.assertRaisesRegex(ConfigError, "managed root must not overlap"):
                    _parallel_config(
                        root,
                        data_roots=(root / protected, root / "data-2"),
                    )

    def test_rejects_parallel_count_greater_than_install_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game = root / "game"
            game.mkdir()
            (game / "hollow_knight.exe").touch()
            content = _document(game, root / "personal", root / "experimental", root / "backup").replace(
                "count = 1\nparallel_enabled = false",
                "count = 2\nparallel_enabled = true\ninstall_roots = ['" + str((root / 'one').as_posix()) + "']",
            )
            config_path = root / "config.toml"
            config_path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "exactly instances.count"):
                load_config(config_path)

    def test_rejects_missing_instance_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game = root / "game"
            game.mkdir()
            (game / "hollow_knight.exe").touch()
            install = root / "hk-ai-1"
            install.mkdir()
            content = _document(game, root / "personal", root / "experimental", root / "backup").replace(
                "parallel_enabled = false", "parallel_enabled = false\ninstall_roots = ['" + install.as_posix() + "']"
            )
            config_path = root / "config.toml"
            config_path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "marker is missing or invalid"):
                load_config(config_path)

    def test_rejects_parallel_flag_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game = root / "game"
            game.mkdir()
            (game / "hollow_knight.exe").touch()
            config_path = root / "config.toml"
            config_path.write_text(
                _document(game, root / "personal", root / "experimental", root / "backup").replace(
                    "parallel_enabled = false", "parallel_enabled = true"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "instances.count"):
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

    def test_resolves_relative_runtime_against_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_root = root / "configuration"
            config_root.mkdir()
            game = root / "game"
            game.mkdir()
            (game / "hollow_knight.exe").touch()
            config_path = config_root / "config.toml"
            config_path.write_text(
                _document(game, root / "personal", root / "experimental", root / "backup")
                + 'runtime_root = "relative/runtime"\n',
                encoding="utf-8",
            )
            original_cwd = Path.cwd()
            elsewhere = root / "elsewhere"
            elsewhere.mkdir()
            try:
                os.chdir(elsewhere)
                config = load_config(config_path)
            finally:
                os.chdir(original_cwd)

            self.assertEqual(config.runtime_root, (config_root / "relative/runtime").resolve())

    def test_parallel_cuda_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game = root / "game"
            game.mkdir()
            (game / "hollow_knight.exe").touch()
            config_path = root / "config.toml"
            contents = _document(
                game, root / "personal", root / "experimental", root / "backup"
            ).replace(
                "count = 1\nparallel_enabled = false",
                "count = 2\nparallel_enabled = true\ntraining_device = 'cuda'",
            )
            config_path.write_text(contents, encoding="utf-8")

            with self.assertRaisesRegex(ConfigError, "GPU reservada"):
                load_config(config_path)

    def test_single_instance_cuda_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game = root / "game"
            game.mkdir()
            (game / "hollow_knight.exe").touch()
            config_path = root / "config.toml"
            contents = _document(
                game, root / "personal", root / "experimental", root / "backup"
            ).replace("parallel_enabled = false", "parallel_enabled = false\ntraining_device = 'cuda'")
            config_path.write_text(contents, encoding="utf-8")

            config = load_config(config_path)

            self.assertEqual(config.training_device, "cuda")


if __name__ == "__main__":
    unittest.main()
