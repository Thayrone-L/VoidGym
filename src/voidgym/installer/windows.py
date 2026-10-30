from __future__ import annotations

import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from voidgym.core.config import process_environment


@dataclass(frozen=True, slots=True)
class GameCandidate:
    path: Path
    store: str
    registry_key: str | None = None


class WindowsSystem:
    """Windows operations behind a small replaceable adapter for tests."""

    def __init__(self) -> None:
        self.environ = process_environment()

    @property
    def local_app_data(self) -> Path:
        value = self.environ.get("LOCALAPPDATA")
        if not value:
            raise RuntimeError("O Windows não informou LOCALAPPDATA.")
        return Path(value).resolve()

    @property
    def user_profile(self) -> Path:
        value = self.environ.get("USERPROFILE")
        return Path(value).resolve() if value else Path.home().resolve()

    @property
    def local_low(self) -> Path:
        return self.local_app_data.parent / "LocalLow"

    @property
    def startup(self) -> Path:
        roaming = self.environ.get("APPDATA", str(self.user_profile / "AppData/Roaming"))
        return Path(roaming) / "Microsoft/Windows/Start Menu/Programs/Startup"

    @property
    def programs(self) -> Path:
        return Path(self.environ.get("APPDATA", str(self.user_profile / "AppData/Roaming"))) / (
            "Microsoft/Windows/Start Menu/Programs"
        )

    @property
    def is_windows(self) -> bool:
        return sys.platform == "win32"

    def is_admin(self) -> bool:
        if not self.is_windows:
            return False
        import ctypes

        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except (AttributeError, OSError):
            return False

    def on_battery(self) -> bool | None:
        if not self.is_windows:
            return None
        import ctypes

        class PowerStatus(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                        ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                        ("BatteryLifeTime", ctypes.c_uint32), ("BatteryFullLifeTime", ctypes.c_uint32)]

        state = PowerStatus()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(state)):
            return None
        return bool(state.ACLineStatus == 0)

    def onedrive_roots(self) -> tuple[Path, ...]:
        roots: set[Path] = set()
        for key in ("OneDrive", "OneDriveCommercial", "OneDriveConsumer"):
            value = self.environ.get(key)
            if value:
                roots.add(Path(value).resolve(strict=False))
        if self.is_windows:
            try:
                import winreg

                base = r"Software\Microsoft\OneDrive\Accounts"
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, base) as accounts:
                    for index in range(winreg.QueryInfoKey(accounts)[0]):
                        name = winreg.EnumKey(accounts, index)
                        with winreg.OpenKey(accounts, name) as account:
                            try:
                                value, _ = winreg.QueryValueEx(account, "UserFolder")
                            except OSError:
                                continue
                            roots.add(Path(value).resolve(strict=False))
            except OSError:
                pass
        return tuple(sorted(roots, key=lambda item: str(item).casefold()))

    def is_onedrive(self, path: Path) -> bool:
        target = path.resolve(strict=False)
        for root in self.onedrive_roots():
            try:
                target.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def disk_free_bytes(self, path: Path) -> int:
        return shutil.disk_usage(path).free

    def game_candidates(self) -> tuple[GameCandidate, ...]:
        candidates: dict[str, GameCandidate] = {}
        defaults = (
            Path(r"C:\GOG Games\Hollow Knight"),
            Path(r"C:\Program Files (x86)\GOG Galaxy\Games\Hollow Knight"),
        )
        for path in defaults:
            if (path / "Hollow Knight.exe").is_file():
                candidates[str(path.resolve(strict=False)).casefold()] = GameCandidate(path, "gog")
        if self.is_windows:
            try:
                import winreg

                roots = (
                    (winreg.HKEY_CURRENT_USER, r"SOFTWARE\GOG.com\Games"),
                    (winreg.HKEY_CURRENT_USER, r"SOFTWARE\WOW6432Node\GOG.com\Games"),
                    (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\GOG.com\Games"),
                    (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\GOG.com\Games"),
                )
                for hive, key_path in roots:
                    try:
                        with winreg.OpenKey(hive, key_path) as root:
                            count = winreg.QueryInfoKey(root)[0]
                            for index in range(count):
                                subkey = winreg.EnumKey(root, index)
                                try:
                                    with winreg.OpenKey(root, subkey) as game:
                                        path, _ = winreg.QueryValueEx(game, "path")
                                        labels = []
                                        for field in ("gameName", "name", "GAMENAME"):
                                            try:
                                                labels.append(str(winreg.QueryValueEx(game, field)[0]))
                                            except OSError:
                                                pass
                                        candidate = Path(path)
                                        if ((any("hollow knight" in name.casefold() for name in labels)
                                             or (candidate / "Hollow Knight.exe").is_file())
                                                and (candidate / "Hollow Knight.exe").is_file()):
                                            resolved = candidate.resolve(strict=False)
                                            candidates[str(resolved).casefold()] = GameCandidate(
                                                resolved, "gog", f"{key_path}\\{subkey}"
                                            )
                                except OSError:
                                    continue
                    except OSError:
                        continue
            except ImportError:
                pass
        return tuple(candidates[key] for key in sorted(candidates))

    def steam_libraries(self) -> tuple[Path, ...]:
        libraries: set[Path] = set()
        vdf = self.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
        path = Path(vdf) / "Steam/steamapps/libraryfolders.vdf"
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            for raw in re.findall(r'"path"\s+"([^"]+)"', text):
                libraries.add(Path(raw.replace("\\\\", "\\")) / "steamapps")
        return tuple(libraries)

    def game_processes(self) -> tuple[str, ...]:
        try:
            import psutil  # type: ignore[import-untyped]
        except ImportError:
            return ()
        paths: list[str] = []
        for process in psutil.process_iter(["name", "exe"]):
            try:
                name = str(process.info.get("name") or "").casefold()
                executable = process.info.get("exe")
                if "hollow knight" in name and executable:
                    paths.append(str(Path(executable).resolve(strict=False)))
            except (psutil.Error, OSError):
                continue
        return tuple(paths)

    def create_shortcut(self, link: Path, target: Path, *, arguments: str = "", working_dir: Path | None = None) -> None:
        if not self.is_windows:
            raise RuntimeError("Atalhos do Windows só podem ser criados no Windows.")
        link.parent.mkdir(parents=True, exist_ok=True)
        import base64
        import subprocess

        def ps(value: str) -> str:
            return "'" + value.replace("'", "''") + "'"

        script = (
            "$shell = New-Object -ComObject WScript.Shell; "
            f"$shortcut = $shell.CreateShortcut({ps(str(link))}); "
            f"$shortcut.TargetPath = {ps(str(target))}; "
            f"$shortcut.Arguments = {ps(arguments)}; "
            f"$shortcut.WorkingDirectory = {ps(str(working_dir or target.parent))}; "
            "$shortcut.Save()"
        )
        encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                       check=True, capture_output=True, text=True)

    def remove_shortcut(self, link: Path) -> None:
        link.unlink(missing_ok=True)

    def write_registry_value(self, key_path: str, name: str, value: str) -> None:
        if not self.is_windows:
            raise RuntimeError("Registro do Windows indisponível.")
        import winreg

        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)

    def delete_registry_key(self, key_path: str) -> None:
        from voidgym.installer.state import safe_uninstall_registry_key

        if not self.is_windows:
            raise RuntimeError("Registro do Windows indisponível.")
        if not safe_uninstall_registry_key(key_path):
            raise RuntimeError(f"Refusing to delete registry key outside managed instance roots: {key_path}")
        import winreg

        def delete_tree(path: str) -> None:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path, 0, winreg.KEY_READ | winreg.KEY_WRITE) as key:
                children = [winreg.EnumKey(key, index) for index in range(winreg.QueryInfoKey(key)[0])]
            for child in children:
                delete_tree(path + "\\" + child)
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, path)

        delete_tree(key_path.removeprefix("HKCU\\"))
