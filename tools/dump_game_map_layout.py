from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hk_agent.control_plane import ControlPlane
from hk_agent.instance_install import (
    INSTANCES_ROOT,
    ORIGINAL_INSTALL,
    PERSONAL_ROOT,
    RUNTIME_ROOT,
)
from hk_agent.instance_marker import InstanceMarker, load_instance_marker


def _request_window_close(pid: int) -> None:
    if os.name != "nt":
        return
    user32 = ctypes.windll.user32
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def close_for_pid(hwnd: int, _unused: int) -> bool:
        process_id = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        if process_id.value == pid and user32.IsWindowVisible(hwnd):
            user32.PostMessageW(hwnd, 0x0010, 0, 0)
            return False
        return True

    user32.EnumWindows(callback_type(close_for_pid), 0)


def _path_overlaps(first: Path, second: Path) -> bool:
    first_path = os.path.normcase(str(first.resolve(strict=False)))
    second_path = os.path.normcase(str(second.resolve(strict=False)))
    try:
        common = os.path.commonpath((first_path, second_path))
    except ValueError:
        return False
    return common in {first_path, second_path}


def _default_status() -> Mapping[str, Any]:
    return ControlPlane(PROJECT_ROOT, observer=True).status()


def _assert_campaign_idle(instance_index: int, status_reader: Callable[[], Mapping[str, Any]]) -> None:
    status = status_reader()
    launches = status.get("active_launches")
    if not isinstance(launches, list):
        raise TypeError("Não foi possível verificar campanhas ativas; dump cancelado por segurança.")
    active_count = status.get("active_count")
    if not isinstance(active_count, int) or active_count < len(launches):
        raise RuntimeError("Contagem de campanhas inconsistente; dump cancelado por segurança.")
    if active_count > len(launches):
        raise RuntimeError("Há campanhas ativas sem registro detalhado; dump cancelado por segurança.")
    for launch in launches:
        if not isinstance(launch, Mapping):
            raise TypeError("Status de campanha inválido; dump cancelado por segurança.")
        resources = launch.get("resources")
        slot_id = resources.get("slot_id") if isinstance(resources, Mapping) else None
        indices = launch.get("instance_indices")
        if isinstance(indices, list) and instance_index in indices:
            raise RuntimeError(f"A instância {instance_index} está sendo usada por uma campanha ativa.")
        if isinstance(indices, list) and all(isinstance(value, int) for value in indices):
            continue
        index = launch.get("instance_index")
        if isinstance(index, int):
            if index == instance_index:
                raise RuntimeError(f"A instância {instance_index} está sendo usada por uma campanha ativa.")
            continue
        if isinstance(slot_id, int):
            if slot_id == instance_index - 1:
                raise RuntimeError(f"A instância {instance_index} está sendo usada por uma campanha ativa.")
            continue
        raise RuntimeError(
            "Há uma campanha ativa sem identificação confiável da instância; dump cancelado por segurança."
        )


def _default_game_processes() -> list[tuple[int, Path]]:
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("psutil é necessário para verificar processos Hollow Knight.") from error
    processes: list[tuple[int, Path]] = []
    for process in psutil.process_iter(["pid", "name", "exe"]):
        try:
            name = str(process.info.get("name") or "").casefold()
            executable = process.info.get("exe")
        except psutil.Error as error:
            if "hollow knight" in str(error).casefold():
                raise RuntimeError("Não foi possível inspecionar um processo Hollow Knight.") from error
            continue
        if name != "hollow knight.exe":
            continue
        if not isinstance(executable, str) or not executable:
            raise RuntimeError("Há um Hollow Knight aberto, mas não foi possível identificar a instalação.")
        processes.append((int(process.info["pid"]), Path(executable)))
    return processes


def _assert_games_closed(
    instance_root: Path,
    process_reader: Callable[[], Iterable[tuple[int, Path]]],
) -> None:
    target_exe = (instance_root / "Hollow Knight.exe").resolve(strict=False)
    personal_exe = (ORIGINAL_INSTALL / "Hollow Knight.exe").resolve(strict=False)
    for _pid, executable in process_reader():
        running = executable.resolve(strict=False)
        if os.path.normcase(str(running)) == os.path.normcase(str(target_exe)):
            raise RuntimeError(f"O jogo da instância está aberto: {running}")
        if os.path.normcase(str(running)) == os.path.normcase(str(personal_exe)):
            raise RuntimeError(f"O Hollow Knight pessoal está aberto: {running}")


def _backup_user_saves(save_root: Path, backup_dir: Path) -> dict[str, str]:
    if _path_overlaps(save_root, PERSONAL_ROOT):
        raise RuntimeError("O caminho de saves da instância coincide com o save pessoal; dump cancelado.")
    backup_dir.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    if not save_root.exists():
        return hashes
    if save_root.is_symlink() or not save_root.is_dir():
        raise RuntimeError(f"Raiz de saves inválida para backup: {save_root}")
    for source in sorted(save_root.glob("user*")):
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"Save de instância não é arquivo regular: {source}")
        destination = backup_dir / source.name
        shutil.copy2(source, destination)
        hashes[source.name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (backup_dir / "sha256.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return hashes


def _restore_user_saves(save_root: Path, backup_dir: Path, hashes: Mapping[str, str]) -> None:
    save_root.mkdir(parents=True, exist_ok=True)
    expected = set(hashes)
    for path in save_root.glob("user*"):
        if path.name not in expected:
            if path.is_symlink() or path.is_file():
                path.unlink()
            else:
                raise RuntimeError(f"Não posso remover com segurança o save criado durante o dump: {path}")
    for name, expected_hash in hashes.items():
        source = backup_dir / name
        target = save_root / name
        if target.is_symlink():
            raise RuntimeError(f"Save foi substituído por link durante o dump: {target}")
        temporary = target.with_name(f".{target.name}.restore.tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
        actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"SHA-256 não confere após restaurar {target}")


def _persistent_data_path(marker: InstanceMarker) -> Path:
    save_root = Path(marker.persistent_data_path_expected).expanduser()
    if _path_overlaps(save_root, PERSONAL_ROOT):
        raise RuntimeError("O marcador aponta para o save pessoal; dump cancelado por segurança.")
    return save_root


def _finish_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    _request_window_close(process.pid)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def dump_layout(
    instance_index: int,
    runtime_root: Path,
    *,
    timeout: float = 120.0,
    instances_root: Path = INSTANCES_ROOT,
    status_reader: Callable[[], Mapping[str, Any]] = _default_status,
    process_reader: Callable[[], Iterable[tuple[int, Path]]] = _default_game_processes,
    process_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
) -> Path:
    if instance_index < 1:
        raise ValueError("--instance deve ser maior que zero")
    base = instances_root.expanduser().resolve(strict=True)
    candidate = base / f"hk-ai-{instance_index}"
    if candidate.is_symlink():
        raise ValueError("a instalação isolada não pode ser link simbólico")
    root = candidate.resolve(strict=True)
    if root.parent != base:
        raise ValueError("a instalação isolada precisa ficar dentro de INSTANCES_ROOT")
    marker = load_instance_marker(root, expected_index=instance_index)
    executable = root / "Hollow Knight.exe"
    if not executable.is_file():
        raise FileNotFoundError(executable)
    _assert_campaign_idle(instance_index, status_reader)
    _assert_games_closed(root, process_reader)

    save_root = _persistent_data_path(marker)
    target = runtime_root / "map-layout"
    target.mkdir(parents=True, exist_ok=True)
    request = target / "DUMP_MAP_LAYOUT.request"
    output = target / "game-map-layout.json"
    if request.exists():
        raise RuntimeError("já existe uma solicitação de dump de mapa pendente")
    if output.exists():
        backup_output = target / f"game-map-layout-{time.strftime('%Y%m%d-%H%M%S')}.json.bak"
        output.replace(backup_output)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup_dir = target / "save-backups" / f"hk-ai-{instance_index}-{stamp}"
    hashes = _backup_user_saves(save_root, backup_dir)
    request.write_text(
        f"instance=hk-ai-{instance_index} marker={marker.bridge_dll_sha256}\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("HK_AGENT_HOST", None)
    environment.pop("HK_AGENT_PORT", None)
    environment["HK_AGENT_MAP_LAYOUT_DUMP_REQUEST"] = str(request)
    process: subprocess.Popen[Any] | None = None
    failure: BaseException | None = None
    result: Path | None = None
    try:
        process = process_factory([str(executable)], cwd=root, env=environment)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if output.is_file() and not request.exists():
                result = output
                break
            error_file = request.with_suffix(request.suffix + ".error.txt")
            if error_file.is_file():
                raise RuntimeError(error_file.read_text("utf-8", errors="replace"))
            if process.poll() is not None:
                raise RuntimeError(
                    f"O Hollow Knight isolado encerrou com status {process.returncode} antes do dump."
                )
            time.sleep(0.25)
        if result is None:
            raise TimeoutError(f"Dump do GameMap não terminou em {timeout:.0f} segundos.")
    except BaseException as error:  # noqa: BLE001 - the finally block must restore saves even on interruption.
        failure = error
    finally:
        if process is not None:
            try:
                _finish_process(process)
            except Exception as error:  # noqa: BLE001 - ainda tenta restaurar os saves.
                failure = RuntimeError(f"{failure or 'dump concluído'}; falha ao fechar o jogo: {error}")
        request.unlink(missing_ok=True)
        try:
            _restore_user_saves(save_root, backup_dir, hashes)
        except Exception as error:
            raise RuntimeError(
                f"Falha ao restaurar os saves: {error}. Backup íntegro esperado em {backup_dir}."
            ) from error
    if failure is not None:
        raise failure
    if result is None:
        raise RuntimeError(f"Dump terminou sem arquivo de saída. Backup dos saves: {backup_dir}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Coleta o layout GameMap de uma instância isolada.")
    parser.add_argument("--instance", type=int, default=2)
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    try:
        path = dump_layout(args.instance, args.runtime_root, timeout=args.timeout)
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        print(f"Falha ao coletar o layout GameMap: {error}", file=sys.stderr)
        return 1
    print(f"Layout GameMap salvo: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
