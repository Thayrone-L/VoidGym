from __future__ import annotations

import argparse
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

from voidgym.installer.log import create_installer_logger
from voidgym.installer.manifest import ManifestError, load_manifest
from voidgym.installer.state import InstallState, safe_uninstall_registry_key
from voidgym.installer.ui import ConsoleUI
from voidgym.installer.windows import WindowsSystem


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m voidgym.installer", description="Instalador local do VoidGym")
    parser.add_argument("comando", choices=("instalar", "verificar", "atualizar", "desinstalar"))
    parser.add_argument("--dry-run", action="store_true", help="mostrar o plano sem alterar o computador")
    parser.add_argument("--root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    return parser


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _verify(root: Path, manifest_path: Path, ui: ConsoleUI, *, dry_run: bool) -> int:
    manifest = load_manifest(manifest_path)
    state = InstallState(root / "install-state.json")
    if dry_run:
        ui.say("Plano: conferir estado, cópias do jogo, marcadores, DLL, configuração e atalhos sem alterar arquivos.")
        ui.say(f"Arte do mapa: {'liberada' if manifest['map_art']['distribution_approved'] else 'bloqueada por consentimento pendente'}.")
        return 0
    failures: list[str] = []
    for raw in state.data["created_paths"]:
        path = Path(raw)
        if not path.exists():
            failures.append(f"Caminho registrado ausente: {path}")
    if manifest["bridge_dll"].get("available") is not True:
        failures.append("A DLL do bridge não tem asset público com SHA-256 fixo.")
    if failures:
        for item in failures:
            ui.say("ERRO: " + item)
        return 1
    ui.say("Estado registrado e artefatos fixados conferidos.")
    return 0


def _uninstall_plan(state: InstallState, root: Path, system: WindowsSystem,
                    ui: ConsoleUI) -> tuple[list[Path], list[str]]:
    from voidgym.installer.state import safe_uninstall_path

    planned: list[Path] = []
    for raw in reversed(state.data["created_paths"]):
        target = Path(raw)
        if not safe_uninstall_path(target, app_root=root, local_low=system.local_low):
            raise RuntimeError(f"Caminho fora das raízes permitidas; plano recusado: {target}")
        planned.append(target)
    for raw in state.data["shortcuts"]:
        target = Path(raw)
        if not (_inside(target, system.startup) or _inside(target, system.programs)):
            raise RuntimeError(f"Atalho fora das pastas do VoidGym; plano recusado: {target}")
        ui.say(f"Atalho registrado: {target}")
    registry_keys = list(state.data["registry_keys"])
    for key in registry_keys:
        if not safe_uninstall_registry_key(key):
            raise RuntimeError(f"Chave de registro fora das instâncias gerenciadas; plano recusado: {key}")
        ui.say(f"Remover chave: HKCU\\{key.removeprefix('HKCU\\')}")
    return planned, registry_keys


def _uninstall(root: Path, system: WindowsSystem, ui: ConsoleUI, *, dry_run: bool) -> int:
    state = InstallState(root / "install-state.json")
    if not state.path.is_file():
        ui.say("Não há install-state.json; nada foi removido.")
        return 0
    planned, registry_keys = _uninstall_plan(state, root, system, ui)
    ui.say("Armazém de experiência e checkpoints serão preservados por padrão.")
    for target in planned:
        ui.say(f"Remover: {target}")
    if dry_run:
        return 0
    if not ui.confirm("Confirma a desinstalação somente dos caminhos registrados?", default=False):
        ui.say("Desinstalação cancelada.")
        return 0
    for raw in state.data["shortcuts"]:
        system.remove_shortcut(Path(raw))
    for key in registry_keys:
        system.delete_registry_key(key)
    for target in planned:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
    state.path.unlink(missing_ok=True)
    ui.say("Componentes registrados removidos; dados de experiência preservados.")
    return 0


def _run_command(args: argparse.Namespace, root: Path, manifest_path: Path,
                 system: WindowsSystem, ui: ConsoleUI, logger: logging.Logger, log_path: Path) -> int:
    if args.comando == "desinstalar":
        return _uninstall(root, system, ui, dry_run=args.dry_run)
    manifest = load_manifest(manifest_path, require_installable=args.comando == "instalar" and not args.dry_run)
    if args.comando == "verificar":
        return _verify(root, manifest_path, ui, dry_run=args.dry_run)
    if args.comando == "atualizar":
        ui.say("Atualização ainda não pode ocorrer: o fluxo de parar e retomar campanha com segurança não foi integrado.")
        return 2
    if args.dry_run:
        ui.say("Plano: detectar hardware/GOG, estimar armazenamento, provisionar cópias isoladas, configurar a bandeja e executar smoke de 60 s.")
        if not manifest["map_art"]["distribution_approved"]:
            ui.say("Mapa: padrão não; consentimento de distribuição pendente.")
        if manifest["bridge_dll"].get("available") is not True:
            ui.say("PENDÊNCIA: a DLL do bridge ainda não tem asset/hash; a instalação real não está liberada.")
        return 0
    if not system.is_windows:
        raise RuntimeError("O instalador só suporta Windows 10/11 de 64 bits.")
    if system.is_admin():
        raise RuntimeError("Execute como usuário normal, sem privilégios de administrador.")
    if system.is_onedrive(root):
        raise RuntimeError("O caminho escolhido fica sob OneDrive; escolha uma pasta local não sincronizada.")
    ui.say("Instalação bloqueada até a Release da DLL do bridge ter URL e SHA-256 verificados.")
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    system = WindowsSystem()
    root = (args.root or (system.local_app_data / "VoidGym")).resolve(strict=False)
    project_root = Path(__file__).resolve().parents[3]
    manifest_path = args.manifest or (project_root / "installer" / "manifest.json")
    ui = ConsoleUI()
    if args.comando == "desinstalar" and args.dry_run:
        return _uninstall(root, system, ui, dry_run=True)
    if args.dry_run:
        import logging
        logger = logging.getLogger("voidgym.installer.dry-run")
        log_path = Path("<dry-run sem gravações>")
    else:
        logger, log_path = create_installer_logger(root / "logs")
    logger.info("Command=%s dry_run=%s", args.comando, args.dry_run)
    try:
        return _run_command(args, root, manifest_path, system, ui, logger, log_path)
    except (OSError, ManifestError, RuntimeError, ValueError) as error:
        logger.exception("Installer failed")
        ui.say(f"ERRO: {error}\nLog: {log_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
