"""Limites de tamanho, legibilidade e localização da configuração."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "voidgym"


def test_python_modules_and_functions_stay_bounded() -> None:
    failures = []
    for path in PACKAGE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if len(text.splitlines()) > 600:
            failures.append(f"{path}: módulo excede 600 linhas")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.end_lineno - node.lineno + 1 > 80
            ):
                failures.append(f"{path}:{node.lineno}: função excede 80 linhas")
    assert not failures, "\n".join(failures)


def test_only_config_reads_environment_and_assets_are_not_minified() -> None:
    failures = []
    for path in PACKAGE.rglob("*.py"):
        if path.name != "config.py" and "os.environ" in path.read_text(encoding="utf-8"):
            failures.append(f"{path}: os.environ fora de core/config.py")
    for suffix in ("*.js", "*.css"):
        for path in PACKAGE.rglob(suffix):
            if any(len(line) > 300 for line in path.read_text(encoding="utf-8").splitlines()):
                failures.append(f"{path}: linha minificada acima de 300 caracteres")
    assert not failures, "\n".join(failures)
