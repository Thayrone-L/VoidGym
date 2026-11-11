import ast
from pathlib import Path

ROOT = Path(__file__).parents[2]
V2 = ROOT / "src" / "voidgym"


def test_python_modules_and_functions_stay_below_limits() -> None:
    for path in V2.rglob("*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 600, f"{path} excede 600 linhas"
        tree = ast.parse("\n".join(lines))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.end_lineno - node.lineno + 1 <= 80, f"{path}:{node.lineno} excede 80 linhas"


def test_v2_web_assets_are_not_minified() -> None:
    for suffix in ("*.js", "*.css"):
        for path in V2.rglob(suffix):
            assert all(len(line) <= 300 for line in path.read_text(encoding="utf-8").splitlines())


def test_runtime_environment_access_stays_in_config_module() -> None:
    for path in V2.rglob("*.py"):
        if path == V2 / "core" / "config.py":
            continue
        source = path.read_text(encoding="utf-8")
        assert "os.environ" not in source and "os.getenv" not in source
