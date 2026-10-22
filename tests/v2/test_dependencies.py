import ast
from pathlib import Path


def test_v2_core_does_not_depend_on_other_layers_or_v1() -> None:
    root = Path(__file__).parents[2] / "src" / "voidgym"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        imports_v1 = any(
            (alias.name if isinstance(node, ast.Import) else node.module or "").startswith("hk_agent")
            for node in imports for alias in (node.names if isinstance(node, ast.Import) else [None])
        )
        if imports_v1:
            assert path.is_relative_to(root / "adapters" / "hollow_knight")
            assert "V1-SHIM: portar no prompt 86" in text
    core = root / "core"
    forbidden_layers = ("voidgym.adapters", "voidgym.envs", "voidgym.models", "voidgym.training")
    for path in core.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = [
            alias.name if isinstance(node, ast.Import) else node.module or ""
            for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in (node.names if isinstance(node, ast.Import) else [None])
        ]
        assert not any(name.startswith(layer) for name in imported for layer in forbidden_layers)
    v1 = Path(__file__).parents[2] / "src" / "hk_agent"
    for path in v1.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = [
            alias.name if isinstance(node, ast.Import) else node.module or ""
            for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in (node.names if isinstance(node, ast.Import) else [None])
        ]
        assert not any(name == "voidgym" or name.startswith("voidgym.") for name in imported)
