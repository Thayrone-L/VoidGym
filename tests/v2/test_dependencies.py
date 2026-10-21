"""Verifica as fronteiras de importação entre v1 e v2."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_only_hollow_knight_shim_imports_v1() -> None:
    package = ROOT / "src" / "voidgym"
    offenders = []
    for path in package.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        imports_v1 = re.findall(
            r"^\s*(?:from\s+hk_agent\b|import\s+hk_agent\b)", text, re.MULTILINE
        )
        if imports_v1 and "adapters/hollow_knight" not in path.as_posix():
            offenders.append(str(path.relative_to(ROOT)))
        if imports_v1 and "adapters/hollow_knight" in path.as_posix():
            for line in text.splitlines():
                if re.match(r"^\s*(?:from\s+hk_agent\b|import\s+hk_agent\b)", line):
                    assert "# V1-SHIM: portar no prompt 86" in line
    assert not offenders


def test_core_does_not_import_outer_packages() -> None:
    package = ROOT / "src" / "voidgym" / "core"
    forbidden = ("voidgym.adapters", "voidgym.envs", "voidgym.models", "voidgym.training")
    offenders = []
    for path in package.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            if f"import {name}" in text or f"from {name}" in text:
                offenders.append(f"{path.name}: {name}")
    assert not offenders


def test_v1_does_not_import_v2() -> None:
    package = ROOT / "src" / "hk_agent"
    offenders = [
        str(path.relative_to(ROOT))
        for path in package.rglob("*.py")
        if re.search(
            r"^\s*(?:from\s+voidgym\b|import\s+voidgym\b)",
            path.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    ]
    assert not offenders
