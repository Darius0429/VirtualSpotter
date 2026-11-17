import ast
from pathlib import Path

from ai_race_engineer.domains import base

PACKAGE: Path = Path(__file__).parents[1] / "src" / "ai_race_engineer"
CONTRACT: str = "ai_race_engineer.domains.base"
DOMAINS: str = "ai_race_engineer.domains"
ENGINEER: str = "ai_race_engineer.engineer"


def test_engineer_uses_only_the_contract_of_the_domains() -> None:
    """Every file of the engineer imports from the domains nothing but the names the contract module exports."""
    crossing = {
        str(path.relative_to(PACKAGE)): sorted(
            f"{module}.{name}"
            for module, name in _imports(path)
            if _within(module, DOMAINS) and (module != CONTRACT or name not in base.__all__)
        )
        for path in (PACKAGE / "engineer").rglob("*.py")
    }
    assert crossing
    assert {file: names for file, names in crossing.items() if names} == {}


def test_domains_never_import_the_engineer() -> None:
    """No file under the domains imports the engineer."""
    reaching = {
        str(path.relative_to(PACKAGE)): sorted(module for module, _ in _imports(path) if _within(module, ENGINEER))
        for path in (PACKAGE / "domains").rglob("*.py")
    }
    assert reaching
    assert {file: modules for file, modules in reaching.items() if modules} == {}


def _imports(path: Path) -> set[tuple[str, str]]:
    """What a file imports.
    In:  path  Path                  a Python file
    Out:       set[tuple[str, str]]  (dotted module, imported name; "" for a plain import of the module)
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    plain = {(alias.name, "") for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    named = {
        (node.module or "", alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    return plain | named | {(f"{module}.{name}", "") for module, name in named if module != CONTRACT}


def _within(module: str, package: str) -> bool:
    """Whether a module is a package or lies inside it.
    In:  module   str  dotted module name
         package  str  dotted package name
    Out:          bool
    """
    return module == package or module.startswith(f"{package}.")
